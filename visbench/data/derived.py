"""Targets computed from the image itself — no target folder, no download.

Every dense probe before this one reads a target somebody else rendered:
Taskonomy's ``edge_texture``, NYUv2's depth, VOC's label maps. A **derived**
target is computed from the RGB frame at read time, which changes two things
that matter and one that does not.

**It removes the alignment hazard structurally, rather than by testing for it.**
The standing rule for dense data is that image and target must survive the same
resize and crop, and every dataset here pays for that separately —
:class:`~visbench.data.dense.DenseFolderDataset` resamples targets
nearest-neighbour, :class:`~visbench.data.detection.DetectionFolderDataset`
transforms boxes by the *achieved* ratio, and the correspondence task paid for a
misalignment bug once at ``recall@1px`` = 0.003. A derived target is computed
**after** the crop, from the exact array the backbone is about to see, so there
is no second geometry to keep in agreement and no resampling of the response at
all. That is the single strongest argument for this class of target.

**It makes the generator part of the protocol.** A stored target is identified
by the dataset it came from; a derived one is identified only by the code that
computed it, so ``sigma``, the gradient kernel, the compression and its scale
all have to travel with the number. They land in ``dataset_params`` via
:meth:`ShiTomasiResponse.describe`, which is what puts two settings of the same
operator into two comparability groups automatically — the same mechanism that
keeps a pixel-unit correspondence record apart from a patch-unit one (step 6f).

**It does not make the target arbitrary.** The response below is a fixed,
deterministic function of the pixels: no sampling, no learned component, no
platform-dependent kernel. Two machines running the same version get the same
target to the bit, which is what a benchmark needs and what the fingerprint
asserts.

See ``visbench/tasks/low_level/README.md`` for why this operator and not
Harris's ``R``, and for the measurements that chose the compression.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from visbench.data.base import BaseDataset, list_files
from visbench.data.dense import load_image

__all__ = ["ShiTomasiResponse", "DerivedTargetDataset", "structure_tensor"]

#: Luminance weights (ITU-R BT.601), the convention PIL's ``convert("L")`` uses.
#: Spelled out rather than delegated because the target is defined by this
#: number and a future Pillow could reasonably change its own.
_LUMA = (0.299, 0.587, 0.114)

#: Sobel, normalised so the response is per unit intensity rather than per unit
#: intensity times eight. The normalisation is arbitrary but it is *fixed*, and
#: it is folded into the recorded scale, so changing it would change the target.
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0


def _gaussian_kernel(sigma: float) -> tuple[torch.Tensor, int]:
    """A normalised 1D Gaussian and its radius, truncated at three sigma."""
    radius = max(1, int(3.0 * sigma))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(offsets**2) / (2.0 * sigma**2))
    return kernel / kernel.sum(), radius


def structure_tensor(gray: torch.Tensor, sigma: float) -> tuple[torch.Tensor, ...]:
    """The Gaussian-windowed structure tensor of a ``(1, 1, H, W)`` grey image.

    Returns ``(Ixx, Ixy, Iyy)``, each the same shape. Separated into its own
    function because both eigenvalues are derived from it and a test asserts the
    derivation against a hand-built case — an operator whose only test is the
    probe's score is an operator nobody has checked.

    Padding is ``replicate`` throughout. Zero padding would manufacture a strong
    intensity step around the whole frame, which for *this* target is not a
    cosmetic border artifact: a step is exactly what the operator responds to,
    so the outermost pixels would carry a fabricated response that the probe
    would then be trained and scored on.
    """
    if gray.ndim != 4 or gray.shape[:2] != (1, 1):
        raise ValueError(f"expected a (1, 1, H, W) grey image, got {tuple(gray.shape)}")

    sobel_x = _SOBEL_X.to(gray.dtype).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded, sobel_x)
    grad_y = F.conv2d(padded, sobel_y)

    kernel, radius = _gaussian_kernel(sigma)
    kernel = kernel.to(gray.dtype)
    row = kernel.view(1, 1, 1, -1)
    column = kernel.view(1, 1, -1, 1)

    def smooth(value: torch.Tensor) -> torch.Tensor:
        value = F.conv2d(F.pad(value, (radius, radius, 0, 0), mode="replicate"), row)
        return F.conv2d(F.pad(value, (0, 0, radius, radius), mode="replicate"), column)

    return smooth(grad_x * grad_x), smooth(grad_x * grad_y), smooth(grad_y * grad_y)


@dataclass(frozen=True)
class ShiTomasiResponse:
    """Shi-Tomasi cornerness — the smaller eigenvalue of the structure tensor.

    Corner strength is "intensity varies in *both* directions here", which is
    the smaller eigenvalue :math:`\\lambda_{min}`: large only where neither
    direction is flat. A straight edge has one large eigenvalue and one near
    zero, so it scores low, and a flat region scores near zero in both.

    **Not Harris's** :math:`R = \\det(M) - k\\,\\mathrm{tr}(M)^2`, and the reason
    is measured rather than stylistic. Harris ``R`` is *signed* — strongly
    negative on edges — so using it as a magnitude needs either a clip, which
    throws away the edge information and leaves the sparsest target of the three
    tried, or an absolute value, which conflates corners with edges and stops
    the target meaning what its name says. Tail mass in the strongest 1% of
    pixels, over 60 Taskonomy frames at 224px, against a working range of ~0.10:

    ==========================  =========
    response                    tail@1%
    ==========================  =========
    Harris ``R``, clipped at 0  0.52
    Harris ``|R|``              0.33
    :math:`\\lambda_{min}`       **0.27**
    ==========================  =========

    All three are too concentrated raw — see ``transform``. :math:`\\lambda_{min}`
    starts closest, is non-negative by construction, and carries **no** ``k``,
    which removes one of the free parameters that makes "Harris corners" a
    family rather than a definition.

    Parameters
    ----------
    sigma:
        Standard deviation, in pixels, of the Gaussian window the tensor is
        summed over. This is the scale the operator sees; it is not a smoothing
        nicety, and 1.0 and 2.0 give measurably different targets.
    transform:
        ``"log1p"`` or ``"none"``. The raw response is far too concentrated for
        the magnitude protocol — at a tail of 0.27 it is closer to
        ``edge_occlusion``'s 0.46, which scored 0.088 and could not separate two
        backbones, than to the ~0.10 of the two probes that work.
    scale:
        Multiplier applied *before* the compression, which is what sets both the
        resulting tail and the target's magnitude. Both matter and they are not
        independent: 6d-1 measured that an L1 loss needs a target of order 1,
        because ``sign()`` gradients do not shrink with the target. At
        ``sigma=2.0`` the default lands tail 0.089 and frame mean 0.593, against
        the ~0.10 and the 0.717 knee those two findings ask for.

        =========  =========  ======
        scale      tail@1%    mean
        =========  =========  ======
        1e2        0.234      0.028
        1e3        0.156      0.166
        **1e4**    **0.089**  0.593
        1e5        0.053      1.457
        =========  =========  ======
    """

    sigma: float = 2.0
    transform: str = "log1p"
    scale: float = 1e4

    #: Named in the record's ``protocol``. The operator, not the family: two
    #: runs differing in sigma or scale are *not* comparable, and rather than
    #: relying on a reader to notice, the settings go into ``dataset_params``
    #: where the comparability key picks them up on its own.
    operator = "shi_tomasi"

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.transform not in ("log1p", "none"):
            raise ValueError(f"transform must be 'log1p' or 'none', got {self.transform!r}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """The response for one PIL image, as ``(H, W)`` float32.

        The image is expected to be **already at the working geometry**. This
        does no resizing and no cropping by design: see the module docstring.
        """
        array = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)
        grey = (
            _LUMA[0] * array[..., 0] + _LUMA[1] * array[..., 1] + _LUMA[2] * array[..., 2]
        ).view(1, 1, *array.shape[:2])

        ixx, ixy, iyy = structure_tensor(grey, self.sigma)
        trace = ixx + iyy
        determinant = ixx * iyy - ixy * ixy
        # Clamped before the square root only to absorb float error: the
        # discriminant of a real symmetric 2x2 is non-negative exactly.
        discriminant = torch.clamp(trace * trace - 4.0 * determinant, min=0.0).sqrt()
        lambda_min = ((trace - discriminant) / 2.0).clamp(min=0.0)

        response = lambda_min.squeeze(0).squeeze(0)
        if self.transform == "log1p":
            return torch.log1p(self.scale * response)
        return self.scale * response

    def describe(self) -> dict[str, Any]:
        """Everything that changes the target, for ``dataset_params``.

        A derived target has no file to point at, so this *is* its identity. It
        is deliberately exhaustive — the gradient kernel and the luminance
        weights are named even though neither is configurable, because a record
        that omits them would silently mean something else if either changed.
        """
        return {
            "target_operator": self.operator,
            "target_sigma": self.sigma,
            "target_transform": self.transform,
            "target_scale": self.scale,
            "target_gradient": "sobel3_over8",
            "target_luma": "bt601",
        }

    def token(self) -> str:
        """Compact stable string for the fingerprint."""
        parts = "|".join(f"{k}={v}" for k, v in sorted(self.describe().items()))
        return hashlib.sha256(parts.encode()).hexdigest()[:12]


class DerivedTargetDataset(BaseDataset):
    """An image folder whose targets are computed from the images.

    ::

        root/
          images/  scene_0001.jpg ...

    One folder, not two. There is nothing to pair by stem and therefore nothing
    that can drift out of pairing, which is the whole class of bug
    :class:`~visbench.data.dense.DenseFolderDataset` spends its constructor
    guarding against.

    The geometry is deliberately identical to that class's — short side resized
    bicubic, then centre-cropped — because a corner number and an edge number
    over the same frames should differ in the operator and nothing else.
    """

    #: Only two, and they are not really at risk: without a target list there is
    #: no second sequence to fall out of step with. Declared so
    #: :meth:`BaseDataset.subset` works, which the CLI's ``--limit`` needs.
    _parallel_attrs: tuple[str, ...] = ("stems", "image_paths")

    def __init__(
        self,
        root: str | Path,
        image_dir: str = "images",
        split: str = "train",
        image_size: int = 224,
        generator: ShiTomasiResponse | None = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
        stems: Sequence[str] | None = None,
        max_images: int | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root does not exist: {self.root}")
        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")

        self.name = self.root.name
        self.split = split
        self.image_size = image_size
        self.generator = generator if generator is not None else ShiTomasiResponse()
        self.extensions = tuple(ext.lower() for ext in extensions)

        image_root = self.root / image_dir
        if not image_root.is_dir():
            raise NotADirectoryError(
                f"image_dir={image_dir!r} is not a directory under {self.root}"
            )
        found = {path.stem: path for path in list_files(image_root, self.extensions)}
        if not found:
            raise ValueError(f"No images with extensions {self.extensions} under {image_root}")

        if stems is None:
            selected = sorted(found)
        else:
            selected = list(stems)
            if not selected:
                raise ValueError("stems= is empty; nothing to load")
            missing = [stem for stem in selected if stem not in found]
            if missing:
                shown = ", ".join(missing[:5])
                more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
                raise ValueError(
                    f"{len(missing)} stem(s) from stems= are absent from {image_dir}/: "
                    f"{shown}{more}. A split naming files that are not there would "
                    "silently evaluate on fewer images than it claims."
                )
        if max_images is not None:
            if max_images < 1:
                raise ValueError(f"max_images must be >= 1, got {max_images}")
            selected = selected[:max_images]

        self.stems = selected
        self.image_paths = [found[stem] for stem in self.stems]

    # -- reading -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        """``(pil_image, target)``, the target computed from that exact image."""
        image = self._crop_image(load_image(self.image_paths[index]))
        return image, self.generator(image)

    def _crop_image(self, image: Image.Image) -> Image.Image:
        """Resize the short side to ``image_size``, then centre-crop.

        Byte-identical to :meth:`DenseFolderDataset._crop_image`. Kept as a copy
        rather than inherited because this class does not extend that one — it
        has no target folder — and a test asserts the two agree pixel for pixel
        on a non-square image, which is the same guarantee step 6c-1 gave the
        detection dataset's crop.
        """
        width, height = image.size
        scale = self.image_size / min(width, height)
        resized = image.resize(
            (
                max(self.image_size, round(width * scale)),
                max(self.image_size, round(height * scale)),
            ),
            Image.Resampling.BICUBIC,
        )
        left = (resized.width - self.image_size) // 2
        top = (resized.height - self.image_size) // 2
        return resized.crop((left, top, left + self.image_size, top + self.image_size))

    def target(self, index: int) -> torch.Tensor:
        """``(image_size, image_size)``, computed from the cropped image.

        Note what is *absent*: no resize, no nearest-neighbour resampling, no
        ``max_target``. The target is generated at the working resolution, so
        there is no second geometry and nothing to keep in step.
        """
        image, target = self[index]
        del image
        return target

    def targets(self) -> torch.Tensor:
        """Every target stacked, ``(N, H, W)``. Decodes every image."""
        return torch.stack([self.target(index) for index in range(len(self))])

    def labels(self) -> list:
        return [self.target(index) for index in range(len(self))]

    # -- identity ------------------------------------------------------------

    def cache_identity(self, index: int) -> str:
        """Token for the image. Excludes the generator, which is correct.

        Cached *features* depend on the image alone; changing the corner
        operator must not invalidate an extraction that is still valid. The
        generator does appear in :meth:`fingerprint`, where the question is
        which data produced a score, and there it is essential — two operators
        over one folder are two different targets.
        """
        path = self.image_paths[index]
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            f"{self.name}|{self.split}|{len(self.stems)}|{self.image_size}|"
            f"{self.generator.token()}".encode()
        )
        for path in self.image_paths:
            digest.update(f"{path.name}|{path.stat().st_size}".encode())
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        info = super().describe()
        info["image_size"] = self.image_size
        info.update(self.generator.describe())
        return info
