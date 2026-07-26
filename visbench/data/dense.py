"""Datasets with a dense per-pixel target — v0.2.

The thing this module exists to get right is **geometry**. Everywhere else in
VisBench a dataset hands over a PIL image and the backbone's ``preprocess``
decides how to resize and crop it. That cannot work here: a depth map has to
survive exactly the same resize and crop as the image it belongs to, and the
dataset is the only place that holds both.

So a dense dataset applies the geometry itself and emits an image already at
the working resolution. The backbone's ``preprocess`` then resizes a square
image to the same square size — geometrically a no-op — and still applies its
own normalisation, which stays a backbone property.

This is the same resolution :class:`~visbench.data.pair_dataset.PairDataset`
reached for correspondence, and for the same reason: a homography expressed in
original pixels while features came from a 224 centre crop scored recall@1px at
0.003. A misaligned depth target fails less visibly — the numbers merely come
out bad — which is worse.
"""

import hashlib
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from PIL import Image

from visbench.data.base import BaseDataset
from visbench.utils.image import load_image

__all__ = ["DenseFolderDataset", "load_depth_map", "load_normal_map", "load_mask"]

#: Target file suffixes understood without a custom ``target_loader``.
_TARGET_SUFFIXES = (".npy", ".png", ".tiff", ".tif")


def load_depth_map(path: Path, scale: float = 1.0) -> torch.Tensor:
    """Read a depth map as a ``(H, W)`` float32 tensor in metres.

    ``.npy`` is taken at face value. A 16-bit PNG or TIFF is read as integers
    and divided by ``scale`` — the convention depth datasets use to store
    millimetres in an image container, and the reason ``scale`` has no default
    that could quietly be wrong for a new dataset: NYUv2 ships 1000, but
    nothing in the file says so.

    Invalid pixels stay 0, which is what every metric here treats as "no
    ground truth".
    """
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as handle:
            array = np.array(handle)

    if array.ndim != 2:
        raise ValueError(
            f"{path.name} holds a {array.ndim}D array {array.shape}; a depth map is 2D. "
            "Pass target_loader= for a layout this does not cover."
        )
    depth = torch.from_numpy(array.astype(np.float32))
    if scale != 1.0:
        depth = depth / scale
    return depth


def load_normal_map(path: Path) -> torch.Tensor:
    """Read a surface-normal map as a ``(3, H, W)`` float32 tensor.

    ``.npy`` is taken at face value, in whatever layout it is stored —
    ``(3, H, W)`` or ``(H, W, 3)``, disambiguated by which axis has length 3.
    An image file is read as 8-bit RGB and mapped ``2 * v / 255 - 1``, the
    encoding GeoNet and every other published NYU normal set uses.

    Vectors are L2-normalised, and any whose length is too small to have a
    meaningful direction becomes exactly ``(0, 0, 0)`` — the convention the
    metric and the loss both read as "no ground truth here", matching the role
    a zero plays in a depth map. That threshold is not fussy: in the 8-bit
    encoding an invalid pixel is stored as ``(128, 128, 128)``, decoding to a
    length of about 0.007, while a genuine unit normal decodes to within a
    percent of 1.
    """
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
        normals = torch.from_numpy(array)
    else:
        with Image.open(path) as handle:
            rgb = handle.convert("RGB")
            array = np.array(rgb).astype(np.float32)
        normals = torch.from_numpy(array) * (2.0 / 255.0) - 1.0

    if normals.ndim != 3:
        raise ValueError(
            f"{path.name} holds a {normals.ndim}D array {tuple(normals.shape)}; a normal map "
            "is 3D. Pass target_loader= for a layout this does not cover."
        )
    if normals.shape[0] != 3:
        if normals.shape[-1] != 3:
            raise ValueError(
                f"{path.name} has shape {tuple(normals.shape)}; expected a length-3 axis "
                "holding the x, y, z components."
            )
        normals = normals.permute(2, 0, 1)

    length = normals.norm(dim=0, keepdim=True)
    return torch.where(length > 0.1, normals / length.clamp(min=1e-8), torch.zeros_like(normals))


def load_mask(path: Path, ignore_index: Optional[int] = None) -> torch.Tensor:
    """Read a binary segmentation mask as a ``(H, W)`` float32 tensor of 0 and 1.

    ``.npy`` is taken at face value; an image file is read as 8-bit greyscale,
    which covers both conventions in circulation — a mask stored as 0/1 and one
    stored as 0/255 — because the rule is simply **non-zero is foreground**. No
    normalisation and no scaling: a mask is a label, not a measurement, and
    dividing it by 255 would turn every foreground pixel into 1/255 and train
    the probe to predict background everywhere.

    For the same reason, do not pass ``max_target`` to
    :class:`DenseFolderDataset` for masks. It exists to mark out-of-range
    *sensor* readings invalid; against a label map it would silently erase the
    foreground class.

    Parameters
    ----------
    ignore_index:
        Raw value marking unlabelled pixels — 255 in the VOC-style palette
        masks, where it outlines objects. Those pixels come back as ``-1``,
        which :func:`~visbench.metrics.dense.binary_iou` and the task's loss
        both read as "no ground truth here". Left ``None`` by default: for a
        plain foreground/background mask every pixel *is* labelled, and
        inventing an ignore region would quietly shrink what the probe is
        scored on. Bind it with ``functools.partial(load_mask,
        ignore_index=255)``.
    """
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as handle:
            array = np.array(handle.convert("L"))

    if array.ndim != 2:
        raise ValueError(
            f"{path.name} holds a {array.ndim}D array {array.shape}; a binary mask is 2D. "
            "Pass target_loader= for a layout this does not cover."
        )

    raw = torch.from_numpy(array)
    mask = (raw != 0).float()
    if ignore_index is not None:
        mask = torch.where(raw == ignore_index, torch.full_like(mask, -1.0), mask)
    return mask


class DenseFolderDataset(BaseDataset):
    """Images and their per-pixel targets, in two parallel folders.

    ::

        root/
          images/  scene_0001.jpg ...
          depths/  scene_0001.npy ...

    Pairing is by **stem**, not by position: two sorted listings can silently
    drift apart when one folder gains a file, and a depth map paired with the
    wrong image produces a plausible bad number rather than an error. A stem
    present in one folder and not the other raises.
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_dir: str = "images",
        target_dir: str = "depths",
        split: str = "train",
        image_size: int = 224,
        target_scale: float = 1.0,
        max_target: Optional[float] = None,
        target_loader: Optional[Callable[[Path], torch.Tensor]] = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> None:
        """Index the folder pair.

        Parameters
        ----------
        image_size:
            Working resolution. Both image and target are resized on their short
            side and centre-cropped to this, together. Must suit the backbone:
            a ViT needs a multiple of its patch size.
        target_scale:
            Divisor applied to integer target files — 1000 for a dataset storing
            millimetres in a 16-bit PNG. Ignored for ``.npy``.
        max_target:
            Values above this are set to 0, i.e. marked invalid. Depth sensors
            report garbage beyond their range; probe3d applies the same cap at
            10 m for NYUv2, inside its loss. Applying it once here means the
            training loss and the reported metric mask identically, which they
            must, or the probe is optimised against pixels it is not scored on.
        """
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root does not exist: {self.root}")

        image_root = self.root / image_dir
        target_root = self.root / target_dir
        for label, directory in (("image_dir", image_root), ("target_dir", target_root)):
            if not directory.is_dir():
                raise NotADirectoryError(
                    f"{label}={directory.name!r} is not a directory under {self.root}"
                )

        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")
        if max_target is not None and max_target <= 0:
            raise ValueError(f"max_target must be positive, got {max_target}")

        self.name = self.root.name
        self.split = split
        self.image_size = image_size
        self.target_scale = target_scale
        self.max_target = max_target
        self.extensions = tuple(ext.lower() for ext in extensions)
        self._target_loader = target_loader

        images = {
            path.stem: path
            for path in sorted(image_root.iterdir())
            if path.is_file() and path.suffix.lower() in self.extensions
        }
        targets = {
            path.stem: path
            for path in sorted(target_root.iterdir())
            if path.is_file() and path.suffix.lower() in _TARGET_SUFFIXES
        }
        if not images:
            raise ValueError(f"No images with extensions {self.extensions} under {image_root}")

        unmatched = sorted(set(images) ^ set(targets))
        if unmatched:
            shown = ", ".join(unmatched[:5])
            more = f" (and {len(unmatched) - 5} more)" if len(unmatched) > 5 else ""
            raise ValueError(
                f"{len(unmatched)} file stem(s) appear in only one of {image_dir}/ and "
                f"{target_dir}/: {shown}{more}. Pairing is by stem, so a partial overlap "
                "would silently drop or mismatch data."
            )

        self.stems = sorted(images)
        self.image_paths = [images[stem] for stem in self.stems]
        self.target_paths = [targets[stem] for stem in self.stems]

    # -- reading -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        """Return ``(pil_image, target)``, both already at the working geometry."""
        image = self._crop_image(load_image(self.image_paths[index]))
        return image, self.target(index)

    def _crop_image(self, image: Image.Image) -> Image.Image:
        """Resize the short side to ``image_size``, then centre-crop."""
        width, height = image.size
        scale = self.image_size / min(width, height)
        resized = image.resize(
            (
                max(self.image_size, round(width * scale)),
                max(self.image_size, round(height * scale)),
            ),
            Image.Resampling.BICUBIC,
        )
        return self._centre_crop(resized)

    def _centre_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        left = (width - self.image_size) // 2
        top = (height - self.image_size) // 2
        return image.crop((left, top, left + self.image_size, top + self.image_size))

    def target(self, index: int) -> torch.Tensor:
        """The target for item ``index``, at the working resolution.

        Shape follows the loader: ``(image_size, image_size)`` for a scalar map
        such as depth, ``(C, image_size, image_size)`` for a vector one such as
        surface normals.

        Resampled **nearest-neighbour**, never bilinear. Interpolating a depth
        map averages across depth discontinuities, inventing surfaces at object
        boundaries that no sensor saw; worse, it averages valid pixels with the
        zeros marking holes, turning a sharp invalid region into a halo of
        plausible-looking wrong depths that the valid mask no longer excludes.
        A normal map has the same problem twice over — averaging two unit
        vectors across an edge gives a direction that is not merely wrong but
        not even unit length.
        """
        path = self.target_paths[index]
        if self._target_loader is not None:
            target = self._target_loader(path)
        else:
            target = load_depth_map(path, scale=self.target_scale)

        if target.ndim not in (2, 3):
            raise ValueError(
                f"target_loader returned {target.ndim}D for {path.name}; expected (H, W) "
                "for a scalar map or (C, H, W) for a vector one"
            )
        scalar = target.ndim == 2
        channelled = target[None] if scalar else target

        height, width = channelled.shape[-2:]
        scale = self.image_size / min(height, width)
        resized = torch.nn.functional.interpolate(
            channelled[None],
            size=(
                max(self.image_size, round(height * scale)),
                max(self.image_size, round(width * scale)),
            ),
            mode="nearest",
        )[0]

        top = (resized.shape[-2] - self.image_size) // 2
        left = (resized.shape[-1] - self.image_size) // 2
        cropped = resized[..., top : top + self.image_size, left : left + self.image_size]

        if self.max_target is not None:
            if not scalar:
                raise ValueError(
                    f"max_target={self.max_target} caps a scalar quantity, but "
                    f"{path.name} holds a {cropped.shape[0]}-channel map. Mark invalid "
                    "pixels inside target_loader instead, where the convention for this "
                    "target type is known."
                )
            # Marked invalid, not clamped: a pixel beyond the sensor's range is
            # unknown, and clamping it to the cap would train and score against
            # a wall of fabricated depth at exactly max_target.
            cropped = torch.where(cropped > self.max_target, torch.zeros_like(cropped), cropped)
        return cropped[0] if scalar else cropped

    def targets(self) -> torch.Tensor:
        """Every target stacked, ``(N, ...)`` over :meth:`target`'s shape.

        Reads every target file. That is unavoidable — a dense task is scored
        per pixel — but it is why this is a separate call from
        :meth:`cache_identity`, which resolves cached *images* without touching
        anything.
        """
        return torch.stack([self.target(index) for index in range(len(self))])

    def labels(self) -> list:
        """Targets in index order, satisfying the :class:`BaseDataset` contract."""
        return [self.target(index) for index in range(len(self))]

    # -- identity ------------------------------------------------------------

    def cache_identity(self, index: int) -> str:
        """Token for the *image* only.

        Deliberately excludes the target: cached features depend on the image
        and nothing else, so editing a depth map must not invalidate an
        extraction that is still perfectly valid. The target does appear in
        :meth:`fingerprint`, where the question is which data produced a score.
        """
        path = self.image_paths[index]
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"

    def fingerprint(self) -> str:
        """Short hash over both file lists, plus the geometry that shaped them.

        ``image_size`` is folded in because it is not a presentation detail
        here: it decides the crop, so the same folder at two resolutions is two
        different sets of targets and the records must not collide.
        """
        digest = hashlib.sha256()
        digest.update(
            f"{self.name}|{self.split}|{len(self.stems)}|{self.image_size}|"
            f"{self.target_scale}|{self.max_target}".encode()
        )
        for image_path, target_path in zip(self.image_paths, self.target_paths):
            digest.update(
                f"{image_path.name}|{image_path.stat().st_size}|"
                f"{target_path.name}|{target_path.stat().st_size}".encode()
            )
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        info = super().describe()
        info["image_size"] = self.image_size
        return info
