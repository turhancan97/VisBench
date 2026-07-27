"""Image-pair datasets for correspondence — v0.1.

Correspondence is the one v0.1 task that consumes pairs plus geometry rather
than single images and labels, so it gets its own interface instead of being
forced into :class:`ImageFolderDataset`.
"""

import copy
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from visbench.data.base import BaseDataset
from visbench.data.image_folder import ImageFolderDataset

__all__ = ["PairDataset", "HomographyPairDataset"]


class PairDataset(BaseDataset):
    """Yields ``(image_0, image_1, geometry)`` for correspondence evaluation.

    .. warning::
       This deliberately widens :meth:`BaseDataset.__getitem__` from a 2-tuple
       to a 3-tuple, which is a Liskov violation and is why
       :meth:`FeatureCache.extract_dataset` rejects a ``PairDataset`` outright
       rather than unpacking it. Anything written against ``BaseDataset`` would
       otherwise read ``item[0]`` and quietly discard the second view.

       It inherits from ``BaseDataset`` for ``fingerprint`` and ``describe``,
       which correspondence records need. Splitting the base into a labelled
       and an unlabelled half would be cleaner; that is a v0.3 refactor, not
       something to do underneath the dense tasks.

    ``geometry`` carries whatever the evaluation protocol needs to verify a
    match — for a depth-and-pose dataset that is depth maps, relative pose and
    intrinsics; for a keypoint dataset, annotated point pairs. Kept as an
    opaque dict here so adding a dataset does not change this interface.

    v0.1 ships :class:`HomographyPairDataset` only, because it needs no
    download. probe3d's depth-and-pose protocol (ScanNet, NAVI) arrives in
    v0.2 alongside the dense tasks that need those datasets anyway.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[Any, Any, dict]:  # type: ignore[override]
        """Return ``(pil_image_0, pil_image_1, geometry_dict)``.

        The ``type: ignore`` marks the widening described in the class
        docstring: a real Liskov violation, kept deliberately and guarded by
        ``extract_dataset`` refusing this type rather than unpacking it.
        """
        raise NotImplementedError

    def labels(self) -> list:
        """Geometry dicts in index order.

        Named ``labels`` to match :class:`BaseDataset`; for correspondence the
        "label" is the geometry a match is verified against.
        """
        raise NotImplementedError


def _solve_homography(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """3x3 homography mapping four ``source`` points onto four ``target`` points.

    The standard direct linear transform: each point pair contributes two rows
    to an 8x8 system, fixing ``h33 = 1``.
    """
    rows = []
    values = []
    for (x, y), (u, v) in zip(source.tolist(), target.tolist(), strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.extend([u, v])

    solution = torch.linalg.solve(
        torch.tensor(rows, dtype=torch.float64),
        torch.tensor(values, dtype=torch.float64),
    )
    return torch.cat([solution, torch.ones(1, dtype=torch.float64)]).reshape(3, 3)


def apply_homography(homography: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Map ``(N, 2)`` xy points through a 3x3 homography."""
    homogeneous = torch.cat([points.double(), torch.ones(len(points), 1, dtype=torch.float64)], 1)
    projected = homogeneous @ homography.double().T
    return projected[:, :2] / projected[:, 2:3].clamp(min=1e-9)


class HomographyPairDataset(PairDataset):
    """Pairs made by warping each image with a known random homography.

    The point of this dataset is that the ground truth is *exact* and free:
    every pixel's true correspondence is given by the homography, so no
    annotation, depth map or pose is needed and nothing has to be downloaded.
    That is what makes correspondence testable in v0.1 under the "small local
    image folder" constraint.

    Its limitation is equally clear and should not be papered over: a
    homography is a planar transform, so this measures viewpoint and
    scale robustness of the features, **not** 3D correspondence across genuine
    parallax. probe3d's ScanNet/NAVI protocol is the real thing and lands in
    v0.2.

    Warps are deterministic per index — same folder, same pairs, every run —
    which is what lets the feature cache work at all.
    """

    def __init__(
        self,
        root: Path,
        split: str = "test",
        max_warp: float = 0.2,
        seed: int = 0,
        image_size: int = 224,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> None:
        """Index ``root`` as a flat or class-structured image folder.

        Parameters
        ----------
        max_warp:
            Corner displacement as a fraction of image size. 0.2 gives a
            clearly non-trivial viewpoint change while keeping most of the
            image content visible in both views.
        image_size:
            Both views are centre-cropped square and resized to this before
            warping, and **the homography is expressed in that frame**.

            This is not a convenience. A backbone's ``preprocess`` resizes and
            centre-crops, so features cover only part of the original image; a
            homography written in original-image coordinates would describe a
            different plane from the one the patches sit on, and every reported
            pixel error would be wrong while still looking plausible. Emitting
            the frame the features will actually occupy makes preprocessing a
            geometric no-op. Match this to the backbone's ``image_size``.
        """
        if not 0 < max_warp < 0.5:
            raise ValueError(f"max_warp must be in (0, 0.5), got {max_warp}")
        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")

        try:
            source = ImageFolderDataset(root, split=split, labeled=True, extensions=extensions)
        except ValueError:
            source = ImageFolderDataset(root, split=split, labeled=False, extensions=extensions)

        self._source = source
        self.root = Path(root)
        self.name = self.root.name
        self.split = split
        self.max_warp = max_warp
        self.seed = seed
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self._source)

    def subset(
        self: "HomographyPairDataset", indices: int | Sequence[int]
    ) -> "HomographyPairDataset":
        """A new pair dataset over ``indices``, by subsetting the image source.

        Overridden because this class holds no per-index sequences of its own:
        it delegates length and reads to ``_source``.

        Note what a subset does to the warps. Each is drawn from ``seed`` and
        the **position**, not from the image, so a prefix (``subset(n)``, the
        ``--limit`` case) keeps exactly the warps those images had in the full
        set, while an arbitrary index list gives them the warps of their *new*
        positions. Neither is wrong — the pairs are synthetic either way, and
        ``cache_identity`` folds the seed in so nothing stale is reused — but a
        reordered subset is not a sub-experiment of the original, and should not
        be reported as one.
        """
        clone = copy.copy(self)
        clone._source = self._source.subset(indices)
        return clone

    def _to_frame(self, image: Image.Image) -> Image.Image:
        """Centre-crop square and resize to ``image_size``.

        Mirrors the standard resize-short-side + centre-crop of a backbone's
        preprocessing, so that applying it again changes nothing.
        """
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        square = image.crop((left, top, left + side, top + side))
        if square.size != (self.image_size, self.image_size):
            square = square.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        return square

    def _homography(self, index: int, width: int, height: int) -> torch.Tensor:
        """Deterministic warp for ``index``, in pixel coordinates."""
        # Seeded per index rather than from a shared stream, so item i is the
        # same whether the dataset is read in order, in reverse, or in part.
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        corners = torch.tensor(
            [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=torch.float64
        )
        scale = torch.tensor([width, height], dtype=torch.float64) * self.max_warp
        offsets = (torch.rand(4, 2, generator=generator, dtype=torch.float64) * 2 - 1) * scale
        return _solve_homography(corners, corners + offsets)

    def __getitem__(self, index: int) -> tuple[Image.Image, Image.Image, dict]:  # type: ignore[override]
        image = self._to_frame(self._source[index][0])
        width, height = image.size
        homography = self._homography(index, width, height)

        # PIL's PERSPECTIVE transform maps *output* coordinates back to input,
        # so it needs the inverse of the forward warp.
        inverse = torch.linalg.inv(homography)
        coefficients = (inverse / inverse[2, 2]).flatten()[:8].tolist()
        warped = image.transform(
            (width, height),
            Image.Transform.PERSPECTIVE,
            coefficients,
            resample=Image.Resampling.BICUBIC,
        )

        return image, warped, {"homography": homography, "size": (width, height)}

    def labels(self) -> list:
        """Geometry for every pair. Cheap: no image is decoded.

        Every view is emitted at ``image_size`` square, so the frame is known
        without opening anything.
        """
        size = (self.image_size, self.image_size)
        return [
            {"homography": self._homography(index, *size), "size": size}
            for index in range(len(self))
        ]

    def cache_identity(self, index: int) -> str | None:
        """Identity of the *source* file; the warped view is derived from it.

        The task extracts both views, so the caller must distinguish them —
        see :meth:`view_identity`.
        """
        return self._source.cache_identity(index)

    def view_identity(self, index: int, view: int) -> str:
        """Identity for one view of a pair.

        The warped image is generated, not read, so it has no file identity of
        its own. Deriving it from the source identity plus the warp parameters
        keeps the memo correct: change the source file, the seed or the warp
        magnitude, and this changes too.
        """
        base = self._source.cache_identity(index)
        return f"{base}|warp={self.seed}:{self.max_warp}:{self.image_size}:{view}"

    def fingerprint(self) -> str:
        """Source files plus the warp settings, which are part of the data."""
        digest = hashlib.sha256()
        digest.update(f"homography|{self.seed}|{self.max_warp}|{self.image_size}|".encode())
        digest.update((self._source.fingerprint() or "").encode())
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        info = super().describe()
        info["pairs"] = len(self)
        info["max_warp"] = self.max_warp
        info["image_size"] = self.image_size
        return info
