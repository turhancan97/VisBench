"""BSDS500 — images with several people's boundary annotations each.

The dataset half of the real boundary benchmark. VisBench's existing `edge`
probe is deliberately *not* this: it is dense magnitude regression on
Taskonomy's ``edge_texture`` scored by per-image Pearson correlation, and its
``protocol`` field says ``visbench_edge_regression`` for exactly this reason.
BSDS500 scores ODS/OIS/AP by matching predicted boundary pixels to *human*
ones, which is a correspondence metric and a different measurement.

``scripts/fetch_bsds500.py`` puts the tree in place::

    root/
      images/       train/ val/ test/  *.jpg
      groundTruth/  train/ val/ test/  *.mat

Three properties of this data shape the class, and all three were measured over
all 500 images rather than assumed.

**Two orientations and nothing else**: 348 images are 481x321 and 152 are
321x481. So native-resolution batching is a matter of grouping by orientation,
not of handling arbitrary sizes — see :meth:`group_by_orientation`.

**The number of annotators varies**, 4 to 9, with 5 the mode (345 images). The
annotation stack is therefore ragged *across* images and no fixed ``A`` may be
promised. :meth:`annotations` returns ``(A, H, W)`` and ``A`` is whatever that
image got.

**The annotators disagree a lot.** Per image, the densest marks a median of
**1.92x** as many boundary pixels as the sparsest, and 4.70x at the 95th
percentile. That spread is the reason the protocol credits a prediction that
matches *any* annotator, and the reason this class refuses to hand back a
single "the" boundary map: collapsing five people's judgement into one would be
a different measurement wearing the benchmark's name.

**Nothing here resizes or crops.** Every other dense dataset in VisBench
resizes the short side and centre-crops to 224 square, because a probe number
only has to be comparable with other VisBench numbers. A BSDS number's whole
purpose is comparability with the published literature, which scores at native
resolution — so a resize would quietly forfeit the only reason to add this
dataset. There is no ``image_size`` argument and adding one would be a mistake.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from visbench.data.base import BaseDataset, list_files
from visbench.data.dense import load_image

__all__ = ["BSDS500Dataset"]

#: The official split sizes, from the January 2013 package.
SPLIT_SIZES = {"train": 200, "val": 100, "test": 200}


class BSDS500Dataset(BaseDataset):
    """BSDS500 images paired with every human annotation of each.

    ``dataset[i]`` is ``(pil_image, consensus)`` where the consensus is the mean
    of the annotators' boundary maps — a float in ``[0, 1]`` giving the fraction
    of people who marked each pixel. :meth:`annotations` returns the individual
    maps, which is what an ODS/OIS/AP metric needs.

    **The consensus is a convenience, not the ground truth**, and the
    distinction matters. It exists because a training loop needs one target per
    image and the boundary-detection literature supervises on exactly this
    quantity; the *scoring* ground truth is the annotator set, and a metric that
    scored against the consensus instead would be measuring agreement with an
    average person nobody is.

    How to treat pixels where the annotators disagree — the usual convention
    ignores ``0 < consensus < threshold`` rather than calling them negative — is
    a **task** decision and is deliberately not made here. This class reports
    what people marked.

    Parameters
    ----------
    root:
        Directory holding ``images/`` and ``groundTruth/``.
    split:
        ``"train"``, ``"val"`` or ``"test"``. The official split is by image and
        is what every published number uses; do not re-partition it.
    stems:
        Restrict to these image ids, in this order. Absent ids raise, rather
        than silently scoring a smaller split than the caller asked for.
    max_images:
        Keep only the first this many, after ``stems``.
    """

    #: Three lists in lockstep. Slicing one alone would pair an image with
    #: another image's annotations, which still trains and still scores.
    _parallel_attrs: tuple[str, ...] = ("stems", "image_paths", "truth_paths")

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        stems: Any = None,
        max_images: int | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root does not exist: {self.root}")
        if split not in SPLIT_SIZES:
            raise ValueError(f"split must be one of {sorted(SPLIT_SIZES)}, got {split!r}")

        self.name = "bsds500"
        self.split = split

        image_dir = self.root / "images" / split
        truth_dir = self.root / "groundTruth" / split
        for directory in (image_dir, truth_dir):
            if not directory.is_dir():
                raise NotADirectoryError(
                    f"{directory} is missing. Run scripts/fetch_bsds500.py to populate {self.root}."
                )

        images = {path.stem: path for path in list_files(image_dir, (".jpg",))}
        truths = {path.stem: path for path in list_files(truth_dir, (".mat",))}
        if not images:
            raise ValueError(f"No .jpg images under {image_dir}")

        # An image with no annotation cannot be scored and an annotation with no
        # image cannot be predicted; either way the split is not what it claims.
        unpaired = set(images) ^ set(truths)
        if unpaired:
            shown = ", ".join(sorted(unpaired)[:5])
            raise ValueError(
                f"{len(unpaired)} stem(s) in {split} appear in images/ or groundTruth/ but "
                f"not both, e.g. {shown}. Re-run scripts/fetch_bsds500.py."
            )

        if stems is None:
            selected = sorted(images)
        else:
            selected = list(stems)
            if not selected:
                raise ValueError("stems= is empty; nothing to load")
            missing = [stem for stem in selected if stem not in images]
            if missing:
                shown = ", ".join(missing[:5])
                more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
                raise ValueError(
                    f"{len(missing)} stem(s) from stems= are absent from {split}: "
                    f"{shown}{more}. A split naming files that are not there would "
                    "silently evaluate on fewer images than it claims."
                )
        if max_images is not None:
            if max_images < 1:
                raise ValueError(f"max_images must be >= 1, got {max_images}")
            selected = selected[:max_images]

        self.stems = selected
        self.image_paths = [images[stem] for stem in self.stems]
        self.truth_paths = [truths[stem] for stem in self.stems]

    # -- reading ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        """``(pil_image, consensus)`` at the image's native resolution."""
        return load_image(self.image_paths[index]), self.target(index)

    def annotations(self, index: int) -> torch.Tensor:
        """Every annotator's boundary map for one image, ``(A, H, W)`` uint8.

        This is the scoring ground truth. ``A`` varies from 4 to 9 by image, so
        a batch of these cannot be stacked — which is correct rather than
        inconvenient, since the metric consumes one image's annotator set at a
        time anyway.
        """
        # Lazy for start-up cost only, NOT because scipy is optional -- it is a
        # core dependency, declared in pyproject.toml because this module
        # imports it directly. Elsewhere in this package a deferred import
        # means "the extra may be absent" (CLIP, timm, hub, datasets), so the
        # distinction is worth stating: `import scipy.io` costs 0.25 s of the
        # 4 s `import visbench` and only a BSDS run needs it.
        import scipy.io

        # squeeze_me / struct_as_record would reshape the 1xA cell array into
        # something whose type depends on A -- an annotator count of 1 would
        # come back unwrapped. Read it in its raw form and index explicitly.
        truth = scipy.io.loadmat(self.truth_paths[index])["groundTruth"]
        maps = [
            np.asarray(truth[0, annotator][0, 0]["Boundaries"], dtype=np.uint8)
            for annotator in range(truth.shape[1])
        ]
        return torch.from_numpy(np.stack(maps))

    def target(self, index: int) -> torch.Tensor:
        """The consensus map, ``(H, W)`` float32 in ``[0, 1]``.

        The fraction of annotators who marked each pixel: 1.0 where everyone
        agreed there is a boundary, 0.0 where nobody did, and a fraction in
        between where they differed — which, on this data, is most of the
        boundary. See the class docstring on why this is not the ground truth.
        """
        return self.annotations(index).to(torch.float32).mean(dim=0)

    def labels(self) -> list:
        """Targets are read per index; there is no label vector to return."""
        return list(range(len(self)))

    # -- geometry --------------------------------------------------------------

    def size(self, index: int) -> tuple[int, int]:
        """``(width, height)`` of one image, read from the header rather than decoded."""
        from PIL import Image

        with Image.open(self.image_paths[index]) as image:
            return image.size

    def group_by_orientation(self) -> dict[tuple[int, int], list[int]]:
        """Indices grouped by ``(width, height)``, for same-size batching.

        Every image here is 481x321 or 321x481, so this returns at most two
        groups and a caller can build one loader per group. A batch mixing the
        two cannot be collated, and the alternative — rotating the portrait
        images to match — would change what is being measured.
        """
        groups: dict[tuple[int, int], list[int]] = {}
        for index in range(len(self)):
            groups.setdefault(self.size(index), []).append(index)
        return groups

    # -- identity --------------------------------------------------------------

    def cache_identity(self, index: int) -> str:
        """``"<abs path>|<size>|<mtime_ns>"``, over the *image* — what gets encoded.

        The annotation file is deliberately absent: the feature cache stores
        what the backbone computed from the pixels, and re-annotating an image
        would not change its features.
        """
        path = self.image_paths[index]
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"

    def fingerprint(self) -> str:
        """Short hash over the stems and both file lists' sizes.

        The annotation sizes are in it although the images' identity alone would
        name the split, because two runs over the same images and *different*
        annotations are different measurements and a record must not present
        them as one.
        """
        digest = hashlib.sha256()
        digest.update(f"{self.name}|{self.split}|{len(self.stems)}".encode())
        for stem, image, truth in zip(self.stems, self.image_paths, self.truth_paths, strict=True):
            digest.update(f"{stem}|{image.stat().st_size}|{truth.stat().st_size}".encode())
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        info = super().describe()
        info["annotators"] = "per_image"
        info["geometry"] = "native"
        return info
