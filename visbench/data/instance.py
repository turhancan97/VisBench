"""Datasets with per-instance mask targets — step 14a-1.

Instance segmentation asks *which one*, not *what*, and that is a different
question from every target already in this codebase. A semantic label map says
"these pixels are sheep"; an instance map says "these pixels are **this**
sheep and those are **that** one". The distinction is only interesting where
two objects of one class touch, which on VOC's segmentation splits is 18.6% of
same-class instance pairs — a minority, but the majority is not free either,
because the obvious shortcut (connected components of the semantic mask) scores
a mean best-IoU of **0.1376** against ground truth: it over-segments, producing
21,819 components for 3,207 instances.

**What makes this expressible as a probe at all** is that VOC's instances are
large. Measured on all 1,449 val images at 224px on a 16x16 feature grid, the
median instance covers **16.92 patches**, 87.7% cover at least one, and — the
number that matters most — **zero of 3,207 instance pairs share a grid cell**.
So no patch is contested, and a per-patch head can carry instance information
even though the instance *index* is only annotation order and can never be a
stable output channel. The same measurement on COCO gives a median of 2.27
patches; this module is VOC-only for that reason, not by accident.

**A mask is a raster, so it resamples**, which makes this the easy half of the
geometry that :mod:`visbench.data.detection` documents at length. Boxes
transform by hand and drift silently; masks ride the same resize and crop as
their image, nearest-neighbour. This module then **derives each box from its
own cropped mask**, so a box cannot disagree with the mask it describes —
the hazard that module is built to test for is *deleted* here rather than
guarded against.

**Two files per item, and the second one is not optional.**
``SegmentationObject`` gives instance indices with no classes;
``SegmentationClass`` gives classes with no instances. An instance's class is
read from the semantic map at that instance's own pixels, and that lookup is
**exact rather than a vote**: over all 6,934 instances in train and val, every
one has exactly one class, with majority share 1.000000 in every case. So a
mixed instance means the two files disagree about the image, and this module
raises instead of taking the majority — a vote would return a confident class
for genuinely broken annotation.

**Three conventions collide in one file, all of them already in this codebase**
(CLAUDE.md, "Validity convention"):

- The palette rule: ``SegmentationObject`` PNGs are mode ``P`` and their raw
  bytes *are* the instance indices, so they are read with **no mode
  conversion**. ``convert("L")`` resolves the palette and turns instances
  ``[0, 1, 2, 255]`` into unrelated greys, which loads, trains and scores
  against supervision that means nothing. :func:`load_instance_map` therefore
  goes through :func:`~visbench.data.dense.load_label_map`, which owns that
  rule and refuses an ``RGB`` file rather than guessing a palette.
- Void is ``255`` and marks the outline VOC draws between touching objects. It
  belongs to **no instance and is not background** — scoring a prediction there
  penalises it for a pixel the annotation declines to label — so it travels as
  a separate ``ignore`` mask, the label-map convention rather than the depth
  one. It is 6.0% of pixels on average, which is far too much to fold into
  either class.
- Class ids are ``0..19`` indexing :data:`~visbench.data.detection.VOC_CLASSES`,
  **one less than the raw palette value**, because ``SegmentationClass`` uses 0
  for background. Matching the detection probe's ids is what lets a mask AP and
  a box AP over the same split be read side by side.

**On the nearest-neighbour convention, which is shared rather than chosen here.**
:meth:`VOCInstanceDataset._crop_map` resamples through
``torch.nn.functional.interpolate(mode="nearest")``, which is the call
:class:`~visbench.data.dense.DenseFolderDataset` already makes for every dense
target, while the *image* is resized by PIL. Those two disagree: torch's nearest
is left-aligned where PIL's is centre-aligned, so a target sits a sub-pixel from
its image. Measured on VOC's segmentation val, the two crops of one instance map
differ on **1.7%** of pixels (1.5% for a semantic map), always at object
boundaries, and on 28 of 60 images a one-pixel roll fits better than none — a
sub-pixel offset, not a shift.

This module keeps the shipped convention deliberately. Every published dense
number was produced under it, and the `semantic_segmentation` board reads *these
same 1,449 images*, so diverging for one probe would buy a fractionally better
alignment at the cost of the comparability that makes a shared split worth
having. The effect on what this dataset is for is small and measured: the
oracle mask mAP@50 is 0.6666 through this class against 0.6638 through a PIL
crop. Whether the shared convention should change is a question about five
existing boards, not about this one.
"""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from visbench.data.base import BaseDataset, list_files
from visbench.data.dense import load_label_map
from visbench.data.detection import VOC_CLASSES
from visbench.utils.image import load_image

__all__ = [
    "ANNOTATION_KEYS",
    "VOCInstanceDataset",
    "load_instance_map",
]

#: Raw palette value marking VOC's inter-object outline. Belongs to no instance
#: and is not background; see the module docstring.
VOID_INDEX = 255

#: Keys every annotation from :meth:`VOCInstanceDataset.target` carries, present
#: even for an image whose instances were all removed by the crop, so a consumer
#: never has to ask whether a particular image has the field. Public because it
#: is the contract; :mod:`visbench.data.detection` keeps its own copy private
#: because there it validates a *pluggable* annotation loader's output, and this
#: dict is built here rather than supplied.
ANNOTATION_KEYS = ("masks", "labels", "boxes", "ignore", "num_original")


def load_instance_map(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read an instance index map as ``(indices, void)``.

    Returns a ``(H, W)`` int64 tensor of instance indices — 0 for background,
    ``1..N`` per instance — and a ``(H, W)`` bool tensor marking the void
    outline, which is *neither* background nor any instance.

    The read goes through :func:`~visbench.data.dense.load_label_map` with
    ``ignore_index=None`` rather than opening the file here, so the palette rule
    and the refusal of ``RGB`` files have exactly one home. That call returns
    the raw values unmapped, which for a mode ``P`` file are the indices
    themselves; ``VOID_INDEX`` is split out afterwards because an instance map
    needs it as a mask rather than as ``-1``.

    Parameters
    ----------
    path:
        A ``SegmentationObject``-style PNG, or a ``.npy`` of raw indices.

    Returns
    -------
    tuple of torch.Tensor
        ``(indices, void)``, both ``(H, W)``, int64 and bool.
    """
    raw = load_label_map(path, ignore_index=None).to(torch.int64)
    void = raw == VOID_INDEX
    indices = torch.where(void, torch.zeros_like(raw), raw)
    return indices, void


class VOCInstanceDataset(BaseDataset):
    """Images paired with per-instance masks, classes and derived boxes.

    Geometry is identical to :class:`~visbench.data.dense.DenseFolderDataset`
    and :class:`~visbench.data.detection.DetectionFolderDataset` — the short
    side is resized to ``image_size`` and a centre crop taken — because the
    dataset is the only place holding both an image and its target, so it is the
    only place that can keep them aligned. Masks resample nearest-neighbour;
    boxes are then read off the cropped masks.

    Parameters
    ----------
    root:
        Dataset root, the directory holding ``SegmentationObject/``. The three
        subdirectories are resolved beneath it unless given as absolute paths.
    image_dir, instance_dir, class_dir:
        Subdirectories holding the JPEGs, the instance maps and the semantic
        maps. VOC's names are the defaults. ``class_dir`` is **required**, not
        optional: an instance map carries no classes. See the module docstring.
    stems:
        Official split list — a path to a file of one stem per line, or the
        stems themselves. VOC names segmentation split membership in
        ``ImageSets/Segmentation/*.txt`` (1,464 train / 1,449 val), so without
        this the folder is trainval rather than a split, and a train number
        would be read against images the head was fitted on. Order is
        preserved, because targets travel by index everywhere in this codebase.
    image_size:
        Side of the square crop, and so the resolution the masks are returned
        at.
    classes:
        Class names in the canonical order, defaulting to
        :data:`~visbench.data.detection.VOC_CLASSES`. Shared with the detection
        probe on purpose; the order is part of the protocol.
    min_instance_pixels:
        Instances covering fewer than this many pixels *after* the crop are
        dropped. A centre crop genuinely removes objects — on VOC val exactly
        one image of 1,449 loses all of its instances — and scoring a probe
        against an object absent from its input measures nothing. ``num_original``
        records what was there first, so "no instances" and "all instances
        dropped" stay distinguishable.
    instance_loader:
        Reader for one instance map, defaulting to :func:`load_instance_map`.
        Must return that function's ``(indices, void)`` pair.

    Notes
    -----
    ``labels()`` reads every mask file, which is unavoidable for a target scored
    per instance, and is why it is a separate call from :meth:`cache_identity` —
    that resolves cached *images* without touching an annotation.
    """

    #: Reindexed together by ``subset()``. Four lists paired by index: slicing
    #: one alone would pair an image with another image's instances, and every
    #: later step would still see equal lengths.
    _parallel_attrs = ("stems", "image_paths", "instance_paths", "class_paths")

    def __init__(
        self,
        root: str | Path,
        image_dir: str | Path = "JPEGImages",
        instance_dir: str | Path = "SegmentationObject",
        class_dir: str | Path = "SegmentationClass",
        *,
        stems: str | Path | Sequence[str] | None = None,
        image_size: int = 224,
        classes: Sequence[str] = VOC_CLASSES,
        min_instance_pixels: int = 1,
        instance_loader: Callable[[Path], tuple[torch.Tensor, torch.Tensor]] | None = None,
        name: str = "voc_instance",
        split: str = "",
    ) -> None:
        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")
        if min_instance_pixels < 1:
            raise ValueError(
                f"min_instance_pixels must be >= 1, got {min_instance_pixels}; an instance "
                "with no pixels after the crop is not in the probe's input at all."
            )

        self.root = Path(root)
        self.image_root = self._resolve(image_dir)
        self.instance_root = self._resolve(instance_dir)
        self.class_root = self._resolve(class_dir)
        self.image_size = int(image_size)
        self.classes = tuple(classes)
        self.min_instance_pixels = int(min_instance_pixels)
        self.name = name
        self.split = split
        self._instance_loader = instance_loader

        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root does not exist: {self.root}")
        for label, directory in (
            ("image_dir", self.image_root),
            ("instance_dir", self.instance_root),
            ("class_dir", self.class_root),
        ):
            if not directory.is_dir():
                raise NotADirectoryError(
                    f"{label}={directory.name!r} is not a directory under {self.root}"
                )

        images = self._index_directory(self.image_root)
        instances = self._index_directory(self.instance_root)
        semantics = self._index_directory(self.class_root)

        if stems is None:
            chosen = sorted(set(images) & set(instances) & set(semantics))
            if not chosen:
                raise ValueError(
                    f"No filename stems are shared between {self.image_root} "
                    f"({len(images)} files), {self.instance_root} ({len(instances)} files) "
                    f"and {self.class_root} ({len(semantics)} files)."
                )
        else:
            chosen = self._read_stems(stems)
            missing = {
                "images": [s for s in chosen if s not in images],
                "instance maps": [s for s in chosen if s not in instances],
                "class maps": [s for s in chosen if s not in semantics],
            }
            absent = {kind: stems_ for kind, stems_ in missing.items() if stems_}
            if absent:
                report = ", ".join(f"{len(v)} from {k}" for k, v in absent.items())
                first = next(iter(absent.values()))[:5]
                raise ValueError(
                    f"The split names stems that are missing: {report}. First few: {first}"
                )

        self.stems = list(chosen)
        self.image_paths = [images[stem] for stem in self.stems]
        self.instance_paths = [instances[stem] for stem in self.stems]
        self.class_paths = [semantics[stem] for stem in self.stems]

    # -- construction helpers ------------------------------------------------

    def _resolve(self, directory: str | Path) -> Path:
        path = Path(directory)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _index_directory(directory: Path) -> dict[str, Path]:
        """Map filename stem to path, refusing a directory with duplicate stems.

        Two files sharing a stem would make the pairing depend on iteration
        order, which is silent. Lifted from
        :class:`~visbench.data.detection.DetectionFolderDataset`, including its
        reason for indexing the whole directory even when ``stems=`` names a
        fraction of it: a stem does not say its extension, and
        :func:`~visbench.data.base.list_files` avoids a stat per entry.
        """
        index: dict[str, Path] = {}
        for path in list_files(directory):
            if path.stem in index:
                raise ValueError(
                    f"{directory} holds two files with stem {path.stem!r} "
                    f"({index[path.stem].name}, {path.name}); pairing would depend on order."
                )
            index[path.stem] = path
        return index

    @staticmethod
    def _read_stems(stems: str | Path | Sequence[str]) -> list[str]:
        """Stems from a split file (one per line) or an explicit sequence."""
        if isinstance(stems, str | Path):
            path = Path(stems)
            if not path.is_file():
                raise FileNotFoundError(f"Split file {path} does not exist")
            chosen = [line.split()[0] for line in path.read_text().splitlines() if line.strip()]
        else:
            chosen = [str(stem).strip() for stem in stems if str(stem).strip()]
        if not chosen:
            raise ValueError("The split list is empty")
        duplicates = len(chosen) - len(set(chosen))
        if duplicates:
            raise ValueError(
                f"The split list names {duplicates} stem(s) more than once, which would "
                "score those images twice."
            )
        return chosen

    # -- reading -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        """Return ``(pil_image, annotation)``, both at the working geometry."""
        image = self._crop_image(load_image(self.image_paths[index]))
        return image, self.target(index)

    def _crop_image(self, image: Image.Image) -> Image.Image:
        """Resize the short side to ``image_size``, then centre-crop.

        Kept identical to ``DenseFolderDataset._crop_image``, including the
        ``max()`` guards and BICUBIC resampling. :meth:`target` derives the mask
        geometry from what this produces, so a divergence would shift every mask
        against its image without raising.
        """
        width, height = image.size
        resized_width, resized_height = self._resized_size(width, height)
        resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        left, top = self._crop_origin(resized_width, resized_height)
        return resized.crop((left, top, left + self.image_size, top + self.image_size))

    def _resized_size(self, width: int, height: int) -> tuple[int, int]:
        """Post-resize ``(width, height)`` for an original of ``(width, height)``."""
        scale = self.image_size / min(width, height)
        return (
            max(self.image_size, round(width * scale)),
            max(self.image_size, round(height * scale)),
        )

    def _crop_origin(self, width: int, height: int) -> tuple[int, int]:
        """Top-left of the centre crop within a resized image."""
        return ((width - self.image_size) // 2, (height - self.image_size) // 2)

    def _crop_map(self, plane: torch.Tensor) -> torch.Tensor:
        """Resize and centre-crop one ``(H, W)`` index map, nearest-neighbour.

        **Nearest, never bilinear**, and for an instance map the reason is
        sharper than the depth case that set the rule: interpolating between
        instance 1 and instance 2 yields 1.5, which is not a wrong instance but
        an instance that does not exist. Averaging a mask against background is
        the same failure in the other direction — a halo of half-membership
        around every object.

        The arithmetic mirrors ``DenseFolderDataset.target`` rather than
        ``_crop_image``: PIL is given ``(width, height)`` and
        ``torch.nn.functional.interpolate`` is given ``(height, width)``, so
        writing this from the image version is how the two silently transpose on
        a non-square frame.
        """
        height, width = plane.shape[-2:]
        resized_width, resized_height = self._resized_size(width, height)
        resized = torch.nn.functional.interpolate(
            plane[None, None].float(),
            size=(resized_height, resized_width),
            mode="nearest",
        )[0, 0]
        left, top = self._crop_origin(resized_width, resized_height)
        cropped = resized[top : top + self.image_size, left : left + self.image_size]
        return cropped.to(plane.dtype)

    def target(self, index: int) -> dict[str, Any]:
        """Instances for item ``index``, at the working geometry.

        Returns a dict whose first three entries are paired by row:

        - ``masks`` ``(N, image_size, image_size)`` bool — one plane per
          instance, disjoint by construction since they come from one index map.
        - ``labels`` ``(N,)`` int64 — class ids into ``classes``, ``0..19``.
        - ``boxes`` ``(N, 4)`` float32 ``xyxy`` — the tight box of each mask, in
          the same post-transform pixel space
          :class:`~visbench.data.detection.DetectionFolderDataset` uses, and
          **derived from the mask rather than transformed alongside it**, so the
          two cannot disagree.

        and two that describe the image:

        - ``ignore`` ``(image_size, image_size)`` bool — VOC's void outline,
          belonging to no instance and not to background.
        - ``num_original`` int — instances present before the crop dropped any.

        Raises
        ------
        ValueError
            If an instance's pixels carry more than one semantic class, or none
            at all. Both mean the instance and class maps disagree about the
            image; over all 6,934 instances in VOC's train and val neither
            happens, so this is a broken-annotation signal rather than a case to
            smooth over with a majority vote.
        """
        indices, void = self._load_instance_map(index)
        semantic = load_label_map(self.class_paths[index], ignore_index=None).to(torch.int64)
        if semantic.shape != indices.shape:
            raise ValueError(
                f"{self.instance_paths[index].name} is {tuple(indices.shape)} but "
                f"{self.class_paths[index].name} is {tuple(semantic.shape)}; the instance "
                "and class maps must describe the same image."
            )

        present = [int(value) for value in torch.unique(indices) if int(value) != 0]
        num_original = len(present)

        # Classes are read at full resolution, before the crop: an instance
        # reduced to a few pixels by cropping could otherwise have its class
        # decided by whichever pixels survived.
        labels_by_instance: dict[int, int] = {}
        for value in present:
            pixels = semantic[indices == value]
            classed = pixels[(pixels != 0) & (pixels != VOID_INDEX)]
            distinct = torch.unique(classed)
            if distinct.numel() != 1:
                stem = self.stems[index]
                found = [int(v) for v in distinct[:5]]
                raise ValueError(
                    f"Instance {value} of {stem} covers {distinct.numel()} semantic "
                    f"classes {found}; expected exactly one. The instance and class maps "
                    "disagree about this image."
                )
            # Palette value 1..20 against class ids 0..19; see the module docstring.
            labels_by_instance[value] = int(distinct[0]) - 1

        cropped = self._crop_map(indices)
        ignore = self._crop_map(void.to(torch.uint8)).bool()

        masks: list[torch.Tensor] = []
        labels: list[int] = []
        for value in present:
            mask = cropped == value
            if int(mask.sum()) < self.min_instance_pixels:
                continue
            masks.append(mask)
            labels.append(labels_by_instance[value])

        stacked = (
            torch.stack(masks)
            if masks
            else torch.zeros((0, self.image_size, self.image_size), dtype=torch.bool)
        )
        return {
            "masks": stacked,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": self._boxes_from_masks(stacked),
            "ignore": ignore,
            "num_original": num_original,
        }

    @staticmethod
    def _boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
        """Tight ``xyxy`` boxes for ``(N, H, W)`` masks, as ``(N, 4)`` float32.

        Exclusive on the far edge, so a single-pixel mask at ``(r, c)`` gives
        ``(c, r, c + 1, r + 1)`` and every box has positive area. That matches
        the half-open convention a width of ``x2 - x1`` implies, which is what
        :func:`~visbench.metrics.detection.box_iou` computes.

        Derived from the mask, which is the point: a box read off the raster it
        describes cannot drift from it, so the rescale-and-shift hazard that
        :mod:`visbench.data.detection` exists to guard does not arise.
        """
        if masks.numel() == 0:
            return torch.zeros((masks.shape[0], 4), dtype=torch.float32)
        boxes = torch.zeros((masks.shape[0], 4), dtype=torch.float32)
        for row, mask in enumerate(masks):
            ys, xs = torch.nonzero(mask, as_tuple=True)
            boxes[row] = torch.tensor(
                [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max()) + 1.0,
                    float(ys.max()) + 1.0,
                ]
            )
        return boxes

    def _load_instance_map(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one instance map through the configured loader, and check it."""
        path = self.instance_paths[index]
        loader = self._instance_loader or load_instance_map
        indices, void = loader(path)
        if indices.ndim != 2 or void.shape != indices.shape:
            raise ValueError(
                f"The instance loader returned {tuple(indices.shape)} indices and "
                f"{tuple(void.shape)} void for {path.name}; both must be (H, W) and equal."
            )
        return indices, void

    def labels(self) -> list[dict[str, Any]]:
        """Annotations in index order, satisfying the :class:`BaseDataset` contract."""
        return [self.target(index) for index in range(len(self))]

    # -- identity ------------------------------------------------------------

    def cache_identity(self, index: int) -> str:
        """Token for the *image* only.

        Excludes both annotation files, exactly as the dense and detection
        datasets do: cached features depend on the image and nothing else, so
        re-exporting a mask must not invalidate an extraction that is still
        valid. The annotations appear in :meth:`fingerprint`, where the question
        is which data produced a score.
        """
        path = self.image_paths[index]
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"

    def fingerprint(self) -> str:
        """Short hash over all three file lists and the settings shaping the targets.

        ``image_size`` decides the crop and ``min_instance_pixels`` decides which
        instances survive it, so the same folder under different values is a
        different set of targets and the records must not collide.
        """
        digest = hashlib.sha256()
        digest.update(
            f"{self.name}|{self.split}|{len(self.stems)}|{self.image_size}|"
            f"{self.min_instance_pixels}|{len(self.classes)}".encode()
        )
        # strict=True: the three lists are paired by index throughout this class,
        # so unequal lengths mean the pairing is already broken. Folding a
        # truncated list into a fingerprint that still looks valid would let a
        # misaligned split train and score without complaint.
        for image_path, instance_path, class_path in zip(
            self.image_paths, self.instance_paths, self.class_paths, strict=True
        ):
            digest.update(
                f"{image_path.name}|{image_path.stat().st_size}|"
                f"{instance_path.name}|{instance_path.stat().st_size}|"
                f"{class_path.name}|{class_path.stat().st_size}".encode()
            )
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        info = super().describe()
        info["image_size"] = self.image_size
        info["min_instance_pixels"] = self.min_instance_pixels
        info["num_classes"] = len(self.classes)
        return info
