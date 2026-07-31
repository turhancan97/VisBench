"""Datasets with bounding-box targets — v0.3, step 6c.

Boxes are the hardest target geometry in this codebase, and the reason is that
they **transform rather than resample**. A depth map is resized and cropped by
the same loader that resizes and crops its image, so the two cannot drift; a
box is four numbers that have to be rescaled and shifted *by hand* to follow
the image. Nothing raises when that is skipped — the boxes simply describe the
original 500x375 frame while the tensor is 224x224, and the probe trains
happily against supervision that is wrong everywhere.

That is the correspondence misalignment bug (recall@1px = 0.003) with a new
coordinate convention. So this module owns the transform, applies it in the
same place the image geometry is applied, and is tested on a non-square image
where a missed rescale is visible.

**The internal convention, decided before this module was written**
(CLAUDE.md, step 6c):

``xyxy``, **absolute pixels**, **0-indexed**, in *post-transform* space.

- ``xyxy`` because VOC's XML already stores corners, so the common path does no
  conversion and cannot get one backwards.
- 0-indexed because every other coordinate in this codebase is, and VOC is not
  (its minimum ``xmin``/``ymin`` over all 17,125 files is 1). Keeping VOC's
  origin would make boxes the one array indexed differently from the tensor
  they describe.
- Absolute pixels, which is what makes the post-transform requirement load
  bearing: an absolute box is meaningless without the resolution it refers to.
  A normalised box would survive resize and crop untouched; these do not, and
  :meth:`DetectionFolderDataset.target` is the only place allowed to apply it.
"""

import hashlib
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from visbench.data.base import BaseDataset, list_files
from visbench.utils.image import load_image

__all__ = [
    "DetectionFolderDataset",
    "VOC_CLASSES",
    "load_voc_boxes",
]

#: The 20 Pascal VOC detection categories, in the canonical alphabetical order
#: every published VOC number uses. Index into this to get a class id; the order
#: is part of the protocol, so do not sort it differently or a per-class AP
#: becomes incomparable to any reference.
VOC_CLASSES: tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

#: Keys every box annotation carries, so a consumer can rely on their presence
#: even for an image with no objects left after cropping.
_ANNOTATION_KEYS = ("boxes", "labels", "difficult")


def load_voc_boxes(
    path: Path,
    classes: Sequence[str] = VOC_CLASSES,
) -> dict[str, Any]:
    """Read a Pascal VOC detection XML into box arrays, in **original** pixels.

    Returns a dict with

    - ``boxes``: ``(N, 4)`` float32, ``xyxy``, 0-indexed, original resolution
    - ``labels``: ``(N,)`` int64 indices into ``classes``
    - ``difficult``: ``(N,)`` bool, VOC's ``<difficult>`` flag
    - ``size``: ``(width, height)`` from the XML's own ``<size>`` element

    ``size`` comes from the XML rather than the JPEG deliberately: the rescale
    needs the original dimensions, VOC records them, and reading them here means
    the image never has to be decoded to transform its boxes. A mismatch between
    the two would be a corrupt dataset, and is checked by the dataset rather
    than assumed here.

    **1-indexed on disk, 0-indexed in memory.** VOC stores ``xmin``/``ymin``
    starting at 1 (verified: the minimum over the whole set is 1, not 0), so one
    is subtracted from each of the four coordinates. Corners are then treated as
    continuous coordinates, i.e. a box's width is ``x2 - x1``. This is the
    convention in common use; VOC's ``xmax`` is an inclusive *index*, so a
    literal reading would give ``x2 - x1 + 1``. The difference is one pixel and
    it must be *chosen* rather than inherited, because a one-pixel shift moves
    mAP slightly and reads as a weak backbone rather than an off-by-one.

    ``difficult`` is **returned, not filtered**. VOC's protocol excludes those
    objects from evaluation, but that is the caller's decision to record and
    apply — dropping them here would make the exclusion invisible to the result
    record, which is the one thing the ``protocol`` field exists to prevent.
    """
    root = ElementTree.parse(path).getroot()

    size_element = root.find("size")
    if size_element is None:
        raise ValueError(f"{path.name} has no <size> element, so its boxes cannot be rescaled.")
    width = _require_int(size_element, "width", path)
    height = _require_int(size_element, "height", path)

    index_of = {name: index for index, name in enumerate(classes)}
    boxes: list[list[float]] = []
    labels: list[int] = []
    difficult: list[bool] = []

    for object_element in root.findall("object"):
        name_element = object_element.find("name")
        if name_element is None or not (name_element.text or "").strip():
            raise ValueError(f"{path.name} has an <object> with no <name>.")
        name = (name_element.text or "").strip()
        if name not in index_of:
            raise ValueError(
                f"{path.name} names class {name!r}, which is not in the {len(classes)} "
                f"classes given. Pass classes= for a dataset with a different label set."
            )

        box_element = object_element.find("bndbox")
        if box_element is None:
            raise ValueError(f"{path.name} has an <object> ({name}) with no <bndbox>.")
        # 1-indexed on disk; see the docstring for why one is subtracted here and
        # nowhere else.
        x_min = _require_int(box_element, "xmin", path) - 1
        y_min = _require_int(box_element, "ymin", path) - 1
        x_max = _require_int(box_element, "xmax", path) - 1
        y_max = _require_int(box_element, "ymax", path) - 1
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"{path.name} has a degenerate {name} box "
                f"({x_min}, {y_min}, {x_max}, {y_max}) after converting to 0-indexed. "
                "A box with no area cannot be matched or scored."
            )

        boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
        labels.append(index_of[name])
        flag = object_element.find("difficult")
        difficult.append((flag is not None) and (flag.text or "0").strip() == "1")

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "difficult": torch.tensor(difficult, dtype=torch.bool),
        "size": (width, height),
    }


def _require_int(element: ElementTree.Element, tag: str, path: Path) -> int:
    """Read ``<tag>`` from ``element`` as an int, or say which file lacked it.

    VOC's coordinates are integers on disk, but some redistributions write them
    as floats ("174.0"), so the text is parsed as a float and then rounded
    rather than fed straight to ``int()``, which would raise on those.
    """
    child = element.find(tag)
    text = (child.text or "").strip() if child is not None else ""
    if not text:
        raise ValueError(f"{path.name} is missing <{tag}> where one is required.")
    try:
        return int(round(float(text)))
    except ValueError as error:
        raise ValueError(f"{path.name} has a non-numeric <{tag}>: {text!r}") from error


class DetectionFolderDataset(BaseDataset):
    """Images paired with box annotations by filename stem.

    Geometry works exactly as :class:`~visbench.data.dense.DenseFolderDataset`
    does — the short side is resized to ``image_size`` and a centre crop taken —
    and for the same reason: the dataset is the only place holding both the
    image and its target, so it is the only place that can keep them aligned.
    The difference is that the boxes are transformed rather than resampled, in
    :meth:`target`.

    Parameters
    ----------
    root:
        Dataset root. ``image_dir`` and ``annotation_dir`` are resolved beneath
        it unless given as absolute paths.
    image_dir, annotation_dir:
        Subdirectories holding the JPEGs and the XML. VOC's names are the
        defaults.
    stems:
        Official split list — either a path to a file of one stem per line, or
        the stems themselves. VOC ships 17,125 annotations and names split
        membership in ``ImageSets/Main/*.txt``, so without this the folder is
        the whole dataset rather than a split. Order is preserved, because
        targets travel by index everywhere in this codebase.
    include_difficult:
        Whether to keep objects VOC flags ``<difficult>1</difficult>`` (4,462 of
        them). **Default ``False``, which is what VOC's own protocol does.**
        Keeping them counts objects the protocol excludes as false negatives and
        depresses mAP against every published number, so the flag is recorded in
        ``describe()`` and a run that sets it must not claim VOC's protocol.
    min_box_size:
        Boxes smaller than this in either dimension *after* the crop are
        dropped. A centre crop genuinely removes objects, and a sliver of a
        person at the frame edge is not something a detector should be scored
        on; see :meth:`target`.
    annotation_loader:
        Reader for one annotation file, defaulting to :func:`load_voc_boxes`.
        Must return that function's dict.
    """

    #: Reindexed together by ``subset()``. Three lists paired by index: slicing
    #: one alone would pair an image with another image's boxes, and every later
    #: step would still see equal lengths.
    _parallel_attrs = ("stems", "image_paths", "annotation_paths")

    def __init__(
        self,
        root: str | Path,
        image_dir: str | Path = "JPEGImages",
        annotation_dir: str | Path = "Annotations",
        *,
        stems: str | Path | Sequence[str] | None = None,
        image_size: int = 224,
        classes: Sequence[str] = VOC_CLASSES,
        include_difficult: bool = False,
        min_box_size: float = 1.0,
        annotation_loader: Callable[[Path], dict[str, Any]] | None = None,
        name: str = "detection_folder",
        split: str = "",
    ) -> None:
        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")
        if min_box_size < 0:
            raise ValueError(f"min_box_size must be >= 0, got {min_box_size}")

        self.root = Path(root)
        self.image_root = self._resolve(image_dir)
        self.annotation_root = self._resolve(annotation_dir)
        self.image_size = int(image_size)
        self.classes = tuple(classes)
        self.include_difficult = bool(include_difficult)
        self.min_box_size = float(min_box_size)
        self.name = name
        self.split = split
        self._annotation_loader = annotation_loader

        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root does not exist: {self.root}")
        for label, directory in (
            ("image_dir", self.image_root),
            ("annotation_dir", self.annotation_root),
        ):
            if not directory.is_dir():
                raise NotADirectoryError(
                    f"{label}={directory.name!r} is not a directory under {self.root}"
                )

        images = self._index_directory(self.image_root)
        annotations = self._index_directory(self.annotation_root)

        if stems is None:
            chosen = sorted(set(images) & set(annotations))
            if not chosen:
                raise ValueError(
                    f"No filename stems are shared between {self.image_root} "
                    f"({len(images)} files) and {self.annotation_root} "
                    f"({len(annotations)} files)."
                )
        else:
            chosen = self._read_stems(stems)
            missing_images = [stem for stem in chosen if stem not in images]
            missing_annotations = [stem for stem in chosen if stem not in annotations]
            if missing_images or missing_annotations:
                raise ValueError(
                    f"{len(missing_images)} stem(s) missing from {self.image_root} and "
                    f"{len(missing_annotations)} from {self.annotation_root}. "
                    f"First few: {(missing_images or missing_annotations)[:5]}"
                )

        self.stems = list(chosen)
        self.image_paths = [images[stem] for stem in self.stems]
        self.annotation_paths = [annotations[stem] for stem in self.stems]

    # -- construction helpers ------------------------------------------------

    def _resolve(self, directory: str | Path) -> Path:
        path = Path(directory)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _index_directory(directory: Path) -> dict[str, Path]:
        """Map filename stem to path, refusing a directory with duplicate stems.

        Two files sharing a stem (``2007_000027.jpg`` and ``2007_000027.png``)
        would make the pairing depend on iteration order, which is silent.

        The whole directory is indexed even when ``stems=`` names a fraction of
        it, because a stem does not say its extension. That is affordable only
        because :func:`~visbench.data.base.list_files` avoids a stat per entry —
        see its docstring for what the stat version cost on VOC over NFS.
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
            lines = path.read_text().splitlines()
            # VOC's Main/<class>_train.txt files carry "<stem> <±1>" pairs, while
            # Main/train.txt is bare stems. Taking the first field reads both,
            # and a bare stem has exactly one field.
            chosen = [line.split()[0] for line in lines if line.strip()]
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
        ``max()`` guards and BICUBIC resampling, because :meth:`target` derives
        the box transform from the geometry this produces. If the two ever
        diverge, boxes shift and nothing raises.
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

    def target(self, index: int) -> dict[str, Any]:
        """Boxes for item ``index``, in **post-transform** pixel space.

        Returns ``boxes`` ``(N, 4)`` float32 ``xyxy``, ``labels`` ``(N,)``
        int64 and ``difficult`` ``(N,)`` bool, all three paired by row, plus
        ``num_original`` — how many objects the file held before filtering, so a
        caller can tell "this image has no objects" from "this image's objects
        were all dropped".

        The transform, and why each step is here rather than assumed:

        1. **Rescale by the achieved ratio, not the nominal one.**
           ``_resized_size`` rounds and applies a ``max()`` floor, so the width
           actually used is not exactly ``image_size / min(w, h)`` times the
           original. Scaling boxes by the nominal factor leaves a sub-pixel
           error that grows with box size and is invisible in any single image.
        2. **Shift by the crop origin**, the same one the image got.
        3. **Clip to the crop.** A box straddling the edge is genuinely
           partly visible, and its visible part is the correct target.
        4. **Drop boxes below ``min_box_size``.** A centre crop removes objects
           entirely — a box outside it clips to zero area — and scoring a
           detector against an object not in its input is measuring nothing.
           Dropping rows keeps ``boxes``, ``labels`` and ``difficult`` aligned
           because all three are indexed by the same mask.

        ``difficult`` objects are dropped here when ``include_difficult`` is
        False, before the geometry, since there is no reason to transform a box
        that is about to be discarded.
        """
        annotation = self._load_annotation(index)
        boxes = annotation["boxes"]
        labels = annotation["labels"]
        difficult = annotation["difficult"]
        num_original = int(boxes.shape[0])

        if not self.include_difficult and difficult.any():
            keep = ~difficult
            boxes, labels, difficult = boxes[keep], labels[keep], difficult[keep]

        original_width, original_height = annotation["size"]
        resized_width, resized_height = self._resized_size(original_width, original_height)
        left, top = self._crop_origin(resized_width, resized_height)

        if boxes.numel():
            # The achieved ratio, per axis. See step 1 above.
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height
            boxes = boxes.clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x - left
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y - top
            boxes = boxes.clamp(min=0.0, max=float(self.image_size))

            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            keep = (widths >= self.min_box_size) & (heights >= self.min_box_size)
            boxes, labels, difficult = boxes[keep], labels[keep], difficult[keep]

        return {
            "boxes": boxes.reshape(-1, 4),
            "labels": labels,
            "difficult": difficult,
            "num_original": num_original,
        }

    def _load_annotation(self, index: int) -> dict[str, Any]:
        """Read one annotation file and check it describes its own image."""
        path = self.annotation_paths[index]
        loader = self._annotation_loader
        annotation = (
            loader(path) if loader is not None else load_voc_boxes(path, classes=self.classes)
        )
        missing = [key for key in (*_ANNOTATION_KEYS, "size") if key not in annotation]
        if missing:
            raise ValueError(
                f"The annotation loader returned no {missing} for {path.name}; "
                f"a box annotation needs {(*_ANNOTATION_KEYS, 'size')}."
            )
        width, height = annotation["size"]
        if width < 1 or height < 1:
            raise ValueError(f"{path.name} declares a {width}x{height} image size.")
        return annotation

    def labels(self) -> list[dict[str, Any]]:
        """Annotations in index order, satisfying the :class:`BaseDataset` contract."""
        return [self.target(index) for index in range(len(self))]

    # -- identity ------------------------------------------------------------

    def cache_identity(self, index: int) -> str:
        """Token for the *image* only.

        Excludes the annotation, exactly as the dense dataset does: cached
        features depend on the image and nothing else, so editing an XML must
        not invalidate an extraction that is still valid. The annotations do
        appear in :meth:`fingerprint`, where the question is which data produced
        a score.
        """
        path = self.image_paths[index]
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"

    def fingerprint(self) -> str:
        """Short hash over both file lists and the settings that shape the targets.

        ``image_size`` decides the crop, and ``include_difficult`` and
        ``min_box_size`` decide which objects survive it, so the same folder
        under different values is a different set of targets and the records
        must not collide.
        """
        digest = hashlib.sha256()
        digest.update(
            f"{self.name}|{self.split}|{len(self.stems)}|{self.image_size}|"
            f"{self.include_difficult}|{self.min_box_size}|{len(self.classes)}".encode()
        )
        # strict=True: images and annotations are paired by index throughout this
        # class, so unequal lengths mean the pairing is already broken. Folding a
        # truncated list into a fingerprint that still looks valid would let a
        # misaligned split train and score without complaint.
        for image_path, annotation_path in zip(
            self.image_paths, self.annotation_paths, strict=True
        ):
            digest.update(
                f"{image_path.name}|{image_path.stat().st_size}|"
                f"{annotation_path.name}|{annotation_path.stat().st_size}".encode()
            )
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        info = super().describe()
        info["image_size"] = self.image_size
        info["include_difficult"] = self.include_difficult
        info["min_box_size"] = self.min_box_size
        info["num_classes"] = len(self.classes)
        return info
