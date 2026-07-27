"""Image-folder dataset — the v0.1 data path.

``root/<class_name>/<image>`` for labeled use, or a flat folder of images for
unlabeled use. Deliberately the only dataset in v0.1: step 3 of the build order
proves the full path end-to-end on a **small local image folder**, with no
dataset-download machinery in the way.
"""

import hashlib
from pathlib import Path
from typing import Any

from visbench.data.base import BaseDataset
from visbench.utils.image import load_image

__all__ = ["ImageFolderDataset"]


class ImageFolderDataset(BaseDataset):
    """Images on disk, optionally labeled by parent directory name."""

    def __init__(
        self,
        root: Path,
        split: str = "train",
        labeled: bool = True,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> None:
        """Index the folder.

        File order is sorted, not filesystem order, so cached features line up
        with labels across machines and reruns.
        """
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root does not exist: {self.root}")

        self.split = split
        self.labeled = labeled
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.name = self.root.name

        self._classes: list[str] = []
        self.paths: list[Path] = []
        self._labels: list[int | None] = []

        if labeled:
            self._classes = sorted(d.name for d in self.root.iterdir() if d.is_dir())
            if not self._classes:
                raise ValueError(
                    f"{self.root} has no class subdirectories. Expected "
                    "root/<class_name>/<image>, or labeled=False for a flat folder."
                )
            for index, class_name in enumerate(self._classes):
                for path in self._image_files(self.root / class_name):
                    self.paths.append(path)
                    self._labels.append(index)
        else:
            for path in self._image_files(self.root):
                self.paths.append(path)
                self._labels.append(None)

        if not self.paths:
            raise ValueError(f"No images with extensions {self.extensions} found under {self.root}")

    def _image_files(self, directory: Path) -> list[Path]:
        """Sorted image files directly inside ``directory``."""
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in self.extensions
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, int | None]:
        """Return ``(pil_image, class_index)``, or ``(pil_image, None)`` if unlabeled."""
        return load_image(self.paths[index]), self._labels[index]

    def labels(self) -> list:
        """Class indices in index order. No image is opened."""
        return list(self._labels)

    def cache_identity(self, index: int) -> str:
        """``"<abs path>|<size>|<mtime_ns>"`` — changes if the file could have.

        mtime is deliberately included here although it is deliberately
        *excluded* from :meth:`fingerprint`. The two answer different
        questions: the fingerprint asks "is this the same dataset", where a
        re-copy must not invalidate past records; this asks "might these bytes
        have changed since I last hashed them", where a re-copy must invalidate
        the memo. Being wrong here would serve one image's features for
        another.
        """
        path = self.paths[index]
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"

    def fingerprint(self) -> str:
        """Short hash over the file list: relative paths, sizes and labels.

        Deliberately built from ``stat()`` rather than file contents. Reading
        every image to fingerprint it would reintroduce, on *every* run, exactly
        the I/O cost the feature cache exists to avoid — and the fingerprint's
        job is to distinguish datasets, not to verify them.

        What it catches: images added, removed, renamed, reordered, relabelled,
        or replaced with a file of a different size. What it misses: an image
        edited in place to exactly the same byte length. That residual case is
        caught downstream anyway, because the feature cache keys on decoded
        pixel content and would treat the edited image as a miss.
        """
        digest = hashlib.sha256()
        digest.update(f"{self.name}|{self.split}|{len(self.paths)}".encode())
        for path, label in zip(self.paths, self._labels):
            relative = path.relative_to(self.root).as_posix()
            digest.update(f"{relative}|{path.stat().st_size}|{label}".encode())
        return digest.hexdigest()[:16]

    @property
    def classes(self) -> list:
        """Sorted class names; empty when ``labeled=False``."""
        return list(self._classes)

    def describe(self) -> dict:
        info = super().describe()
        info["num_classes"] = len(self._classes)
        return info
