"""Image-folder dataset — the v0.1 data path.

``root/<class_name>/<image>`` for labeled use, or a flat folder of images for
unlabeled use. Deliberately the only dataset in v0.1: step 3 of the build order
proves the full path end-to-end on a **small local image folder**, with no
dataset-download machinery in the way.
"""

from pathlib import Path
from typing import Any, Optional

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
        self._labels: list[Optional[int]] = []

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

    def __getitem__(self, index: int) -> tuple[Any, Optional[int]]:
        """Return ``(pil_image, class_index)``, or ``(pil_image, None)`` if unlabeled."""
        return load_image(self.paths[index]), self._labels[index]

    def labels(self) -> list:
        """Class indices in index order. No image is opened."""
        return list(self._labels)

    @property
    def classes(self) -> list:
        """Sorted class names; empty when ``labeled=False``."""
        return list(self._classes)

    def describe(self) -> dict:
        info = super().describe()
        info["num_classes"] = len(self._classes)
        return info
