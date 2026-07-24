"""Image-folder dataset — the v0.1 data path.

``root/<class_name>/<image>`` for labeled use, or a flat folder of images for
unlabeled use. Deliberately the only dataset in v0.1: step 3 of the build order
proves the full path end-to-end on a **small local image folder**, with no
dataset-download machinery in the way.
"""

from pathlib import Path
from typing import Any, Optional

from visbench.data.base import BaseDataset

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
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[Any, Optional[int]]:
        """Return ``(pil_image, class_index)``, or ``(pil_image, None)`` if unlabeled."""
        raise NotImplementedError

    @property
    def classes(self) -> list:
        """Sorted class names; empty when ``labeled=False``."""
        raise NotImplementedError
