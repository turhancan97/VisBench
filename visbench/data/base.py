"""Dataset interface.

Datasets stay thin: they yield PIL images plus labels and leave all tensor
conversion to the backbone's ``preprocess()``, because normalisation and input
resolution are backbone properties, not dataset properties. This is what allows
the same dataset object to feed DINOv2 and CLIP unchanged.
"""

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["BaseDataset"]


class BaseDataset(ABC):
    """Indexed collection of images and labels.

    Yields *unpreprocessed* PIL images. The cache hashes decoded pixel content
    (:func:`visbench.cache.hash_image`), so this must be deterministic — no
    random augmentation in the read path.
    """

    #: Identifier recorded in the result log.
    name: str = ""

    #: Split identifier ("train" / "val" / "test"), also logged.
    split: str = ""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Return ``(pil_image, label)``. ``label`` is ``None`` for unlabeled data."""
        raise NotImplementedError

    def describe(self) -> dict:
        """Dataset metadata (name, split, size) for the result record."""
        raise NotImplementedError
