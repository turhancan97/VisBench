"""Dataset interface.

Datasets stay thin: they yield PIL images plus labels and leave all tensor
conversion to the backbone's ``preprocess()``, because normalisation and input
resolution are backbone properties, not dataset properties. This is what allows
the same dataset object to feed DINOv2 and CLIP unchanged.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
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

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        """Iterate in index order.

        Defined explicitly rather than relying on the old ``__getitem__``
        iteration protocol, so that ``FeatureCache.extract_dataset`` gets a
        lazy, one-item-at-a-time read and never materialises the whole folder.
        """
        for index in range(len(self)):
            yield self[index]

    def labels(self) -> list:
        """All labels in index order, without decoding any image.

        Tasks need labels alongside cached features; loading every image to get
        them would defeat the cache entirely.
        """
        raise NotImplementedError

    def cache_identity(self, index: int) -> str | None:
        """Cheap stable token for item ``index``, or ``None`` if unavailable.

        Lets :meth:`FeatureCache.extract_dataset` decide whether an item is
        already cached **without decoding it**. On a fully cached dataset that
        is the difference between reading thousands of JPEGs and reading none.

        This does not replace content hashing: the cache key is still derived
        from decoded pixels, so the same image under two filenames shares one
        entry. This only memoises "which content hash did this file last
        produce", and must therefore change whenever the file's bytes might
        have (size and mtime, not path alone).

        Returning ``None`` is always safe — it simply costs a decode.
        """
        return None

    def fingerprint(self) -> str | None:
        """Short hash identifying *which* data this is, for the result record.

        A record naming only ``"imagenette"`` and ``"val"`` cannot distinguish
        two folders that share a name, so two runs over different images
        produce records that look identical and mean different things.

        Returns ``None`` when a subclass cannot compute one cheaply — the
        record then carries no fingerprint, which is honest, rather than a
        misleading one.
        """
        return None

    def describe(self) -> dict:
        """Dataset metadata (name, split, size, fingerprint) for the result record."""
        return {
            "dataset": self.name,
            "split": self.split,
            "dataset_size": len(self),
            "dataset_fingerprint": self.fingerprint(),
        }
