"""Dataset interface.

Datasets stay thin: they yield PIL images plus labels and leave all tensor
conversion to the backbone's ``preprocess()``, because normalisation and input
resolution are backbone properties, not dataset properties. This is what allows
the same dataset object to feed DINOv2 and CLIP unchanged.
"""

import copy
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

__all__ = ["BaseDataset", "list_files"]


def list_files(directory: Path, extensions: Sequence[str] | None = None) -> list[Path]:
    """Sorted regular files directly inside ``directory``, optionally filtered.

    ``extensions`` is matched case-insensitively against the suffix and should
    include the dot (``(".jpg", ".png")``). ``None`` accepts every file.

    **Use this rather than ``iterdir()`` + ``Path.is_file()``, which costs one
    stat per entry and is the slowest thing a dataset constructor does on a
    network filesystem.** ``Path.is_file()`` cannot reuse what ``readdir``
    already returned, so over NFS it is a round trip per file with nothing to
    amortise it; :func:`os.scandir` answers the same question from the file
    type the directory listing carried anyway. Measured on a cold 2,913-file
    directory over NFSv4.2: **0.05 s against 5.69 s**, and on VOC's
    17,125-file ``Annotations`` the stat version reached **76 s** — paid in
    full even by a split naming 600 stems, and paid *before* ``run()`` starts
    timing, so no result record shows it.

    A symlink still costs a stat, because ``readdir`` reports it as a link
    rather than as whatever it resolves to. That is correct, and rare.
    """
    suffixes = tuple(ext.lower() for ext in extensions) if extensions is not None else None
    with os.scandir(directory) as entries:
        listing = sorted(entries, key=lambda entry: entry.name)
    return [
        Path(entry.path)
        for entry in listing
        if entry.is_file()
        and (suffixes is None or os.path.splitext(entry.name)[1].lower() in suffixes)
    ]


#: Lets subset() return the caller's own type rather than the base, so a
#: DenseFolderDataset stays one through a --limit and keeps its dense methods.
DatasetT = TypeVar("DatasetT", bound="BaseDataset")


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

    #: Attribute names holding sequences in dataset order, which :meth:`subset`
    #: reindexes **together**. A dense dataset carries three of them — stems,
    #: image paths, target paths — and slicing one without the others pairs a
    #: target with the wrong image, which nothing downstream would catch. Naming
    #: them here is what lets one implementation get that right for every
    #: subclass instead of each caller re-deriving it.
    _parallel_attrs: tuple[str, ...] = ()

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

    def labels(self) -> Any:
        """Whatever the task scores against, in index order, without decoding.

        Tasks need labels alongside cached features; loading every image to get
        them would defeat the cache entirely.

        Deliberately ``Any``, because "label" means something different per
        task and the base cannot know which: a class index for classification,
        a geometry dict for correspondence, a ``(T, 4)`` tensor of triplets for
        2AFC similarity. Subclasses narrow it to what they actually return, and
        a caller should read that rather than this.
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

    def subset(self: DatasetT, indices: int | Sequence[int]) -> DatasetT:
        """A new dataset over ``indices``, or over the first ``n`` if an int.

        The public way to shorten a split — for a quick run, a smoke test, or a
        CLI ``--limit``. Before this existed every example did it by hand, and
        the dense ones had to slice three parallel lists in step while a comment
        explained that dropping one would pair a target with the wrong image.
        That hazard belongs in one tested place.

        The original is left untouched: a shallow copy shares the heavy state
        (loaders, roots) and gets its own reindexed sequences, so subsetting a
        dataset you are still using cannot disturb it.

        ``fingerprint`` follows automatically, because it is computed from the
        very sequences this reindexes — so a limited run and a full one are
        distinguishable in the cache and in the result record, rather than the
        short run poisoning the long one's entry.

        Parameters
        ----------
        indices:
            An ``int`` takes the first ``n`` in order and is **clamped** to the
            dataset's length, matching "use at most N". An explicit sequence is
            validated strictly: an out-of-range index raises rather than being
            skipped, since a silently shorter split is the failure this method
            exists to prevent.
        """
        if not self._parallel_attrs:
            raise NotImplementedError(
                f"{type(self).__name__} does not declare _parallel_attrs, so it cannot "
                "be subset generically. Set them, or override subset()."
            )

        if isinstance(indices, int):
            if indices < 1:
                raise ValueError(f"subset(n) needs n >= 1, got {indices}")
            chosen = list(range(min(indices, len(self))))
        else:
            chosen = [int(index) for index in indices]
            if not chosen:
                raise ValueError("subset() got an empty index sequence")
            out_of_range = [index for index in chosen if not 0 <= index < len(self)]
            if out_of_range:
                raise IndexError(
                    f"{len(out_of_range)} index/indices out of range for a dataset of "
                    f"{len(self)}: {out_of_range[:5]}"
                )

        clone = copy.copy(self)
        for attribute in self._parallel_attrs:
            sequence = getattr(self, attribute)
            setattr(clone, attribute, [sequence[index] for index in chosen])
        return clone

    def describe(self) -> dict:
        """Dataset metadata (name, split, size, fingerprint) for the result record."""
        return {
            "dataset": self.name,
            "split": self.split,
            "dataset_size": len(self),
            "dataset_fingerprint": self.fingerprint(),
        }
