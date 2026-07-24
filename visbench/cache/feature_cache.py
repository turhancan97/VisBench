"""Disk-backed feature cache.

Mandatory in v0.1, not an optional speed-up bolted on later (CLAUDE.md,
"Feature cache"). Every task reads through this; the backbone forward pass runs
**at most once per image per backbone**.
"""

import hashlib
import itertools
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from visbench.cache.keys import SEPARATOR, hash_image, make_key
from visbench.types import FeatureDict, Pooling

__all__ = ["FeatureCache", "DEFAULT_CACHE_DIR"]

#: Default location, relative to the working directory. In .gitignore.
DEFAULT_CACHE_DIR = Path(".visbench_cache")

_ENTRY_SUFFIX = ".pt"

#: Which outputs :meth:`FeatureCache.extract_dataset` accumulates in memory.
_KEEP_CHOICES = ("both", "pooled", "dense")


def _chunks(items: Iterable, size: int) -> Iterator[list]:
    """Yield lists of at most ``size`` items, pulling from ``items`` lazily.

    Distinct from :func:`visbench.utils.device.batched`, which slices a
    ``Sequence`` and therefore needs the whole thing in memory first. Here the
    point is precisely that it never is.
    """
    if size < 1:
        raise ValueError(f"batch_size must be >= 1, got {size}")
    iterator = iter(items)
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


class FeatureCache:
    """Key-value store mapping cache keys to extracted features on disk.

    Stores ``dense``, ``pooled`` and ``grid_hw`` together, since they come from
    one forward pass and splitting them would allow a partial hit that still
    requires re-running the backbone.

    Entries are stored on CPU regardless of the extraction device, so a cache
    written on a GPU box is readable on a laptop.
    """

    def __init__(self, root: Optional[Path] = None, enabled: bool = True) -> None:
        """Open (creating if needed) the cache directory.

        ``enabled=False`` turns every lookup into a miss without changing call
        sites — useful for tests and for measuring true extraction cost.
        """
        self.root = Path(root) if root is not None else DEFAULT_CACHE_DIR
        self.enabled = enabled
        self._hits = 0
        self._misses = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------------

    def _path(self, key: str) -> Path:
        """Map a key to its file: ``<root>/<backbone>/<shard>/<digest>.pt``.

        The backbone directory is what makes :meth:`clear` per-backbone cheap,
        and the two-character shard keeps any single directory from collecting
        every entry in a large dataset.
        """
        backbone_key = key.split(SEPARATOR, 1)[0]
        # "/" is legal in a backbone key ("dinov2/dinov2_vitb14/224") but would
        # nest directories arbitrarily; flatten it to one level.
        safe_backbone = backbone_key.replace("/", "__").replace(os.sep, "__")
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.root / safe_backbone / digest[:2] / f"{digest}{_ENTRY_SUFFIX}"

    # -- single entries ------------------------------------------------------

    def get(self, key: str) -> Optional[FeatureDict]:
        """Return the cached entry, or ``None`` on a miss."""
        if not self.enabled:
            self._misses += 1
            return None

        path = self._path(key)
        if not path.exists():
            self._misses += 1
            return None

        try:
            entry = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # A truncated or corrupt file counts as a miss, never as a hit. The
            # subsequent put() overwrites it, so the cache self-heals.
            self._misses += 1
            return None

        self._hits += 1
        return {
            "dense": entry["dense"],
            "pooled": entry["pooled"],
            "grid_hw": tuple(entry["grid_hw"]),
        }

    def put(self, key: str, features: FeatureDict) -> None:
        """Write an entry. Writes atomically so an interrupted run cannot leave
        a half-written file that later reads as a corrupt hit."""
        if not self.enabled:
            return

        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dense": features["dense"].detach().cpu(),
            "pooled": features["pooled"].detach().cpu(),
            # Saved as a list: weights_only=True unpickles lists but not tuples
            # of arbitrary origin. Restored to a tuple on read.
            "grid_hw": list(features["grid_hw"]),
        }

        # Same directory as the target, so os.replace stays on one filesystem
        # and is therefore atomic.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(payload, handle)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def get_or_compute(self, key: str, compute: Callable[[], FeatureDict]) -> FeatureDict:
        """Return the cached entry, computing and storing it on a miss."""
        cached = self.get(key)
        if cached is not None:
            return cached
        features = compute()
        self.put(key, features)
        return features

    # -- datasets ------------------------------------------------------------

    def extract_dataset(
        self,
        backbone: Any,
        dataset: Iterable,
        pooling: str = Pooling.DEFAULT,
        layer: Optional[int] = None,
        batch_size: int = 32,
        keep: str = "both",
    ) -> FeatureDict:
        """Extract (or load) features for a whole dataset.

        The main entry point tasks use: batches only the cache misses through
        the backbone, then returns features for the full dataset in dataset
        order.

        ``dataset`` may yield PIL images or ``(image, label)`` pairs; labels are
        ignored here, since a task reads them from the dataset directly. It is
        consumed lazily, one batch at a time — a dataset that decodes on
        ``__getitem__`` never holds more than ``batch_size`` images in memory,
        which is what makes a 50k-image run possible at all.

        Parameters
        ----------
        keep:
            Which outputs to accumulate: ``"both"``, ``"pooled"`` or ``"dense"``.
            ``dense`` is the memory risk — 5k images at 16x16x768 is roughly
            4 GB in fp32 — so a task needing only pooled features should say so
            and never materialise the rest. The *cache* always stores both,
            since they come from one forward pass; ``keep`` only controls what
            is held in RAM and returned.
        """
        if keep not in _KEEP_CHOICES:
            raise ValueError(f"keep must be one of {_KEEP_CHOICES}, got {keep!r}")

        backbone_key = backbone.cache_key()
        layers = None if layer is None else [layer]

        dense_chunks: list[torch.Tensor] = []
        pooled_chunks: list[torch.Tensor] = []
        grids: set = set()
        count = 0

        for batch_items in _chunks(dataset, batch_size):
            images = [item[0] if isinstance(item, (tuple, list)) else item for item in batch_items]
            keys = [make_key(hash_image(img), backbone_key, layer, pooling) for img in images]

            entries: list[Optional[FeatureDict]] = [self.get(key) for key in keys]
            missing = [i for i, entry in enumerate(entries) if entry is None]

            if missing:
                tensor_batch = backbone.preprocess([images[i] for i in missing])
                features = backbone.extract_features(tensor_batch, pooling=pooling, layers=layers)
                for position, index in enumerate(missing):
                    single: FeatureDict = {
                        "dense": features["dense"][position : position + 1].cpu(),
                        "pooled": features["pooled"][position : position + 1].cpu(),
                        "grid_hw": features["grid_hw"],
                    }
                    self.put(keys[index], single)
                    entries[index] = single

            for entry in entries:
                assert entry is not None  # every miss was filled above
                grids.add(entry["grid_hw"])
                count += 1
                if keep in ("both", "dense"):
                    dense_chunks.append(entry["dense"])
                if keep in ("both", "pooled"):
                    pooled_chunks.append(entry["pooled"])

        if count == 0:
            raise ValueError("Cannot extract features from an empty dataset")
        if len(grids) > 1:
            raise ValueError(
                f"Dataset produced more than one dense grid shape ({sorted(grids)}); "
                "features cannot be stacked. Use a fixed input resolution."
            )

        result: FeatureDict = {"grid_hw": grids.pop()}  # type: ignore[typeddict-item]
        if keep in ("both", "dense"):
            result["dense"] = torch.cat(dense_chunks)
        if keep in ("both", "pooled"):
            result["pooled"] = torch.cat(pooled_chunks)
        return result

    # -- maintenance ---------------------------------------------------------

    def clear(self, backbone_key: Optional[str] = None) -> int:
        """Delete cached entries, optionally only those for one backbone.

        Returns the number of entries removed.
        """
        if backbone_key is None:
            targets = [self.root]
        else:
            safe = backbone_key.replace("/", "__").replace(os.sep, "__")
            targets = [self.root / safe]

        removed = 0
        for target in targets:
            if not target.exists():
                continue
            removed += sum(1 for _ in target.rglob(f"*{_ENTRY_SUFFIX}"))
            shutil.rmtree(target)

        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
        return removed

    def stats(self) -> dict:
        """Entry count, on-disk size, and hit/miss counts for this session."""
        entries = list(self.root.rglob(f"*{_ENTRY_SUFFIX}")) if self.root.exists() else []
        return {
            "entries": len(entries),
            "bytes": sum(path.stat().st_size for path in entries),
            "hits": self._hits,
            "misses": self._misses,
            "enabled": self.enabled,
            "root": str(self.root),
        }
