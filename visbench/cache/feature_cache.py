"""Disk-backed feature cache.

Mandatory in v0.1, not an optional speed-up bolted on later (CLAUDE.md,
"Feature cache"). Every task reads through this; the backbone forward pass runs
**at most once per image per backbone**.
"""

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from visbench.cache.keys import SEPARATOR, hash_image, make_key
from visbench.types import FeatureDict, Pooling
from visbench.utils.device import batched

__all__ = ["FeatureCache", "DEFAULT_CACHE_DIR"]

#: Default location, relative to the working directory. In .gitignore.
DEFAULT_CACHE_DIR = Path(".visbench_cache")

_ENTRY_SUFFIX = ".pt"


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
    ) -> FeatureDict:
        """Extract (or load) features for a whole dataset.

        The main entry point tasks use: batches only the cache misses through
        the backbone, then returns features for the full dataset in dataset
        order.

        ``dataset`` may yield PIL images or ``(image, label)`` pairs; labels are
        ignored here, since a task reads them from the dataset directly.

        Returns one :class:`FeatureDict` for the whole dataset, with ``dense``
        ``(N, C, H, W)`` and ``pooled`` ``(N, C)``. ``dense`` is the memory
        risk — a 5k-image dataset at 16x16x768 is roughly 4 GB in fp32 — so a
        task that only needs ``pooled`` should keep just that and drop the rest.
        """
        images = [item[0] if isinstance(item, (tuple, list)) else item for item in dataset]
        if not images:
            raise ValueError("Cannot extract features from an empty dataset")

        backbone_key = backbone.cache_key()
        keys = [make_key(hash_image(img), backbone_key, layer, pooling) for img in images]

        results: list[Optional[FeatureDict]] = [self.get(key) for key in keys]
        missing = [i for i, entry in enumerate(results) if entry is None]

        layers = None if layer is None else [layer]
        for chunk in batched(missing, batch_size):
            batch = backbone.preprocess([images[i] for i in chunk])
            features = backbone.extract_features(batch, pooling=pooling, layers=layers)
            for position, index in enumerate(chunk):
                single: FeatureDict = {
                    "dense": features["dense"][position : position + 1].cpu(),
                    "pooled": features["pooled"][position : position + 1].cpu(),
                    "grid_hw": features["grid_hw"],
                }
                self.put(keys[index], single)
                results[index] = single

        grids = {entry["grid_hw"] for entry in results}  # type: ignore[index]
        if len(grids) > 1:
            raise ValueError(
                f"Dataset produced more than one dense grid shape ({sorted(grids)}); "
                "features cannot be stacked. Use a fixed input resolution."
            )

        return {
            "dense": torch.cat([entry["dense"] for entry in results]),  # type: ignore[index]
            "pooled": torch.cat([entry["pooled"] for entry in results]),  # type: ignore[index]
            "grid_hw": grids.pop(),
        }

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
