"""Disk-backed feature cache.

Mandatory in v0.1, not an optional speed-up bolted on later (CLAUDE.md,
"Feature cache"). Every task reads through this; the backbone forward pass runs
**at most once per image per backbone**.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Optional

from visbench.types import FeatureDict, Pooling

__all__ = ["FeatureCache"]


class FeatureCache:
    """Key-value store mapping cache keys to extracted features on disk.

    Stores ``dense``, ``pooled`` and ``grid_hw`` together, since they come from
    one forward pass and splitting them would allow a partial hit that still
    requires re-running the backbone.
    """

    def __init__(self, root: Optional[Path] = None, enabled: bool = True) -> None:
        """Open (creating if needed) the cache directory.

        ``enabled=False`` turns every lookup into a miss without changing call
        sites — useful for tests and for measuring true extraction cost.
        """
        raise NotImplementedError

    def get(self, key: str) -> Optional[FeatureDict]:
        """Return the cached entry, or ``None`` on a miss."""
        raise NotImplementedError

    def put(self, key: str, features: FeatureDict) -> None:
        """Write an entry. Writes atomically so an interrupted run cannot leave
        a half-written file that later reads as a corrupt hit."""
        raise NotImplementedError

    def get_or_compute(self, key: str, compute: Callable[[], FeatureDict]) -> FeatureDict:
        """Return the cached entry, computing and storing it on a miss."""
        raise NotImplementedError

    def extract_dataset(
        self,
        backbone: Any,
        dataset: Iterable,
        pooling: str = Pooling.DEFAULT,
        layer: Optional[int] = None,
        batch_size: int = 32,
    ) -> Any:
        """Extract (or load) features for a whole dataset.

        The main entry point tasks use: batches only the cache misses through
        the backbone, then returns features for the full dataset in dataset
        order.
        """
        raise NotImplementedError

    def clear(self, backbone_key: Optional[str] = None) -> int:
        """Delete cached entries, optionally only those for one backbone.

        Returns the number of entries removed.
        """
        raise NotImplementedError

    def stats(self) -> dict:
        """Entry count, on-disk size, and hit/miss counts for this session."""
        raise NotImplementedError
