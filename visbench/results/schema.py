"""The structured result record — one schema from the start.

Every task run logs a record with this shape so that leaderboard tooling never
needs a retrofit (CLAUDE.md, "Engineering conventions"). Fields are added only
additively; renaming or removing one is a breaking change to every historical
record.
"""

from dataclasses import dataclass
from typing import Optional

__all__ = ["ResultRecord", "SCHEMA_VERSION"]

#: Bumped whenever the record shape changes, so consumers can migrate.
SCHEMA_VERSION = 1


@dataclass
class ResultRecord:
    """One task run on one backbone over one dataset.

    Attributes
    ----------
    backbone / backbone_key:
        Registered name, and the weights-and-resolution identifier from
        :meth:`BaseBackbone.cache_key`. Both, because the name alone does not
        pin down what actually ran.
    task / level:
        Registered task name and its level (high/mid/low).
    dataset:
        Dataset identifier plus split.
    pooling / feature_mode:
        Exactly what representation the task requested — without these two the
        metrics are not reproducible.
    metrics:
        The flat dict returned by :meth:`BaseTask.evaluate`.
    """

    backbone: str
    backbone_key: str
    task: str
    level: str
    dataset: str
    split: str
    pooling: str
    feature_mode: str
    metrics: dict[str, float]
    timestamp: str
    visbench_version: str
    schema_version: int = SCHEMA_VERSION
    layer: Optional[int] = None
    seed: Optional[int] = None
    duration_seconds: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        """JSON-serialisable dict, with ``None`` fields retained.

        Retained rather than dropped so every record has identical keys, which
        keeps downstream tabular loading trivial.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, payload: dict) -> "ResultRecord":
        """Rebuild a record, rejecting unknown ``schema_version`` values."""
        raise NotImplementedError
