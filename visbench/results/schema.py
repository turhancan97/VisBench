"""The structured result record — one schema from the start.

Every task run logs a record with this shape so that leaderboard tooling never
needs a retrofit (CLAUDE.md, "Engineering conventions"). Fields are added only
additively; renaming or removing one is a breaking change to every historical
record.
"""

from dataclasses import MISSING, asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Optional

__all__ = ["ResultRecord", "SCHEMA_VERSION", "utc_timestamp"]

#: Bumped whenever the record shape changes, so consumers can migrate.
#:
#: History
#: -------
#: 1. Initial schema.
#: 2. Added ``dataset_size`` and ``dataset_fingerprint``. Without them, two
#:    runs over different folders that happen to share a name produced records
#:    that were byte-identical and meant different things.
SCHEMA_VERSION = 2


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
    dataset / split:
        Dataset identifier plus split.
    dataset_size / dataset_fingerprint:
        How many images, and a short hash of the file list from
        :meth:`BaseDataset.fingerprint`. The name alone does not identify data
        — two folders can share one — so without these, a run before and after
        the images changed produces indistinguishable records.
    pooling / feature_mode:
        Exactly what representation the task requested — without these two the
        metrics are not reproducible.
    metrics:
        The flat dict returned by :meth:`BaseTask.evaluate`.
    seed:
        From :func:`visbench.utils.set_seed`, which returns the seed it used
        even when the caller passed ``None``. Irrelevant for zero-shot tasks;
        required to reproduce anything that trains.
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
    dataset_size: Optional[int] = None
    dataset_fingerprint: Optional[str] = None
    layer: Optional[int] = None
    seed: Optional[int] = None
    duration_seconds: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        """JSON-serialisable dict, with ``None`` fields retained.

        Retained rather than dropped so every record has identical keys, which
        keeps downstream tabular loading trivial.
        """
        payload = asdict(self)
        # Metrics arrive from evaluate() and are the one field a task fills
        # freely; coerce here so a stray tensor or numpy float cannot make a
        # whole results file unreadable.
        payload["metrics"] = {key: float(value) for key, value in self.metrics.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ResultRecord":
        """Rebuild a record, rejecting a ``schema_version`` newer than this one.

        Older versions are read, not rejected: the schema is additive-only, so
        every field a v1 record has still exists, and the ones it predates come
        back as ``None``. Refusing them would throw away exactly the history a
        benchmark library exists to accumulate.
        """
        version = payload.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(f"schema_version must be an int, got {version!r}")
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version}; this VisBench reads up to "
                f"version {SCHEMA_VERSION}. Records are additive-only, so a "
                "newer file needs a newer VisBench."
            )

        known = {field.name for field in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                f"Unknown fields for schema_version {SCHEMA_VERSION}: {sorted(unknown)}"
            )
        missing = {f.name for f in fields(cls) if f.default is MISSING} - set(payload)
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        return cls(**payload)


def utc_timestamp() -> str:
    """ISO 8601 UTC timestamp, e.g. ``2026-07-24T09:15:04+00:00``.

    UTC always: a leaderboard aggregating runs from several machines cannot
    order local timestamps.
    """
    return datetime.now(timezone.utc).isoformat()
