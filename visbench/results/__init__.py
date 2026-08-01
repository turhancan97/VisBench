"""Structured result logging.

Named ``results`` rather than ``logging`` to avoid shadowing the stdlib module.
"""

from visbench.results.leaderboard import (
    ComparabilityKey,
    IncomparableRecords,
    UnknownMetric,
    comparability_key,
    group_comparable,
    latest_per_backbone,
    metric_direction,
    rank,
    ranking_disagreements,
    shared_metrics,
)
from visbench.results.schema import SCHEMA_VERSION, ResultRecord
from visbench.results.writer import ResultWriter, read_records

__all__ = [
    "ResultRecord",
    "ResultWriter",
    "read_records",
    "SCHEMA_VERSION",
    "ComparabilityKey",
    "IncomparableRecords",
    "UnknownMetric",
    "comparability_key",
    "group_comparable",
    "latest_per_backbone",
    "metric_direction",
    "rank",
    "ranking_disagreements",
    "shared_metrics",
]
