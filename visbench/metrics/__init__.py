"""Metric implementations, kept out of task classes.

Separate so that a metric can be unit-tested against known inputs without
constructing a backbone or a task, and so two tasks can share one definition.
Every function returns a flat dict (or a scalar), matching the contract of
:meth:`BaseTask.evaluate`.
"""

from visbench.metrics.classification import top_k_accuracy
from visbench.metrics.correspondence import (
    correspondence_recall,
    error_auc,
    nn_match,
    ratio_test,
)
from visbench.metrics.retrieval import mean_average_precision, recall_at_k

__all__ = [
    "top_k_accuracy",
    "recall_at_k",
    "mean_average_precision",
    "nn_match",
    "ratio_test",
    "correspondence_recall",
    "error_auc",
]
