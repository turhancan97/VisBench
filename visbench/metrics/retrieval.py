"""Retrieval metrics."""

from collections.abc import Sequence

import torch

from visbench.types import MetricsDict

__all__ = ["recall_at_k", "mean_average_precision"]


def recall_at_k(
    ranked_labels: torch.Tensor,
    query_labels: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
) -> MetricsDict:
    """Fraction of queries with a correct match in the top k, per k.

    The query itself must already be excluded from its own ranking by the
    caller; otherwise recall@1 is trivially 1.0.
    """
    raise NotImplementedError


def mean_average_precision(
    ranked_labels: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    """mAP over all queries."""
    raise NotImplementedError
