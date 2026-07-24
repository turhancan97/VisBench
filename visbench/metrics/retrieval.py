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

    Parameters
    ----------
    ranked_labels:
        ``(num_queries, num_ranked)`` — the label of each retrieved item, in
        rank order.
    query_labels:
        ``(num_queries,)``.
    ks:
        A ``k`` larger than ``num_ranked`` is skipped rather than silently
        clamped: reporting recall@10 computed over 4 candidates would be a
        different number wearing the same name.
    """
    if ranked_labels.ndim != 2:
        raise ValueError(
            f"ranked_labels must be (num_queries, num_ranked), got {tuple(ranked_labels.shape)}"
        )
    if len(ranked_labels) != len(query_labels):
        raise ValueError(f"Got {len(ranked_labels)} ranked rows for {len(query_labels)} queries")

    num_ranked = ranked_labels.shape[1]
    hits = ranked_labels == query_labels[:, None]

    metrics: MetricsDict = {}
    for k in ks:
        if k > num_ranked:
            continue
        metrics[f"recall@{k}"] = hits[:, :k].any(dim=1).float().mean().item()
    return metrics


def mean_average_precision(
    ranked_labels: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    """mAP over all queries.

    Average precision is computed over the full ranking, so it rewards putting
    *every* same-class item near the top — unlike recall@k, which saturates on
    the first hit. A query with no same-class item anywhere in the gallery
    contributes 0, which keeps mAP comparable across datasets where some
    classes are singletons.
    """
    hits = (ranked_labels == query_labels[:, None]).float()
    ranks = torch.arange(1, hits.shape[1] + 1, device=hits.device, dtype=hits.dtype)

    precision_at_hit = hits.cumsum(dim=1) / ranks
    relevant = hits.sum(dim=1)
    # clamp guards the 0/0 for a query with no relevant item; the numerator is
    # already 0 there, so the result stays 0.
    average_precision = (precision_at_hit * hits).sum(dim=1) / relevant.clamp(min=1)
    return average_precision.mean().item()
