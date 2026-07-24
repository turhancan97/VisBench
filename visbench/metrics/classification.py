"""Classification metrics."""

from collections.abc import Sequence

import torch

from visbench.types import MetricsDict

__all__ = ["top_k_accuracy"]


def top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Sequence[int] = (1, 5),
) -> MetricsDict:
    """Top-k accuracy for each k, as ``{"top1": ..., "top5": ...}``.

    A ``k`` at or above the number of classes is dropped rather than reported:
    top-5 on a 3-class dataset is 1.0 by construction, and a metric that is
    always 1.0 on a leaderboard is worse than an absent one.

    Note this differs from :func:`visbench.metrics.retrieval.recall_at_k`,
    which drops ``k`` larger than the *gallery*. Same instinct, different
    quantity: there, k can exceed what is rankable; here, k can exceed what is
    distinguishable.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be (N, num_classes), got {tuple(logits.shape)}")
    if len(logits) != len(targets):
        raise ValueError(f"Got {len(logits)} predictions for {len(targets)} targets")

    num_classes = logits.shape[1]
    usable = [k for k in ks if k < num_classes]
    if not usable:
        # Every requested k is degenerate; report top-1 so the caller still
        # gets a number rather than an empty dict.
        usable = [1]

    largest = max(usable)
    ranked = logits.topk(largest, dim=1).indices
    hits = ranked == targets[:, None]

    return {f"top{k}": hits[:, :k].any(dim=1).float().mean().item() for k in usable}
