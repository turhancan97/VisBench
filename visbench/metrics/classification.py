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

    Clamps k to the number of classes so that top-5 on a 3-class dataset
    reports something meaningful instead of erroring.
    """
    raise NotImplementedError
