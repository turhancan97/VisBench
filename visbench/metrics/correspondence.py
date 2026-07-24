"""Correspondence matching and metrics.

Protocol follows probe3d (El Banani et al., CVPR 2024, arXiv:2404.08476).
Nearest-neighbour matching with a ratio test is standard practice inherited
from classical local-feature matching, and is also what vismatch
(https://github.com/gmberton/vismatch) does with dedicated matchers.
"""

from collections.abc import Sequence

import torch

from visbench.types import MetricsDict

__all__ = ["nn_match", "ratio_test", "correspondence_recall", "error_auc"]


def nn_match(feats_0: torch.Tensor, feats_1: torch.Tensor, k: int = 2):
    """k-nearest-neighbour match every feature in ``feats_0`` against ``feats_1``.

    ``k=2`` by default because the ratio test needs the second neighbour.
    Returns ``(distances, indices)``, both ``(N, k)``.
    """
    raise NotImplementedError


def ratio_test(distances: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    """Lowe's ratio test: keep matches whose first/second neighbour ratio is low.

    Rejects matches in repetitive regions, where the nearest neighbour is no
    more convincing than the runner-up.
    """
    raise NotImplementedError


def correspondence_recall(
    errors: torch.Tensor,
    thresholds: Sequence[float] = (1, 2, 5, 10),
) -> MetricsDict:
    """Fraction of correspondences within each pixel-error threshold."""
    raise NotImplementedError


def error_auc(errors: torch.Tensor, thresholds: Sequence[float]) -> MetricsDict:
    """Area under the cumulative error curve up to each threshold.

    Summarises the whole error distribution rather than a single cut-off, which
    is why probe3d reports it alongside thresholded recall.
    """
    raise NotImplementedError
