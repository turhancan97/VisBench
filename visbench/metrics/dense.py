"""Dense-prediction metrics (depth, surface normals, segmentation) — v0.2.

Definitions to be taken from probe3d (arXiv:2404.08476) rather than
re-derived, so VisBench numbers stay comparable to published ones. Small
differences in masking and averaging convention move these numbers
noticeably — match the reference implementation, and say so in comments.
"""

from collections.abc import Sequence

import torch

from visbench.types import MetricsDict

__all__ = ["depth_metrics", "surface_normal_metrics", "binary_iou"]


def depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: Sequence[float] = (1.25, 1.25**2, 1.25**3),
) -> MetricsDict:
    """``{"d1", "d2", "d3", "rmse", "abs_rel"}`` over valid pixels only."""
    raise NotImplementedError("Depth metrics land in v0.2.")


def surface_normal_metrics(pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
    """Angular error mean/median plus within-11.25/22.5/30-degree percentages."""
    raise NotImplementedError("Surface normal metrics land in v0.2.")


def binary_iou(pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
    """Foreground IoU and pixel accuracy for generic object segmentation."""
    raise NotImplementedError("Segmentation metrics land in v0.2.")
