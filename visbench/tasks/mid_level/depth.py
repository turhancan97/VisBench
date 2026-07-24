"""Monocular depth estimation — v0.2, not implemented.

Evaluation protocol to be taken from probe3d (El Banani et al., CVPR 2024,
arXiv:2404.08476) rather than re-derived; metrics live in
:mod:`visbench.metrics.dense`.

Requires a dense training loop, which is explicitly outside the v0.1 boundary.
"""

from typing import Any, Optional

from visbench.tasks.base import BaseTask
from visbench.types import FeatureMode, MetricsDict

__all__ = ["DepthTask"]


class DepthTask(BaseTask):
    """Pixel-wise depth from a single image, via a pluggable head (linear or DPT)."""

    level = "mid_level"
    feature_mode = FeatureMode.DENSE_ONLY
    zero_shot = False

    def predict(self, features: Any) -> Any:
        raise NotImplementedError("Depth estimation lands in v0.2.")

    def evaluate(self, features: Any, labels: Optional[Any] = None) -> MetricsDict:
        """Will return ``{"d1", "d2", "d3", "rmse", ...}`` per probe3d."""
        raise NotImplementedError("Depth estimation lands in v0.2.")
