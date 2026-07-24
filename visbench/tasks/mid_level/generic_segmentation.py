"""Generic (binary) object segmentation — v0.2, not implemented.

Figure-ground separation with **no semantics**: split object from background
without naming either. This is what makes it mid-level, and what distinguishes
it from :mod:`visbench.tasks.high_level.semantic_segmentation`. The two should
be runnable on the same backbone for direct comparison.
"""

from typing import Any, Optional

from visbench.tasks.base import BaseTask
from visbench.types import FeatureMode, MetricsDict

__all__ = ["GenericSegmentationTask"]


class GenericSegmentationTask(BaseTask):
    """Binary foreground/background prediction over dense features."""

    level = "mid_level"
    feature_mode = FeatureMode.DENSE_ONLY
    zero_shot = False

    def predict(self, features: Any) -> Any:
        raise NotImplementedError("Generic object segmentation lands in v0.2.")

    def evaluate(self, features: Any, labels: Optional[Any] = None) -> MetricsDict:
        """Will return ``{"iou", "pixel_acc", ...}``."""
        raise NotImplementedError("Generic object segmentation lands in v0.2.")
