"""Semantic (multi-class) segmentation — v0.2, not implemented.

Exists in v0.1 as a stub so that the high-level/mid-level pairing is visible in
the structure: this is the multi-class counterpart to mid-level *generic*
(binary) object segmentation, and the two are meant to be compared directly on
the same backbone (CLAUDE.md, v0.2).
"""

from typing import Any

from visbench.tasks.base import BaseTask
from visbench.types import FeatureMode, MetricsDict

__all__ = ["SemanticSegmentationTask"]


class SemanticSegmentationTask(BaseTask):
    """Dense per-pixel category prediction. Deferred to v0.2.

    Not registered yet — registering it would advertise a task that raises.
    """

    level = "high_level"
    feature_mode = FeatureMode.DENSE_ONLY
    zero_shot = False

    def predict(self, features: Any) -> Any:
        raise NotImplementedError("Semantic segmentation lands in v0.2.")

    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        raise NotImplementedError("Semantic segmentation lands in v0.2.")
