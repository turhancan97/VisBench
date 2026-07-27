"""Object detection — v0.3 groundwork, not implemented.

Expected to take longer than any other single addition: it is the hardest task
to do cheaply on limited compute (CLAUDE.md, v0.3). Stubbed now only so the
structure does not need rearranging later.
"""

from typing import Any

from visbench.tasks.base import BaseTask
from visbench.types import FeatureMode, MetricsDict

__all__ = ["DetectionTask"]


class DetectionTask(BaseTask):
    """Lightweight detection head over frozen features. Deferred to v0.3."""

    level = "high_level"
    feature_mode = FeatureMode.DENSE_ONLY
    zero_shot = False

    def predict(self, features: Any) -> Any:
        raise NotImplementedError("Detection lands in v0.3.")

    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        raise NotImplementedError("Detection lands in v0.3.")
