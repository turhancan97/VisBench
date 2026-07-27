"""Mid-level image similarity — v0.2, not implemented.

Judges perceptual/geometric resemblance (scene layout, geometry, viewpoint),
**not** category membership. This is a distinct task class from high-level
retrieval and the two must not be merged, even though both are
"similarity"-flavored (CLAUDE.md, "Task categorization").

Typically evaluated as a two-alternative forced choice against human
judgements.
"""

from typing import Any

from visbench.tasks.base import BaseTask
from visbench.types import MetricsDict, Pooling

__all__ = ["MidLevelSimilarityTask"]


class MidLevelSimilarityTask(BaseTask):
    """Reference-vs-two-candidates perceptual similarity in frozen feature space."""

    level = "mid_level"
    pooling = Pooling.DEFAULT
    zero_shot = True

    def predict(self, features: Any) -> Any:
        raise NotImplementedError("Mid-level image similarity lands in v0.2.")

    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """Will return ``{"2afc_agreement": ...}`` against human judgements."""
        raise NotImplementedError("Mid-level image similarity lands in v0.2.")
