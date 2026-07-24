"""Surface normal estimation — v0.2, not implemented.

Protocol from probe3d (arXiv:2404.08476). Note that NYU surface normals
conventionally come from GeoNet's extracted normals, not the raw dataset —
record the normal source in the result log, since it changes the numbers.
"""

from typing import Any, Optional

from visbench.tasks.base import BaseTask
from visbench.types import FeatureMode, MetricsDict

__all__ = ["SurfaceNormalTask"]


class SurfaceNormalTask(BaseTask):
    """Pixel-wise 3D surface orientation, via a pluggable head."""

    level = "mid_level"
    feature_mode = FeatureMode.DENSE_ONLY
    zero_shot = False

    def predict(self, features: Any) -> Any:
        raise NotImplementedError("Surface normal estimation lands in v0.2.")

    def evaluate(self, features: Any, labels: Optional[Any] = None) -> MetricsDict:
        """Will return angular error mean/median and within-threshold percentages."""
        raise NotImplementedError("Surface normal estimation lands in v0.2.")
