"""Geometric correspondence — zero-shot dense feature matching. v0.1.

The only v0.1 task that uses dense features, which makes it the real test of
the dense path in :meth:`BaseBackbone.extract_features`.

Matching logic is conceptually the same as vismatch
(https://github.com/gmberton/vismatch), applied to raw backbone features
instead of dedicated matcher networks. Evaluation protocol follows probe3d
(El Banani et al., CVPR 2024, arXiv:2404.08476).
"""

from typing import Any, Optional

from visbench.tasks.base import BaseTask
from visbench.types import FeatureMode, MetricsDict

__all__ = ["CorrespondenceTask"]


# Activated at build step 3, once ``register_task`` has a body.
# @register_task("correspondence")
class CorrespondenceTask(BaseTask):
    """Match dense features between image pairs and score against known geometry.

    Given a pair of views of the same scene/object, nearest-neighbour match
    patch features (with a ratio test to reject ambiguous matches), then score
    the surviving matches against ground-truth geometry.
    """

    level = "mid_level"
    feature_mode = FeatureMode.DENSE_ONLY
    zero_shot = True

    def __init__(
        self,
        num_corr: int = 1000,
        ratio_threshold: float = 0.9,
        thresholds: tuple = (1, 2, 5, 10),
    ) -> None:
        """Configure how many correspondences to keep and the error thresholds scored."""
        raise NotImplementedError

    def fit(self, features: Any, labels: Optional[Any] = None) -> "CorrespondenceTask":
        """No-op — correspondence is zero-shot. Returns ``self``."""
        raise NotImplementedError

    def predict(self, features: Any) -> Any:
        """Return matched patch-index pairs and their similarity scores."""
        raise NotImplementedError

    def evaluate(self, features: Any, labels: Optional[Any] = None) -> MetricsDict:
        """Return recall at each pixel threshold plus the error AUC."""
        raise NotImplementedError
