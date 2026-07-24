"""Image classification — linear probe on cached pooled features. v0.1.

The canonical "does this representation separate categories" probe. Trains a
single linear layer; the backbone stays frozen and is never re-run.
"""

from typing import Any, Optional

from visbench.tasks.base import BaseTask
from visbench.types import MetricsDict, Pooling

__all__ = ["ClassificationTask"]


# Activated at build step 3, once ``register_task`` has a body.
# @register_task("classification")
class ClassificationTask(BaseTask):
    """Linear probe over pooled features.

    Uses the backbone's default pooling (CLS for ViTs) unless overridden —
    worth revisiting per backbone, since mean-pooled patch tokens sometimes
    beat CLS for classification.
    """

    level = "high_level"
    pooling = Pooling.DEFAULT
    zero_shot = False

    def __init__(
        self,
        num_classes: int,
        pooling: str = Pooling.DEFAULT,
        max_iter: int = 1000,
    ) -> None:
        """Configure the linear head; weights are created lazily in :meth:`fit`."""
        raise NotImplementedError

    def fit(self, features: Any, labels: Optional[Any] = None) -> "ClassificationTask":
        """Fit the linear classifier on ``(N, C)`` pooled features."""
        raise NotImplementedError

    def predict(self, features: Any) -> Any:
        """Return predicted class indices, ``(N,)``."""
        raise NotImplementedError

    def evaluate(self, features: Any, labels: Optional[Any] = None) -> MetricsDict:
        """Return ``{"top1": ..., "top5": ...}``."""
        raise NotImplementedError
