"""Image retrieval — zero-shot cosine similarity over pooled features. v0.1.

Semantic retrieval: is the retrieved image the same *category*. Deliberately
distinct from mid-level image similarity (:mod:`visbench.tasks.mid_level.similarity`),
which judges perceptual/geometric resemblance instead. Do not merge the two
even though both are "similarity"-flavored (CLAUDE.md, "Task categorization").
"""

from typing import Any, Optional

from visbench.tasks.base import BaseTask
from visbench.types import MetricsDict, Pooling

__all__ = ["RetrievalTask"]


# Activated at build step 3, once ``register_task`` has a body.
# @register_task("retrieval")
class RetrievalTask(BaseTask):
    """Nearest-neighbour retrieval in pooled feature space, no training."""

    level = "high_level"
    pooling = Pooling.DEFAULT
    zero_shot = True

    def __init__(self, topk: tuple = (1, 5, 10), metric: str = "cosine") -> None:
        """Configure the ranking metric and the k values reported."""
        raise NotImplementedError

    def fit(self, features: Any, labels: Optional[Any] = None) -> "RetrievalTask":
        """No-op — retrieval is zero-shot. Returns ``self``."""
        raise NotImplementedError

    def predict(self, features: Any) -> Any:
        """Return ranked gallery indices per query, ``(N_query, max(topk))``."""
        raise NotImplementedError

    def evaluate(self, features: Any, labels: Optional[Any] = None) -> MetricsDict:
        """Return ``{"recall@1": ..., "recall@5": ..., "mAP": ...}``."""
        raise NotImplementedError
