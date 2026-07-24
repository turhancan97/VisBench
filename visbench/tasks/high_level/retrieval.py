"""Image retrieval — zero-shot cosine similarity over pooled features. v0.1.

Semantic retrieval: is the retrieved image the same *category*. Deliberately
distinct from mid-level image similarity (:mod:`visbench.tasks.mid_level.similarity`),
which judges perceptual/geometric resemblance instead. Do not merge the two
even though both are "similarity"-flavored (CLAUDE.md, "Task categorization").
"""

from typing import Any, Optional

import torch

from visbench.metrics.retrieval import mean_average_precision, recall_at_k
from visbench.registry import register_task
from visbench.tasks.base import BaseTask
from visbench.types import MetricsDict, Pooling

__all__ = ["RetrievalTask"]

_METRICS = ("cosine", "l2")


@register_task("retrieval")
class RetrievalTask(BaseTask):
    """Nearest-neighbour retrieval in pooled feature space, no training."""

    level = "high_level"
    zero_shot = True

    def __init__(
        self,
        topk: tuple = (1, 5, 10),
        metric: str = "cosine",
        pooling: str = Pooling.DEFAULT,
    ) -> None:
        """Configure the ranking metric and the k values reported."""
        if metric not in _METRICS:
            raise ValueError(f"Unknown metric {metric!r}; expected one of {_METRICS}")
        if not topk:
            raise ValueError("topk must contain at least one k")
        if any(k < 1 for k in topk):
            raise ValueError(f"topk values must be >= 1, got {topk}")

        self.name = "retrieval"
        self.topk = tuple(sorted(topk))
        self.metric = metric
        self.pooling = pooling

    def fit(self, features: Any, labels: Optional[Any] = None) -> "RetrievalTask":
        """No-op — retrieval is zero-shot. Returns ``self``."""
        return self

    def _similarity(self, queries: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
        """``(num_queries, num_gallery)`` scores, higher is better."""
        queries = queries.float()
        gallery = gallery.float()
        if self.metric == "cosine":
            queries = torch.nn.functional.normalize(queries, dim=1)
            gallery = torch.nn.functional.normalize(gallery, dim=1)
            return queries @ gallery.T
        # Negated distance, so "higher is better" holds for both metrics and
        # ranking never needs to branch on which one is active.
        return -torch.cdist(queries, gallery)

    def predict(
        self,
        features: Any,
        gallery_features: Optional[Any] = None,
    ) -> torch.Tensor:
        """Return ranked gallery indices per query, ``(N_query, N_ranked)``.

        With no ``gallery_features``, this is leave-one-out retrieval over a
        single set: every image queries every *other* image. Self-matches are
        removed, not merely down-weighted — leaving them in makes recall@1
        trivially 1.0 and the whole metric meaningless.
        """
        queries = self._as_pooled(features)
        self_retrieval = gallery_features is None
        gallery = queries if self_retrieval else self._as_pooled(gallery_features)

        if queries.shape[1] != gallery.shape[1]:
            raise ValueError(
                f"Query and gallery feature dims differ ({queries.shape[1]} vs "
                f"{gallery.shape[1]}); they must come from the same backbone."
            )
        if self_retrieval and len(queries) < 2:
            raise ValueError("Leave-one-out retrieval needs at least 2 images")

        scores = self._similarity(queries, gallery)
        if self_retrieval:
            scores.fill_diagonal_(float("-inf"))

        ranking = scores.argsort(dim=1, descending=True)
        # Drop the last column, which is the masked self-match for every row.
        return ranking[:, :-1] if self_retrieval else ranking

    def evaluate(
        self,
        features: Any,
        labels: Optional[Any] = None,
        gallery_features: Optional[Any] = None,
        gallery_labels: Optional[Any] = None,
    ) -> MetricsDict:
        """Return ``{"recall@1": ..., "recall@5": ..., "mAP": ...}``."""
        query_labels = self._as_label_tensor(labels)
        queries = self._as_pooled(features)
        if len(queries) != len(query_labels):
            raise ValueError(f"Got {len(queries)} features for {len(query_labels)} labels")

        ranking = self.predict(features, gallery_features=gallery_features)

        if gallery_features is None:
            lookup = query_labels
        else:
            lookup = self._as_label_tensor(gallery_labels)
            if len(lookup) != len(self._as_pooled(gallery_features)):
                raise ValueError("gallery_features and gallery_labels have different lengths")

        ranked_labels = lookup[ranking]

        metrics: MetricsDict = dict(recall_at_k(ranked_labels, query_labels, ks=self.topk))
        metrics["mAP"] = mean_average_precision(ranked_labels, query_labels)
        return metrics
