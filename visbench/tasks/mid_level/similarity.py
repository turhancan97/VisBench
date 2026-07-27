"""Mid-level image similarity — zero-shot 2AFC, v0.2.

Judges perceptual/geometric resemblance (scene layout, geometry, viewpoint),
**not** category membership. This is a distinct task class from high-level
retrieval and the two must not be merged, even though both are
"similarity"-flavoured (CLAUDE.md, "Task categorization"). The difference is in
the ground truth, not the machinery: retrieval scores against class labels,
this against human judgements of which of two candidates *looks* more like a
reference.

**Zero-shot, and that is not an approximation.** The protocol in Chen, Marks &
Cheng (arXiv:2411.17474) trains nothing — its ``evaluate_model_percepture.py``
builds a test loader, freezes the backbone, and compares two cosine
similarities. Their README says "train a mid-level image similarity estimator",
which the code does not do; the code is what was followed here. There is no
head, and ``fit`` is a no-op like retrieval's and correspondence's.

The whole probe is two cosine similarities and a comparison::

    prefer_right = cos(ref, right) > cos(ref, left)

scored against the human vote as binary classification. That is deliberately
the same shape as the reference so the numbers are comparable; a margin or a
correlation would measure something else, and something no published number
exists for.
"""

from typing import Any

import torch
import torch.nn.functional as F

from visbench.metrics.similarity import two_afc_metrics
from visbench.registry import register_task
from visbench.tasks.base import BaseTask
from visbench.types import MetricsDict, Pooling

__all__ = ["MidLevelSimilarityTask"]


@register_task("similarity")
class MidLevelSimilarityTask(BaseTask):
    """Reference-vs-two-candidates perceptual similarity in frozen feature space.

    Reads **pooled** features — CLS on a ViT, mean on a CNN — because a single
    global vector per image is what the comparison needs and what the reference
    implementation uses. ``uses_dense`` is therefore False, so extraction keeps
    the cheap half of the cache.
    """

    level = "mid_level"
    pooling = Pooling.DEFAULT
    zero_shot = True
    uses_dense = False

    def __init__(self, min_votes: int | None = None) -> None:
        """``min_votes`` is recorded, not applied.

        The filter belongs to the dataset, which is where the triplets are; but
        a score computed over near-unanimous triplets is not comparable to one
        computed over contested ones, so the value travels with the record. Pass
        the dataset's ``min_votes`` here, or leave it and the record says
        nothing rather than something wrong.
        """
        self.name = "similarity"
        self.min_votes = min_votes

    def fit(self, features: Any, labels: Any | None = None) -> "MidLevelSimilarityTask":
        """No-op: nothing is trained. Present so every task has the same shape."""
        return self

    def _pooled(self, features: Any) -> torch.Tensor:
        """The ``(N, C)`` pooled block, from a feature dict or a bare tensor."""
        pooled = features["pooled"] if isinstance(features, dict) else features
        if not isinstance(pooled, torch.Tensor):
            raise TypeError(f"Expected pooled features as a tensor, got {type(pooled).__name__}")
        if pooled.ndim != 2:
            raise ValueError(f"Expected (N, C) pooled features, got {tuple(pooled.shape)}")
        return pooled

    @staticmethod
    def _as_triplets(labels: Any) -> torch.Tensor:
        triplets = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)
        if triplets.ndim != 2 or triplets.shape[1] != 4:
            raise ValueError(
                f"Expected (T, 4) triplets of (ref, left, right, vote), got {tuple(triplets.shape)}"
            )
        return triplets.long()

    def _similarities(
        self, features: Any, labels: Any | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(left, right, triplets)`` — the whole probe, computed once.

        Shared by :meth:`predict` and :meth:`evaluate` rather than each deriving
        its own, so the choice they report and the ties they count can never be
        computed from different numbers.
        """
        if labels is None:
            raise ValueError(
                "Mid-level similarity needs the triplets naming which images to compare; "
                "pass dataset.labels()."
            )
        pooled = self._pooled(features)
        triplets = self._as_triplets(labels)

        indices = triplets[:, :3]
        out_of_range = indices[(indices < 0) | (indices >= len(pooled))]
        if len(out_of_range):
            raise IndexError(
                f"{len(out_of_range)} triplet index/indices fall outside the {len(pooled)} "
                "extracted features. The triplets and the features must come from the "
                "same dataset."
            )

        reference = pooled[triplets[:, 0]]
        left = F.cosine_similarity(reference, pooled[triplets[:, 1]], dim=-1)
        right = F.cosine_similarity(reference, pooled[triplets[:, 2]], dim=-1)
        return left, right, triplets

    def predict(self, features: Any, labels: Any | None = None) -> torch.Tensor:
        """``(T,)`` of 0 (left preferred) or 1 (right preferred).

        ``labels`` carries the triplet structure, not supervision — the indices
        say which three of the extracted images form each comparison. It is
        required, because a set of feature vectors alone does not say what is
        being compared with what.
        """
        left, right, _ = self._similarities(features, labels)
        # Ties go to the *left* candidate here, where the reference sends them
        # right (`torch.where(sim_left > sim_right, 0, 1)`). Either is arbitrary
        # and neither should decide a number, which is why `evaluate` reports
        # how often it happened instead of leaving the choice invisible.
        return (right > left).long()

    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """Agreement with the human vote, as binary classification."""
        left, right, triplets = self._similarities(features, labels)
        predictions = (right > left).long()
        return two_afc_metrics(predictions, triplets[:, 3], ties=left == right)

    def describe(self) -> dict:
        described = super().describe()
        described["task_params"] = {
            "protocol": "midvision_2afc",
            "similarity": "cosine",
            "min_votes": self.min_votes,
        }
        return described
