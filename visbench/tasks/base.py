"""The task (probe) abstraction.

A task owns the decision of *what representation it needs* — it passes
``pooling`` and ``feature_mode`` down into ``extract_features()``. Backbones
never choose. This keeps that decision in one place and backbones swappable
(CLAUDE.md, "BaseTask").
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from visbench.types import FeatureMode, MetricsDict, Pooling

__all__ = ["BaseTask"]


class BaseTask(ABC):
    """Base class for every probe, zero-shot or trained.

    The three-method contract is uniform across task levels; zero-shot tasks
    (retrieval, correspondence) simply no-op in :meth:`fit`.
    """

    #: Registered name, set by the ``@register_task`` decorator.
    name: str = ""

    #: ``"high_level"`` | ``"mid_level"`` | ``"low_level"`` — recorded in the
    #: result log so leaderboard tooling can group without a lookup table.
    level: str = ""

    #: Pooling this task requests from the backbone. ``DEFAULT`` defers to the
    #: architecture; dense tasks override to use the grid instead.
    pooling: str = Pooling.DEFAULT

    #: Dense feature presentation. Ignored by tasks that only use ``pooled``.
    feature_mode: str = FeatureMode.DENSE_ONLY

    #: True for tasks where :meth:`fit` is a no-op.
    zero_shot: bool = False

    def fit(self, features: Any, labels: Optional[Any] = None) -> "BaseTask":
        """Train the probe head on cached features.

        No-op for zero-shot tasks; returns ``self`` so calls can chain.
        Features arrive pre-extracted and cached — a task never runs a backbone
        forward pass itself.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: Any) -> Any:
        """Produce predictions for the given features."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, features: Any, labels: Optional[Any] = None) -> MetricsDict:
        """Score predictions and return a **flat** metrics dict.

        Never prints or logs results directly — the caller writes the
        structured JSON record (see :mod:`visbench.results`). Flat because
        nested metrics make leaderboard schemas painful later.
        """
        raise NotImplementedError

    def requires_labels(self) -> bool:
        """Whether :meth:`evaluate` needs ground-truth labels."""
        raise NotImplementedError

    def describe(self) -> dict:
        """Task metadata (name, level, pooling, feature mode) for the result record."""
        raise NotImplementedError
