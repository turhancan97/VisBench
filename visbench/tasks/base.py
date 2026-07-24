"""The task (probe) abstraction.

A task owns the decision of *what representation it needs* — it passes
``pooling`` and ``feature_mode`` down into ``extract_features()``. Backbones
never choose. This keeps that decision in one place and backbones swappable
(CLAUDE.md, "BaseTask").
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from visbench.types import FeatureMode, MetricsDict, Pooling

__all__ = ["BaseTask"]


class BaseTask(ABC):
    """Base class for every probe, zero-shot or trained.

    The three-method contract is uniform across task levels; zero-shot tasks
    (retrieval, correspondence) simply no-op in :meth:`fit`.
    """

    #: Registered name. Set per instance in ``__init__``, matching the backbone
    #: convention (see :mod:`visbench.registry`).
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

    #: Whether the task reads the dense grid rather than the pooled vector.
    #: Declared, not inferred: it tells the cache which half of the extraction
    #: to keep and store, and dense features are ~250x the size of pooled ones,
    #: so guessing it from the task's level would be an expensive guess.
    uses_dense: bool = False

    def fit(self, features: Any, labels: Optional[Any] = None) -> "BaseTask":
        """Train the probe head on cached features.

        No-op for zero-shot tasks; returns ``self`` so calls can chain.
        Features arrive pre-extracted and cached — a task never runs a backbone
        forward pass itself.
        """
        if self.zero_shot:
            return self
        raise NotImplementedError(
            f"{type(self).__name__} is not zero-shot and must implement fit()"
        )

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
        """Whether :meth:`evaluate` needs ground-truth labels.

        Note this is not the same question as :attr:`zero_shot`: retrieval
        needs no *training* labels but cannot be scored without them.
        """
        return True

    def describe(self) -> dict:
        """Task metadata (name, level, pooling, feature mode) for the result record.

        ``task_params`` is always present, empty by default: a caller building
        a record should never have to ask whether this particular task has
        hyperparameters. Trained probes override it with theirs.
        """
        return {
            "task": self.name,
            "level": self.level,
            "pooling": self.pooling,
            "feature_mode": self.feature_mode,
            "zero_shot": self.zero_shot,
            "task_params": {},
        }

    # -- shared helpers ------------------------------------------------------

    @staticmethod
    def _as_pooled(features: Any) -> torch.Tensor:
        """Accept a :class:`FeatureDict` or a bare tensor, return ``(N, C)``.

        Tasks are handed whatever ``FeatureCache.extract_dataset`` returned, but
        are also useful on raw tensors in tests and notebooks; normalising here
        keeps that branch out of every task.
        """
        if isinstance(features, dict):
            if "pooled" not in features:
                raise KeyError(
                    "Feature dict has no 'pooled' entry. If it came from "
                    "extract_dataset(keep='dense'), re-extract with "
                    "keep='both' or keep='pooled'."
                )
            pooled = features["pooled"]
        else:
            pooled = features

        if not isinstance(pooled, torch.Tensor):
            raise TypeError(f"Expected pooled features as a tensor, got {type(pooled).__name__}")
        if pooled.ndim != 2:
            raise ValueError(f"Expected pooled features of shape (N, C), got {tuple(pooled.shape)}")
        return pooled

    @staticmethod
    def _as_label_tensor(labels: Any) -> torch.Tensor:
        """Coerce a list of labels to a 1-D tensor, rejecting missing entries."""
        if labels is None:
            raise ValueError("This task requires labels; got None")
        if isinstance(labels, torch.Tensor):
            tensor = labels
        else:
            values = list(labels)
            if any(value is None for value in values):
                raise ValueError(
                    "Labels contain None. An unlabeled ImageFolderDataset "
                    "(labeled=False) cannot be scored."
                )
            tensor = torch.as_tensor(values)
        if tensor.ndim != 1:
            raise ValueError(f"Expected labels of shape (N,), got {tuple(tensor.shape)}")
        return tensor
