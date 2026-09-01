"""The task (probe) abstraction.

A task owns the decision of *what representation it needs* — it passes
``pooling`` and ``feature_mode`` down into ``extract_features()``. Backbones
never choose. This keeps that decision in one place and backbones swappable
(CLAUDE.md, "BaseTask").
"""

from abc import ABC, abstractmethod
from typing import Any

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

    #: Whether the task consumes **image pairs plus geometry** rather than
    #: images plus labels. Declared for the same reason ``uses_dense`` is: it
    #: decides how :func:`visbench.run` extracts, and the two shapes are not
    #: distinguishable from anything else the task exposes.
    #:
    #: Correspondence is the only task with it set, and one explicit fork is the
    #: honest way to carry a genuinely different data shape — the alternative,
    #: a general "dataset adapter" mechanism, would be built to fit one case
    #: while pretending to anticipate others.
    uses_pairs: bool = False

    #: How many trailing backbone blocks this probe fine-tunes. ``0`` — every
    #: task that is not a trained dense one, and the default everywhere — means
    #: frozen features, which is what makes them cacheable. Declared on the base
    #: so :func:`visbench.run` can ask any probe without a hasattr dance.
    finetune_blocks: int = 0

    #: Backbone depths this task wants, or ``None`` for the last layer alone.
    #: A multiscale head needs several, and the task is what knows that — the
    #: same reason pooling is chosen here rather than on the backbone.
    layers: list[int] | None = None

    def fit(self, features: Any, labels: Any | None = None) -> "BaseTask":
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
    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """Score predictions and return a **flat** metrics dict.

        Never prints or logs results directly — the caller writes the
        structured JSON record (see :mod:`visbench.results`). Flat because
        nested metrics make leaderboard schemas painful later.
        """
        raise NotImplementedError

    def context_metrics(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """Numbers that *qualify* the score without being it, merged into the record.

        Empty for almost every task. It exists because correspondence's score is
        close to meaningless on its own: matches can only land on patch centres,
        so ``recall@1px`` on DINOv2 ViT-S/14 at 224px has a ceiling of 0.015, and
        a reader who sees the score without the ceiling concludes the backbone
        failed when the grid is simply coarse.

        ``examples/correspond.py`` has always printed both. This is what makes
        :func:`visbench.run` — and therefore the CLI — do the same, rather than
        the qualification being available only to whoever read the example.

        **The dense probes that declare an oracle answer it too**, for the same
        reason one step removed: their head reads one feature vector per patch,
        so part of the target is out of reach before the backbone is chosen and
        how much varies by backbone. See
        :meth:`~visbench.tasks.dense_base.DenseTrainingTask.context_metrics`.

        Keys should be namespaced by the implementer (both implementers prefix
        ``ceiling_``), since these land in the same flat metrics dict as the
        score and a collision would overwrite a result.

        **What comes back here is never a score.** It qualifies one, so it must
        not be ranked, averaged or used as a denominator — a ceiling falls with
        the feature grid, and a board ordered by one would be a board ordered by
        resolution.
        """
        return {}

    def finetune(self) -> dict | None:
        """What this run unfroze, or ``None`` for a frozen probe.

        ``None`` for every task that cannot fine-tune, which is the same value
        a v0.1 or v0.2 record carries by absence — so a reader distinguishing
        frozen from fine-tuned numbers gets one answer for both.
        """
        return None

    def training_summary(self) -> dict | None:
        """How the fit itself went, or ``None`` for a probe that trains nothing.

        This is the diagnostic that separates the two readings of a low score,
        and the library says everywhere that it is the one to check: a probe
        that has not converged **understates** a backbone, which is a different
        finding from a representation that genuinely does not carry the answer.
        ``GenericSegmentationTask`` on 80 images reads 0.16 IoU at the defaults
        and 0.87 at ``epochs=40`` on identical features — same number, opposite
        conclusion, and ``train_loss`` is what tells them apart.

        Until schema v8 every probe computed this and then dropped it on the
        floor: it reached a log line and never the record, so a corpus of 156
        trained runs could not answer "did this underfit?" without re-running
        them. Found while writing up the CUB board, where the claim could be
        made for the six backbones run by hand and not for the six run on the
        cluster.

        **An open dict, deliberately**, for the reason ``task_params`` is one:
        every trained probe here has ``train_loss``, the classification family
        also has ``train_top1``, and a future probe's natural diagnostic should
        not need another schema bump. Keys are the implementer's choice; values
        must be JSON primitives.

        ``None`` rather than ``{}`` for the zero-shot probes — retrieval,
        correspondence and similarity fit nothing, so there is no fit to
        describe, and that is a different statement from "trained and reported
        nothing about it". Same convention as :meth:`finetune`.
        """
        return None

    def head_spec(self) -> dict | None:
        """How to rebuild this probe's trained head, or ``None`` if it has none.

        Serialising a ``state_dict`` alone is not enough to reload a probe: the
        module has to exist first, at exactly the shape the weights expect. This
        returns the recipe, as JSON-primitives only, so the whole artifact can
        be read back under ``torch.load(weights_only=True)`` — a head fetched
        from a hub must not be able to execute code on the way in.

        Two kinds, both explicit rather than inferred:

        - ``{"kind": "registered", "name": ..., "kwargs": {...}}`` for anything
          built through :func:`visbench.heads.build_head`.
        - ``{"kind": "linear", "in_features": ..., "out_features": ...}`` for
          the bare ``nn.Linear`` the classification probe fits on pooled
          features, which is not a registered dense head and would not survive
          being pretended into one.

        ``None`` is the honest answer for a zero-shot probe. Retrieval,
        correspondence and similarity train nothing, so there is nothing to
        share and :func:`visbench.hub.save_probe` refuses them by name rather
        than writing an artifact holding no weights.
        """
        return None

    def probe_state(self) -> dict[str, torch.Tensor]:
        """Fitted tensors that are **not** in the head's ``state_dict``.

        Empty for most probes, and not an abstraction added speculatively:
        :class:`ClassificationTask` fits a feature standardiser (``_mean``,
        ``_std``) on its training split and applies it in ``predict``. A head
        saved without those loads cleanly, has the right shapes, and scores
        against unstandardised features — confidently wrong, with nothing to
        say so. Any per-probe state learned outside ``self.head`` belongs here.
        """
        return {}

    def load_probe_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore what :meth:`probe_state` saved.

        Raises on an unexpected key rather than ignoring it: a probe told to
        restore state it does not understand is a probe about to run without it.
        """
        if state:
            raise ValueError(
                f"{type(self).__name__} has no probe state to restore, but the "
                f"artifact carries {sorted(state)}. Refusing rather than "
                "dropping it: the weights were fitted alongside these tensors."
            )

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
            "layers": self.layers,
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
