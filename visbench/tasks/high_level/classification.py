"""Image classification — linear probe on cached pooled features. v0.1.

The canonical "does this representation separate categories" probe. Trains a
single linear layer; the backbone stays frozen and is never re-run.
"""

from typing import Any

import torch
import torch.nn as nn

from visbench.metrics.classification import top_k_accuracy
from visbench.registry import register_task
from visbench.tasks.base import BaseTask
from visbench.types import MetricsDict, Pooling
from visbench.utils.device import resolve_device

__all__ = ["ClassificationTask"]


@register_task("classification")
class ClassificationTask(BaseTask):
    """Linear probe over pooled features.

    Uses the backbone's default pooling (CLS for ViTs) unless overridden —
    worth revisiting per backbone, since mean-pooled patch tokens sometimes
    beat CLS for classification.

    Trained with AdamW on the cached features. That makes the reported number
    depend on the optimiser settings, so every one of them is returned by
    :meth:`describe` and lands in the result record; a probe accuracy without
    its hyperparameters is not reproducible.
    """

    level = "high_level"
    zero_shot = False

    def __init__(
        self,
        num_classes: int | None = None,
        pooling: str = Pooling.DEFAULT,
        epochs: int = 200,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        standardize: bool = False,
        device: str | None = None,
    ) -> None:
        """Configure the linear head; weights are created lazily in :meth:`fit`.

        Parameters
        ----------
        num_classes:
            Inferred from the training labels when ``None``, so
            ``get_probe("classification")`` works without knowing the dataset
            up front. Pass it explicitly when a class may be absent from the
            training split.
        epochs / lr:
            Defaults chosen by measurement, not convention. On a synthetic
            linearly-separable problem ``lr=1e-3`` reached only 0.66 *training*
            accuracy after 100 epochs — it could not fit data it provably
            could separate — while ``1e-2`` reached 0.96 and 5e-2 reached 0.99.
            An underfitting probe understates a backbone, which is the worst
            failure mode this library has, so the default errs toward
            converging. Check :attr:`train_top1` if a number looks low.
        standardize:
            Zero-mean/unit-variance the features using training statistics.
            Off by default to stay comparable with published linear-probe
            protocols, which fit on raw features. Worth turning on if a
            backbone's features have an awkward scale and the probe underfits.
        """
        if num_classes is not None and num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.name = "classification"
        self.num_classes = num_classes
        self.pooling = pooling
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.standardize = standardize
        self.device = resolve_device(device)

        self.head: nn.Linear | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

        #: Set by :meth:`fit`. Diagnostics, not results — a low test score with
        #: a low ``train_top1`` means the probe underfitted (raise ``lr`` or
        #: ``epochs``); a low test score with ``train_top1`` near 1.0 means the
        #: representation genuinely does not separate the classes. Without
        #: these, the two are indistinguishable from the outside.
        self.train_top1: float | None = None
        self.train_loss: float | None = None

    # -- training ------------------------------------------------------------

    def fit(self, features: Any, labels: Any | None = None) -> "ClassificationTask":
        """Fit the linear classifier on ``(N, C)`` pooled features.

        Seeding is the caller's job (:func:`visbench.utils.set_seed`), so that
        the seed recorded alongside the metrics is the one that actually
        governed this run.
        """
        pooled = self._as_pooled(features).float()
        targets = self._as_label_tensor(labels).long()
        if len(pooled) != len(targets):
            raise ValueError(f"Got {len(pooled)} features for {len(targets)} labels")

        num_classes = self.num_classes
        if num_classes is None:
            num_classes = int(targets.max().item()) + 1
            if num_classes < 2:
                raise ValueError(
                    "Training labels contain a single class; a classifier needs at least two"
                )
            self.num_classes = num_classes
        elif targets.max().item() >= num_classes:
            raise ValueError(
                f"Label {int(targets.max().item())} is out of range for num_classes={num_classes}"
            )

        pooled = pooled.to(self.device)
        targets = targets.to(self.device)

        if self.standardize:
            self._mean = pooled.mean(dim=0, keepdim=True)
            # clamp: a constant feature dimension would divide by zero.
            self._std = pooled.std(dim=0, keepdim=True).clamp(min=1e-6)
            pooled = (pooled - self._mean) / self._std

        self.head = nn.Linear(pooled.shape[1], num_classes).to(self.device)
        optimiser = torch.optim.AdamW(
            self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        self.head.train()
        for _ in range(self.epochs):
            # Reshuffled every epoch; with cached features the permutation is
            # the only source of stochasticity, so the caller's seed fully
            # determines the result.
            order = torch.randperm(len(pooled), device=self.device)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                optimiser.zero_grad()
                loss = criterion(self.head(pooled[batch]), targets[batch])
                loss.backward()
                optimiser.step()
        self.head.eval()

        with torch.no_grad():
            scores = self.head(pooled)
            self.train_loss = criterion(scores, targets).item()
            self.train_top1 = (scores.argmax(dim=1) == targets).float().mean().item()

        return self

    # -- serialisation -------------------------------------------------------

    def head_spec(self) -> dict | None:
        """The bare ``nn.Linear`` this probe fits, not a registered dense head.

        Declared as its own kind rather than squeezed through ``build_head``:
        the registered heads all map ``(B, C, H, W)`` feature maps, and this one
        maps a pooled ``(B, C)`` vector. Pretending otherwise would produce a
        head that reloads at the wrong rank.
        """
        if self.head is None:
            return None
        return {
            "kind": "linear",
            "in_features": int(self.head.in_features),
            "out_features": int(self.head.out_features),
        }

    def probe_state(self) -> dict[str, torch.Tensor]:
        """The standardiser, which lives outside the head and decides the answer.

        With ``standardize=True`` the linear layer was fitted on
        ``(x - mean) / std`` and is meaningless applied to raw features. Both
        tensors travel with the weights or the artifact is not reloadable.
        """
        if self._mean is None or self._std is None:
            return {}
        return {"mean": self._mean, "std": self._std}

    def load_probe_state(self, state: dict[str, torch.Tensor]) -> None:
        if not state:
            # A probe fitted with standardize=True cannot run without them, and
            # the flag alone is what says whether they were ever written.
            if self.standardize:
                raise ValueError(
                    "This probe standardises its features, but the artifact "
                    "carries no mean/std. The head was fitted on standardised "
                    "inputs and would score against raw ones."
                )
            return
        missing = {"mean", "std"} - set(state)
        if missing:
            raise ValueError(f"Standardiser is incomplete; missing {sorted(missing)}.")
        self._mean = state["mean"].to(self.device)
        self._std = state["std"].to(self.device)
        self.standardize = True

    # -- inference -----------------------------------------------------------

    def _require_head(self) -> nn.Linear:
        if self.head is None:
            raise RuntimeError(
                "This probe has not been fitted. Call fit(train_features, train_labels) "
                "before predict() or evaluate()."
            )
        return self.head

    @torch.no_grad()
    def logits(self, features: Any) -> torch.Tensor:
        """Raw ``(N, num_classes)`` scores. Exposed because top-k needs them."""
        head = self._require_head()
        pooled = self._as_pooled(features).float().to(self.device)

        if pooled.shape[1] != head.in_features:
            raise ValueError(
                f"Features have {pooled.shape[1]} channels but this probe was fitted on "
                f"{head.in_features}; train and test features must come from the same backbone."
            )
        if self.standardize:
            assert self._mean is not None and self._std is not None  # set by fit()
            pooled = (pooled - self._mean) / self._std

        return head(pooled)

    def predict(self, features: Any) -> torch.Tensor:
        """Return predicted class indices, ``(N,)``."""
        return self.logits(features).argmax(dim=1)

    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """Return ``{"top1": ..., "top5": ...}``."""
        targets = self._as_label_tensor(labels).long().to(self.device)
        scores = self.logits(features)
        if len(scores) != len(targets):
            raise ValueError(f"Got {len(scores)} features for {len(targets)} labels")
        return top_k_accuracy(scores, targets)

    # -- provenance ----------------------------------------------------------

    def describe(self) -> dict:
        """Task metadata plus the optimiser settings that produced the number."""
        described = super().describe()
        described["task_params"] = {
            "num_classes": self.num_classes,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "standardize": self.standardize,
            "optimizer": "adamw",
        }
        return described
