"""Relative camera pose — two views of one rigid scene, and the transform between them (16a-2).

probe3d's pairwise protocol: concatenate the two frames' **pooled** features,
regress the 7-vector ``[quat, trans]`` under an MSE loss, and score the
geodesic rotation error. It reads pooled features, so there is no grid, no
streaming and no dense cache — the cheapest geometry probe here, and **the
first whose reported head is not a linear map of the features**. That is a
narrower claim than "the first nonlinear head": `DPTHead` has been here since
v0.2, but it is a *control* rather than a board, and the heads the published
boards do use — a bare `nn.Linear`, `LinearHead`, `DetectionHead`,
`InstanceHead` — are all 1x1 convolutions or affine maps. This is the first
board whose number comes from a head with hidden layers.

**Three things about this probe are protocol rather than configuration**, each
measured before it was built and each recorded so a second run has to match:

* **The head is an MLP, deliberately.** Every other board here is quoted with
  the least expressive head that can express the task, and for every one of
  them that has turned out to be a linear map. A single affine map
  cannot express this one: on the same pairs it underfits at ``train_loss``
  0.0725-0.0732, clears the no-feature floor by 2.7-3.5 degrees against this
  head's 24.5-42.6, and produces an ordering that nearly inverts this one's top
  two. See :class:`~visbench.heads.pose.PoseHead`, and the linear control
  committed beside the board.
* **The pair count.** Rotation error keeps falling as partners are added and
  nothing has converged at any count measured, so the number of training pairs
  is in ``task_params`` and therefore in the comparability key. Two pose
  numbers drawn from different pair sets are not two measurements of the same
  thing.
* **The floor travels with the score.** A rotation error is uninterpretable
  alone — pairs drawn within 120 degrees have a median near 65, so predicting a
  constant already scores about 67 — and a backbone landing there is *at
  chance*, which is a different claim from weak. :meth:`context_metrics` emits
  it as ``floor_*``, the convention ``ceiling_*`` set.

**This probe has no oracle, and that is not an omission.**
:meth:`~visbench.tasks.dense_base.DenseTrainingTask.evaluate_oracle` asks what
a *dense* target would score if the features contained it, by pooling it to the
feature grid — the bottleneck a per-patch head imposes. Nothing here is per
patch: the head reads two pooled vectors, so there is no grid to pool to and no
signal that is finer than the input. The instrument this probe needs is the
opposite one, and it has it: the gate that matters for pose is the **floor**,
because the failure mode is a target a constant already predicts rather than a
target the features cannot reach. See the relative-depth rejection for the same
lesson arrived at from the other side.

**A low score here has three readings and ``train_loss`` separates them**, as
it does everywhere in this project, with one extra failure the others do not
have. High loss with the score at the floor is underfitting. Loss near zero
with the score at the floor is *memorisation* — the head reads 1,536 numbers,
so at a few thousand training pairs it interpolates the split and generalises
nothing, which is what made two of four backbones read as at chance in the
first pre-measurement. And a score **worse than the floor** is neither: a
trained head cannot lose to a constant unless the loss is not optimising the
scored term, which is what an unconverted millimetre translation does.
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from visbench.heads.pose import POSE_HIDDEN_DIMS, PoseHead
from visbench.metrics.pose import mean_pose_floor, pose_metrics
from visbench.registry import register_task
from visbench.tasks.base import BaseTask
from visbench.types import MetricsDict, Pooling
from visbench.utils.device import resolve_device

__all__ = ["RelativePoseTask"]


@register_task("relative_pose")
class RelativePoseTask(BaseTask):
    """Pairwise 7D pose regression over pooled features.

    ``labels`` carries the pairing as indices into the extracted features, plus
    the relative pose of each pair — see
    :class:`~visbench.data.navi.PosePairs`. The indices are structure rather
    than supervision: a set of feature vectors alone does not say which two of
    them form a pair.
    """

    level = "mid_level"
    zero_shot = False
    uses_dense = False

    def __init__(
        self,
        hidden_dims: Sequence[int] = POSE_HIDDEN_DIMS,
        pooling: str = Pooling.DEFAULT,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        """Configure the head and its optimiser; weights are built in :meth:`fit`.

        Parameters
        ----------
        hidden_dims:
            Widths between the concatenated pair and the 7-vector. The default
            is probe3d's. ``()`` makes it one affine map, which is the control
            rather than an option — it underfits, and the board says so.
        epochs, lr, weight_decay, batch_size:
            AdamW with a cosine schedule and **no warmup**, which is the pose
            reference's recipe rather than this project's dense-probe schedule.
            The two answer different protocols and mixing them would make this
            board's numbers comparable to neither.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 2:
            # The head's first layer is a BatchNorm, which cannot normalise one
            # row. Refused here rather than in the middle of the first epoch.
            raise ValueError(f"batch_size must be >= 2, got {batch_size}")

        self.name = "relative_pose"
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.pooling = pooling
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = resolve_device(device)

        self.head: PoseHead | None = None

        #: The constant a featureless head converges to, fitted in :meth:`fit`
        #: and the yardstick every score here is read against. Kept because a
        #: probe reloaded from an artifact must still be able to report the
        #: floor its number is quoted over.
        self._train_mean: torch.Tensor | None = None

        #: Set by :meth:`fit`. A diagnostic, never a result — see the module
        #: docstring for the three readings it separates.
        self.train_loss: float | None = None
        self.train_pairs: int | None = None

    # -- the pairing ---------------------------------------------------------

    @staticmethod
    def _as_pairs(labels: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """``(indices, pose)`` from a :class:`~visbench.data.navi.PosePairs`.

        Accepts any 2-sequence of the same shape, so a caller can pass plain
        tensors, but checks both halves: an ``(P, 2)`` index block and an
        ``(P, 7)`` target. The checks are here because every way of getting
        this wrong produces a number rather than an error.
        """
        if labels is None:
            raise ValueError(
                "Relative pose needs the pairs naming which two frames to compare and "
                "the transform between them; pass dataset.labels()."
            )
        try:
            indices, pose = labels
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Expected (indices, pose) — a PosePairs — got "
                f"{type(labels).__name__}. See visbench.data.navi.PosePairs."
            ) from error

        indices = torch.as_tensor(indices)
        pose = torch.as_tensor(pose)
        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError(f"Expected (P, 2) pair indices, got {tuple(indices.shape)}")
        if pose.ndim != 2 or pose.shape[1] != 7:
            raise ValueError(f"Expected (P, 7) relative poses, got {tuple(pose.shape)}")
        if len(indices) != len(pose):
            raise ValueError(f"Got {len(indices)} pairs for {len(pose)} poses")
        return indices.long(), pose.float()

    def _paired(self, features: Any, labels: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """``(paired_features, pose)`` — the whole input to the head, computed once.

        Shared by :meth:`fit`, :meth:`predict` and :meth:`evaluate` rather than
        each assembling its own, so the three cannot pair the same features
        differently. Column 0 is the anchor and column 1 the partner, in that
        order, because the target is the partner's pose *with respect to* the
        anchor and the reverse is the same magnitude with the wrong sign.
        """
        pooled = self._as_pooled(features).float()
        indices, pose = self._as_pairs(labels)

        out_of_range = indices[(indices < 0) | (indices >= len(pooled))]
        if len(out_of_range):
            raise IndexError(
                f"{len(out_of_range)} pair index/indices fall outside the {len(pooled)} "
                "extracted features. The pairs and the features must come from the "
                "same dataset."
            )
        return torch.cat([pooled[indices[:, 0]], pooled[indices[:, 1]]], dim=1), pose

    # -- training ------------------------------------------------------------

    def fit(self, features: Any, labels: Any | None = None) -> "RelativePoseTask":
        """Fit the head on the training pairs.

        Seeding is the caller's job (:func:`visbench.utils.set_seed`), so the
        seed a record names is the one that governed the run.
        """
        paired, pose = self._paired(features, labels)
        paired = paired.to(self.device)
        target = pose.to(self.device)

        self.head = PoseHead(embed_dim=paired.shape[1] // 2, hidden_dims=self.hidden_dims).to(
            self.device
        )
        optimiser = torch.optim.AdamW(
            self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs)
        criterion = nn.MSELoss()

        self.head.train()
        for _ in range(self.epochs):
            # Reshuffled every epoch, as every other trained probe here does:
            # with cached features the permutation is the only stochasticity,
            # so the caller's seed fully determines the result.
            order = torch.randperm(len(paired), device=self.device)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                if len(batch) < 2:
                    # A trailing singleton: the head's BatchNorm has no variance
                    # to normalise by. Dropped rather than crashing a run on the
                    # one split whose size happens to be 1 mod batch_size.
                    continue
                optimiser.zero_grad()
                loss = criterion(self.head(paired[batch]), target[batch])
                loss.backward()
                optimiser.step()
            schedule.step()
        self.head.eval()

        with torch.no_grad():
            # In eval mode and over the whole split, so the diagnostic is
            # measured under the same BatchNorm statistics `predict` uses
            # rather than on whichever batch happened to be last. Summed in
            # batches rather than forwarded in one tensor: at sixteen partners
            # the training split is 92,521 pairs and the activations would be
            # larger than the features.
            squared = 0.0
            for start in range(0, len(paired), self.batch_size):
                block = slice(start, start + self.batch_size)
                squared += float((self.head(paired[block]) - target[block]).pow(2).sum())
            self.train_loss = squared / target.numel()
        self._train_mean = target.mean(dim=0, keepdim=True).cpu()
        self.train_pairs = len(target)
        return self

    # -- inference -----------------------------------------------------------

    def _require_head(self) -> PoseHead:
        if self.head is None:
            raise RuntimeError(
                "This probe has not been fitted. Call fit(train_features, train_labels) "
                "before predict() or evaluate()."
            )
        return self.head

    @torch.no_grad()
    def predict(self, features: Any, labels: Any | None = None) -> torch.Tensor:
        """``(P, 7)`` predicted relative poses, raw.

        The quaternion is **not** normalised, because nothing in this protocol
        normalises it: the loss is MSE over the 7-vector as probe3d's reference
        writes it, and :func:`~visbench.metrics.pose.rotation_error_deg`
        projects whatever it is given through the rotation matrix. Normalising
        here and not in the loss would make the prediction, the loss and the
        metric three slightly different functions.
        """
        head = self._require_head()
        paired, _ = self._paired(features, labels)
        if paired.shape[1] != head.in_features:
            raise ValueError(
                f"Features are {paired.shape[1] // 2} wide but this probe was fitted on "
                f"{head.embed_dim}; train and test features must come from the same backbone."
            )
        # The head's device, not this task's: a probe rebuilt from an artifact
        # gets its head from `build_head`, and where that landed is a fact
        # about the load rather than about the configuration this was
        # constructed with.
        return head(paired.to(next(head.parameters()).device)).cpu()

    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """Rotation and translation error over the pairs — see
        :func:`~visbench.metrics.pose.pose_metrics`.

        The prediction comes from :meth:`predict` rather than from a second
        forward pass here, so the number reported and the number a caller can
        inspect are the same one.
        """
        _, pose = self._as_pairs(labels)
        return pose_metrics(self.predict(features, labels), pose)

    def context_metrics(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """The no-feature floor, as ``floor_*``.

        What predicting the *training* mean scores on these pairs: the
        yardstick, not a result. **Never rank on it** — it describes the draw
        rather than a backbone, and two boards with different floors cannot be
        compared by subtracting them.

        Empty before :meth:`fit`, because the constant is fitted like anything
        else and a floor quoted from the evaluation split would be unbeatable
        in a way no head is.
        """
        if self._train_mean is None:
            return {}
        _, pose = self._as_pairs(labels)
        return mean_pose_floor(self._train_mean, pose)

    # -- serialisation -------------------------------------------------------

    def training_summary(self) -> dict | None:
        """Final training loss, in eval mode over the whole training split."""
        if self.train_loss is None:
            return None
        return {"train_loss": self.train_loss}

    def head_spec(self) -> dict | None:
        if self.head is None:
            return None
        return {
            "kind": "registered",
            "name": "pose",
            "kwargs": {
                "embed_dim": int(self.head.embed_dim),
                "hidden_dims": list(self.head.hidden_dims),
            },
        }

    def probe_state(self) -> dict[str, torch.Tensor]:
        """The fitted floor, which lives outside the head.

        Not needed to *predict*, which is why it is easy to forget, and needed
        to report the number a prediction is read against. A reloaded probe
        without it scores fine and cannot say whether the score beats a
        constant.
        """
        if self._train_mean is None:
            return {}
        return {"train_pose_mean": self._train_mean}

    def load_probe_state(self, state: dict[str, torch.Tensor]) -> None:
        if not state:
            return
        if set(state) != {"train_pose_mean"}:
            raise ValueError(
                f"Expected a 'train_pose_mean' tensor, got {sorted(state)}. Refusing "
                "rather than dropping it: the floor was fitted alongside these weights."
            )
        mean = state["train_pose_mean"]
        if mean.shape != (1, 7):
            raise ValueError(f"train_pose_mean must be (1, 7), got {tuple(mean.shape)}")
        self._train_mean = mean

    # -- provenance ----------------------------------------------------------

    def describe(self) -> dict:
        """Task metadata plus everything that decides what the number means.

        ``train_pairs`` is here rather than left to the dataset record because
        the *scored* split is the validation one, whose ``describe()`` says
        nothing about how many pairs the head was fitted on — and that count is
        the protocol parameter this probe has not converged in.
        """
        described = super().describe()
        described["task_params"] = {
            "protocol": "probe3d_pose",
            "head": "mlp" if self.hidden_dims else "linear",
            "hidden_dims": list(self.hidden_dims),
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "optimizer": "adamw",
            "schedule": "cosine",
            "train_pairs": self.train_pairs,
        }
        return described
