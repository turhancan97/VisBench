"""Monocular depth estimation — the first dense task. v0.2.

The evaluation protocol, the prediction parameterisation and the loss all come
from probe3d (El Banani et al., CVPR 2024, arXiv:2404.08476), whose
``evals/utils/metrics.py``, ``evals/utils/losses.py`` and
``evals/models/probes.py`` are MIT licensed. This reproduces its
``configs/probe/depth_dpt.yaml`` plus ``configs/optimizer/ten_epoch.yaml``:

===============  ==========================================================
prediction       256 uniform bins over [0.001, 10] m, then the expectation
loss             10 x scale-invariant log  +  0.5 x gradient
optimiser        AdamW, lr 5e-4, 10 epochs, 1.5 warmup, cosine decay
backbone         frozen (their ``model_lr: 0``)
===============  ==========================================================

**Why bins rather than one number.** Regressing a scalar per pixel pushes a
linear head towards predicting the dataset's mean depth almost everywhere.
Predicting a distribution over depths and taking its expectation lets a
*linear* map express a multi-modal belief, which is most of why probe3d's
linear probe is a fair baseline rather than a straw man. The parameterisation
is AdaBins' (arXiv:2011.14141).

**Memory.** Features are held in RAM: N images at C x H x W floats. DINOv2-B at
224 is 768 x 16 x 16 x 4 B, about 786 KB per image, so all 24k NYUv2 images
would need roughly 19 GB. Small splits are fine; a full-size dense run needs
the cache to stream batches from disk, which it cannot do yet. :meth:`fit`
raises with that arithmetic rather than letting a run discover it by being
killed.
"""

import math
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from visbench.heads import build_head
from visbench.metrics.dense import depth_metrics
from visbench.registry import register_task
from visbench.tasks.base import BaseTask
from visbench.types import FeatureMode, MetricsDict, Pooling
from visbench.utils.device import resolve_device

__all__ = ["DepthTask", "DepthBinPrediction", "depth_loss"]


class DepthBinPrediction(nn.Module):
    """Turn ``(B, n_bins, H, W)`` scores into one depth per pixel.

    probe3d's ``DepthBinPrediction`` at its defaults: uniformly spaced bin
    centres, ``linear`` normalisation (ReLU, add 0.1, divide by the sum), depth
    as the expectation over bins.

    The 0.1 is not decoration. After a ReLU, a pixel whose scores are all
    negative sums to zero and the normalisation would divide by it; the offset
    makes such a pixel fall back to a uniform distribution — mid-range depth —
    instead of producing NaN and poisoning the rest of the epoch.
    """

    def __init__(self, min_depth: float = 0.001, max_depth: float = 10.0, n_bins: int = 256):
        super().__init__()
        if not 0 < min_depth < max_depth:
            raise ValueError(f"Need 0 < min_depth < max_depth, got {min_depth} and {max_depth}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.register_buffer("bins", torch.linspace(min_depth, max_depth, n_bins))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """``(B, n_bins, H, W)`` scores to ``(B, 1, H, W)`` depth."""
        if scores.ndim != 4 or scores.shape[1] != self.n_bins:
            raise ValueError(
                f"Expected (B, {self.n_bins}, H, W) bin scores, got {tuple(scores.shape)}"
            )
        probabilities = torch.relu(scores) + 0.1
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        depth = torch.einsum("bkhw,k->bhw", probabilities, self.bins)
        return depth.unsqueeze(1)


def _scale_invariant_log_loss(
    pred: torch.Tensor, target: torch.Tensor, sigma: float = 0.85, eps: float = 1e-3
) -> torch.Tensor:
    """probe3d's ``sig_loss`` — the Eigen et al. scale-invariant log error.

    Penalises the *variance* of the log-ratio rather than its mean, so a
    prediction that is uniformly too deep is barely punished while one that
    gets the relative arrangement wrong is. ``sigma=0.85`` leaves some absolute
    scale pressure, as AdaBins and DINOv2's own depth head both do.
    """
    valid = target > 0
    if not valid.any():
        # Keeps the graph connected: a bare zero would detach the head and
        # silently skip this batch's gradient.
        return pred.sum() * 0.0
    difference = torch.log(pred[valid] + eps) - torch.log(target[valid] + eps)
    return (difference.pow(2).mean() - sigma * difference.mean().pow(2)).clamp(min=0).sqrt()


def _gradient_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """probe3d's ``gradient_loss`` — multi-scale log-depth gradient matching.

    Compares horizontal and vertical differences of log depth at four
    subsamplings. The scale-invariant term alone is content with a blurry
    prediction; this is the term that asks for edges in the right places, which
    is exactly the mid-level structure a depth probe exists to measure.
    """
    total = pred.new_zeros(())
    for step in (1, 2, 4, 6):
        pred_s = pred[..., ::step, ::step]
        target_s = target[..., ::step, ::step]
        valid = (target_s > 0).float()
        count = valid.sum().clamp(min=1)

        difference = (torch.log(pred_s + eps) - torch.log(target_s + eps)) * valid
        vertical = (difference[..., :-2, :] - difference[..., 2:, :]).abs()
        vertical = vertical * valid[..., :-2, :] * valid[..., 2:, :]
        horizontal = (difference[..., :, :-2] - difference[..., :, 2:]).abs()
        horizontal = horizontal * valid[..., :, :-2] * valid[..., :, 2:]

        total = total + (vertical.sum() + horizontal.sum()) / count
    return total


def depth_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_si: float = 10.0,
    weight_gradient: float = 0.5,
) -> torch.Tensor:
    """probe3d's ``DepthLoss``: weighted scale-invariant plus gradient terms.

    Unlike the reference, this does **not** mutate ``target`` in place to apply
    a maximum-depth cap. That cap belongs to the dataset
    (:class:`~visbench.data.dense.DenseFolderDataset` ``max_target``), so the
    pixels the loss trains on and the pixels the metric scores are one set — a
    loss masking more than the metric would optimise for a number nobody
    reports.
    """
    return weight_si * _scale_invariant_log_loss(pred, target) + weight_gradient * _gradient_loss(
        pred, target
    )


@register_task("depth")
class DepthTask(BaseTask):
    """Pixel-wise depth from a single image, via a pluggable head.

    ``head="linear"`` is the number to quote when comparing representations: it
    is the only head under which a difference between two backbones is a
    difference between two *feature maps*. ``head="dpt"`` is probe3d's own
    choice and scores higher for everyone, so report both, or say which.

    The DPT head is multiscale and needs several backbone depths: set
    :attr:`layers`, and extraction reads them all in one forward pass.
    """

    level = "mid_level"
    feature_mode = FeatureMode.DENSE_ONLY
    zero_shot = False
    uses_dense = True
    #: Dense tasks read the grid, never the pooled vector — but extraction
    #: still needs a pooling name for the cache key. Mean is the honest choice:
    #: it is what a CNN would use anyway, and it keeps these entries from
    #: colliding with a CLS-pooled run over the same images.
    pooling = Pooling.MEAN

    #: Roughly 8 GB of float32 features, past which :meth:`fit` refuses rather
    #: than letting the OOM killer end a run half an hour in.
    max_feature_elements = 2_000_000_000

    def __init__(
        self,
        head: str = "linear",
        layers: Optional[list[int]] = None,
        min_depth: float = 0.001,
        max_depth: float = 10.0,
        n_bins: int = 256,
        hidden_dim: int = 512,
        epochs: int = 10,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        warmup_epochs: float = 1.5,
        scale_invariant: bool = False,
        head_kwargs: Optional[dict] = None,
        device: Optional[str] = None,
    ) -> None:
        """Configure the probe; the head is built lazily in :meth:`fit`.

        Parameters
        ----------
        head:
            Any registered head name — ``"linear"``, ``"dpt"``, or a
            contributor's — built once :meth:`fit` knows the feature width.
        layers:
            Backbone depths to extract. ``None`` means the last layer only,
            which a linear head is happy with and a DPT head is not: it refuses
            a single map rather than duplicating it. probe3d reads four evenly
            spaced blocks.
        epochs, lr, warmup_epochs:
            probe3d's ten-epoch schedule — AdamW at 5e-4, linear warmup over
            the first 1.5 epochs, then cosine decay to zero.
        scale_invariant:
            Fit a per-image scale and shift before scoring. Off by default; see
            :func:`~visbench.metrics.dense.depth_metrics`.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if warmup_epochs < 0 or warmup_epochs >= epochs:
            raise ValueError(
                f"warmup_epochs must be in [0, epochs), got {warmup_epochs} with "
                f"epochs={epochs} — the schedule would still be warming up when training "
                "ended. Pass warmup_epochs=0 for a short run; the default 1.5 assumes "
                "probe3d's 10 epochs. Clamping it silently would report a number produced "
                "by a schedule nobody chose."
            )
        if layers is not None and len(layers) == 0:
            raise ValueError("layers=[] requests nothing; pass None for the last layer")

        self.name = "depth"
        self.head_name = head
        self.layers = layers
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.scale_invariant = scale_invariant
        self.head_kwargs = dict(head_kwargs or {})
        self.device = resolve_device(device)

        self.head: Optional[nn.Module] = None
        self.predict_depth = DepthBinPrediction(min_depth, max_depth, n_bins).to(self.device)

        #: Set by :meth:`fit`. A diagnostic, not a result — a poor score with a
        #: high training loss means the probe underfitted, which is a different
        #: finding from a representation that does not carry depth.
        self.train_loss: Optional[float] = None

    # -- features ------------------------------------------------------------

    def _dense(self, features: Any) -> Union[torch.Tensor, list]:
        """Pull what the head wants: a list of maps if multiscale, else one."""
        if isinstance(features, dict):
            if self.layers is not None:
                if "dense_layers" not in features:
                    raise KeyError(
                        f"This task asked for layers={self.layers} but the features hold "
                        "only a single map. Extract with the same layers= the task "
                        "declares, or use visbench.run(), which does that for you."
                    )
                return [layer.float() for layer in features["dense_layers"]]
            if "dense" not in features:
                raise KeyError(
                    "Feature dict has no 'dense' entry. A dense task needs "
                    "extract_dataset(keep='dense') or keep='both'."
                )
            return features["dense"].float()
        if isinstance(features, (list, tuple)):
            return [layer.float() for layer in features]
        return features.float()

    @staticmethod
    def _count(dense: Union[torch.Tensor, list]) -> int:
        return len(dense[0]) if isinstance(dense, list) else len(dense)

    @staticmethod
    def _slice(dense: Union[torch.Tensor, list], index: torch.Tensor) -> Any:
        return [layer[index] for layer in dense] if isinstance(dense, list) else dense[index]

    def _check_size(self, dense: Union[torch.Tensor, list]) -> None:
        maps = dense if isinstance(dense, list) else [dense]
        elements = sum(layer.numel() for layer in maps)
        if elements > self.max_feature_elements:
            raise ValueError(
                f"These features are about {elements * 4 / 1e9:.1f} GB in float32, over this "
                f"task's {self.max_feature_elements * 4 / 1e9:.0f} GB ceiling. Dense features "
                "are held in memory for training; use a smaller split or fewer layers until "
                "the cache can stream batches from disk. Raise max_feature_elements to "
                "override."
            )

    def _build_head(self, dense: Union[torch.Tensor, list], output_size: int) -> nn.Module:
        """Instantiate the configured head, sized to these features."""
        if isinstance(dense, list):
            channels: Any = [layer.shape[1] for layer in dense]
            kwargs: dict = {"num_layers": len(dense), "hidden_dim": self.hidden_dim}
        else:
            channels = dense.shape[1]
            kwargs = {}
        kwargs.update(self.head_kwargs)
        return build_head(
            self.head_name,
            in_channels=channels,
            out_channels=self.n_bins,
            output_size=output_size,
            **kwargs,
        ).to(self.device)

    # -- training ------------------------------------------------------------

    def fit(self, features: Any, labels: Optional[Any] = None) -> "DepthTask":
        """Train the head on cached dense features and ``(N, H, W)`` depth targets.

        Seeding is the caller's job (:func:`visbench.utils.set_seed`), so the
        seed recorded next to the metrics is the one that governed the run.
        """
        dense = self._dense(features)
        self._check_size(dense)
        targets = self._as_targets(labels)
        count = self._count(dense)
        if count != len(targets):
            raise ValueError(f"Got {count} feature maps for {len(targets)} targets")

        output_size = int(targets.shape[-1])
        if targets.shape[-2] != output_size:
            raise ValueError(
                f"Targets are {tuple(targets.shape[-2:])}; this task assumes square maps, "
                "which is what DenseFolderDataset produces."
            )

        self.head = self._build_head(dense, output_size)
        self.head.train()
        optimiser = torch.optim.AdamW(
            self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        steps_per_epoch = max(1, math.ceil(count / self.batch_size))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimiser, self._schedule(steps_per_epoch * self.epochs, steps_per_epoch)
        )

        targets = targets.to(self.device)
        running = 0.0
        for _ in range(self.epochs):
            # Reshuffled each epoch; with cached features the permutation is
            # the only stochasticity, so the caller's seed fully determines it.
            order = torch.randperm(count)
            running = 0.0
            for start in range(0, count, self.batch_size):
                batch = order[start : start + self.batch_size]
                optimiser.zero_grad()
                predicted = self._forward(self._slice(dense, batch))
                loss = depth_loss(predicted, targets[batch].unsqueeze(1))
                loss.backward()
                optimiser.step()
                scheduler.step()
                running += loss.item()
            running /= steps_per_epoch

        self.head.eval()
        self.train_loss = running
        return self

    def _schedule(self, total_steps: int, steps_per_epoch: int):
        """Linear warmup then cosine decay, as a step-indexed multiplier."""
        warmup_steps = int(self.warmup_epochs * steps_per_epoch)

        def multiplier(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            remaining = max(1, total_steps - warmup_steps)
            progress = min(1.0, (step - warmup_steps) / remaining)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return multiplier

    def _forward(self, dense: Any) -> torch.Tensor:
        """Features to ``(B, 1, H, W)`` depth."""
        head = self._require_head()
        if isinstance(dense, list):
            dense = [layer.to(self.device) for layer in dense]
        else:
            dense = dense.to(self.device)
        return self.predict_depth(head(dense))

    def _require_head(self) -> nn.Module:
        if self.head is None:
            raise RuntimeError(
                "This probe has not been fitted. Call fit(train_features, train_targets) "
                "before predict() or evaluate()."
            )
        return self.head

    @staticmethod
    def _as_targets(labels: Any) -> torch.Tensor:
        """Coerce targets to a ``(N, H, W)`` float tensor."""
        if labels is None:
            raise ValueError("Depth estimation requires target maps; got None")
        targets = labels if isinstance(labels, torch.Tensor) else torch.stack(list(labels))
        targets = targets.float()
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if targets.ndim != 3:
            raise ValueError(f"Expected targets of shape (N, H, W), got {tuple(targets.shape)}")
        return targets

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def predict(self, features: Any) -> torch.Tensor:
        """Predicted depth, ``(N, 1, H, W)``, batched to bound memory."""
        self._require_head()
        dense = self._dense(features)
        count = self._count(dense)
        outputs = []
        for start in range(0, count, self.batch_size):
            index = torch.arange(start, min(start + self.batch_size, count))
            outputs.append(self._forward(self._slice(dense, index)).cpu())
        return torch.cat(outputs)

    def evaluate(self, features: Any, labels: Optional[Any] = None) -> MetricsDict:
        """Return ``{"d1", "d2", "d3", "rmse", "abs_rel"}`` per probe3d."""
        targets = self._as_targets(labels)
        predicted = self.predict(features)
        if len(predicted) != len(targets):
            raise ValueError(f"Got {len(predicted)} predictions for {len(targets)} targets")
        return depth_metrics(predicted.squeeze(1), targets, scale_invariant=self.scale_invariant)

    # -- provenance ----------------------------------------------------------

    def describe(self) -> dict:
        """Task metadata plus everything that shaped the number."""
        described = super().describe()
        described["task_params"] = {
            "head": self.head_name,
            "layers": self.layers,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "n_bins": self.n_bins,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "warmup_epochs": self.warmup_epochs,
            "scale_invariant": self.scale_invariant,
            "optimizer": "adamw",
            "protocol": "probe3d",
        }
        return described
