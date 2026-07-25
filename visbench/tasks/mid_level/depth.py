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

Everything not specific to depth — streaming features, head construction, the
optimiser schedule, the training loop, batch-wise scoring — lives in
:class:`~visbench.tasks.dense_base.DenseTrainingTask` and is shared with the
other dense probes.
"""

from typing import Optional

import torch
import torch.nn as nn

from visbench.metrics.dense import depth_metrics
from visbench.registry import register_task
from visbench.tasks.dense_base import DenseTrainingTask
from visbench.types import MetricsDict

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
class DepthTask(DenseTrainingTask):
    """Pixel-wise depth from a single image, via a pluggable head."""

    level = "mid_level"
    display_name = "Depth estimation"
    target_noun = "target depth maps"
    target_channels = 1

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
        head, layers, epochs, lr, warmup_epochs:
            See :class:`~visbench.tasks.dense_base.DenseTrainingTask`.
        min_depth, max_depth, n_bins:
            The bin grid the expectation is taken over. probe3d's NYUv2
            defaults are 256 bins across [0.001, 10] metres.
        scale_invariant:
            Fit a per-image scale and shift before scoring. Off by default; see
            :func:`~visbench.metrics.dense.depth_metrics`.
        """
        super().__init__(
            head=head,
            layers=layers,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            head_kwargs=head_kwargs,
            device=device,
        )
        self.name = "depth"
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.scale_invariant = scale_invariant
        self.predict_depth = DepthBinPrediction(min_depth, max_depth, n_bins).to(self.device)

    @property
    def out_channels(self) -> int:
        """One score per depth bin; the expectation collapses them to one map."""
        return self.n_bins

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        """Bin scores to ``(B, 1, H, W)`` metric depth."""
        return self.predict_depth(raw)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return depth_loss(pred, target)

    def _batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
        """``{"d1", "d2", "d3", "rmse", "abs_rel"}`` per probe3d."""
        return depth_metrics(
            pred.squeeze(1), target.squeeze(1), scale_invariant=self.scale_invariant
        )

    def _task_params(self) -> dict:
        return {
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "n_bins": self.n_bins,
            "scale_invariant": self.scale_invariant,
        }
