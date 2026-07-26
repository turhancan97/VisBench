"""Generic (binary) object segmentation — v0.2.

Figure-ground separation with **no semantics**: split object from background
without naming either. This is what makes it mid-level, and what distinguishes
it from :mod:`visbench.tasks.high_level.semantic_segmentation`. The two are
meant to be runnable on the same backbone for direct comparison, and share
:class:`~visbench.tasks.dense_base.DenseTrainingTask` so that the comparison is
between the tasks rather than between two training loops.

**This protocol is not probe3d's.** Depth and surface normals borrow theirs
outright, because comparability with published numbers is the whole reason to
use someone else's protocol. probe3d has no binary segmentation task, so there
is nothing to borrow here — what is kept is its *optimiser* schedule (AdamW at
5e-4, ten epochs, 1.5 warmup, cosine decay, backbone frozen), so that a
backbone's segmentation number sits alongside its depth and normal numbers
under one training budget. The loss and metric are the field's conventional
choices, recorded as ``protocol: "visbench_binary_seg"`` so no reader mistakes
them for probe3d's:

===============  ==========================================================
prediction       1 channel, sigmoid, thresholded at 0.5
loss             masked binary cross-entropy
metric           foreground IoU (plus Dice and pixel accuracy)
optimiser        AdamW, lr 5e-4, 10 epochs, 1.5 warmup, cosine decay
===============  ==========================================================

**Class imbalance is not corrected in the loss.** Objects cover a minority of
most frames, so plain BCE does pull a weak probe towards predicting background;
that is answered by *reporting* foreground IoU, which ignores true negatives and
so scores such a probe near zero, rather than by reweighting. A pos_weight
tuned per dataset would make two datasets' numbers incomparable, and tuned per
backbone would flatter whichever backbone got the most tuning — the two failures
a probe suite exists to avoid.

**Validity.** A mask has no equivalent of a depth hole: 0 is a real label. So
the convention shifts by one — a pixel is unlabelled where the target is
**negative**, which is what :func:`~visbench.data.dense.load_mask` writes for a
dataset's explicit ignore region and what nothing at all writes otherwise. See
:func:`~visbench.metrics.dense.binary_iou`, which masks identically, so the
pixels trained on and the pixels scored are one set.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from visbench.metrics.dense import SEGMENTATION_THRESHOLD, binary_iou
from visbench.registry import register_task
from visbench.tasks.dense_base import DenseTrainingTask
from visbench.types import MetricsDict

__all__ = ["GenericSegmentationTask", "masked_bce_loss"]


def masked_bce_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary cross-entropy over labelled pixels only.

    ``pred`` holds foreground **probabilities**, not logits — the base class
    applies :meth:`GenericSegmentationTask._activate` before the loss, the
    metric and :meth:`predict` alike, precisely so those three cannot disagree
    about what the model predicts. The usual advice to prefer
    ``binary_cross_entropy_with_logits`` is about the fused version's numerical
    stability; the clamp below buys the same guarantee, which is the part that
    matters, at the cost of the fused kernel.

    A pixel is skipped where ``target < 0``. If every pixel of a batch is
    unlabelled this returns a zero that is still *connected to the graph*: a
    bare zero would detach the head and silently skip the batch's gradient,
    the same trap the depth and normal losses guard against.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction {tuple(pred.shape)} and target {tuple(target.shape)} must match"
        )

    valid = target >= 0
    if not valid.any():
        return pred.sum() * 0.0

    # Clamped away from 0 and 1 before the log: a confident, correct sigmoid
    # saturates to exactly 1.0 in float32, and log(0) on the other side of a
    # single mislabelled pixel would put inf into the batch's gradient.
    probability = pred[valid].clamp(min=eps, max=1.0 - eps)
    return F.binary_cross_entropy(probability, target[valid])


@register_task("generic_segmentation")
class GenericSegmentationTask(DenseTrainingTask):
    """Binary foreground/background prediction over dense features.

    As with depth and normals, ``head="linear"`` is the number that compares
    *representations* — it is the only head under which a difference between two
    backbones is a difference between two feature maps — and ``head="dpt"``
    scores higher for everyone. Report both, or say which.
    """

    level = "mid_level"
    display_name = "Generic object segmentation"
    target_noun = "target masks"
    target_channels = 1

    def __init__(
        self,
        head: str = "linear",
        layers: Optional[list[int]] = None,
        hidden_dim: int = 512,
        epochs: int = 10,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        warmup_epochs: float = 1.5,
        head_kwargs: Optional[dict] = None,
        device: Optional[str] = None,
    ) -> None:
        """Configure the probe; the head is built lazily in :meth:`fit`.

        Parameters
        ----------
        head, layers, epochs, lr, warmup_epochs:
            See :class:`~visbench.tasks.dense_base.DenseTrainingTask`. There are
            no task-specific hyperparameters: the decision threshold is fixed at
            :data:`~visbench.metrics.dense.SEGMENTATION_THRESHOLD`, since a
            swept threshold reports each backbone's best operating point rather
            than its representation.
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
        self.name = "generic_segmentation"

    @property
    def out_channels(self) -> int:
        """One logit per pixel: foreground against everything else."""
        return 1

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        """Logits to ``(B, 1, H, W)`` foreground probabilities.

        A probability rather than a hard mask, so that :meth:`predict` hands
        back something a caller can threshold themselves, overlay, or calibrate.
        :func:`~visbench.metrics.dense.binary_iou` applies the threshold.
        """
        return torch.sigmoid(raw)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return masked_bce_loss(pred, target)

    def _batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
        """``{"iou", "f1", "pixel_acc"}``; quote ``iou``."""
        return binary_iou(pred, target)

    def _task_params(self) -> dict:
        """Override the inherited ``protocol``: this one is not probe3d's.

        Only the optimiser schedule is theirs, and it is already recorded under
        ``optimizer``. A record claiming ``protocol: "probe3d"`` for a loss and
        metric that paper never defined would be worse than no record.
        """
        return {
            "protocol": "visbench_binary_seg",
            "loss": "masked_bce",
            "threshold": SEGMENTATION_THRESHOLD,
        }
