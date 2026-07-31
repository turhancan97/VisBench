"""Edge detection — the first low-level task, v0.3 (step 6d-1).

Low-level in this library's taxonomy means a property recoverable from the
signal without naming an object, and edge magnitude is the canonical one. The
probe asks whether a backbone's dense features still carry *where the intensity
structure is*, after however many layers of increasingly semantic pooling.

**This protocol is not BSDS500's, and the record says so.** BSDS is the
canonical edge benchmark, but its ODS/OIS/AP is a correspondence metric —
predicted edge pixels are matched to several annotators' by bipartite matching
after non-maximum suppression, swept over thresholds — not a per-pixel one.
Borrowing a protocol is only worth anything if it is borrowed exactly (see the
depth probe, whose 256-bin expectation a from-memory reconstruction would have
turned into scalar regression), and that is a step of its own. What is kept
here is the same thing :class:`GenericSegmentationTask` keeps: probe3d's
*optimiser* schedule, so a backbone's edge number sits alongside its depth,
normal and segmentation numbers under one training budget.

===============  ==========================================================
prediction       1 channel, identity — the output is the magnitude
loss             L1
metric           per-image Pearson correlation (plus RMSE and MAE)
optimiser        AdamW, lr 5e-4, 10 epochs, 1.5 warmup, cosine decay
protocol         ``visbench_edge_regression``
===============  ==========================================================

**Every pixel is scored; there is no validity mask.** Depth has holes and
normals have zero-length vectors, both meaning "no ground truth". An edge map
has neither: 0 means *no edge*, which is a real reading covering most of most
frames. Masking it away — the obvious thing to copy from the three dense tasks
that came before — would score the probe only where an edge already is, which
is the one place the answer is easy.
"""

import torch
import torch.nn.functional as F

from visbench.metrics.dense import edge_metrics
from visbench.registry import register_task
from visbench.tasks.dense_base import DenseTrainingTask
from visbench.types import MetricsDict

__all__ = ["EdgeTask"]


@register_task("edge")
class EdgeTask(DenseTrainingTask):
    """Dense edge-magnitude regression over frozen features.

    As with every other dense probe here, ``head="linear"`` is the number that
    compares *representations* — it is the only head under which a difference
    between two backbones is a difference between two feature maps — and
    ``head="dpt"`` scores higher for everyone. Report both, or say which.
    """

    level = "low_level"
    display_name = "Edge detection"
    target_noun = "target edge maps"
    target_channels = 1

    def __init__(
        self,
        head: str = "linear",
        layers: list[int] | None = None,
        hidden_dim: int = 512,
        epochs: int = 10,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        warmup_epochs: float = 1.5,
        head_kwargs: dict | None = None,
        device: str | None = None,
        finetune_blocks: int = 0,
        backbone_lr: float | None = None,
    ) -> None:
        """Configure the probe; the head is built lazily in :meth:`fit`.

        Parameters
        ----------
        head, layers, epochs, lr, warmup_epochs:
            See :class:`~visbench.tasks.dense_base.DenseTrainingTask`. There are
            no task-specific hyperparameters — in particular no threshold, since
            nothing here is binarised.
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
            finetune_blocks=finetune_blocks,
            backbone_lr=backbone_lr,
        )
        self.name = "edge"

    @property
    def out_channels(self) -> int:
        """One magnitude per pixel."""
        return 1

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        """The identity: the head's output *is* the edge magnitude.

        An edge magnitude cannot be negative, so the obvious move is to impose
        that with a rectifying activation. **Both ways of doing so were tried
        and both destroy the probe**, because the target lives in a narrow band
        just above zero — a frame's mean is about 0.011 of the container range.
        Measured on features that literally encode the answer, where the ceiling
        is 1.0:

        ============  ==================  =========================================
        activation    ``edge_correlation``  what happens
        ============  ==================  =========================================
        ``relu``      0.0000              dies outright; prediction std is exactly 0
        ``softplus``  -0.9851             collapses to a near-constant
        identity      **0.9997**          recovers the target
        ============  ==================  =========================================

        ReLU fails the way its reputation suggests: with the target this close
        to zero the pre-activation sits near or below the origin, the gradient
        is zero there, and the output never recovers. Softplus fails less
        obviously and so is the more dangerous of the two — to emit 0.065 it
        needs a raw value near -2.7, where its own gradient is
        ``sigmoid(-2.7)`` ~ 0.06, so it attenuates the signal about sixteenfold
        in exactly the region this target occupies. Its -0.985 is not learning
        the inverse; it is a constant prediction whose residual noise happens to
        anti-correlate.

        So non-negativity is **learned rather than imposed** — predictions come
        out non-negative because the targets are, and the loss is what says so.
        A prediction may dip slightly below zero in a flat region; that costs a
        little MAE and nothing else, which is a far smaller price than either
        row above. Not a sigmoid either: the magnitude has no upper bound the
        task knows about, and real maps reach only a fraction of the container
        range, so squashing to ``[0, 1]`` would spend most of the output range
        on values that never occur.
        """
        return raw

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 over every pixel.

        L1 and not MSE because the magnitude distribution is heavy-tailed — a
        frame's mean is around 0.011 of the container range and its peak around
        0.13. Squared error is dominated by that handful of strongest edges, so
        the probe would be optimised to fit the brightest contours and left free
        to ignore the rest of the map, which is most of it.

        No mask, deliberately: see the module docstring. Every pixel of an edge
        map is a measurement.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction {tuple(pred.shape)} and target {tuple(target.shape)} must match"
            )
        return F.l1_loss(pred, target)

    def _batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
        """``{"edge_correlation", "rmse", "mae"}``; quote ``edge_correlation``."""
        return edge_metrics(pred, target)

    def _task_params(self) -> dict:
        """Override the inherited ``protocol``: this one is neither probe3d's nor BSDS's.

        probe3d has no edge task, and BSDS's is a correspondence metric this does
        not implement. Only the optimiser schedule is borrowed, and that is
        already recorded under ``optimizer``. The whole value of this field is
        that it says what a number may be compared with.
        """
        return {
            "protocol": "visbench_edge_regression",
            "loss": "l1",
            "activation": "identity",
        }
