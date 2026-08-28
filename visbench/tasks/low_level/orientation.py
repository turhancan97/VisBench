"""Gradient orientation estimation — the second derived-target probe.

Like ``corner`` this computes its target from the RGB frame, so it runs on any
image folder with no download. Unlike ``corner`` — and unlike ``edge`` and
``keypoints2d`` — the target is **not a magnitude**: it is the local *direction*
structure runs, read from the same Gaussian-windowed structure tensor whose
smaller eigenvalue is the corner response. So this is the first derived probe
that cannot reuse :class:`~visbench.tasks.magnitude_base.DenseMagnitudeTask`.

===============  ==========================================================
prediction       2 channels ``(cos 2t, sin 2t)``, L2-normalised
target           ``coherence * (cos 2t, sin 2t)`` — length carries validity
loss             coherence-weighted angular error (doubled angle)
metric           coherence-weighted orientation error in degrees (halved:
                 0-90, chance 45), plus within-11.25/22.5 fractions
optimiser        AdamW, lr 5e-4, 10 epochs, 1.5 warmup, cosine decay
protocol         ``visbench_structure_tensor_orientation_regression``
===============  ==========================================================

**Why it earns a place beside the three magnitude probes.** The point of the
low-level tier is that a backbone can be strong at one signal-level property and
weak at another. Orientation is close to *independent* of every target that
already ships: per-image ``|r|`` with ``edge_texture`` is 0.07 and with
``corner`` 0.08, where ``corner`` and ``edge`` themselves sit at 0.53. It
measures phase, and no other probe here does — which is exactly the gap the DoG
blob candidate could not fill (it landed at 0.51 with ``corner``).

**The angle is defined modulo pi, and the encoding is what makes it
learnable.** An edge and its reverse run the same way, so orientation wraps at
pi, not 2pi, and a raw-angle regression target would be discontinuous across
that wrap. The double-angle unit vector ``(cos 2t, sin 2t)`` is single-valued
under it — the standard trick, and the same shape as a surface normal, which is
why the metric is modelled on :func:`~visbench.metrics.dense.surface_normal_metrics`.

**Coherence is a weight, not a mask.** ``(lambda_max - lambda_min) /
(lambda_max + lambda_min)`` is 1 where one direction dominates and 0 in a flat
or isotropic patch where no orientation is defined. It is folded into the
target's *length*, and both the loss and the metric weight by it, so an
undefined pixel contributes ~0 rather than being dropped by a threshold nobody
chose. On Taskonomy tiny val only 1.4% of pixels fall below coherence 0.1.

**No compression.** An angle has no heavy tail, so unlike ``corner`` there is
no ``log1p`` and no scale to sweep — see
:class:`~visbench.data.derived.OrientationResponse`. The one operator setting is
``sigma``, and it travels in ``dataset_params``.

**Not BSDS500's and not probe3d's.** probe3d has no orientation task; only its
optimiser schedule is borrowed, which ``optimizer`` already records. The
``protocol`` string says so.
"""

import torch
import torch.nn.functional as F

from visbench.metrics.dense import orientation_metrics
from visbench.registry import register_task
from visbench.tasks.dense_base import DenseTrainingTask
from visbench.types import MetricsDict

__all__ = ["OrientationTask"]


@register_task("orientation")
class OrientationTask(DenseTrainingTask):
    """Dense gradient-orientation regression over frozen features.

    The constructor is :class:`~visbench.tasks.dense_base.DenseTrainingTask`'s,
    unchanged. Pair it with
    :class:`~visbench.data.derived.DerivedTargetDataset` carrying an
    :class:`~visbench.data.derived.OrientationResponse`, which supplies the
    ``(2, H, W)`` target and records the operator that produced it.

    As with every dense probe here ``head="linear"`` is the number that
    compares *representations*; ``head="dpt"`` scores higher for everyone.
    """

    name = "orientation"
    level = "low_level"
    display_name = "Gradient orientation estimation"
    target_noun = "derived orientation fields"
    target_channels = 2
    protocol = "visbench_structure_tensor_orientation_regression"

    #: Tighter than the metric's clamp: ``acos`` has infinite derivative at the
    #: ends, so a prediction landing exactly on the target would blow up the
    #: head in one step. The metric has no gradient and uses the exact bound.
    _loss_eps = 1e-4

    @property
    def out_channels(self) -> int:
        """Two: ``cos 2theta`` and ``sin 2theta``."""
        return 2

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        """L2-normalise to a unit double-angle vector.

        The loss and metric both compare by cosine, so normalising changes
        neither — it is done so :meth:`predict` hands back an actual unit vector.
        ``eps`` guards the zero vector an untrained head predicts on step one.
        """
        return F.normalize(raw[:, :2], dim=1, eps=1e-8)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Coherence-weighted mean angular error, in the doubled angle.

        The target's per-pixel length is the coherence, so weighting by it
        discounts flat and isotropic pixels smoothly. A batch that is entirely
        incoherent returns a graph-connected zero rather than a NaN.
        """
        if pred.shape[1] < 2 or pred.shape[2:] != target.shape[2:]:
            raise ValueError(
                f"Prediction {tuple(pred.shape)} and target {tuple(target.shape)} must match"
            )
        weight = target.norm(dim=1)
        if not bool(weight.any()):
            return pred.sum() * 0.0
        cosine = torch.cosine_similarity(pred[:, :2], target, dim=1)
        error = cosine.clamp(min=-1 + self._loss_eps, max=1 - self._loss_eps).acos()
        return (weight * error).sum() / weight.sum().clamp(min=1e-6)

    def _batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
        """``{"orientation_error", "d1", "d2", "rmse", "median"}``; quote the error."""
        return orientation_metrics(pred, target)

    def _task_params(self) -> dict:
        """Override the inherited ``protocol``: this is not probe3d's."""
        return {
            "protocol": self.protocol,
            "loss": "coherence_weighted_angular",
            "activation": "l2_normalize",
        }
