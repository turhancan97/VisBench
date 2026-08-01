"""Dense magnitude regression — the shared half of the edge-style probes, v0.4 (step 6d-2).

Lifted out of a *working* :class:`~visbench.tasks.low_level.edge.EdgeTask` when
the second and third magnitude targets arrived, which is the same move that
produced :class:`~visbench.tasks.dense_base.DenseTrainingTask` from a working
``DepthTask`` and ``tasks/schedule.py`` from a working ``DetectionTask``. Nothing
here was designed up front; every line of it had already run against real
weights under one name.

A **magnitude target** is a single-channel, non-negative, heavy-tailed map whose
interesting content is *where* it is large rather than how large — edge
strength, keypoint response. Three properties follow, and all three were
measured rather than reasoned:

* the activation is the **identity**, because both ways of imposing
  non-negativity destroy the probe;
* the loss is **L1**, because squared error is dominated by the few strongest
  pixels;
* the headline metric is **per-image Pearson correlation**, because a constant
  predictor achieves a small RMSE on a target concentrated near zero.

See :meth:`DenseMagnitudeTask._activate` and
:func:`~visbench.metrics.dense.magnitude_metrics` for the numbers behind each.

Subclasses supply ``name``, ``display_name``, ``correlation_key``, ``protocol``
and ``level`` — nothing else. That the three probes differ only in what they are
*called* is the honest description: they are the same measurement applied to
three targets, and keeping them as three registered names with three protocol
strings is what stops their numbers being pooled.
"""

import torch
import torch.nn.functional as F

from visbench.metrics.dense import magnitude_metrics
from visbench.tasks.dense_base import DenseTrainingTask
from visbench.types import MetricsDict

__all__ = ["DenseMagnitudeTask"]


class DenseMagnitudeTask(DenseTrainingTask):
    """Dense single-channel magnitude regression over frozen features.

    As with every other dense probe here, ``head="linear"`` is the number that
    compares *representations* — it is the only head under which a difference
    between two backbones is a difference between two feature maps — and
    ``head="dpt"`` scores higher for everyone. Report both, or say which.
    """

    target_channels = 1

    #: Key the correlation is reported under. Per-probe, so an occlusion-edge
    #: correlation and a texture-edge correlation cannot be mistaken for the
    #: same measurement in a record — they are not comparable.
    correlation_key = "correlation"

    #: Value of the record's ``protocol`` field. Never ``"probe3d"``: that paper
    #: has no magnitude task, and only its *optimiser* schedule is borrowed
    #: here, which ``optimizer`` already records.
    protocol = "visbench_magnitude_regression"

    @property
    def out_channels(self) -> int:
        """One magnitude per pixel."""
        return 1

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        """The identity: the head's output *is* the magnitude.

        A magnitude cannot be negative, so the obvious move is to impose that
        with a rectifying activation. **Both ways of doing so were tried and
        both destroy the probe**, because the target lives in a narrow band just
        above zero — a frame's mean is about 0.011 of the container range.
        Measured on features that literally encode the answer, where the ceiling
        is 1.0:

        ============  ====================  =========================================
        activation    correlation           what happens
        ============  ====================  =========================================
        ``relu``      0.0000                dies outright; prediction std is exactly 0
        ``softplus``  -0.9851               collapses to a near-constant
        identity      **0.9997**            recovers the target
        ============  ====================  =========================================

        ReLU fails the way its reputation suggests: with the target this close
        to zero the pre-activation sits near or below the origin, the gradient is
        zero there, and the output never recovers. Softplus fails less obviously
        and so is the more dangerous of the two — to emit 0.065 it needs a raw
        value near -2.7, where its own gradient is ``sigmoid(-2.7)`` ~ 0.06, so
        it attenuates the signal about sixteenfold in exactly the region this
        target occupies. Its -0.985 is not learning the inverse; it is a constant
        prediction whose residual noise happens to anti-correlate.

        So non-negativity is **learned rather than imposed** — predictions come
        out non-negative because the targets are, and the loss is what says so.
        A prediction may dip slightly below zero in a flat region; that costs a
        little MAE and nothing else, which is a far smaller price than either row
        above. Not a sigmoid either: the magnitude has no upper bound the task
        knows about, and real maps reach only a fraction of the container range,
        so squashing to ``[0, 1]`` would spend most of the output range on values
        that never occur.
        """
        return raw

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 over the valid pixels.

        L1 and not MSE because the magnitude distribution is heavy-tailed — on
        Taskonomy's ``edge_texture`` a frame's mean is around 0.011 of the
        container range and its peak around 0.13. Squared error is dominated by
        that handful of strongest edges, so the probe would be optimised to fit
        the brightest contours and left free to ignore the rest of the map, which
        is most of it.

        **Invalid pixels are ``NaN`` and are masked here**, matching
        :func:`~visbench.metrics.dense.magnitude_metrics` exactly — the loss and
        the metric must drop the same pixels, or the probe is optimised against
        pixels it is not scored on. For an image-derived target
        (``edge_texture``, ``keypoints2d``) nothing is ever masked: every pixel
        is a real measurement, and a 0 means "no edge here". See
        :func:`~visbench.data.dense.load_edge_map`.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction {tuple(pred.shape)} and target {tuple(target.shape)} must match"
            )
        valid = torch.isfinite(target)
        if bool(valid.all()):
            # The common path, and kept exact rather than routed through the
            # masked one: boolean indexing would reduce over a flattened copy
            # and is measurably slower for no difference in result.
            return F.l1_loss(pred, target)
        if not bool(valid.any()):
            # Every pixel of the batch is a reconstruction hole. Returning a
            # detached zero would break the graph; this keeps it and contributes
            # nothing, so the step is a no-op rather than a NaN.
            return pred.sum() * 0.0
        return (pred[valid] - target[valid]).abs().mean()

    def _batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
        """``{correlation_key, "rmse", "mae"}``; quote the correlation."""
        return magnitude_metrics(pred, target, correlation_key=self.correlation_key)

    def _task_params(self) -> dict:
        """Override the inherited ``protocol``: none of these is probe3d's or BSDS's.

        probe3d has no magnitude task, and BSDS500's edge protocol is a
        correspondence metric none of these implements. Only the optimiser
        schedule is borrowed, and that is already recorded under ``optimizer``.
        The whole value of this field is that it says what a number may be
        compared with.
        """
        return {
            "protocol": self.protocol,
            "loss": "l1",
            "activation": "identity",
        }
