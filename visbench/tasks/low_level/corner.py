"""Corner detection — the first probe whose target needs no dataset.

Every other probe here reads a target somebody else rendered. This one computes
it, deterministically, from whatever images are already on hand, which makes the
whole low-level tier reachable without a Taskonomy download. The measurement is
otherwise the one ``edge`` and ``keypoints2d`` already make: dense magnitude
regression over frozen features, scored by per-image Pearson correlation.

===============  ==========================================================
prediction       1 channel, identity — the output is the response
loss             L1
metric           per-image Pearson correlation (plus RMSE and MAE)
optimiser        AdamW, lr 5e-4, 10 epochs, 1.5 warmup, cosine decay
protocol         ``visbench_shi_tomasi_regression``
===============  ==========================================================

**The protocol names the operator, not the family.** "Harris corners" is not a
definition — the ``k`` parameter, the window, the smoothing and any non-maximum
suppression all move the target, so two records both claiming ``"harris"`` need
not be comparable at all. Naming the operator is only half the fix; the other
half is that every setting of it travels in ``dataset_params`` via
:meth:`~visbench.data.derived.ShiTomasiResponse.describe`, so two sigmas land in
two comparability groups without anyone having to notice. That is the same
mechanism that keeps step 6f's pixel-unit correspondence records apart from the
patch-unit ones.

**This target overlaps measurably with the edge target, and the record cannot
say so — so it is said here.** Per-image correlation between the corner target
and Taskonomy's ``edge_texture``, over the same 60 frames, is **0.52**; against
``keypoints2d``, the learned response this is the classical counterpart to, it
is **0.27**. For scale, ``edge_texture`` and ``keypoints2d`` — two probes this
library ships separately — correlate at **0.147** with each other. So this
target is *more* redundant with edges than those two are with one another, and
the overlap is intrinsic rather than an artifact of the compression: it holds
at 0.46-0.54 across eight transforms including near-linear ones, because a
corner is a pixel where the gradient is large in two directions and an edge map
is gradient magnitude.

What that means in practice is that a corner score and an edge score should not
be read as independent evidence about a backbone. It does **not** mean the probe
is redundant, because correlated targets can still rank backbones differently —
see the module's entry in ``visbench/tasks/low_level/README.md`` for the
measured ranking, which is the number that decides it.

**No validity mask.** The target is computed from the image, so every pixel is a
real measurement — the same fact that holds for ``edge_texture`` and
``keypoints2d``, and here it holds by construction rather than by observation:
there is no reconstruction to have holes in.
"""

from visbench.registry import register_task
from visbench.tasks.magnitude_base import DenseMagnitudeTask

__all__ = ["CornerTask"]


@register_task("corner")
class CornerTask(DenseMagnitudeTask):
    """Dense corner-response regression over frozen features.

    Everything mechanical — the identity activation, the L1 loss, the per-image
    correlation, and the measurement behind each — lives in
    :class:`~visbench.tasks.magnitude_base.DenseMagnitudeTask`. The constructor
    is ``DenseTrainingTask``'s, unchanged.

    Pair it with :class:`~visbench.data.derived.DerivedTargetDataset`, which
    supplies the target and records the operator that produced it.
    """

    name = "corner"
    level = "low_level"
    display_name = "Corner detection"
    target_noun = "derived corner response maps"
    correlation_key = "corner_correlation"
    protocol = "visbench_shi_tomasi_regression"
