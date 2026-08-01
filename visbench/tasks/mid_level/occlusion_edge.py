"""Occlusion-edge detection — a mid-level probe, v0.4 (step 6d-2).

An *occlusion* edge is a depth discontinuity: the boundary where one surface
stops and a farther one begins. A *texture* edge is an intensity discontinuity.
They coincide often enough to be confusable and differ exactly where the
interesting cases are — a painted line on a wall is a texture edge and not an
occlusion edge, the silhouette of a chair against a similarly-toned wall is an
occlusion edge with almost no intensity gradient.

That is why this probe is **mid-level and the texture-edge one is low-level**,
following Chen, Marks & Cheng (arXiv:2411.17474): recovering it requires scene
geometry, not just the signal. It is also the first pair of probes in VisBench
that share their entire implementation and differ only in what they read, which
makes it about as clean a comparison of the two levels as this library can
offer: run both on the same frames and the gap is the geometry.

===============  ==========================================================
prediction       1 channel, identity — the output is the edge magnitude
loss             L1, over valid pixels only
metric           per-image Pearson correlation (plus RMSE and MAE)
optimiser        AdamW, lr 5e-4, 10 epochs, 1.5 warmup, cosine decay
protocol         ``visbench_occlusion_edge_regression``
===============  ==========================================================

**This one masks, and the texture-edge probe does not.** Taskonomy computes
``edge_occlusion`` from the 3D reconstruction, so it has holes — about 3% of an
average frame and 12% of the worst sampled — and those holes hold a plain 0,
which is indistinguishable from a real "no occlusion edge here". So validity
travels out of band as ``NaN``, and both the loss and the metric mask on
``isfinite``. Reusing the texture-edge probe's "score every pixel" rule would
train this one to predict "no edge" wherever the reconstruction failed, and
nothing would raise. See :func:`~visbench.metrics.dense.magnitude_metrics` for
why ``NaN`` rather than an in-band sentinel.
"""

from visbench.registry import register_task
from visbench.tasks.magnitude_base import DenseMagnitudeTask

__all__ = ["OcclusionEdgeTask"]


@register_task("occlusion_edge")
class OcclusionEdgeTask(DenseMagnitudeTask):
    """Dense occlusion-edge regression over frozen features.

    Everything mechanical — the identity activation, the masked L1 loss, the
    per-image correlation, and the measurement behind each — lives in
    :class:`~visbench.tasks.magnitude_base.DenseMagnitudeTask`. The constructor
    is ``DenseTrainingTask``'s, unchanged.
    """

    name = "occlusion_edge"
    level = "mid_level"
    display_name = "Occlusion-edge detection"
    target_noun = "target occlusion-edge maps"
    correlation_key = "occlusion_edge_correlation"
    protocol = "visbench_occlusion_edge_regression"
