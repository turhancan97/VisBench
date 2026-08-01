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
normals have zero-length vectors, both meaning "no ground truth". Taskonomy's
``edge_texture`` has neither: 0 means *no edge*, which is a real reading
covering most of most frames, and because the map is computed from the RGB
frame it is a real reading even inside the 3D reconstruction's holes — verified
on real data in step 6d-2, where those regions hold ordinary magnitudes.
Masking them away — the obvious thing to copy from the three dense tasks that
came before — would score the probe only where an edge already is, which is the
one place the answer is easy.

The *occlusion*-edge probe next door is the contrasting case: it comes from the
reconstruction, so it does have holes, and it masks them. Same machinery, one
different fact about the target. See
:class:`~visbench.tasks.mid_level.occlusion_edge.OcclusionEdgeTask`.
"""

from visbench.registry import register_task
from visbench.tasks.magnitude_base import DenseMagnitudeTask

__all__ = ["EdgeTask"]


@register_task("edge")
class EdgeTask(DenseMagnitudeTask):
    """Dense edge-magnitude regression over frozen features.

    Everything mechanical lives in
    :class:`~visbench.tasks.magnitude_base.DenseMagnitudeTask` — the identity
    activation, the L1 loss and the correlation metric, each with the
    measurement that chose it. What is left here is what makes this probe *this*
    probe: the target it reads, the level it sits at, and the protocol name its
    records carry.

    There are no task-specific hyperparameters — in particular no threshold,
    since nothing here is binarised. The constructor is
    ``DenseTrainingTask``'s, unchanged.
    """

    name = "edge"
    level = "low_level"
    display_name = "Edge detection"
    target_noun = "target edge maps"
    correlation_key = "edge_correlation"
    protocol = "visbench_edge_regression"
