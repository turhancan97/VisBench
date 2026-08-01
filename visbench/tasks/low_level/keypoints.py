"""2D keypoint detection — the second low-level task, v0.4 (step 6d-2).

Taskonomy's ``keypoints2d`` is a dense corner/interest-point response computed
from the RGB frame alone — a SURF-style detector response rendered as a map, not
a sparse list of coordinates. That makes it low-level in exactly the sense this
library's taxonomy means, and it makes it a **regression** target rather than a
detection one: there is nothing to match, only a response surface to reproduce.

**Why this is a separate probe from edges rather than a second dataset for the
same one.** Mechanically they are identical, and they share
:class:`~visbench.tasks.magnitude_base.DenseMagnitudeTask` for exactly that
reason. But they ask different questions of a representation — an edge response
fires along intensity *contours*, a keypoint response at *corners and blobs*,
and a backbone can be good at one and weak at the other. Two registered names
with two protocol strings is what stops the two numbers being pooled or
mistaken for repeats of each other.

===============  ==========================================================
prediction       1 channel, identity — the output is the response
loss             L1
metric           per-image Pearson correlation (plus RMSE and MAE)
optimiser        AdamW, lr 5e-4, 10 epochs, 1.5 warmup, cosine decay
protocol         ``visbench_keypoint2d_regression``
===============  ==========================================================

**No validity mask, and that is a fact about the target rather than a default.**
``keypoints2d`` is derived from the image, so its response is a real measurement
everywhere — including inside the 3D reconstruction's holes, verified on real
data in step 6d-2. Its 3D sibling ``keypoints3d`` is the opposite case and does
need masking; the two names differ by one character and their validity
conventions differ completely.
"""

from visbench.registry import register_task
from visbench.tasks.magnitude_base import DenseMagnitudeTask

__all__ = ["Keypoint2DTask"]


@register_task("keypoints2d")
class Keypoint2DTask(DenseMagnitudeTask):
    """Dense 2D keypoint-response regression over frozen features.

    Everything mechanical — the identity activation, the L1 loss, the per-image
    correlation, and the measurement behind each — lives in
    :class:`~visbench.tasks.magnitude_base.DenseMagnitudeTask`. The constructor
    is ``DenseTrainingTask``'s, unchanged.
    """

    name = "keypoints2d"
    level = "low_level"
    display_name = "2D keypoint detection"
    target_noun = "target keypoint response maps"
    correlation_key = "keypoint_correlation"
    protocol = "visbench_keypoint2d_regression"
