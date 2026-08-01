"""Low-level tasks — signal-level properties, recoverable without naming objects.

A placeholder from v0.1 until step 6d-1, which added the first entry: edge
detection, on Taskonomy's ``edge_texture`` maps. Step 6d-2 added 2D keypoint
detection beside it, on ``keypoints2d``. Both are dense magnitude regression and
share :class:`~visbench.tasks.magnitude_base.DenseMagnitudeTask`; what makes
them two probes rather than one is that an edge response fires along contours
and a keypoint response at corners, and a backbone can be good at one and weak
at the other.

See ``README.md`` in this folder for the remaining intended scope (optical flow,
texture/reflectance, image quality assessment) and what each would cost.
"""

from visbench.tasks.low_level.edge import EdgeTask
from visbench.tasks.low_level.keypoints import Keypoint2DTask

__all__ = ["EdgeTask", "Keypoint2DTask"]
