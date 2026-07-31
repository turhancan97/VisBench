"""Low-level tasks — signal-level properties, recoverable without naming objects.

A placeholder from v0.1 until step 6d-1, which added the first entry: edge
detection, on Taskonomy's ``edge_texture`` maps. See ``README.md`` in this
folder for the remaining intended scope (optical flow, texture/reflectance,
image quality assessment) and what each would cost.
"""

from visbench.tasks.low_level.edge import EdgeTask

__all__ = ["EdgeTask"]
