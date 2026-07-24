"""Mid-level tasks — geometry and generic structure prior to semantic labeling.

The paper's core contribution area (Chen, Marks & Cheng, arXiv:2411.17474) and
where VisBench should be strongest relative to existing tools.

v0.1: geometric correspondence (zero-shot).
v0.2: depth, surface normals, generic segmentation, mid-level similarity.
"""

from visbench.tasks.mid_level.correspondence import CorrespondenceTask

# Depth, surface normals, generic segmentation and mid-level similarity are
# added as they are implemented; exporting a name whose module is still a stub
# would make this package unimportable.
__all__ = ["CorrespondenceTask"]
