"""Mid-level tasks — geometry and generic structure prior to semantic labeling.

The paper's core contribution area (Chen, Marks & Cheng, arXiv:2411.17474) and
where VisBench should be strongest relative to existing tools.

v0.1: geometric correspondence (zero-shot).
v0.2: depth, surface normals and generic segmentation (trained heads on frozen
features); mid-level similarity still to come.
"""

from visbench.tasks.mid_level.correspondence import CorrespondenceTask
from visbench.tasks.mid_level.depth import DepthTask
from visbench.tasks.mid_level.generic_segmentation import GenericSegmentationTask
from visbench.tasks.mid_level.surface_normal import SurfaceNormalTask

# Mid-level similarity is added when it is implemented; exporting a name whose
# module is still a stub would make this package unimportable.
__all__ = [
    "CorrespondenceTask",
    "DepthTask",
    "GenericSegmentationTask",
    "SurfaceNormalTask",
]
