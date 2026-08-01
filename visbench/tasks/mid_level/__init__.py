"""Mid-level tasks — geometry and generic structure prior to semantic labeling.

The paper's core contribution area (Chen, Marks & Cheng, arXiv:2411.17474) and
where VisBench should be strongest relative to existing tools.

v0.1: geometric correspondence (zero-shot).
v0.2: depth, surface normals and generic segmentation (trained heads on frozen
features), plus mid-level image similarity (zero-shot 2AFC).
v0.4: occlusion-edge detection, the geometric counterpart of the low-level
texture-edge probe — same implementation, different target, one level apart.
"""

from visbench.tasks.mid_level.correspondence import CorrespondenceTask
from visbench.tasks.mid_level.depth import DepthTask
from visbench.tasks.mid_level.generic_segmentation import GenericSegmentationTask
from visbench.tasks.mid_level.occlusion_edge import OcclusionEdgeTask
from visbench.tasks.mid_level.similarity import MidLevelSimilarityTask
from visbench.tasks.mid_level.surface_normal import SurfaceNormalTask

__all__ = [
    "CorrespondenceTask",
    "DepthTask",
    "GenericSegmentationTask",
    "MidLevelSimilarityTask",
    "OcclusionEdgeTask",
    "SurfaceNormalTask",
]
