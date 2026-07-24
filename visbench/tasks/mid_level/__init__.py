"""Mid-level tasks — geometry and generic structure prior to semantic labeling.

The paper's core contribution area (Chen, Marks & Cheng, arXiv:2411.17474) and
where VisBench should be strongest relative to existing tools.

v0.1: geometric correspondence (zero-shot).
v0.2: depth, surface normals, generic segmentation, mid-level similarity.
"""

__all__ = [
    "CorrespondenceTask",
    "DepthTask",
    "SurfaceNormalTask",
    "GenericSegmentationTask",
    "MidLevelSimilarityTask",
]
