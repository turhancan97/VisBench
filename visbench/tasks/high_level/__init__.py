"""High-level tasks — semantic / category understanding.

v0.1: classification (linear probe), retrieval (zero-shot).
v0.2: semantic segmentation (trained head on frozen features).
v0.3: detection (anchor-free single-scale head, step 6c-3).
"""

from visbench.tasks.high_level.classification import ClassificationTask
from visbench.tasks.high_level.detection import DetectionTask
from visbench.tasks.high_level.retrieval import RetrievalTask
from visbench.tasks.high_level.semantic_segmentation import SemanticSegmentationTask

__all__ = [
    "ClassificationTask",
    "DetectionTask",
    "RetrievalTask",
    "SemanticSegmentationTask",
]
