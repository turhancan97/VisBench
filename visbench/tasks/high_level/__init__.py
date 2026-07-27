"""High-level tasks — semantic / category understanding.

v0.1: classification (linear probe), retrieval (zero-shot).
v0.2: semantic segmentation (trained head on frozen features). v0.3: detection.
"""

from visbench.tasks.high_level.classification import ClassificationTask
from visbench.tasks.high_level.retrieval import RetrievalTask
from visbench.tasks.high_level.semantic_segmentation import SemanticSegmentationTask

# DetectionTask (v0.3) is added when it is implemented; exporting a name whose
# module is still a stub would make this package unimportable.
__all__ = ["ClassificationTask", "RetrievalTask", "SemanticSegmentationTask"]
