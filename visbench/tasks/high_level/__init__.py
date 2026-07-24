"""High-level tasks — semantic / category understanding.

v0.1: classification (linear probe), retrieval (zero-shot).
v0.2: semantic segmentation. v0.3: detection.
"""

from visbench.tasks.high_level.retrieval import RetrievalTask

# ClassificationTask, SemanticSegmentationTask and DetectionTask are added as
# they are implemented; exporting a name whose module is still a stub would
# make this package unimportable.
__all__ = ["RetrievalTask"]
