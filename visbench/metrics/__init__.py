"""Metric implementations, kept out of task classes.

Separate so that a metric can be unit-tested against known inputs without
constructing a backbone or a task, and so two tasks can share one definition.
Every function returns a flat dict (or a scalar), matching the contract of
:meth:`BaseTask.evaluate`.
"""

from visbench.metrics.boundary import (
    boundary_metrics,
    correspond_pixels,
    image_counts,
    thin_boundaries,
)
from visbench.metrics.classification import top_k_accuracy
from visbench.metrics.correspondence import (
    correspondence_recall,
    error_auc,
    nn_match,
    ratio_test,
)
from visbench.metrics.dense import (
    depth_metrics,
    match_scale_and_shift,
    surface_normal_metrics,
)
from visbench.metrics.detection import (
    COCO_IOU_THRESHOLDS,
    average_precision,
    box_iou,
    detection_metrics,
)
from visbench.metrics.retrieval import mean_average_precision, recall_at_k

__all__ = [
    "top_k_accuracy",
    "boundary_metrics",
    "correspond_pixels",
    "image_counts",
    "thin_boundaries",
    "recall_at_k",
    "mean_average_precision",
    "nn_match",
    "ratio_test",
    "correspondence_recall",
    "error_auc",
    "depth_metrics",
    "match_scale_and_shift",
    "surface_normal_metrics",
    "box_iou",
    "average_precision",
    "detection_metrics",
    "COCO_IOU_THRESHOLDS",
]
