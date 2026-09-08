"""Instance-segmentation metrics — mask AP and mask mAP, step 14a-2.

**This is the detection protocol with the overlap swapped, and nothing else.**
:func:`~visbench.metrics.detection.average_precision` does the ranking and
matching — every detection in the split ranked globally, each matched to the
ground-truth shape it overlaps *most*, that shape's difficult and claimed state
consulted only then — and it reads *which* geometry to overlap from a listed
table. So mask AP is that same function with ``shapes="masks"``, and the whole
of this module is naming its outputs.

That sharing is deliberate rather than convenient. The matching loop reproduces
``VOCevaldet.m`` including the subtlety that there is **no fallback to the
second-best shape**, and it is the piece a second implementation would get
subtly wrong — the same argument the gallery's duplicated ``_row`` lost the hard
way. A mask AP computed by a parallel matcher would be incomparable to this
codebase's own box AP, which is the one comparison the shared split exists to
make.

**Every convention is inherited, and each one moves the number by points.**
All-points interpolation (VOC2010+, not VOC2007's 11 points); ``difficult``
objects **ignored rather than dropped**; AP taken at the **dataset level**, not
per image then averaged; and a class with no non-difficult instances scored
``None`` and excluded rather than counted as 0. See
:mod:`visbench.metrics.detection` for what each is worth.

**The keys are prefixed ``mask_``**, so ``mask_map_50`` never sits in a flat
metrics dict beside detection's ``map_50`` meaning something else. Two probes
over the same 1,449 VOC images reporting a bare ``map_50`` each would be one
rename away from a leaderboard averaging them.

**Calibration, and why it is a test rather than a remark.** On predictions that
*are* the ground truth this reports exactly ``1.0000``, which is what makes any
number below it attributable to the probe. The same check on box AP (step 6c-2)
is what caught the difficult-objects convention being worth 4.3 mAP. A mask
matcher has one further way to be quietly wrong — comparing masks of two
resolutions — and :func:`~visbench.metrics.detection.mask_iou` refuses that
outright rather than returning a small IoU.
"""

from collections.abc import Sequence

import torch

from visbench.metrics.detection import (
    COCO_IOU_THRESHOLDS,
    average_precision,
    sweep_average_precision,
)
from visbench.types import MetricsDict

__all__ = [
    "instance_metrics",
    "mask_average_precision",
    "masks_from_instance_map",
]


def mask_average_precision(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    *,
    class_id: int,
    iou_threshold: float = 0.5,
) -> float | None:
    """Mask average precision for one class, over a whole split.

    A thin naming of
    :func:`~visbench.metrics.detection.average_precision` with
    ``shapes="masks"``; see that function for the parameters and for what
    ``None`` means.

    Parameters
    ----------
    predictions:
        One dict per image, in the same order as ``targets``, each with
        ``masks`` ``(N, H, W)`` bool, ``scores`` ``(N,)`` and ``labels``
        ``(N,)``. An image may have no detections.
    targets:
        One dict per image with ``masks`` ``(M, H, W)``, ``labels`` ``(M,)`` and
        optionally ``difficult`` ``(M,)`` bool — which is *ignored*, never
        dropped, so targets must not be pre-filtered.
    class_id:
        Which class to score. AP is per class by definition.
    iou_threshold:
        Minimum mask IoU for a detection to match an instance.

    Returns
    -------
    float or None
        The AP, or ``None`` when the class has no non-difficult instances in
        the split, which is undefined rather than zero.
    """
    return average_precision(
        predictions,
        targets,
        class_id=class_id,
        iou_threshold=iou_threshold,
        shapes="masks",
    )


def instance_metrics(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    *,
    num_classes: int,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
) -> MetricsDict:
    """Mask mAP@50 and mAP@50:95 over a split, plus how many classes were scored.

    ``mask_map_50`` is the VOC-comparable number. ``mask_map_50_95`` averages
    ``iou_thresholds`` and is COCO-*style* rather than a COCO number, because
    COCO quantises the recall axis to 101 points where this integrates all of
    them — the same caveat detection's ``map_50_95`` carries.

    ``classes_scored`` is reported for the reason detection reports it: mAP is a
    mean over classes and the denominator is not always ``num_classes``, since a
    class absent from the split has undefined AP and is excluded rather than
    scored 0. A caller comparing two runs should check it matches.

    Parameters
    ----------
    predictions, targets:
        As :func:`mask_average_precision`, one entry per image, paired by index.
    num_classes:
        How many classes to score, ``0..num_classes - 1``.
    iou_thresholds:
        The sweep to average for ``mask_map_50_95``. Must contain 0.5, or
        ``mask_map_50`` could not be reported from it.

    Returns
    -------
    MetricsDict
        ``mask_map_50``, ``mask_map_50_95`` and ``classes_scored``.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if not iou_thresholds:
        raise ValueError("iou_thresholds is empty, so mask_map_50_95 would have nothing to average")
    sweep = [round(float(threshold), 4) for threshold in iou_thresholds]
    if 0.5 not in sweep:
        raise ValueError(
            f"iou_thresholds {tuple(iou_thresholds)} does not include 0.5, so mask_map_50 "
            "cannot be reported from it. Pass a sweep containing 0.5."
        )

    # One overlap pass per class, reused across the sweep. Calling
    # mask_average_precision per threshold gives identical numbers and
    # recomputes every 224x224 mask IoU ten times over.
    per_class = sweep_average_precision(
        predictions,
        targets,
        num_classes=num_classes,
        iou_thresholds=sweep,
        shapes="masks",
    )

    scored = [class_id for class_id, values in per_class.items() if values[0.5] is not None]

    def mean_at(threshold: float) -> float:
        values = [
            value
            for class_id in scored
            if (value := per_class[class_id][round(threshold, 4)]) is not None
        ]
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "mask_map_50": mean_at(0.5),
        "mask_map_50_95": float(sum(mean_at(threshold) for threshold in sweep) / len(sweep)),
        "classes_scored": float(len(scored)),
    }


def masks_from_instance_map(indices: torch.Tensor) -> torch.Tensor:
    """Split a ``(H, W)`` instance index map into ``(N, H, W)`` boolean masks.

    Background (0) is not an instance and gets no plane. Present purely so a
    caller holding a single index map — a prediction from a head that emits one,
    or a target read straight off disk — can reach this module's shape without
    writing the same two lines again and getting the background case wrong.

    Order follows ascending index, which is the order
    :meth:`~visbench.data.instance.VOCInstanceDataset.target` produces, so the
    two are directly comparable.
    """
    if indices.ndim != 2:
        raise ValueError(f"indices must be (H, W), got {tuple(indices.shape)}")
    present = [int(value) for value in torch.unique(indices) if int(value) != 0]
    if not present:
        return torch.zeros((0, *indices.shape), dtype=torch.bool)
    return torch.stack([indices == value for value in present])
