"""Detection metrics — average precision and mAP, v0.3 step 6c-2.

Follows the Pascal VOC protocol as its own reference implementation defines it
(``VOCevaldet.m``, VOC2010 onward), because the whole value of quoting mAP is
that it is comparable to published numbers, and three separate conventions here
move it by several points each.

**AP is a dataset-level quantity, and this is the one place the codebase's
"per image, then averaged" rule does not apply.** Every other metric here scores
each image and averages, so uneven coverage cannot reweight the split. Average
precision cannot work that way: it is the area under one precision-recall curve
built by ranking *every* detection in the split against *every* ground-truth
box, and a mean of per-image APs is a different, lower number with no published
counterpart. Semantic segmentation already carries the same tension and resolves
it by reporting both (``miou`` and ``miou_per_image``); here there is no
defensible per-image version to report, so there is only the dataset-level one.

**`difficult` objects are ignored, not dropped.** VOC flags 4,462 of its 40,138
objects ``<difficult>1</difficult>``, and its protocol removes a detection that
matches one from the tally entirely — neither true positive nor false positive.
That is not the same as dropping difficult boxes from the ground truth, which
would make a *correct* detection of a difficult object a false positive and
depress mAP below every published figure. The distinction is the reason
:func:`average_precision` takes a ``difficult`` mask rather than pre-filtered
targets, and the reason
:class:`~visbench.data.DetectionFolderDataset` must be constructed with
``include_difficult=True`` when its targets are headed for this module.

Two details that are choices rather than inherited, spelled out because a
reader comparing numbers needs them:

* **All-points interpolation**, as VOC2010+ and COCO use — not the 11-point
  sampling of VOC2007. A number computed the 2007 way is systematically higher
  and is not comparable.
* **mAP@50:95 averages ten thresholds**, 0.50 to 0.95 in steps of 0.05, which is
  COCO's *range*. COCO also quantises the recall axis to 101 points; this module
  integrates all points at every threshold, so ``map_50`` is directly
  VOC-comparable while ``map_50_95`` is COCO-**style** rather than a COCO number.
"""

from collections.abc import Sequence

import torch

from visbench.types import MetricsDict

__all__ = [
    "box_iou",
    "average_precision",
    "detection_metrics",
    "COCO_IOU_THRESHOLDS",
]

#: COCO's IoU sweep, 0.50 to 0.95 inclusive in steps of 0.05. Ten values, and
#: the reason ``map_50_95`` is not simply ``map_50``'s stricter cousin: a
#: detector good at localising coarsely can score well at 0.5 and poorly here.
COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of ``xyxy`` boxes, as an ``(N, M)`` tensor.

    Corners are treated as continuous coordinates, so a box's width is
    ``x2 - x1`` — matching :mod:`visbench.data.detection`, which converts VOC's
    inclusive integer ``xmax`` on read. Using ``x2 - x1 + 1`` here instead would
    make the metric disagree with the dataset about how big every box is, which
    shifts IoU slightly and therefore shifts which detections match.

    An empty input gives a correctly shaped empty output rather than raising:
    an image with no detections or no objects is ordinary, not an error.
    """
    if boxes_a.ndim != 2 or boxes_a.shape[-1] != 4:
        raise ValueError(f"boxes_a must be (N, 4), got {tuple(boxes_a.shape)}")
    if boxes_b.ndim != 2 or boxes_b.shape[-1] != 4:
        raise ValueError(f"boxes_b must be (M, 4), got {tuple(boxes_b.shape)}")
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(
        min=0
    )
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(
        min=0
    )

    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    overlap = (bottom_right - top_left).clamp(min=0)
    intersection = overlap[..., 0] * overlap[..., 1]

    union = area_a[:, None] + area_b[None, :] - intersection
    # A degenerate box gives zero union; 0/0 is 0 IoU, not nan, or one bad box
    # would poison every mean it takes part in.
    return torch.where(union > 0, intersection / union.clamp(min=1e-12), torch.zeros_like(union))


def _interpolated_average_precision(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """Area under the precision-recall curve, all-points interpolated.

    Precision is made monotonically non-increasing from the right — at each
    recall level the best precision achievable at that recall *or higher* — then
    integrated exactly. This is VOC2010+ and COCO's definition. VOC2007 instead
    sampled 11 fixed recall points, which yields a systematically higher number,
    so the two must not be mixed in one table.
    """
    # Sentinels at both ends so the curve starts at recall 0 and closes at 1.
    padded_recall = torch.cat([recall.new_zeros(1), recall, recall.new_ones(1)])
    padded_precision = torch.cat([precision.new_zeros(1), precision, precision.new_zeros(1)])

    for index in range(padded_precision.numel() - 2, -1, -1):
        padded_precision[index] = torch.maximum(
            padded_precision[index], padded_precision[index + 1]
        )

    changes = padded_recall[1:] != padded_recall[:-1]
    return float(
        ((padded_recall[1:] - padded_recall[:-1])[changes] * padded_precision[1:][changes]).sum()
    )


def average_precision(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    *,
    class_id: int,
    iou_threshold: float = 0.5,
) -> float | None:
    """VOC average precision for one class, over a whole split.

    Parameters
    ----------
    predictions:
        One dict per image, in the same order as ``targets``, each with
        ``boxes`` ``(N, 4)`` ``xyxy``, ``scores`` ``(N,)`` and ``labels``
        ``(N,)``. An image may have no detections.
    targets:
        One dict per image with ``boxes`` ``(M, 4)``, ``labels`` ``(M,)`` and
        ``difficult`` ``(M,)`` bool. ``difficult`` may be omitted, in which case
        no object is treated as difficult — but note that passing targets from
        which difficult objects have already been *removed* silently changes the
        protocol; see the module docstring.
    class_id:
        Which class to score. AP is per class by definition; mAP is the mean over
        classes, taken by :func:`detection_metrics`.
    iou_threshold:
        Minimum IoU for a detection to match an object.

    Returns
    -------
    The AP, or ``None`` when the class has **no non-difficult objects anywhere
    in the split**. That is undefined rather than zero: recall has no
    denominator, and scoring it 0 would drag a mAP down in proportion to how
    many classes the split happens not to contain. :func:`detection_metrics`
    excludes those classes and reports how many it scored.

    Notes
    -----
    Matching follows ``VOCevaldet.m`` exactly, including a subtlety worth
    stating: each detection is matched to the ground-truth box it overlaps
    **most**, and only then is that box's state consulted. If the best-matching
    box is difficult, the detection is ignored; if it is already claimed by a
    higher-scoring detection, this one is a false positive. There is deliberately
    no fallback to the second-best box — a greedy alternative that reassigned
    duplicates would score higher than the reference implementation and stop
    being comparable to it.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"{len(predictions)} prediction entries against {len(targets)} target entries; "
            "detections and objects are paired by image index."
        )
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    # Ground truth for this class, per image, plus the claimed/difficult state
    # the matching loop mutates.
    per_image: list[dict] = []
    num_positives = 0
    for target in targets:
        labels = _as_tensor(target.get("labels"), torch.int64)
        boxes = _as_float_boxes(target.get("boxes"))
        difficult = target.get("difficult")
        if difficult is None:
            difficult_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
        else:
            difficult_mask = _as_tensor(difficult, torch.bool)
        if not (labels.shape[0] == boxes.shape[0] == difficult_mask.shape[0]):
            raise ValueError(
                f"A target has {boxes.shape[0]} boxes, {labels.shape[0]} labels and "
                f"{difficult_mask.shape[0]} difficult flags; the three are paired by row."
            )

        keep = labels == class_id
        entry_difficult = difficult_mask[keep]
        per_image.append(
            {
                "boxes": boxes[keep],
                "difficult": entry_difficult,
                "claimed": torch.zeros(int(keep.sum()), dtype=torch.bool),
            }
        )
        # The recall denominator counts only objects the protocol expects to be
        # found. Difficult ones are excluded here as well as in the tally.
        num_positives += int((~entry_difficult).sum())

    if num_positives == 0:
        return None

    # Every detection of this class in the split, ranked globally. This is the
    # step that makes AP dataset-level: ranking within an image and averaging
    # would measure something else.
    image_indices: list[int] = []
    scores: list[torch.Tensor] = []
    boxes_list: list[torch.Tensor] = []
    for index, prediction in enumerate(predictions):
        labels = _as_tensor(prediction.get("labels"), torch.int64)
        prediction_scores = _as_tensor(prediction.get("scores"), torch.float32)
        prediction_boxes = _as_float_boxes(prediction.get("boxes"))
        if not (labels.shape[0] == prediction_scores.shape[0] == prediction_boxes.shape[0]):
            raise ValueError(
                f"A prediction has {prediction_boxes.shape[0]} boxes, "
                f"{prediction_scores.shape[0]} scores and {labels.shape[0]} labels; "
                "the three are paired by row."
            )
        keep = labels == class_id
        count = int(keep.sum())
        if count:
            image_indices.extend([index] * count)
            scores.append(prediction_scores[keep])
            boxes_list.append(prediction_boxes[keep])

    if not scores:
        # Objects exist and nothing was detected: recall 0 everywhere, so the
        # area under the curve is 0. Distinct from the None above.
        return 0.0

    all_scores = torch.cat(scores)
    all_boxes = torch.cat(boxes_list)
    all_images = torch.tensor(image_indices, dtype=torch.int64)

    # Descending score. Stable so that equal scores keep a deterministic order
    # and the metric reproduces exactly across runs, as every other number here
    # does.
    order = torch.argsort(all_scores, descending=True, stable=True)

    true_positives = torch.zeros(order.numel(), dtype=torch.float64)
    false_positives = torch.zeros(order.numel(), dtype=torch.float64)
    ignored = torch.zeros(order.numel(), dtype=torch.bool)

    for rank, detection_index in enumerate(order.tolist()):
        entry = per_image[int(all_images[detection_index])]
        if entry["boxes"].numel() == 0:
            false_positives[rank] = 1.0
            continue

        overlaps = box_iou(all_boxes[detection_index][None, :], entry["boxes"])[0]
        best = int(torch.argmax(overlaps))
        if float(overlaps[best]) < iou_threshold:
            false_positives[rank] = 1.0
        elif bool(entry["difficult"][best]):
            # Neither TP nor FP: removed from the tally. This is the whole
            # difference between VOC's protocol and dropping difficult targets.
            ignored[rank] = True
        elif bool(entry["claimed"][best]):
            false_positives[rank] = 1.0  # a duplicate detection
        else:
            true_positives[rank] = 1.0
            entry["claimed"][best] = True

    keep = ~ignored
    cumulative_tp = torch.cumsum(true_positives[keep], dim=0)
    cumulative_fp = torch.cumsum(false_positives[keep], dim=0)
    if cumulative_tp.numel() == 0:
        return 0.0

    recall = cumulative_tp / num_positives
    precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp(min=1e-12)
    return _interpolated_average_precision(recall, precision)


def detection_metrics(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    *,
    num_classes: int,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
) -> MetricsDict:
    """mAP@50 and mAP@50:95 over a split, plus how many classes were scored.

    ``map_50`` is the VOC number and is directly comparable to published VOC
    mAP. ``map_50_95`` averages ``iou_thresholds`` and is COCO-*style*; see the
    module docstring for why it is not a COCO number.

    ``classes_scored`` is reported because mAP is a mean over classes and the
    denominator is not always ``num_classes``: a class with no non-difficult
    objects in the split has undefined AP and is excluded rather than scored 0,
    which would otherwise make mAP depend on how many categories the split
    happens to omit. A caller comparing two runs should check this matches.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if not iou_thresholds:
        raise ValueError("iou_thresholds is empty, so mAP@50:95 would have nothing to average")
    if 0.5 not in tuple(round(float(value), 4) for value in iou_thresholds):
        raise ValueError(
            f"iou_thresholds {tuple(iou_thresholds)} does not include 0.5, so map_50 "
            "cannot be reported from it. Pass a sweep containing 0.5."
        )

    # class -> threshold -> AP, keeping None (undefined) distinct from 0.0.
    per_class: dict[int, dict[float, float | None]] = {}
    for class_id in range(num_classes):
        per_class[class_id] = {
            round(float(threshold), 4): average_precision(
                predictions, targets, class_id=class_id, iou_threshold=float(threshold)
            )
            for threshold in iou_thresholds
        }

    scored = [class_id for class_id, values in per_class.items() if values[0.5] is not None]

    def mean_at(threshold: float) -> float:
        values: list[float] = []
        for class_id in scored:
            value = per_class[class_id][round(threshold, 4)]
            if value is not None:
                values.append(value)
        return float(sum(values) / len(values)) if values else 0.0

    sweep = [round(float(threshold), 4) for threshold in iou_thresholds]
    return {
        "map_50": mean_at(0.5),
        "map_50_95": float(sum(mean_at(threshold) for threshold in sweep) / len(sweep)),
        "classes_scored": float(len(scored)),
    }


def _as_tensor(value: object, dtype: torch.dtype) -> torch.Tensor:
    """A 1-D tensor of ``dtype``, accepting a tensor, sequence or ``None``."""
    if value is None:
        return torch.zeros(0, dtype=dtype)
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.reshape(-1).to(dtype)


def _as_float_boxes(value: object) -> torch.Tensor:
    """An ``(N, 4)`` float32 box tensor, accepting a tensor, sequence or ``None``."""
    if value is None:
        return torch.zeros((0, 4), dtype=torch.float32)
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(torch.float32).reshape(-1, 4)
    return tensor
