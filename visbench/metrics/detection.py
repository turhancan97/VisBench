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

from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch

from visbench.types import MetricsDict

__all__ = [
    "box_iou",
    "mask_iou",
    "average_precision",
    "detection_metrics",
    "COCO_IOU_THRESHOLDS",
    "SHAPE_KINDS",
    "sweep_average_precision",
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


def mask_iou(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of binary masks, as an ``(N, M)`` tensor.

    ``masks_a`` is ``(N, H, W)`` and ``masks_b`` is ``(M, H, W)``, both boolean
    or castable to it. The counterpart of :func:`box_iou` for the same matching
    protocol: an instance-segmentation AP differs from a detection AP only in
    which of these two computes the overlap.

    Computed as a matrix product over flattened masks rather than an
    ``(N, M, H, W)`` broadcast, which at 224px and twenty instances a side would
    allocate about 4 GB to answer a 20x20 question.

    An empty input gives a correctly shaped empty output rather than raising: an
    image with no detections or no instances is ordinary, not an error. The
    masks must agree on ``(H, W)`` though — comparing masks of two resolutions
    is not a degenerate case but a pipeline that has already gone wrong, and the
    IoU it would produce is meaningless rather than merely small.
    """
    for name, masks in (("masks_a", masks_a), ("masks_b", masks_b)):
        if masks.ndim != 3:
            raise ValueError(f"{name} must be (N, H, W), got {tuple(masks.shape)}")
    if masks_a.shape[1:] != masks_b.shape[1:]:
        raise ValueError(
            f"masks_a is {tuple(masks_a.shape[1:])} and masks_b is "
            f"{tuple(masks_b.shape[1:])}; an IoU between two resolutions is meaningless."
        )
    if masks_a.numel() == 0 or masks_b.numel() == 0:
        return masks_a.new_zeros((masks_a.shape[0], masks_b.shape[0]), dtype=torch.float32)

    flat_a = masks_a.reshape(masks_a.shape[0], -1).to(torch.float32)
    flat_b = masks_b.reshape(masks_b.shape[0], -1).to(torch.float32)
    intersection = flat_a @ flat_b.T
    area_a = flat_a.sum(dim=1)
    area_b = flat_b.sum(dim=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    # An empty mask gives zero union; 0/0 is 0 IoU, not nan, exactly as in
    # box_iou -- one degenerate instance must not poison a whole curve.
    return torch.where(union > 0, intersection / union.clamp(min=1e-12), torch.zeros_like(union))


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


class _ShapeKind(NamedTuple):
    """How one geometry is read out of an annotation and overlapped.

    The VOC matching protocol -- rank every detection in the split, match each
    to the ground-truth shape it overlaps *most*, consult that shape's
    difficult/claimed state -- says nothing about what a shape is. Only three
    lines of :func:`average_precision` do, so they read this instead: the key
    the annotations carry, the reader that validates one, and the overlap.

    A table rather than a callable parameter, and listed rather than inferred,
    for the reason ``TARGET_STYLES`` and ``METRIC_DIRECTIONS`` are: guessing the
    geometry from whichever key happens to be present would silently score a
    mask run against absent boxes and report the 0.0 as a result.
    """

    key: str
    read: Callable[[object], torch.Tensor]
    overlap: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


#: The geometries :func:`average_precision` can match, by name. ``"boxes"`` is
#: the detection protocol and the default, so every existing caller is
#: unchanged; ``"masks"`` is the same protocol with the overlap swapped, which
#: is what makes an instance-segmentation AP comparable to a detection one.
SHAPE_KINDS: dict[str, _ShapeKind] = {}


def _as_bool_masks(value: object) -> torch.Tensor:
    """An ``(N, H, W)`` bool mask tensor, accepting a tensor, sequence or ``None``.

    ``None`` and an empty tensor both give ``(0, 0, 0)``: there is no resolution
    to infer from no masks, and :func:`mask_iou` short-circuits on an empty side
    before it would compare shapes.
    """
    if value is None:
        return torch.zeros((0, 0, 0), dtype=torch.bool)
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.numel() == 0:
        return torch.zeros((0, 0, 0), dtype=torch.bool)
    if tensor.ndim != 3:
        raise ValueError(
            f"masks must be (N, H, W), got {tuple(tensor.shape)}. A single mask needs a "
            "leading instance axis."
        )
    return tensor.to(torch.bool)


SHAPE_KINDS["boxes"] = _ShapeKind(key="boxes", read=_as_float_boxes, overlap=box_iou)
SHAPE_KINDS["masks"] = _ShapeKind(key="masks", read=_as_bool_masks, overlap=mask_iou)


def _refuse_the_other_geometry(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    kind: _ShapeKind,
) -> None:
    """Raise when the annotations carry a geometry other than the one asked for.

    This is the guard the whole adapter table exists for. Without it, scoring
    mask annotations with ``shapes="boxes"`` reads a ``"boxes"`` key that is not
    there, coerces the absence to an empty tensor, and returns **0.0** — a
    number that looks like a detector finding nothing rather than like a caller
    naming the wrong geometry. Every other failure in this module raises; this
    one would not, which is exactly the class of silent-wrong-number this
    codebase keeps paying for.

    An entry with *neither* key is fine and not checked: an image with no
    detections is ordinary, and the row-pairing check downstream catches a
    genuinely malformed one.
    """
    other_keys = {name: spec.key for name, spec in SHAPE_KINDS.items() if spec.key != kind.key}
    for label, entries in (("prediction", predictions), ("target", targets)):
        for entry in entries:
            if kind.key in entry:
                continue
            for other_name, other_key in other_keys.items():
                if other_key in entry:
                    raise ValueError(
                        f"A {label} carries {other_key!r} but not {kind.key!r}, and "
                        f"shapes={kind.key!r} was requested. Pass shapes={other_name!r} "
                        f"to score {other_key}; scoring them as {kind.key} would silently "
                        "report 0.0."
                    )


def _resolve_shape_kind(name: str) -> _ShapeKind:
    """The adapter named ``name``, or a refusal listing what exists."""
    try:
        return SHAPE_KINDS[name]
    except KeyError:
        raise ValueError(f"Unknown shapes={name!r}. Known: {sorted(SHAPE_KINDS)}.") from None


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


class _ClassMatches(NamedTuple):
    """One class's ranked detections, with everything the threshold does not move.

    The split that makes this worth existing: the matcher takes ``argmax`` over
    *every* ground-truth shape in the image and only then consults that shape's
    difficult and claimed state, so **which shape a detection matches best, and
    by how much, does not depend on the IoU threshold**. Only the tally does.

    So a ten-threshold sweep can overlap once and tally ten times. That is pure
    waste avoided rather than an approximation — :func:`_ap_at_threshold` still
    starts each threshold with a fresh ``claimed`` state, because *that* is
    threshold-dependent. On mask AP the saving is what makes the metric usable
    at all: 20 classes x 10 thresholds over 1,449 VOC images was recomputing
    every 224x224 mask overlap ten times.
    """

    #: Ranked detection -> index of the image it belongs to.
    image_indices: list[int]
    #: Ranked detection -> its best overlap with any shape of this class.
    best_overlap: list[float]
    #: Ranked detection -> which shape of that image it overlapped most, or -1
    #: when the image has no shapes of this class.
    best_index: list[int]
    #: Image -> ``difficult`` flags for this class's shapes, in target order.
    difficult: list[torch.Tensor]
    #: Non-difficult shapes of this class in the split: the recall denominator.
    num_positives: int


def _class_matches(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    *,
    class_id: int,
    kind: _ShapeKind,
) -> _ClassMatches | None:
    """Rank one class's detections and overlap each, independently of any threshold.

    Returns ``None`` when the class has no non-difficult shapes in the split,
    which is AP undefined rather than zero.
    """
    per_image_shapes: list[torch.Tensor] = []
    per_image_difficult: list[torch.Tensor] = []
    num_positives = 0
    for target in targets:
        labels = _as_tensor(target.get("labels"), torch.int64)
        target_shapes = kind.read(target.get(kind.key))
        difficult = target.get("difficult")
        if difficult is None:
            difficult_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
        else:
            difficult_mask = _as_tensor(difficult, torch.bool)
        if not (labels.shape[0] == target_shapes.shape[0] == difficult_mask.shape[0]):
            raise ValueError(
                f"A target has {target_shapes.shape[0]} {kind.key}, {labels.shape[0]} labels "
                f"and {difficult_mask.shape[0]} difficult flags; the three are paired by row."
            )
        keep = labels == class_id
        per_image_shapes.append(target_shapes[keep])
        per_image_difficult.append(difficult_mask[keep])
        # The recall denominator counts only shapes the protocol expects to be
        # found. Difficult ones are excluded here as well as in the tally.
        num_positives += int((~difficult_mask[keep]).sum())

    if num_positives == 0:
        return None

    # Every detection of this class in the split, ranked globally. This is the
    # step that makes AP dataset-level: ranking within an image and averaging
    # would measure something else.
    image_indices: list[int] = []
    scores: list[torch.Tensor] = []
    shapes_list: list[torch.Tensor] = []
    for index, prediction in enumerate(predictions):
        labels = _as_tensor(prediction.get("labels"), torch.int64)
        prediction_scores = _as_tensor(prediction.get("scores"), torch.float32)
        prediction_shapes = kind.read(prediction.get(kind.key))
        if not (labels.shape[0] == prediction_scores.shape[0] == prediction_shapes.shape[0]):
            raise ValueError(
                f"A prediction has {prediction_shapes.shape[0]} {kind.key}, "
                f"{prediction_scores.shape[0]} scores and {labels.shape[0]} labels; "
                "the three are paired by row."
            )
        keep = labels == class_id
        count = int(keep.sum())
        if count:
            image_indices.extend([index] * count)
            scores.append(prediction_scores[keep])
            shapes_list.append(prediction_shapes[keep])

    if not scores:
        return _ClassMatches([], [], [], per_image_difficult, num_positives)

    all_scores = torch.cat(scores)
    all_shapes = torch.cat(shapes_list)
    all_images = torch.tensor(image_indices, dtype=torch.int64)

    # Descending score. Stable so that equal scores keep a deterministic order
    # and the metric reproduces exactly across runs, as every other number here
    # does.
    order = torch.argsort(all_scores, descending=True, stable=True)

    ranked_images: list[int] = []
    ranked_overlap: list[float] = []
    ranked_index: list[int] = []
    for detection_index in order.tolist():
        image_index = int(all_images[detection_index])
        entry_shapes = per_image_shapes[image_index]
        ranked_images.append(image_index)
        if entry_shapes.numel() == 0:
            ranked_overlap.append(0.0)
            ranked_index.append(-1)
            continue
        overlaps = kind.overlap(all_shapes[detection_index][None], entry_shapes)[0]
        best = int(torch.argmax(overlaps))
        ranked_overlap.append(float(overlaps[best]))
        ranked_index.append(best)

    return _ClassMatches(
        ranked_images, ranked_overlap, ranked_index, per_image_difficult, num_positives
    )


def _ap_at_threshold(matches: _ClassMatches, iou_threshold: float) -> float:
    """Tally :func:`_class_matches` at one threshold and integrate the curve.

    ``claimed`` is rebuilt here rather than carried on ``matches``, because a
    shape claimed at IoU 0.5 need not be claimed at 0.75 — that state is the
    one part of the matching the threshold does move.
    """
    if not matches.image_indices:
        # Shapes exist and nothing was detected: recall 0 everywhere, so the
        # area under the curve is 0. Distinct from the None _class_matches gives.
        return 0.0

    claimed = [torch.zeros(flags.numel(), dtype=torch.bool) for flags in matches.difficult]
    count = len(matches.image_indices)
    true_positives = torch.zeros(count, dtype=torch.float64)
    false_positives = torch.zeros(count, dtype=torch.float64)
    ignored = torch.zeros(count, dtype=torch.bool)

    for rank in range(count):
        image_index = matches.image_indices[rank]
        best = matches.best_index[rank]
        if best < 0 or matches.best_overlap[rank] < iou_threshold:
            false_positives[rank] = 1.0
        elif bool(matches.difficult[image_index][best]):
            # Neither TP nor FP: removed from the tally. This is the whole
            # difference between VOC's protocol and dropping difficult targets.
            ignored[rank] = True
        elif bool(claimed[image_index][best]):
            false_positives[rank] = 1.0  # a duplicate detection
        else:
            true_positives[rank] = 1.0
            claimed[image_index][best] = True

    keep = ~ignored
    cumulative_tp = torch.cumsum(true_positives[keep], dim=0)
    cumulative_fp = torch.cumsum(false_positives[keep], dim=0)
    if cumulative_tp.numel() == 0:
        return 0.0

    recall = cumulative_tp / matches.num_positives
    precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp(min=1e-12)
    return _interpolated_average_precision(recall, precision)


def average_precision(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    *,
    class_id: int,
    iou_threshold: float = 0.5,
    shapes: str = "boxes",
) -> float | None:
    """VOC average precision for one class, over a whole split.

    Parameters
    ----------
    predictions:
        One dict per image, in the same order as ``targets``, each with the
        geometry named by ``shapes``, plus ``scores`` ``(N,)`` and ``labels``
        ``(N,)``. An image may have no detections.
    targets:
        One dict per image with the same geometry key, ``labels`` ``(M,)`` and
        ``difficult`` ``(M,)`` bool. ``difficult`` may be omitted, in which case
        no object is treated as difficult — but note that passing targets from
        which difficult objects have already been *removed* silently changes the
        protocol; see the module docstring.
    class_id:
        Which class to score. AP is per class by definition; mAP is the mean over
        classes, taken by :func:`detection_metrics`.
    iou_threshold:
        Minimum overlap for a detection to match an object.
    shapes:
        Which geometry to match, a key of :data:`SHAPE_KINDS`. ``"boxes"`` is
        the detection protocol and the default, so every existing caller is
        unchanged; ``"masks"`` is the same protocol with the overlap swapped,
        which is what :func:`~visbench.metrics.instance.mask_average_precision`
        names. **Annotations carrying the other geometry are refused** rather
        than read as absent and scored 0.0.

    Returns
    -------
    float or None
        The AP, or ``None`` when the class has **no non-difficult objects** in
        the split. :func:`detection_metrics` excludes those classes and reports
        how many it scored.

    Notes
    -----
    Matching follows ``VOCevaldet.m`` exactly, including a subtlety worth
    stating: each detection is matched to the ground-truth shape it overlaps
    **most**, and only then is that shape's state consulted. If the best-matching
    shape is difficult, the detection is ignored; if it is already claimed by a
    higher-scoring detection, this one is a false positive. There is deliberately
    no fallback to the second-best shape — a greedy alternative that reassigned
    duplicates would score higher than the reference implementation and stop
    being comparable to it.

    Because that ``argmax`` precedes every threshold comparison, the overlap
    work is threshold-independent and is factored into :func:`_class_matches`,
    which :func:`detection_metrics` reuses across its whole sweep.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"{len(predictions)} prediction entries against {len(targets)} target entries; "
            "detections and objects are paired by image index."
        )
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
    kind = _resolve_shape_kind(shapes)
    _refuse_the_other_geometry(predictions, targets, kind)

    matches = _class_matches(predictions, targets, class_id=class_id, kind=kind)
    if matches is None:
        return None
    return _ap_at_threshold(matches, iou_threshold)


def sweep_average_precision(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    *,
    num_classes: int,
    iou_thresholds: Sequence[float],
    shapes: str = "boxes",
) -> dict[int, dict[float, float | None]]:
    """AP for every class at every threshold, overlapping each class **once**.

    Returns ``{class_id: {threshold: ap_or_None}}`` with thresholds rounded to
    four places, which is how :func:`detection_metrics` and
    :func:`~visbench.metrics.instance.instance_metrics` key their lookups.

    Calling :func:`average_precision` per class per threshold gives identical
    numbers and recomputes every overlap once per threshold. For boxes that is
    merely wasteful; for masks it is the difference between a usable metric and
    an unusable one, so both mAP functions come through here. ``None`` is kept
    distinct from ``0.0``: a class absent from the split has undefined AP and
    must be excluded from the mean rather than dragging it down.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"{len(predictions)} prediction entries against {len(targets)} target entries; "
            "detections and objects are paired by image index."
        )
    kind = _resolve_shape_kind(shapes)
    _refuse_the_other_geometry(predictions, targets, kind)
    sweep = [round(float(threshold), 4) for threshold in iou_thresholds]
    for threshold in sweep:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"iou_thresholds must lie in (0, 1], got {threshold}")

    per_class: dict[int, dict[float, float | None]] = {}
    for class_id in range(num_classes):
        matches = _class_matches(predictions, targets, class_id=class_id, kind=kind)
        if matches is None:
            per_class[class_id] = dict.fromkeys(sweep)
        else:
            per_class[class_id] = {
                threshold: _ap_at_threshold(matches, threshold) for threshold in sweep
            }
    return per_class


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

    per_class = sweep_average_precision(
        predictions,
        targets,
        num_classes=num_classes,
        iou_thresholds=iou_thresholds,
        shapes="boxes",
    )
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
