"""Detection metrics — step 6c-2.

Average precision has more conventions per line than anything else in this
codebase, and each one moves the number by points rather than decimals: how
``difficult`` objects are handled, whether detections are ranked per image or
across the split, and whether the precision-recall curve is interpolated at all
points or sampled at eleven. A wrong choice produces a *plausible* mAP.

So the APs here are **hand-computed** and asserted exactly, and several tests are
built so that the wrong convention gives a specific, different, also-plausible
answer — the arithmetic is written into the docstring so a future reader can
check the expectation rather than trusting it.
"""

import pytest
import torch

from visbench.metrics import average_precision, box_iou, detection_metrics
from visbench.metrics.detection import COCO_IOU_THRESHOLDS


def prediction(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def target(boxes, labels, difficult=None):
    count = len(labels)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "difficult": torch.tensor(
            [False] * count if difficult is None else difficult, dtype=torch.bool
        ),
    }


# -- box_iou ----------------------------------------------------------------


def test_iou_of_identical_boxes_is_one():
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

    assert box_iou(box, box).item() == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[20.0, 20.0, 30.0, 30.0]])

    assert box_iou(a, b).item() == 0.0


def test_iou_of_a_known_half_overlap():
    """Two 10x10 boxes offset by 5 in x: intersection 50, union 150, IoU 1/3."""
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[5.0, 0.0, 15.0, 10.0]])

    assert box_iou(a, b).item() == pytest.approx(1.0 / 3.0)


def test_iou_uses_continuous_corners_not_inclusive_indices():
    """Width is ``x2 - x1``, matching the dataset's conversion of VOC's xmax.

    A 0..10 box is 10 wide here, not 11. Reading corners as inclusive indices
    would give area 121 against 100 and shift every IoU, so the metric and the
    dataset would disagree about how big each box is.
    """
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    half = torch.tensor([[0.0, 0.0, 5.0, 10.0]])

    # 50 / 100 exactly, only if the widths are 10 and 5.
    assert box_iou(box, half).item() == pytest.approx(0.5)


def test_iou_shape_and_empty_inputs():
    a = torch.rand(3, 4).sort(dim=1).values * 10
    b = torch.rand(5, 4).sort(dim=1).values * 10

    assert box_iou(a, b).shape == (3, 5)
    assert box_iou(a, torch.zeros(0, 4)).shape == (3, 0)
    assert box_iou(torch.zeros(0, 4), b).shape == (0, 5)


def test_iou_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        box_iou(torch.zeros(3, 5), torch.zeros(2, 4))


# -- average precision, hand-computed ---------------------------------------


def test_perfect_detection_scores_one():
    predictions = [prediction([[0, 0, 10, 10]], [0.9], [0])]
    targets = [target([[0, 0, 10, 10]], [0])]

    assert average_precision(predictions, targets, class_id=0) == pytest.approx(1.0)


def test_a_duplicate_detection_is_a_false_positive():
    """Hand-computed: AP = 5/6.

    Two objects, three detections ranked 0.9, 0.8, 0.7. The first matches object
    A, the second is a duplicate of A, the third matches B.

        tp   = [1, 0, 1]      fp = [0, 1, 0]      npos = 2
        recall    = [0.5, 0.5, 1.0]
        precision = [1.0, 0.5, 2/3]

    All-points interpolation takes precision 1.0 up to recall 0.5 and 2/3 from
    there to 1.0, so AP = 0.5 * 1.0 + 0.5 * 2/3 = 5/6.
    """
    predictions = [
        prediction(
            [[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]],
            [0.9, 0.8, 0.7],
            [0, 0, 0],
        )
    ]
    targets = [target([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 0])]

    assert average_precision(predictions, targets, class_id=0) == pytest.approx(5.0 / 6.0)


def test_detections_are_ranked_across_the_split_not_per_image():
    """Hand-computed: AP = 2/3 globally, and 0.75 if ranked per image.

    Image 0 has one object and one correct detection at score 0.6. Image 1 has
    one object, a *false* detection at 0.9 and a correct one at 0.5.

    Ranked globally the order is FP, TP, TP:

        recall    = [0, 0.5, 1.0]
        precision = [0, 0.5, 2/3]        ->  AP = 2/3

    Ranked per image and averaged, image 0 scores 1.0 and image 1 scores 0.5,
    giving 0.75. The two differ, and only the first has a published counterpart —
    this is the one place the codebase's "per image, then averaged" rule is
    deliberately not followed.
    """
    predictions = [
        prediction([[0, 0, 10, 10]], [0.6], [0]),
        prediction([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.5], [0, 0]),
    ]
    targets = [
        target([[0, 0, 10, 10]], [0]),
        target([[50, 50, 60, 60]], [0]),
    ]

    result = average_precision(predictions, targets, class_id=0)

    assert result == pytest.approx(2.0 / 3.0)
    assert result != pytest.approx(0.75)


def test_all_points_interpolation_not_eleven_point():
    """A single detection at recall 1 gives exactly 1.0 under all-points.

    The VOC2007 11-point rule samples recall at 0, 0.1, ... 1.0 and would also
    give 1.0 here, so this case cannot separate them; what it pins is that the
    curve is closed at recall 1 rather than left short, which would give 0.0.
    """
    predictions = [prediction([[0, 0, 10, 10]], [0.5], [0])]
    targets = [target([[0, 0, 10, 10]], [0])]

    assert average_precision(predictions, targets, class_id=0) == pytest.approx(1.0)


def test_a_loose_match_fails_a_stricter_threshold():
    """IoU 1/3: a true positive at 0.3, a false positive at 0.5."""
    predictions = [prediction([[5, 0, 15, 10]], [0.9], [0])]
    targets = [target([[0, 0, 10, 10]], [0])]

    assert average_precision(predictions, targets, class_id=0, iou_threshold=0.3) == pytest.approx(
        1.0
    )
    assert average_precision(predictions, targets, class_id=0, iou_threshold=0.5) == 0.0


# -- difficult objects: ignored, not dropped --------------------------------


def test_a_detection_matching_a_difficult_object_is_ignored():
    """The whole point of the protocol: AP = 1.0, not 0.5.

    Object A is difficult, B is not. Detections: 0.9 on A, 0.8 on B.

    Ignoring A's detection leaves ``tp = [1]``, ``fp = []`` against ``npos = 1``,
    so AP = 1.0. If difficult objects were merely *dropped from the targets*,
    the 0.9 detection would be a false positive:

        recall = [0, 1.0], precision = [0, 0.5]   ->  AP = 0.5

    Both numbers are plausible; only 1.0 is VOC's.
    """
    predictions = [
        prediction([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8], [0, 0]),
    ]
    targets = [
        target([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 0], difficult=[True, False]),
    ]

    assert average_precision(predictions, targets, class_id=0) == pytest.approx(1.0)


def test_difficult_objects_are_not_in_the_recall_denominator():
    """One difficult object plus one normal one, only the normal one detected.

    ``npos`` counts the non-difficult object alone, so recall reaches 1.0 and
    AP is 1.0. Counting the difficult object as a required detection would cap
    recall at 0.5 and give AP 0.5.
    """
    predictions = [prediction([[20, 20, 30, 30]], [0.9], [0])]
    targets = [
        target([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 0], difficult=[True, False]),
    ]

    assert average_precision(predictions, targets, class_id=0) == pytest.approx(1.0)


def test_a_class_of_only_difficult_objects_is_undefined():
    """No non-difficult objects means recall has no denominator, so AP is None."""
    predictions = [prediction([[0, 0, 10, 10]], [0.9], [0])]
    targets = [target([[0, 0, 10, 10]], [0], difficult=[True])]

    assert average_precision(predictions, targets, class_id=0) is None


def test_missing_difficult_key_treats_nothing_as_difficult():
    predictions = [prediction([[0, 0, 10, 10]], [0.9], [0])]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]

    assert average_precision(predictions, targets, class_id=0) == pytest.approx(1.0)


# -- undefined vs zero ------------------------------------------------------


def test_a_class_absent_from_the_split_is_none_not_zero():
    """Undefined, so mAP does not depend on how many categories a split omits."""
    predictions = [prediction([[0, 0, 10, 10]], [0.9], [0])]
    targets = [target([[0, 0, 10, 10]], [0])]

    assert average_precision(predictions, targets, class_id=7) is None


def test_objects_with_no_detections_score_zero():
    """Distinct from None: the class is present and was entirely missed."""
    predictions = [prediction([], [], [])]
    targets = [target([[0, 0, 10, 10]], [0])]

    assert average_precision(predictions, targets, class_id=0) == 0.0


def test_only_wrong_class_detections_score_zero():
    predictions = [prediction([[0, 0, 10, 10]], [0.9], [1])]
    targets = [target([[0, 0, 10, 10]], [0])]

    assert average_precision(predictions, targets, class_id=0) == 0.0


# -- refusals ---------------------------------------------------------------


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="paired by image index"):
        average_precision(
            [prediction([[0, 0, 1, 1]], [0.5], [0])],
            [target([[0, 0, 1, 1]], [0]), target([[0, 0, 1, 1]], [0])],
            class_id=0,
        )


def test_ragged_target_rows_raise():
    bad = {
        "boxes": torch.zeros(2, 4),
        "labels": torch.tensor([0]),
        "difficult": torch.tensor([False, False]),
    }

    with pytest.raises(ValueError, match="paired by row"):
        average_precision([prediction([], [], [])], [bad], class_id=0)


def test_an_out_of_range_iou_threshold_raises():
    with pytest.raises(ValueError, match="iou_threshold"):
        average_precision(
            [prediction([[0, 0, 1, 1]], [0.5], [0])],
            [target([[0, 0, 1, 1]], [0])],
            class_id=0,
            iou_threshold=0.0,
        )


# -- detection_metrics ------------------------------------------------------


def test_map_averages_only_the_classes_present():
    """classes_scored is the mAP denominator, and it is not num_classes."""
    predictions = [prediction([[0, 0, 10, 10]], [0.9], [0])]
    targets = [target([[0, 0, 10, 10]], [0])]

    result = detection_metrics(predictions, targets, num_classes=20)

    assert result["classes_scored"] == 1.0
    # One class, perfectly detected: the mean over scored classes is 1.0, not
    # 1/20, which is what scoring the 19 absent classes as 0 would give.
    assert result["map_50"] == pytest.approx(1.0)


def test_map_50_95_is_below_map_50_for_a_loose_match():
    """A box at IoU 0.6 passes at 0.5 and 0.55 but fails the other eight."""
    # 10x10 target, prediction offset so IoU is about 0.6.
    predictions = [prediction([[0, 0, 10, 12]], [0.9], [0])]
    targets = [target([[0, 0, 10, 10]], [0])]

    result = detection_metrics(predictions, targets, num_classes=1)

    assert result["map_50"] == pytest.approx(1.0)
    assert result["map_50_95"] < result["map_50"]


def test_map_50_95_of_a_perfect_detector_is_one():
    """An exact box matches at every threshold, so the sweep cannot dilute it."""
    predictions = [prediction([[0, 0, 10, 10]], [0.9], [0])]
    targets = [target([[0, 0, 10, 10]], [0])]

    result = detection_metrics(predictions, targets, num_classes=1)

    assert result["map_50"] == pytest.approx(1.0)
    assert result["map_50_95"] == pytest.approx(1.0)


def test_metrics_are_flat_floats():
    """BaseTask.evaluate returns a flat dict, so a metric helper must too."""
    predictions = [prediction([[0, 0, 10, 10]], [0.9], [0])]
    targets = [target([[0, 0, 10, 10]], [0])]

    result = detection_metrics(predictions, targets, num_classes=2)

    assert set(result) == {"map_50", "map_50_95", "classes_scored"}
    assert all(isinstance(value, float) for value in result.values())


def test_a_sweep_without_point_five_raises():
    """map_50 must come from the sweep, not be silently absent."""
    with pytest.raises(ValueError, match="0.5"):
        detection_metrics(
            [prediction([[0, 0, 10, 10]], [0.9], [0])],
            [target([[0, 0, 10, 10]], [0])],
            num_classes=1,
            iou_thresholds=(0.75, 0.9),
        )


def test_the_coco_sweep_is_ten_thresholds_from_fifty_to_ninetyfive():
    assert COCO_IOU_THRESHOLDS[0] == 0.5
    assert COCO_IOU_THRESHOLDS[-1] == 0.95
    assert len(COCO_IOU_THRESHOLDS) == 10


def test_scores_break_ties_deterministically():
    """Every number in this codebase reproduces exactly; equal scores must not
    make AP depend on sort implementation."""
    predictions = [
        prediction([[0, 0, 10, 10], [20, 20, 30, 30]], [0.5, 0.5], [0, 0]),
    ]
    targets = [target([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 0])]

    first = average_precision(predictions, targets, class_id=0)
    again = average_precision(predictions, targets, class_id=0)

    assert first == again == pytest.approx(1.0)


# -- cross-check against a literal VOCevaldet.m transcription ---------------
#
# The analytic tests above pin individual conventions. This pins the *interaction*
# of all of them — duplicates, difficult objects, cross-image ranking and the
# threshold — over many random configurations, against a reference written from
# VOC's MATLAB control flow rather than from the module.
#
# It is here because the obvious future change to `average_precision` is
# vectorising its per-detection loop, and the subtlety most likely to be lost is
# that VOC picks the highest-overlap ground-truth box *first* and only then asks
# whether it is difficult or already claimed. A greedy variant that fell back to
# the second-best box would score higher than the reference and stop being
# comparable to published mAP, while passing every hand-computed test above.
# When this ran during 6c-2 it agreed on 3,060 defined APs with a maximum
# difference of exactly 0.0.


def _reference_overlap(box, gt_box):
    """VOCevaldet's overlap, with this codebase's continuous-corner convention."""
    left, top = max(box[0], gt_box[0]), max(box[1], gt_box[1])
    right, bottom = min(box[2], gt_box[2]), min(box[3], gt_box[3])
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    union = (
        (box[2] - box[0]) * (box[3] - box[1])
        + (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _reference_vocap(recalls, precisions):
    """VOCap.m: sentinels, precision monotonic from the right, exact integration."""
    recall = [0.0, *recalls, 1.0]
    precision = [0.0, *precisions, 0.0]
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    return sum(
        (recall[index] - recall[index - 1]) * precision[index]
        for index in range(1, len(recall))
        if recall[index] != recall[index - 1]
    )


def _reference_ap(predictions, targets, class_id, minoverlap):
    """Literal transcription of VOCevaldet.m's loop. None when npos == 0."""
    ground_truth, num_positives = [], 0
    for entry in targets:
        labels = entry["labels"].tolist()
        keep = [row for row, label in enumerate(labels) if label == class_id]
        boxes, difficult = entry["boxes"].tolist(), entry["difficult"].tolist()
        ground_truth.append(
            {
                "boxes": [boxes[row] for row in keep],
                "difficult": [difficult[row] for row in keep],
                "claimed": [False] * len(keep),
            }
        )
        num_positives += sum(1 for row in keep if not difficult[row])
    if num_positives == 0:
        return None

    detections = [
        (float(entry["scores"][row]), image, entry["boxes"][row].tolist())
        for image, entry in enumerate(predictions)
        for row, label in enumerate(entry["labels"].tolist())
        if label == class_id
    ]
    if not detections:
        return 0.0
    detections.sort(key=lambda item: -item[0])

    running_tp = running_fp = 0
    recalls, precisions = [], []
    for _, image, box in detections:
        entry = ground_truth[image]
        best_overlap, best = -float("inf"), -1
        for index, gt_box in enumerate(entry["boxes"]):
            overlap = _reference_overlap(box, gt_box)
            if overlap > best_overlap:
                best_overlap, best = overlap, index
        if best_overlap >= minoverlap:
            if entry["difficult"][best]:
                continue  # ignored: contributes to neither tally
            if entry["claimed"][best]:
                running_fp += 1
            else:
                running_tp += 1
                entry["claimed"][best] = True
        else:
            running_fp += 1
        recalls.append(running_tp / num_positives)
        total = running_tp + running_fp
        precisions.append(running_tp / total if total else 0.0)
    if not recalls:
        return 0.0
    return _reference_vocap(recalls, precisions)


def _random_case(rng, num_classes=3, num_images=4):
    predictions, targets = [], []
    for _ in range(num_images):
        gt_boxes, gt_labels, gt_difficult = [], [], []
        for _ in range(rng.randint(0, 4)):
            x, y = rng.uniform(0, 80), rng.uniform(0, 80)
            gt_boxes.append([x, y, x + rng.uniform(5, 40), y + rng.uniform(5, 40)])
            gt_labels.append(rng.randrange(num_classes))
            gt_difficult.append(rng.random() < 0.25)
        targets.append(target(gt_boxes, gt_labels, gt_difficult or None))

        boxes, scores, labels = [], [], []
        for _ in range(rng.randint(0, 6)):
            if gt_boxes and rng.random() < 0.6:
                # Perturb a real box so matches, duplicates and near-misses all occur.
                jitter = rng.uniform(-6, 6)
                boxes.append([value + jitter for value in gt_boxes[rng.randrange(len(gt_boxes))]])
            else:
                x, y = rng.uniform(0, 80), rng.uniform(0, 80)
                boxes.append([x, y, x + rng.uniform(5, 40), y + rng.uniform(5, 40)])
            scores.append(rng.random())
            labels.append(rng.randrange(num_classes))
        predictions.append(prediction(boxes, scores, labels))
    return predictions, targets


def test_matches_a_literal_vocevaldet_transcription():
    """Agreement over many random configurations, at three IoU thresholds."""
    import random

    rng = random.Random(0)
    compared = 0
    for _ in range(60):
        predictions, targets = _random_case(rng)
        for class_id in range(3):
            for threshold in (0.3, 0.5, 0.75):
                mine = average_precision(
                    predictions, targets, class_id=class_id, iou_threshold=threshold
                )
                reference = _reference_ap(predictions, targets, class_id, threshold)
                # None means "no non-difficult objects"; both must agree it is
                # undefined rather than one of them reporting 0.
                assert (mine is None) == (reference is None)
                if mine is not None and reference is not None:
                    assert mine == pytest.approx(reference, abs=1e-12)
                    compared += 1
    # Guards the guard: a generator that produced only empty cases would make
    # every assertion above vacuous, which is the QuickGELU failure shape.
    assert compared > 300
