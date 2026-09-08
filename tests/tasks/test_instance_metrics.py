"""Mask AP and mask mAP — step 14a-2.

The claim this file has to defend is that instance segmentation and detection
share **one** matching protocol, differing only in which overlap is computed.
Two tests carry that weight and the rest are conventions:

:meth:`TestMaskApEqualsBoxAp.test_rectangle_masks_score_exactly_as_their_boxes`
is the strongest available check. A mask that *is* an axis-aligned rectangle has
a pixel-set IoU identical to the half-open box IoU of its corners, so mask AP
must equal box AP **exactly** on such input — not approximately. If the two
paths ever diverge in ranking, matching, difficult handling or integration, this
fails, and it fails on a number rather than on a shape.

:meth:`TestTheSweepIsAnOptimisationNotAnApproximation.test_sweep_agrees_with_per_threshold_calls`
defends the other half. ``sweep_average_precision`` overlaps once per class and
tallies per threshold, on the grounds that the best-matching shape is
threshold-independent. That reasoning is only worth as much as its equality with
the naive per-threshold path, so the equality is asserted directly, exactly.

The rest exists because each convention here is worth points of mAP and none of
them raises when wrong: ``difficult`` ignored rather than dropped, a class with
no instances scored ``None`` rather than 0, and the calibration at 1.0000
without which no lower number is attributable to anything.
"""

import pytest
import torch

from visbench.metrics import (
    SHAPE_KINDS,
    average_precision,
    detection_metrics,
    instance_metrics,
    mask_average_precision,
    mask_iou,
    masks_from_instance_map,
)
from visbench.metrics.detection import sweep_average_precision

SIZE = 16


def rect(x_min, y_min, x_max, y_max, size=SIZE):
    """A boolean mask filling the half-open box ``[x_min, x_max) x [y_min, y_max)``."""
    mask = torch.zeros((size, size), dtype=torch.bool)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def rect_pair(boxes, size=SIZE):
    """``(masks, boxes)`` for a list of ``(x_min, y_min, x_max, y_max)``."""
    if not boxes:
        return torch.zeros((0, size, size), dtype=torch.bool), torch.zeros((0, 4))
    masks = torch.stack([rect(*box, size=size) for box in boxes])
    return masks, torch.tensor(boxes, dtype=torch.float32)


class TestMaskIou:
    def test_identical_masks_score_one(self):
        masks = rect(2, 2, 6, 6)[None]
        assert float(mask_iou(masks, masks)[0, 0]) == 1.0

    def test_disjoint_masks_score_zero(self):
        assert float(mask_iou(rect(0, 0, 4, 4)[None], rect(8, 8, 12, 12)[None])[0, 0]) == 0.0

    def test_a_known_overlap(self):
        """Half of one square inside the other: 8 shared of 24 covered."""
        a = rect(0, 0, 4, 4)[None]  # 16 px
        b = rect(2, 0, 6, 4)[None]  # 16 px, overlapping 2..4 in x -> 8 px
        assert float(mask_iou(a, b)[0, 0]) == pytest.approx(8 / 24)

    def test_the_matrix_is_pairwise(self):
        a, _ = rect_pair([(0, 0, 4, 4), (8, 8, 12, 12)])
        b, _ = rect_pair([(0, 0, 4, 4), (2, 2, 6, 6), (8, 8, 12, 12)])
        overlaps = mask_iou(a, b)
        assert overlaps.shape == (2, 3)
        assert float(overlaps[0, 0]) == 1.0
        assert float(overlaps[1, 2]) == 1.0
        assert float(overlaps[0, 2]) == 0.0

    def test_an_empty_side_gives_a_shaped_empty_result(self):
        """An image with no detections is ordinary, not an error."""
        a, _ = rect_pair([(0, 0, 4, 4)])
        empty = torch.zeros((0, SIZE, SIZE), dtype=torch.bool)
        assert mask_iou(a, empty).shape == (1, 0)
        assert mask_iou(empty, a).shape == (0, 1)

    def test_an_empty_mask_gives_zero_not_nan(self):
        """0/0 must not poison a curve, exactly as in box_iou."""
        blank = torch.zeros((1, SIZE, SIZE), dtype=torch.bool)
        value = float(mask_iou(blank, blank)[0, 0])
        assert value == 0.0

    def test_two_resolutions_are_refused(self):
        """Not a degenerate case but a pipeline already gone wrong."""
        small = torch.zeros((1, 8, 8), dtype=torch.bool)
        large = torch.zeros((1, 16, 16), dtype=torch.bool)
        with pytest.raises(ValueError, match="meaningless"):
            mask_iou(small, large)

    def test_a_single_mask_without_an_instance_axis_is_refused(self):
        with pytest.raises(ValueError, match=r"must be \(N, H, W\)"):
            mask_iou(rect(0, 0, 4, 4), rect(0, 0, 4, 4)[None])


class TestMaskApEqualsBoxAp:
    """The load-bearing claim: one protocol, two overlaps."""

    @staticmethod
    def _split():
        """Three images whose instances are all axis-aligned rectangles."""
        layout = [
            # (detections as (box, score, label)), (targets as (box, label, difficult))
            (
                [((0, 0, 6, 6), 0.9, 0), ((8, 8, 14, 14), 0.4, 0)],
                [((0, 0, 6, 6), 0, False), ((8, 8, 13, 13), 0, False)],
            ),
            (
                [((2, 2, 8, 8), 0.8, 1), ((2, 2, 7, 7), 0.7, 1)],
                [((2, 2, 8, 8), 1, False)],
            ),
            (
                [((0, 0, 4, 4), 0.6, 0)],
                [((10, 10, 15, 15), 0, True), ((0, 0, 4, 4), 0, False)],
            ),
        ]
        box_preds, box_tgts, mask_preds, mask_tgts = [], [], [], []
        for detections, objects in layout:
            d_masks, d_boxes = rect_pair([d[0] for d in detections])
            scores = torch.tensor([d[1] for d in detections], dtype=torch.float32)
            d_labels = torch.tensor([d[2] for d in detections], dtype=torch.int64)
            t_masks, t_boxes = rect_pair([o[0] for o in objects])
            t_labels = torch.tensor([o[1] for o in objects], dtype=torch.int64)
            difficult = torch.tensor([o[2] for o in objects], dtype=torch.bool)

            box_preds.append({"boxes": d_boxes, "scores": scores, "labels": d_labels})
            box_tgts.append({"boxes": t_boxes, "labels": t_labels, "difficult": difficult})
            mask_preds.append({"masks": d_masks, "scores": scores, "labels": d_labels})
            mask_tgts.append({"masks": t_masks, "labels": t_labels, "difficult": difficult})
        return box_preds, box_tgts, mask_preds, mask_tgts

    def test_rectangle_masks_score_exactly_as_their_boxes(self):
        """A rectangle's pixel IoU *is* its half-open box IoU, so AP must match.

        Exact equality, not approximate: any divergence in ranking, matching,
        difficult handling or integration shows up here as a different number.
        """
        box_preds, box_tgts, mask_preds, mask_tgts = self._split()
        for class_id in (0, 1):
            for threshold in (0.5, 0.75, 0.9):
                boxed = average_precision(
                    box_preds, box_tgts, class_id=class_id, iou_threshold=threshold
                )
                masked = mask_average_precision(
                    mask_preds, mask_tgts, class_id=class_id, iou_threshold=threshold
                )
                assert boxed == masked, f"class {class_id} at {threshold}"

    def test_the_map_tables_agree_too(self):
        """The same equality through both mAP entry points, which sweep."""
        box_preds, box_tgts, mask_preds, mask_tgts = self._split()
        boxed = detection_metrics(box_preds, box_tgts, num_classes=2)
        masked = instance_metrics(mask_preds, mask_tgts, num_classes=2)
        assert boxed["map_50"] == masked["mask_map_50"]
        assert boxed["map_50_95"] == masked["mask_map_50_95"]
        assert boxed["classes_scored"] == masked["classes_scored"]


class TestTheSweepIsAnOptimisationNotAnApproximation:
    @staticmethod
    def _random_split(seed, *, shapes):
        generator = torch.Generator().manual_seed(seed)
        preds, tgts = [], []
        for _ in range(4):
            boxes = []
            for _ in range(int(torch.randint(0, 4, (1,), generator=generator))):
                x = int(torch.randint(0, 10, (1,), generator=generator))
                y = int(torch.randint(0, 10, (1,), generator=generator))
                boxes.append((x, y, x + 4, y + 4))
            t_boxes = []
            for _ in range(int(torch.randint(0, 3, (1,), generator=generator))):
                x = int(torch.randint(0, 10, (1,), generator=generator))
                y = int(torch.randint(0, 10, (1,), generator=generator))
                t_boxes.append((x, y, x + 5, y + 5))
            d_masks, d_boxes = rect_pair(boxes)
            t_masks, t_boxes_tensor = rect_pair(t_boxes)
            key_d = d_masks if shapes == "masks" else d_boxes
            key_t = t_masks if shapes == "masks" else t_boxes_tensor
            preds.append(
                {
                    shapes: key_d,
                    "scores": torch.rand(len(boxes), generator=generator),
                    "labels": torch.randint(0, 2, (len(boxes),), generator=generator),
                }
            )
            tgts.append(
                {
                    shapes: key_t,
                    "labels": torch.randint(0, 2, (len(t_boxes),), generator=generator),
                    "difficult": torch.rand(len(t_boxes), generator=generator) < 0.3,
                }
            )
        return preds, tgts

    @pytest.mark.parametrize("shapes", ["boxes", "masks"])
    def test_sweep_agrees_with_per_threshold_calls(self, shapes):
        """The equality the optimisation rests on, asserted exactly.

        ``sweep_average_precision`` overlaps once per class because the
        best-matching shape does not depend on the threshold. If that reasoning
        were wrong, these two would differ.
        """
        thresholds = (0.5, 0.6, 0.75, 0.9)
        for seed in range(25):
            preds, tgts = self._random_split(seed, shapes=shapes)
            swept = sweep_average_precision(
                preds, tgts, num_classes=2, iou_thresholds=thresholds, shapes=shapes
            )
            for class_id in range(2):
                for threshold in thresholds:
                    naive = average_precision(
                        preds,
                        tgts,
                        class_id=class_id,
                        iou_threshold=threshold,
                        shapes=shapes,
                    )
                    assert swept[class_id][threshold] == naive, (
                        f"seed {seed} class {class_id} at {threshold}"
                    )


class TestTheGeometryIsListedNotGuessed:
    def test_masks_scored_as_boxes_raises_rather_than_reporting_zero(self):
        """The silent failure the adapter table exists to prevent."""
        masks, _ = rect_pair([(0, 0, 4, 4)])
        preds = [
            {"masks": masks, "scores": torch.ones(1), "labels": torch.zeros(1, dtype=torch.int64)}
        ]
        tgts = [{"masks": masks, "labels": torch.zeros(1, dtype=torch.int64)}]
        with pytest.raises(ValueError, match="silently report 0.0"):
            average_precision(preds, tgts, class_id=0, shapes="boxes")

    def test_boxes_scored_as_masks_raises_too(self):
        _, boxes = rect_pair([(0, 0, 4, 4)])
        preds = [
            {"boxes": boxes, "scores": torch.ones(1), "labels": torch.zeros(1, dtype=torch.int64)}
        ]
        tgts = [{"boxes": boxes, "labels": torch.zeros(1, dtype=torch.int64)}]
        with pytest.raises(ValueError, match="silently report 0.0"):
            average_precision(preds, tgts, class_id=0, shapes="masks")

    def test_an_annotation_carrying_both_is_accepted(self):
        """VOCInstanceDataset returns masks *and* derived boxes, so this is the
        ordinary case and must not trip the guard."""
        masks, boxes = rect_pair([(0, 0, 4, 4)])
        labels = torch.zeros(1, dtype=torch.int64)
        preds = [{"masks": masks, "boxes": boxes, "scores": torch.ones(1), "labels": labels}]
        tgts = [{"masks": masks, "boxes": boxes, "labels": labels}]
        assert average_precision(preds, tgts, class_id=0, shapes="masks") == 1.0
        assert average_precision(preds, tgts, class_id=0, shapes="boxes") == 1.0

    def test_an_unknown_geometry_is_refused_by_name(self):
        with pytest.raises(ValueError, match="Unknown shapes"):
            average_precision([], [], class_id=0, shapes="polygons")

    def test_the_table_lists_exactly_the_two_supported_geometries(self):
        assert sorted(SHAPE_KINDS) == ["boxes", "masks"]


class TestConventionsInheritedFromDetection:
    def test_perfect_predictions_calibrate_at_one(self):
        """Without this, no lower number is attributable to a probe."""
        masks, _ = rect_pair([(0, 0, 6, 6), (8, 8, 14, 14)])
        labels = torch.tensor([0, 1], dtype=torch.int64)
        preds = [{"masks": masks, "labels": labels, "scores": torch.ones(2)}]
        tgts = [{"masks": masks, "labels": labels}]
        result = instance_metrics(preds, tgts, num_classes=2)
        assert result["mask_map_50"] == 1.0
        assert result["mask_map_50_95"] == 1.0
        assert result["classes_scored"] == 2.0

    def test_a_difficult_instance_is_ignored_not_counted_against(self):
        """Ignoring scores 1.0; dropping would make a correct detection an FP."""
        masks, _ = rect_pair([(0, 0, 6, 6), (8, 8, 14, 14)])
        labels = torch.zeros(2, dtype=torch.int64)
        preds = [{"masks": masks, "labels": labels, "scores": torch.tensor([0.9, 0.8])}]
        tgts = [
            {
                "masks": masks,
                "labels": labels,
                "difficult": torch.tensor([False, True]),
            }
        ]
        assert mask_average_precision(preds, tgts, class_id=0) == 1.0

    def test_a_class_absent_from_the_split_is_none_not_zero(self):
        masks, _ = rect_pair([(0, 0, 6, 6)])
        labels = torch.zeros(1, dtype=torch.int64)
        preds = [{"masks": masks, "labels": labels, "scores": torch.ones(1)}]
        tgts = [{"masks": masks, "labels": labels}]
        assert mask_average_precision(preds, tgts, class_id=5) is None
        # ...and it is excluded from the mean rather than dragging it to zero.
        result = instance_metrics(preds, tgts, num_classes=6)
        assert result["mask_map_50"] == 1.0
        assert result["classes_scored"] == 1.0

    def test_instances_present_and_nothing_detected_is_zero(self):
        masks, _ = rect_pair([(0, 0, 6, 6)])
        empty = torch.zeros((0, SIZE, SIZE), dtype=torch.bool)
        preds = [
            {"masks": empty, "labels": torch.zeros(0, dtype=torch.int64), "scores": torch.zeros(0)}
        ]
        tgts = [{"masks": masks, "labels": torch.zeros(1, dtype=torch.int64)}]
        assert mask_average_precision(preds, tgts, class_id=0) == 0.0

    def test_a_duplicate_detection_is_a_false_positive(self):
        """With a second, undetected instance so the answer is not 1.0 either way.

        A single target plus a duplicate scores 1.0 whether the duplicate is
        counted as a false positive or (wrongly) as a true one, because
        interpolation flattens the curve. Adding an instance that *is* found
        later makes the middle of the curve matter:

        ranked TP, FP, TP over two positives gives recall 0.5, 0.5, 1.0 and
        precision 1.0, 0.5, 0.667; interpolated that is
        ``0.5 * 1.0 + 0.5 * 0.667 = 0.8333``. Scoring the duplicate as a true
        positive would push recall past 1.0 and the area above it.
        """
        masks, _ = rect_pair([(0, 0, 6, 6), (0, 0, 6, 6), (8, 8, 14, 14)])
        preds = [
            {
                "masks": masks,
                "labels": torch.zeros(3, dtype=torch.int64),
                "scores": torch.tensor([0.9, 0.8, 0.7]),
            }
        ]
        targets, _ = rect_pair([(0, 0, 6, 6), (8, 8, 14, 14)])
        tgts = [{"masks": targets, "labels": torch.zeros(2, dtype=torch.int64)}]
        assert mask_average_precision(preds, tgts, class_id=0) == pytest.approx(
            0.5 * 1.0 + 0.5 * (2 / 3)
        )

    def test_mismatched_row_counts_are_refused_and_name_the_geometry(self):
        masks, _ = rect_pair([(0, 0, 6, 6), (8, 8, 14, 14)])
        preds = [
            {
                "masks": masks,
                "labels": torch.zeros(1, dtype=torch.int64),
                "scores": torch.ones(1),
            }
        ]
        tgts = [{"masks": masks, "labels": torch.zeros(2, dtype=torch.int64)}]
        with pytest.raises(ValueError, match="masks"):
            mask_average_precision(preds, tgts, class_id=0)

    def test_unequal_split_lengths_are_refused(self):
        masks, _ = rect_pair([(0, 0, 6, 6)])
        entry = {
            "masks": masks,
            "labels": torch.zeros(1, dtype=torch.int64),
            "scores": torch.ones(1),
        }
        with pytest.raises(ValueError, match="paired by image index"):
            mask_average_precision([entry, entry], [entry], class_id=0)

    def test_a_sweep_without_one_half_is_refused(self):
        masks, _ = rect_pair([(0, 0, 6, 6)])
        labels = torch.zeros(1, dtype=torch.int64)
        preds = [{"masks": masks, "labels": labels, "scores": torch.ones(1)}]
        tgts = [{"masks": masks, "labels": labels}]
        with pytest.raises(ValueError, match="does not include 0.5"):
            instance_metrics(preds, tgts, num_classes=1, iou_thresholds=(0.75, 0.9))


class TestMasksFromInstanceMap:
    def test_background_gets_no_plane(self):
        indices = torch.zeros((4, 4), dtype=torch.int64)
        indices[0, 0] = 1
        indices[3, 3] = 2
        masks = masks_from_instance_map(indices)
        assert masks.shape == (2, 4, 4)
        assert masks[0, 0, 0] and masks[1, 3, 3]
        assert int(masks.long().sum()) == 2

    def test_an_empty_map_gives_no_masks(self):
        masks = masks_from_instance_map(torch.zeros((5, 5), dtype=torch.int64))
        assert masks.shape == (0, 5, 5)

    def test_order_follows_ascending_index(self):
        """So it matches VOCInstanceDataset.target's order and the two compare."""
        indices = torch.zeros((4, 4), dtype=torch.int64)
        indices[0, :] = 3
        indices[1, :] = 1
        masks = masks_from_instance_map(indices)
        assert masks[0][1, 0] and not masks[0][0, 0]

    def test_a_non_2d_map_is_refused(self):
        with pytest.raises(ValueError, match=r"must be \(H, W\)"):
            masks_from_instance_map(torch.zeros((1, 4, 4), dtype=torch.int64))
