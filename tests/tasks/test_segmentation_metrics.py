"""Binary segmentation metrics.

Unlike the depth and normal metrics next door, these do not reproduce a
published implementation — probe3d has no binary segmentation task — so the
tests here pin the definitions themselves against hand-computed values, and pin
the two conventions that are easy to get quietly wrong: that 0 is a *label*
rather than a hole, and that each image contributes one number.
"""

import pytest
import torch

from visbench.metrics.dense import SEGMENTATION_THRESHOLD, binary_iou


def mask(rows, size=4):
    """A ``(1, 1, size, size)`` mask whose first ``rows`` rows are foreground."""
    out = torch.zeros(1, 1, size, size)
    out[:, :, :rows] = 1.0
    return out


class TestBinaryIoU:
    def test_a_perfect_prediction_scores_one(self):
        target = mask(2)
        assert binary_iou(target.clone(), target) == {"iou": 1.0, "f1": 1.0, "pixel_acc": 1.0}

    def test_a_hand_computed_overlap(self):
        """Predict 3 rows where 2 are foreground: TP 8, FP 4, FN 0 of 16 pixels.

        IoU 8/12, Dice 16/20, accuracy 12/16 — all three by hand, because a
        metric that agrees with itself proves nothing.
        """
        metrics = binary_iou(mask(3), mask(2))
        assert metrics["iou"] == pytest.approx(8 / 12)
        assert metrics["f1"] == pytest.approx(16 / 20)
        assert metrics["pixel_acc"] == pytest.approx(12 / 16)

    def test_the_complement_scores_zero(self):
        assert binary_iou(1.0 - mask(2), mask(2))["iou"] == 0.0

    def test_it_thresholds_at_a_half(self):
        target = mask(2)
        assert binary_iou(target * 0.51, target)["iou"] == 1.0
        assert binary_iou(target * 0.49, target)["iou"] == 0.0
        assert SEGMENTATION_THRESHOLD == 0.5

    def test_accuracy_flatters_a_probe_that_iou_does_not(self):
        """The reason all three are reported. A probe predicting background
        everywhere on a frame that is a quarter object gets 75% accuracy and
        the zero it deserves on IoU."""
        metrics = binary_iou(torch.zeros(1, 1, 4, 4), mask(1))
        assert metrics["pixel_acc"] == pytest.approx(0.75)
        assert metrics["iou"] == 0.0

    def test_channelless_maps_are_accepted(self):
        target = mask(2)
        assert binary_iou(target.squeeze(1), target.squeeze(1))["iou"] == 1.0

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="must match"):
            binary_iou(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 8, 8))

    def test_a_multi_channel_prediction_raises(self):
        """Named for masks, not for depth — the message has to say so, or it
        sends the reader looking at the wrong probe."""
        with pytest.raises(ValueError, match="one mask channel"):
            binary_iou(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4))


class TestEmptyTargets:
    def test_an_empty_target_correctly_predicted_scores_one(self):
        """Nothing was there and nothing was claimed. 0/0 would be the
        alternative, and averaging a NaN takes the whole split with it."""
        empty = torch.zeros(1, 1, 4, 4)
        assert binary_iou(empty.clone(), empty) == {"iou": 1.0, "f1": 1.0, "pixel_acc": 1.0}

    def test_an_empty_target_wrongly_predicted_scores_zero(self):
        assert binary_iou(torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))["iou"] == 0.0


class TestIgnoredPixels:
    """A negative target is unlabelled — the one place this module's validity
    convention differs from depth's, because here 0 is a real label."""

    def test_ignored_pixels_do_not_count_against_the_prediction(self):
        target = mask(2)
        target[:, :, 2:] = -1.0
        assert binary_iou(torch.ones(1, 1, 4, 4), target)["iou"] == 1.0

    def test_ignoring_changes_accuracy_too(self):
        target = mask(2)
        target[:, :, 3:] = -1.0
        # 12 labelled pixels, all correct.
        assert binary_iou(mask(2), target)["pixel_acc"] == 1.0

    def test_a_fully_unlabelled_image_contributes_nothing(self):
        """Zero across the board, matching how depth treats an image with no
        valid pixels — not the 1.0 that an empty union would otherwise give."""
        target = torch.full((1, 1, 4, 4), -1.0)
        assert binary_iou(torch.ones(1, 1, 4, 4), target) == {
            "iou": 0.0,
            "f1": 0.0,
            "pixel_acc": 0.0,
        }


class TestPerImageAveraging:
    def test_images_are_weighted_equally(self):
        """A big object and a small one count the same. Pooling every pixel of
        the split instead would let object size reweight the dataset."""
        target = torch.cat([mask(1), mask(3)])
        pred = torch.cat([mask(1), mask(2)])  # perfect, then 2/3 of the object

        metrics = binary_iou(pred, target)
        assert metrics["iou"] == pytest.approx((1.0 + 8 / 12) / 2)

    def test_a_batch_is_the_mean_of_its_images(self):
        target = torch.cat([mask(1), mask(3)])
        pred = torch.cat([mask(2), mask(2)])

        singles = [binary_iou(pred[i : i + 1], target[i : i + 1])["iou"] for i in range(2)]
        assert binary_iou(pred, target)["iou"] == pytest.approx(sum(singles) / 2)
