"""Multi-class segmentation metrics, against hand-computed values.

Every expected number here is worked out in the docstring rather than taken
from a run, so the tests disagree with the implementation when it is wrong
instead of agreeing with it whatever it does.

The two reductions are the point of this module. Dataset-level mIoU takes one
confusion matrix over every image and divides once; per-image mIoU divides per
image and averages. They are different numbers, and only the first is what a
VOC leaderboard reports.
"""

import pytest
import torch

from visbench.metrics.dense import (
    confusion_matrix,
    metrics_from_confusion,
    semantic_metrics,
)


def labels(rows):
    """A ``(1, H, W)`` long map from nested lists."""
    return torch.tensor([rows], dtype=torch.long)


class TestConfusionMatrix:
    def test_perfect_prediction_is_diagonal(self):
        gt = labels([[0, 1], [2, 2]])
        matrix = confusion_matrix(gt, gt, 3)

        assert torch.equal(matrix.diag(), torch.tensor([1, 1, 2]))
        assert matrix.sum() == 4

    def test_rows_are_ground_truth_columns_are_prediction(self):
        """One pixel of class 2 predicted as class 1 lands at [2, 1]."""
        gt = labels([[2]])
        pred = labels([[1]])

        matrix = confusion_matrix(pred, gt, 3)
        assert matrix[2, 1] == 1
        assert matrix.sum() == 1

    def test_ignored_pixels_are_not_counted(self):
        """-1 is unlabelled; 0 is background and very much counted."""
        gt = labels([[0, -1], [-1, 1]])
        pred = labels([[0, 0], [0, 1]])

        matrix = confusion_matrix(pred, gt, 2)
        assert matrix.sum() == 2, "only the two labelled pixels"
        assert matrix[0, 0] == 1
        assert matrix[1, 1] == 1

    def test_out_of_range_labels_are_dropped_not_folded(self):
        """A stray index must not be wrapped into a real class's score."""
        gt = labels([[0, 7]])
        pred = labels([[0, 0]])

        matrix = confusion_matrix(pred, gt, 3)
        assert matrix.sum() == 1, "the class-7 pixel is discarded, not counted as class 1"

    def test_a_fully_ignored_image_gives_an_empty_matrix(self):
        gt = labels([[-1, -1]])
        matrix = confusion_matrix(labels([[0, 0]]), gt, 3)

        assert matrix.sum() == 0
        assert matrix.shape == (3, 3)

    def test_scores_and_labels_agree(self):
        """(B, C, H, W) logits reduce to the same matrix as their argmax."""
        gt = labels([[0, 1], [2, 1]])
        pred = labels([[0, 1], [1, 1]])
        one_hot = torch.nn.functional.one_hot(pred, 3).permute(0, 3, 1, 2).float()

        assert torch.equal(confusion_matrix(one_hot, gt, 3), confusion_matrix(pred, gt, 3))

    def test_argmax_is_indifferent_to_monotone_transforms(self):
        """Softmax before scoring must not change the answer."""
        gt = labels([[0, 1], [2, 1]])
        scores = torch.randn(1, 3, 2, 2)

        plain = confusion_matrix(scores, gt, 3)
        softmaxed = confusion_matrix(scores.softmax(dim=1), gt, 3)
        assert torch.equal(plain, softmaxed)

    def test_channel_count_must_match_num_classes(self):
        gt = labels([[0, 1]])
        with pytest.raises(ValueError, match="Expected"):
            confusion_matrix(torch.randn(1, 5, 1, 2), gt, 3)

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="Resize the prediction"):
            confusion_matrix(labels([[0, 1, 2]]), labels([[0, 1]]), 3)

    def test_binary_num_classes_is_the_floor(self):
        with pytest.raises(ValueError, match="num_classes must be >= 2"):
            confusion_matrix(labels([[0]]), labels([[0]]), 1)


class TestMetricsFromConfusion:
    def test_perfect_prediction_scores_one(self):
        matrix = torch.diag(torch.tensor([3, 5, 2]))
        metrics = metrics_from_confusion(matrix)

        assert metrics["miou"] == 1.0
        assert metrics["pixel_acc"] == 1.0
        assert metrics["mean_acc"] == 1.0

    def test_hand_computed_partial_credit(self):
        """gt=[0,1,2], pred=[0,1,1]: class 2's pixel is called class 1.

        class 0: tp 1, gt 1, pred 1 -> IoU 1/1 = 1
        class 1: tp 1, gt 1, pred 2 -> IoU 1/(1+2-1) = 0.5
        class 2: tp 0, gt 1, pred 0 -> IoU 0/1 = 0
        mIoU = 1.5/3 = 0.5, pixel accuracy = 2/3
        """
        gt = labels([[0, 1, 2]])
        pred = labels([[0, 1, 1]])

        metrics = metrics_from_confusion(confusion_matrix(pred, gt, 3))
        assert metrics["miou"] == pytest.approx(0.5)
        assert metrics["pixel_acc"] == pytest.approx(2 / 3)

    def test_absent_classes_are_excluded_not_scored_zero(self):
        """A class in neither prediction nor truth says nothing about the probe.

        Only class 0 appears, and it is perfect. Counting the two unused classes
        as 0 would give 1/3; excluding them gives 1.0.
        """
        matrix = torch.zeros(3, 3, dtype=torch.long)
        matrix[0, 0] = 4

        assert metrics_from_confusion(matrix)["miou"] == 1.0

    def test_a_class_predicted_but_absent_from_truth_counts_against(self):
        """Predicting a class nowhere in the ground truth is a false positive.

        class 0: tp 2, gt 2, pred 3 -> IoU 2/3
        class 1: tp 0, gt 0, pred 1 -> IoU 0/1 = 0, and it *is* in the union
        mIoU = (2/3 + 0)/2 = 1/3
        """
        matrix = torch.tensor([[2, 1], [0, 0]], dtype=torch.long)
        assert metrics_from_confusion(matrix)["miou"] == pytest.approx(1 / 3)

    def test_mean_acc_averages_over_labelled_classes_only(self):
        """gt has 4 of class 0 (3 right) and 2 of class 1 (1 right).

        recall_0 = 3/4, recall_1 = 1/2, mean = 0.625, while pixel accuracy is
        4/6 = 0.667. The two differ whenever classes are unbalanced.
        """
        matrix = torch.tensor([[3, 1], [1, 1]], dtype=torch.long)
        metrics = metrics_from_confusion(matrix)

        assert metrics["mean_acc"] == pytest.approx(0.625)
        assert metrics["pixel_acc"] == pytest.approx(4 / 6)

    def test_an_empty_matrix_scores_zero(self):
        metrics = metrics_from_confusion(torch.zeros(3, 3, dtype=torch.long))
        assert metrics == {"miou": 0.0, "pixel_acc": 0.0, "mean_acc": 0.0}

    def test_a_non_square_matrix_is_refused(self):
        with pytest.raises(ValueError, match="square confusion matrix"):
            metrics_from_confusion(torch.zeros(2, 3))


class TestTwoReductionsDisagree:
    """The reason both are reported instead of one.

    Image A is all class 0 and perfect. Image B is one pixel of class 1, got
    wrong. Per image: A scores mIoU 1, B scores 0, mean 0.5. Dataset-level:
    class 0 has tp 4, pred 5, gt 4 -> IoU 4/5; class 1 has tp 0 -> IoU 0; mean
    0.4. Neither is wrong; they answer different questions.
    """

    @pytest.fixture
    def split(self):
        gt = torch.tensor([[[0, 0], [0, 0]], [[1, -1], [-1, -1]]], dtype=torch.long)
        pred = torch.tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=torch.long)
        return pred, gt

    def test_per_image_value(self, split):
        pred, gt = split
        assert semantic_metrics(pred, gt, 2)["miou_per_image"] == pytest.approx(0.5)

    def test_dataset_level_value(self, split):
        pred, gt = split
        assert metrics_from_confusion(confusion_matrix(pred, gt, 2))["miou"] == pytest.approx(0.4)

    def test_they_are_reported_under_different_names(self, split):
        pred, gt = split
        assert "miou_per_image" in semantic_metrics(pred, gt, 2)
        assert "miou" in metrics_from_confusion(confusion_matrix(pred, gt, 2))


class TestSemanticMetricsPerImage:
    def test_each_image_counts_equally_regardless_of_size(self):
        """A 1-pixel image and a 3-pixel one both weigh one image.

        Both are perfect here, so the mean is 1.0 either way — the point is that
        pooling pixels would let the larger frame dominate.
        """
        gt = torch.tensor([[[0, -1, -1]], [[1, 1, 1]]], dtype=torch.long)
        assert semantic_metrics(gt, gt, 2)["miou_per_image"] == pytest.approx(1.0)

    def test_a_fully_ignored_image_contributes_zero(self):
        """Matching how binary_iou treats a frame with no ground truth."""
        gt = torch.tensor([[[0, 0]], [[-1, -1]]], dtype=torch.long)
        pred = torch.tensor([[[0, 0]], [[0, 0]]], dtype=torch.long)

        assert semantic_metrics(pred, gt, 2)["miou_per_image"] == pytest.approx(0.5)
