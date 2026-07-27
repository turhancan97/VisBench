"""Two-alternative-forced-choice metrics.

The reference implementation scores this with scikit-learn, so the strongest
check available is that these agree with it exactly. scikit-learn is already a
VisBench dependency, so this costs nothing and removes any doubt about averaging
conventions or how a zero denominator is handled.

Hand-computed cases are kept alongside, because agreeing with sklearn only
proves the two match — not that either is the quantity intended.
"""

import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from visbench.metrics.similarity import two_afc_metrics


class TestAgainstSklearn:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_predictions_match_exactly(self, seed):
        torch.manual_seed(seed)
        predicted = torch.randint(0, 2, (200,))
        actual = torch.randint(0, 2, (200,))

        mine = two_afc_metrics(predicted, actual)
        assert mine["accuracy"] == pytest.approx(accuracy_score(actual, predicted), abs=1e-12)
        assert mine["f1"] == pytest.approx(f1_score(actual, predicted), abs=1e-12)
        assert mine["precision"] == pytest.approx(precision_score(actual, predicted), abs=1e-12)
        assert mine["recall"] == pytest.approx(recall_score(actual, predicted), abs=1e-12)

    def test_accuracy_is_exact_not_float32(self):
        """101 of 200 correct is 0.505, not 0.50499999.

        `.float().mean()` on a bool tensor gives the latter, and a benchmark
        number that disagrees with the reference in the third decimal sends
        someone hunting for a difference that is not there.
        """
        predicted = torch.zeros(200, dtype=torch.long)
        actual = torch.zeros(200, dtype=torch.long)
        actual[:99] = 1

        assert two_afc_metrics(predicted, actual)["accuracy"] == 0.505


class TestHandComputed:
    def test_perfect_agreement(self):
        values = torch.tensor([0, 1, 1, 0])
        metrics = two_afc_metrics(values, values)

        assert metrics == {"accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}

    def test_total_disagreement(self):
        predicted = torch.tensor([0, 1, 0, 1])
        actual = torch.tensor([1, 0, 1, 0])

        metrics = two_afc_metrics(predicted, actual)
        assert metrics["accuracy"] == 0.0
        assert metrics["f1"] == 0.0

    def test_partial_credit(self):
        """predicted 1 on three, of which two were right; four were truly 1.

        precision = 2/3, recall = 2/4 = 0.5, f1 = 2 * (2/3 * 0.5)/(2/3 + 0.5)
        = 4/7. Accuracy: 5 of 8 correct.
        """
        predicted = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0])
        actual = torch.tensor([1, 1, 0, 1, 1, 0, 0, 0])

        metrics = two_afc_metrics(predicted, actual)
        assert metrics["precision"] == pytest.approx(2 / 3)
        assert metrics["recall"] == pytest.approx(0.5)
        assert metrics["f1"] == pytest.approx(4 / 7)
        assert metrics["accuracy"] == pytest.approx(5 / 8)

    def test_right_is_the_positive_class(self):
        """Matching the reference's use of the right_vote column.

        Predicting 'right' everywhere against all-left truth is 0 precision;
        the mirror image would be 0 too only if the convention were symmetric,
        which precision and recall are not.
        """
        metrics = two_afc_metrics(torch.ones(4, dtype=torch.long), torch.zeros(4, dtype=torch.long))
        assert metrics["precision"] == 0.0
        assert metrics["accuracy"] == 0.0


class TestDegenerate:
    def test_never_predicting_positive_scores_zero_not_nan(self):
        """A NaN here would poison any average taken over backbones."""
        predicted = torch.zeros(10, dtype=torch.long)
        actual = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

        metrics = two_afc_metrics(predicted, actual)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["accuracy"] == 0.8

    def test_no_positives_anywhere_is_still_finite(self):
        zeros = torch.zeros(5, dtype=torch.long)
        metrics = two_afc_metrics(zeros, zeros)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 0.0, "no positive class to score"

    def test_empty_input_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            two_afc_metrics(torch.zeros(0), torch.zeros(0))

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="must match"):
            two_afc_metrics(torch.zeros(3), torch.zeros(4))

    def test_non_binary_values_are_refused(self):
        """A third option means this is not the task it claims to be."""
        with pytest.raises(ValueError, match="must be 0 or 1"):
            two_afc_metrics(torch.tensor([0, 2]), torch.tensor([0, 1]))

    def test_a_two_dimensional_input_is_refused(self):
        with pytest.raises(ValueError, match=r"\(T,\)"):
            two_afc_metrics(torch.zeros(2, 2), torch.zeros(2, 2))


class TestTies:
    def test_tie_rate_is_reported_when_given(self):
        values = torch.tensor([0, 1])
        metrics = two_afc_metrics(values, values, ties=torch.tensor([True, False]))
        assert metrics["tie_rate"] == 0.5

    def test_absent_when_not_given(self):
        values = torch.tensor([0, 1])
        assert "tie_rate" not in two_afc_metrics(values, values)

    def test_ties_do_not_change_the_other_metrics(self):
        """It is a diagnostic, not part of the score."""
        values = torch.tensor([0, 1, 1])
        without = two_afc_metrics(values, values)
        with_ties = two_afc_metrics(values, values, ties=torch.ones(3, dtype=torch.bool))

        assert all(with_ties[name] == without[name] for name in without)
