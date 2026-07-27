"""Mid-level image similarity metrics — v0.2.

Two-alternative forced choice: given a reference and two candidates, does the
representation prefer the same one a human did? The protocol follows Chen, Marks
& Cheng (arXiv:2411.17474), whose ``evaluate_model_percepture.py`` scores the
choice as binary classification — accuracy, F1, precision and recall — rather
than as a ranking or a correlation. Reported the same way here so the numbers
are comparable to theirs.

The positive class is **"right"** (vote 1), matching the reference's use of the
``right_vote`` column. That only affects precision, recall and F1; accuracy is
symmetric.
"""

import torch

from visbench.types import MetricsDict

__all__ = ["two_afc_metrics"]


def two_afc_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, ties: torch.Tensor | None = None
) -> MetricsDict:
    """Accuracy, F1, precision and recall over a forced binary choice.

    Parameters
    ----------
    predictions, targets:
        ``(T,)`` of 0 (left preferred) and 1 (right preferred).
    ties:
        Optional ``(T,)`` boolean marking triplets where the two similarities
        were exactly equal. Reported as ``tie_rate`` and not otherwise used: a
        forced choice has to resolve them somehow, and knowing how often that
        happened is the difference between a real score and one propped up by a
        coin flip. It should be ~0 for real features and can be large for a
        degenerate backbone that maps everything to one vector.

    Notes
    -----
    A zero denominator yields 0.0 rather than NaN — matching scikit-learn's
    ``zero_division=0`` — so a model that never predicts the positive class
    scores 0 precision instead of poisoning an average with NaN.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Predictions {tuple(predictions.shape)} and targets {tuple(targets.shape)} "
            "must match, one per triplet"
        )
    if predictions.ndim != 1:
        raise ValueError(f"Expected (T,) predictions, got {tuple(predictions.shape)}")
    if len(predictions) == 0:
        raise ValueError("Cannot score an empty set of triplets")

    predicted = predictions.long()
    actual = targets.long()
    for name, values in (("predictions", predicted), ("targets", actual)):
        if not torch.isin(values, torch.tensor([0, 1])).all():
            raise ValueError(f"{name} must be 0 or 1; a forced choice has two options")

    true_positive = ((predicted == 1) & (actual == 1)).sum().item()
    false_positive = ((predicted == 1) & (actual == 0)).sum().item()
    false_negative = ((predicted == 0) & (actual == 1)).sum().item()

    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics: MetricsDict = {
        # An integer ratio, not a float32 mean: `.float().mean()` returns
        # 0.50499999... where the exact answer is 0.505, and a benchmark number
        # that disagrees with the reference implementation in the third decimal
        # invites someone to go looking for a difference that is not there.
        "accuracy": (predicted == actual).sum().item() / len(predicted),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
    if ties is not None:
        metrics["tie_rate"] = ties.float().mean().item()
    return metrics
