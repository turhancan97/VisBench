"""BSDS500's boundary protocol — the metric that matches rather than compares.

Every other dense metric here puts a prediction and a target side by side and
reduces per pixel. This one asks a different question: *did a person also mark a
boundary near here*, which needs a correspondence between two sets of pixels and
has no per-pixel form at all.

Three things carry the risk, and each has its own group below.

**The matching must be exact.** Only cardinality reaches the score, so a greedy
matcher looks defensible — and it is not even maximum-cardinality, let alone
minimum-cost. The measured cost of the cheap substitute is in
`correspond_pixels`' docstring: an arbitrary maximum-cardinality matching moved
precision by up to 0.013 on real images, in both directions, on a benchmark
quoted to three decimals.

**Thinning is not cosmetic.** A boundary predicted three pixels thick can match
an annotator's one-pixel curve once; the other two are false positives at every
threshold.

**Precision unions over annotators and recall sums over them.** Getting that
backwards is silent — it still produces a number between 0 and 1.

The external check lives in `tests/tasks/test_boundary_agreement.py`, which
reads the real dataset and reproduces the published human ODS.
"""

import numpy as np
import pytest

from visbench.metrics.boundary import (
    boundary_metrics,
    correspond_pixels,
    image_counts,
    thin_boundaries,
)


def line(shape=(24, 24), row=12, cols=(2, 22), thickness=1):
    """A horizontal boundary, optionally thick."""
    canvas = np.zeros(shape, dtype=bool)
    canvas[row : row + thickness, cols[0] : cols[1]] = True
    return canvas


# -- thinning -----------------------------------------------------------------


def test_thinning_reduces_a_thick_line_to_one_pixel():
    thick = line(thickness=5)
    thinned = thin_boundaries(thick)
    rows = sorted({int(y) for y, _ in np.argwhere(thinned)})
    assert len(rows) == 1
    assert thinned.sum() < thick.sum()


def test_thinning_leaves_an_already_thin_line_alone():
    """Idempotent on the shape the benchmark wants, or it would erode real curves."""
    thin = line(thickness=1)
    assert np.array_equal(thin_boundaries(thin), thin)


def test_thinning_preserves_connectivity():
    """A thinned curve must stay one curve; a skeleton that breaks it loses recall."""
    from scipy.ndimage import label

    diagonal = np.zeros((30, 30), dtype=bool)
    for i in range(2, 28):
        diagonal[i, i] = diagonal[i, i + 1] = diagonal[i + 1, i] = True
    thinned = thin_boundaries(diagonal)
    _, pieces = label(thinned, structure=np.ones((3, 3)))
    assert pieces == 1


def test_thinning_an_empty_map_is_empty():
    assert not thin_boundaries(np.zeros((8, 8), dtype=bool)).any()


# -- correspondence -----------------------------------------------------------


def test_an_identical_map_matches_completely():
    boundary = line()
    hit_pred, hit_truth = correspond_pixels(boundary, boundary, 4.3)
    assert hit_pred.sum() == boundary.sum()
    assert hit_truth.sum() == boundary.sum()


def test_a_shift_inside_the_tolerance_still_matches():
    hit, _ = correspond_pixels(line(row=12), line(row=14), 4.3)
    assert hit.sum() == line().sum()


def test_a_shift_beyond_the_tolerance_matches_nothing():
    """The tolerance is the whole content of the metric; past it there is no credit."""
    hit, _ = correspond_pixels(line(row=12), line(row=19), 4.3)
    assert hit.sum() == 0


def test_no_pixel_is_matched_twice():
    """A thick prediction cannot claim the same annotator pixel repeatedly.

    This is the property that makes precision fall for an unthinned map, and the
    one a naive "is there a truth pixel nearby" test would lose.
    """
    hit_pred, hit_truth = correspond_pixels(line(thickness=3), line(thickness=1), 4.3)
    assert hit_pred.sum() == hit_truth.sum()
    assert hit_truth.sum() <= line(thickness=1).sum()


def test_the_matching_is_maximum_cardinality_not_greedy():
    """The case a nearest-first matcher gets wrong.

    Greedy pairs the closest available and can strand a prediction whose only
    partner was taken; the optimum matches both.
    """
    prediction = np.zeros((9, 9), dtype=bool)
    truth = np.zeros((9, 9), dtype=bool)
    prediction[4, 4] = prediction[4, 6] = True
    truth[4, 5] = truth[4, 7] = True

    hit, _ = correspond_pixels(prediction, truth, 1.5)
    assert hit.sum() == 2


@pytest.mark.parametrize("seed", range(12))
def test_it_agrees_with_a_brute_force_reference(seed):
    """Randomised check against a dense assignment, which is exact by construction.

    The fast path is sparse, drops candidate-less pixels, pads only the smaller
    side and flips orientation to do so. Each is a chance to lose a match, and
    none of them is visible in the output.
    """
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(seed)
    prediction = rng.random((18, 18)) < 0.08
    truth = rng.random((18, 18)) < 0.08
    tolerance = float(rng.choice([1.5, 3.0, 4.3]))

    pred_yx, truth_yx = np.argwhere(prediction), np.argwhere(truth)
    if len(pred_yx) == 0 or len(truth_yx) == 0:
        pytest.skip("degenerate draw")
    gaps = np.linalg.norm(pred_yx[:, None, :] - truth_yx[None, :, :], axis=2)
    forbidden = (len(pred_yx) + len(truth_yx) + 1) * (tolerance + 1) + 1
    cost = np.where(gaps <= tolerance, gaps, forbidden)
    left, right = linear_sum_assignment(cost)
    expected = int((cost[left, right] < forbidden).sum())

    hit_pred, hit_truth = correspond_pixels(prediction, truth, tolerance)
    assert int(hit_pred.sum()) == expected
    assert int(hit_truth.sum()) == expected


def test_an_empty_side_matches_nothing():
    empty = np.zeros((10, 10), dtype=bool)
    drawn = line((10, 10), row=5, cols=(1, 9))
    assert correspond_pixels(empty, drawn, 4.3)[0].sum() == 0
    assert correspond_pixels(drawn, empty, 4.3)[0].sum() == 0


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="Maps must match"):
        correspond_pixels(np.zeros((4, 4), bool), np.zeros((5, 5), bool), 2.0)


# -- counting -----------------------------------------------------------------


def test_precision_unions_over_annotators_and_recall_sums_over_them():
    """The protocol's asymmetry, as counts rather than prose.

    Three annotators draw the *same* line. A prediction of that line matches all
    three, so it is correct once, while the recall denominator is three lines.
    """
    boundary = line((20, 20), row=10, cols=(2, 18))
    annotations = np.stack([boundary, boundary, boundary])

    matched_pred, predicted, matched_truth, truth_total = image_counts(
        boundary.astype(float), annotations, thresholds=[0.5], thin=False
    )[0]

    assert predicted == boundary.sum()
    assert matched_pred == boundary.sum()  # union: counted once
    assert truth_total == 3 * boundary.sum()  # sum: three annotators
    assert matched_truth == 3 * boundary.sum()


def test_a_prediction_matching_one_annotator_of_three_has_full_precision():
    """Anyone drawing it makes it right; the other two only cost recall."""
    drawn = line((20, 20), row=10, cols=(2, 18))
    elsewhere = line((20, 20), row=18, cols=(2, 18))
    annotations = np.stack([drawn, elsewhere, elsewhere])

    matched_pred, predicted, matched_truth, truth_total = image_counts(
        drawn.astype(float), annotations, thresholds=[0.5], thin=False
    )[0]
    assert matched_pred == predicted  # precision 1.0
    assert matched_truth < truth_total  # recall well below it


def test_thinning_changes_precision():
    """The reason `thin` exists: an unthinned map is punished for its own width."""
    truth = line((24, 24), row=12, cols=(2, 22))
    thick = line((24, 24), row=11, cols=(2, 22), thickness=3).astype(float)

    thinned = image_counts(thick, truth[None], thresholds=[0.5], thin=True)[0]
    raw = image_counts(thick, truth[None], thresholds=[0.5], thin=False)[0]
    assert thinned[0] / thinned[1] > raw[0] / raw[1]


def test_thinning_is_on_by_default():
    """Asserted by *omitting* the argument, which is the only way to test a default.

    The obvious version of this test passes `thin=True` and `thin=False`
    explicitly and names the default in its title -- it then passes unchanged
    when the default is flipped, which is how this gap was found: flipping it
    was one of five mutations and the only one nothing caught.
    """
    truth = line((24, 24), row=12, cols=(2, 22))
    thick = line((24, 24), row=11, cols=(2, 22), thickness=3).astype(float)

    default = image_counts(thick, truth[None], thresholds=[0.5])
    explicit = image_counts(thick, truth[None], thresholds=[0.5], thin=True)
    assert np.array_equal(default, explicit)


def test_counts_have_one_row_per_threshold():
    boundary = line((16, 16), row=8, cols=(1, 15))
    assert image_counts(boundary.astype(float), boundary[None], thresholds=9).shape == (9, 4)


def test_a_prediction_that_does_not_match_the_annotations_is_refused():
    with pytest.raises(ValueError, match="does not match annotations"):
        image_counts(np.zeros((8, 8)), np.zeros((2, 9, 9), dtype=bool))


# -- aggregation --------------------------------------------------------------


def _perfect(images=3, thresholds=5):
    """Counts for a detector that is exactly right at one threshold."""
    counts = np.zeros((images, thresholds, 4), dtype=np.int64)
    counts[:, :, 1] = 100
    counts[:, :, 3] = 100
    counts[:, 2, 0] = 100
    counts[:, 2, 2] = 100
    return counts


def test_a_perfect_detector_scores_one():
    result = boundary_metrics(_perfect(), thresholds=5)
    assert result["ods"] == pytest.approx(1.0)
    assert result["ois"] == pytest.approx(1.0)


def test_ods_reports_the_threshold_it_chose():
    """A score without its operating point cannot be reproduced or compared."""
    levels = np.linspace(0.1, 0.9, 5)
    result = boundary_metrics(_perfect(thresholds=5), thresholds=levels)
    assert result["ods_threshold"] == pytest.approx(levels[2])


def test_ois_is_at_least_ods():
    """Per-image thresholds can only help: ODS is OIS restricted to one choice."""
    rng = np.random.default_rng(0)
    counts = np.zeros((6, 7, 4), dtype=np.int64)
    counts[:, :, 1] = rng.integers(50, 150, (6, 7))
    counts[:, :, 3] = rng.integers(50, 150, (6, 7))
    counts[:, :, 0] = rng.integers(0, 50, (6, 7))
    counts[:, :, 2] = rng.integers(0, 50, (6, 7))
    result = boundary_metrics(counts, thresholds=7)
    assert result["ois"] >= result["ods"] - 1e-12


def test_ois_is_not_the_mean_of_per_image_bests():
    """It aggregates counts and divides once, so a sparse image cannot outweigh a dense one."""
    counts = np.array(
        [
            [[10, 10, 10, 10], [0, 1000, 0, 10]],
            [[0, 1000, 0, 1000], [900, 1000, 900, 1000]],
        ],
        dtype=np.int64,
    )
    result = boundary_metrics(counts, thresholds=2)
    assert result["ois"] != pytest.approx((1.0 + 0.9) / 2)


def test_the_sweep_that_produced_the_counts_must_summarise_them():
    with pytest.raises(ValueError, match="thresholds"):
        boundary_metrics(_perfect(thresholds=5), thresholds=9)


def test_an_empty_dataset_is_refused():
    with pytest.raises(ValueError, match="empty"):
        boundary_metrics(np.zeros((0, 5, 4), dtype=np.int64), thresholds=5)


def test_zero_predictions_do_not_divide_by_zero():
    """A threshold above every value predicts nothing, which is normal at the top."""
    counts = np.zeros((2, 3, 4), dtype=np.int64)
    counts[:, :, 3] = 50
    result = boundary_metrics(counts, thresholds=3)
    assert result["ods"] == 0.0
    assert np.isfinite(result["ap"])
