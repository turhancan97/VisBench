"""Geometric correspondence — the only v0.1 task that uses dense features.

Metrics are checked against hand-computed values, and the task against a case
where the right answer is known exactly: matching an image to an unwarped copy
of itself must recover patch i -> patch i with zero pixel error.
"""

import numpy as np
import pytest
import torch
from PIL import Image

import visbench
from visbench.data.pair_dataset import HomographyPairDataset
from visbench.metrics.correspondence import (
    correspondence_recall,
    error_auc,
    nn_match,
    ratio_test,
)
from visbench.tasks.mid_level.correspondence import patch_centers


@pytest.fixture
def probe():
    return visbench.get_probe("correspondence")


# -- matching primitives -----------------------------------------------------


def test_nn_match_finds_the_identical_vector():
    feats = torch.eye(4)
    distances, indices = nn_match(feats, feats, k=2)

    assert torch.equal(indices[:, 0], torch.arange(4))
    assert torch.allclose(distances[:, 0], torch.zeros(4), atol=1e-6)


def test_nn_match_ignores_magnitude():
    """Features are normalised, so a scaled copy is still the same direction."""
    feats = torch.randn(6, 8)
    _, indices = nn_match(feats, feats * 7.0, k=2)
    assert torch.equal(indices[:, 0], torch.arange(6))


def test_nn_match_rejects_mismatched_dims():
    with pytest.raises(ValueError, match="same backbone"):
        nn_match(torch.rand(4, 8), torch.rand(4, 16))


def test_ratio_test_hand_computed():
    # ratios 0.5, 0.95 against a 0.9 threshold
    distances = torch.tensor([[0.5, 1.0], [0.95, 1.0]])
    assert ratio_test(distances, 0.9).tolist() == [True, False]


def test_ratio_test_rejects_a_tie():
    """Nearest and runner-up equally good means the match says nothing."""
    assert ratio_test(torch.tensor([[0.4, 0.4]]), 0.9).tolist() == [False]


def test_ratio_test_needs_two_neighbours():
    with pytest.raises(ValueError, match="at least 2 neighbours"):
        ratio_test(torch.tensor([[0.1]]))


# -- metrics -----------------------------------------------------------------


def test_correspondence_recall_hand_computed():
    errors = torch.tensor([0.5, 1.5, 3.0, 20.0])
    metrics = correspondence_recall(errors, thresholds=(1, 2, 5))

    assert metrics["recall@1px"] == 0.25
    assert metrics["recall@2px"] == 0.5
    assert metrics["recall@5px"] == 0.75


def test_perfect_matches_score_one():
    metrics = correspondence_recall(torch.zeros(10), thresholds=(1,))
    assert metrics["recall@1px"] == 1.0
    assert error_auc(torch.zeros(10), thresholds=(1,))["auc@1px"] == pytest.approx(1.0)


def test_auc_is_zero_when_everything_misses():
    assert error_auc(torch.full((5,), 99.0), thresholds=(1, 5))["auc@5px"] == 0.0


def test_auc_hand_computed():
    """One match at 2px, threshold 4px.

    The curve is linearly interpolated, per the probe3d / pose-AUC convention:
    (0,0) -> (2,1) -> (4,1). Area = 1 + 2 = 3, normalised by 4 gives 0.75.
    Treating it as a step function instead would give 0.5, which is why this
    is pinned — the choice is invisible in the output and changes every number.
    """
    assert error_auc(torch.tensor([2.0]), thresholds=(4,))["auc@4px"] == pytest.approx(0.75)


def test_auc_convention_matches_the_reference():
    """Spot-check against the published formulation on a second case."""
    # Errors 1 and 3 under a 4px threshold: (0,0) -> (1,0.5) -> (3,1) -> (4,1).
    # Area = 0.25 + 1.5 + 1 = 2.75, normalised by 4 = 0.6875.
    result = error_auc(torch.tensor([1.0, 3.0]), thresholds=(4,))["auc@4px"]
    assert result == pytest.approx(0.6875)


def test_auc_separates_distributions_recall_cannot():
    """Two sets with identical recall@5 but different error concentration."""
    tight = torch.tensor([0.1, 0.1, 0.1, 0.1])
    loose = torch.tensor([4.9, 4.9, 4.9, 4.9])

    assert correspondence_recall(tight, (5,)) == correspondence_recall(loose, (5,))
    assert error_auc(tight, (5,))["auc@5px"] > error_auc(loose, (5,))["auc@5px"]


def test_no_matches_scores_zero_not_nan():
    """A backbone that produces no usable match scored 0, which is a result."""
    empty = torch.zeros(0)
    assert correspondence_recall(empty, (1,))["recall@1px"] == 0.0
    assert error_auc(empty, (1,))["auc@1px"] == 0.0


# -- patch geometry ----------------------------------------------------------


def test_patch_centers_are_centres_not_corners():
    """Using the corner would bias every error by half a patch — 7px at /14."""
    centres = patch_centers((2, 2), size=(100, 100))
    assert centres[0].tolist() == [25.0, 25.0]
    assert centres[-1].tolist() == [75.0, 75.0]


def test_patch_centers_are_row_major():
    """Must match dense.flatten(1), or every match is transposed."""
    centres = patch_centers((2, 3), size=(60, 40))
    # Second entry moves along x (same row), not down a column.
    assert centres[1][1] == centres[0][1]
    assert centres[1][0] > centres[0][0]


# -- the task ----------------------------------------------------------------


def _feature_dict(dense):
    return {"dense": dense, "grid_hw": (dense.shape[-2], dense.shape[-1])}


def test_identical_views_match_exactly(probe):
    """An image against an unwarped copy: patch i must match patch i at 0px."""
    torch.manual_seed(0)
    dense = torch.randn(1, 16, 4, 4)
    pair = (_feature_dict(dense), _feature_dict(dense.clone()))

    identity = {"homography": torch.eye(3, dtype=torch.float64), "size": (64, 64)}
    metrics = probe.evaluate([pair], [identity])

    assert metrics["recall@1px"] == 1.0
    assert metrics["auc@1px"] == pytest.approx(1.0)
    assert metrics["num_matches"] == 16


def test_match_returns_the_diagonal_for_identical_views(probe):
    torch.manual_seed(0)
    dense = torch.randn(1, 16, 4, 4)
    source, target = probe.match(_feature_dict(dense), _feature_dict(dense.clone()))
    assert torch.equal(source, target)


def test_shuffled_features_score_badly(probe):
    """A sanity floor: matching against a permuted grid must not look good."""
    torch.manual_seed(0)
    dense = torch.randn(1, 16, 6, 6)
    permuted = dense.flatten(2)[:, :, torch.randperm(36)].reshape(1, 16, 6, 6)

    identity = {"homography": torch.eye(3, dtype=torch.float64), "size": (96, 96)}
    metrics = probe.evaluate([(_feature_dict(dense), _feature_dict(permuted))], [identity])
    assert metrics["recall@1px"] < 0.2


def test_num_corr_caps_the_matches(probe_factory=None):
    probe = visbench.get_probe("correspondence", num_corr=5)
    torch.manual_seed(0)
    dense = torch.randn(1, 16, 4, 4)
    source, _ = probe.match(_feature_dict(dense), _feature_dict(dense.clone()))
    assert len(source) == 5


def test_errors_pool_across_pairs(probe):
    """A pair with more matches must weigh more than one with fewer."""
    torch.manual_seed(0)
    dense = torch.randn(1, 16, 4, 4)
    identity = {"homography": torch.eye(3, dtype=torch.float64), "size": (64, 64)}
    pair = (_feature_dict(dense), _feature_dict(dense.clone()))

    one = probe.evaluate([pair], [identity])
    two = probe.evaluate([pair, pair], [identity, identity])
    assert two["num_matches"] == 2 * one["num_matches"]


def test_batched_features_are_rejected(probe):
    with pytest.raises(ValueError, match="one at a time"):
        probe.match(_feature_dict(torch.rand(4, 8, 4, 4)), _feature_dict(torch.rand(4, 8, 4, 4)))


def test_missing_geometry_names_what_is_needed(probe):
    dense = torch.randn(1, 8, 4, 4)
    pair = (_feature_dict(dense), _feature_dict(dense))
    with pytest.raises(KeyError, match="homography"):
        probe.evaluate([pair], [{"depth": None, "size": (64, 64)}])


def test_no_geometry_at_all_raises(probe):
    dense = torch.randn(1, 8, 4, 4)
    with pytest.raises(ValueError, match="needs geometry"):
        probe.evaluate([(_feature_dict(dense), _feature_dict(dense))])


def test_length_mismatch_raises(probe):
    dense = torch.randn(1, 8, 4, 4)
    pair = (_feature_dict(dense), _feature_dict(dense))
    with pytest.raises(ValueError, match="feature pairs for"):
        probe.evaluate([pair, pair], [{"homography": torch.eye(3), "size": (64, 64)}])


def test_fit_is_a_noop(probe):
    assert probe.fit(None) is probe


def test_task_reports_dense_only(probe):
    described = probe.describe()
    assert described["feature_mode"] == "dense_only"
    assert described["level"] == "mid_level"
    assert described["zero_shot"] is True
    assert described["task_params"]["ratio_threshold"] == 0.9


def test_invalid_configuration_raises():
    with pytest.raises(ValueError, match="num_corr"):
        visbench.get_probe("correspondence", num_corr=0)
    with pytest.raises(ValueError, match="ratio_threshold"):
        visbench.get_probe("correspondence", ratio_threshold=1.5)


# -- against the synthetic dataset -------------------------------------------


class TestCeiling:
    """Patch quantisation caps what any matcher can score.

    Without this, a low recall@1px reads as "the backbone failed" when it
    actually means "patches are 14px apart".
    """

    def test_identity_warp_has_a_perfect_ceiling(self, probe):
        """No warp: every target lands exactly on a patch centre."""
        dense = torch.randn(1, 8, 4, 4)
        pair = (_feature_dict(dense), _feature_dict(dense))
        identity = {"homography": torch.eye(3, dtype=torch.float64), "size": (64, 64)}

        ceiling = probe.evaluate_ceiling([pair], [identity])
        assert ceiling["recall@1px"] == 1.0

    def test_ceiling_is_below_one_for_a_real_warp(self, probe):
        """A shifted target falls between patch centres and cannot be hit."""
        shift = torch.eye(3, dtype=torch.float64)
        shift[0, 2] = 7.0  # half a patch on a 64px image with a 4x4 grid
        dense = torch.randn(1, 8, 4, 4)
        pair = (_feature_dict(dense), _feature_dict(dense))

        ceiling = probe.evaluate_ceiling([pair], [{"homography": shift, "size": (64, 64)}])
        assert ceiling["recall@1px"] < 1.0

    def test_score_never_exceeds_its_ceiling(self, probe):
        """The invariant that makes the ceiling meaningful."""
        torch.manual_seed(0)
        warp = torch.eye(3, dtype=torch.float64)
        warp[0, 2] = 3.0
        dense_0 = torch.randn(1, 16, 8, 8)
        dense_1 = torch.randn(1, 16, 8, 8)
        pair = (_feature_dict(dense_0), _feature_dict(dense_1))
        geometry = [{"homography": warp, "size": (112, 112)}]

        scored = probe.evaluate([pair], geometry)
        ceiling = probe.evaluate_ceiling([pair], geometry)
        for threshold in (1, 2, 5, 10):
            key = f"recall@{threshold}px"
            assert scored[key] <= ceiling[key] + 1e-9, f"{key} beat its own ceiling"

    def test_finer_grid_raises_the_ceiling(self, probe):
        """More patches, smaller spacing, more of the fine thresholds reachable.

        The shift has to exceed half the fine grid's spacing for this to show
        at all: a translation smaller than that lands nearest the *same* patch
        on either grid, so both ceilings are identical. 4x4 over 112px is 28px
        spacing; 28x28 is 4px.
        """
        warp = torch.eye(3, dtype=torch.float64)
        warp[0, 2] = 10.0
        geometry = [{"homography": warp, "size": (112, 112)}]

        coarse = (_feature_dict(torch.randn(1, 8, 4, 4)),) * 2
        fine = (_feature_dict(torch.randn(1, 8, 28, 28)),) * 2

        assert probe.evaluate_ceiling([coarse], geometry)["recall@5px"] == 0.0
        # Not 1.0: patches near the right edge shift outside the grid entirely,
        # so their true target has no candidate near it. 26 of 28 columns.
        assert probe.evaluate_ceiling([fine], geometry)["recall@5px"] == pytest.approx(26 / 28)


def test_end_to_end_with_a_real_warp(tmp_path, fake_vit, probe):
    """Full path on generated pairs; the fake backbone makes no claim about quality."""
    root = tmp_path / "imgs"
    root.mkdir()
    for i in range(2):
        array = np.random.RandomState(i).randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(array).save(root / f"{i}.png")

    dataset = HomographyPairDataset(root, max_warp=0.1)
    pairs = []
    for index in range(len(dataset)):
        image_0, image_1, _ = dataset[index]
        pairs.append(
            (
                fake_vit.extract_features(fake_vit.preprocess([image_0])),
                fake_vit.extract_features(fake_vit.preprocess([image_1])),
            )
        )

    metrics = probe.evaluate(pairs, dataset.labels())
    assert set(metrics) >= {"recall@1px", "recall@10px", "auc@10px", "num_matches"}
    assert all(isinstance(value, float) for value in metrics.values())
