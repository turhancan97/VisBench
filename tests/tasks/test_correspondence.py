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
    """Default probe: thresholds in pixels since v0.6.1."""
    return visbench.get_probe("correspondence")


@pytest.fixture
def patch_probe():
    """Explicit patch units, for the within-backbone cases that reason in them."""
    return visbench.get_probe("correspondence", threshold_units="patch")


@pytest.fixture
def pixel_probe():
    """Explicit pixel units, for the cases that reason in raw pixels."""
    return visbench.get_probe("correspondence", threshold_units="pixel")


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


def test_ratio_test_rejects_a_zero_tie():
    """A constant feature region ties at distance 0 and must also be rejected.

    Flooring the runner-up to avoid a zero turns this into `0 < 0.9e-12`, which
    is true — so the single most ambiguous case in an image would be kept.
    """
    assert ratio_test(torch.tensor([[0.0, 0.0]]), 0.9).tolist() == [False]


def test_ratio_test_needs_two_neighbours():
    with pytest.raises(ValueError, match="at least 2 neighbours"):
        ratio_test(torch.tensor([[0.1]]))


# -- metrics -----------------------------------------------------------------


def test_correspondence_recall_hand_computed():
    errors = torch.tensor([0.5, 1.5, 3.0, 20.0])
    metrics = correspondence_recall(errors, thresholds=(1, 2, 5), unit="px")

    assert metrics["recall@1px"] == 0.25
    assert metrics["recall@2px"] == 0.5
    assert metrics["recall@5px"] == 0.75


def test_perfect_matches_score_one():
    metrics = correspondence_recall(torch.zeros(10), thresholds=(1,), unit="px")
    assert metrics["recall@1px"] == 1.0
    assert error_auc(torch.zeros(10), thresholds=(1,), unit="px")["auc@1px"] == pytest.approx(1.0)


def test_auc_is_zero_when_everything_misses():
    assert error_auc(torch.full((5,), 99.0), thresholds=(1, 5), unit="px")["auc@5px"] == 0.0


def test_auc_hand_computed():
    """One match at 2px, threshold 4px.

    The curve is linearly interpolated, per the probe3d / pose-AUC convention:
    (0,0) -> (2,1) -> (4,1). Area = 1 + 2 = 3, normalised by 4 gives 0.75.
    Treating it as a step function instead would give 0.5, which is why this
    is pinned — the choice is invisible in the output and changes every number.
    """
    assert error_auc(torch.tensor([2.0]), thresholds=(4,), unit="px")["auc@4px"] == pytest.approx(
        0.75
    )


def test_auc_convention_matches_the_reference():
    """Spot-check against the published formulation on a second case."""
    # Errors 1 and 3 under a 4px threshold: (0,0) -> (1,0.5) -> (3,1) -> (4,1).
    # Area = 0.25 + 1.5 + 1 = 2.75, normalised by 4 = 0.6875.
    result = error_auc(torch.tensor([1.0, 3.0]), thresholds=(4,), unit="px")["auc@4px"]
    assert result == pytest.approx(0.6875)


def test_auc_separates_distributions_recall_cannot():
    """Two sets with identical recall@5 but different error concentration."""
    tight = torch.tensor([0.1, 0.1, 0.1, 0.1])
    loose = torch.tensor([4.9, 4.9, 4.9, 4.9])

    assert correspondence_recall(tight, (5,), unit="px") == correspondence_recall(
        loose, (5,), unit="px"
    )
    assert (
        error_auc(tight, (5,), unit="px")["auc@5px"] > error_auc(loose, (5,), unit="px")["auc@5px"]
    )


def test_no_matches_scores_zero_not_nan():
    """A backbone that produces no usable match scored 0, which is a result."""
    empty = torch.zeros(0)
    assert correspondence_recall(empty, (1,), unit="px")["recall@1px"] == 0.0
    assert error_auc(empty, (1,), unit="px")["auc@1px"] == 0.0


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_features_straight_off_a_gpu_backbone(pixel_probe):
    """Features from a CUDA backbone must work without a manual .cpu().

    They previously did not: match() returned CUDA indices and the geometry is
    built on CPU, so this raised. It went unnoticed because the cache hands
    back CPU tensors, so every example and test took the working path.
    """
    dense = torch.randn(1, 16, 4, 4, device="cuda")
    pair = (_feature_dict(dense), _feature_dict(dense.clone()))
    identity = {"homography": torch.eye(3, dtype=torch.float64), "size": (64, 64)}

    assert pixel_probe.evaluate([pair], [identity])["recall@1px"] == 1.0
    assert pixel_probe.evaluate_ceiling([pair], [identity])["recall@1px"] == 1.0


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

    Stated in pixels throughout, so these use the explicit pixel probe: the
    cases are "shift by 3px against a 28px grid", which patch units would
    obscure.

    Without this, a low recall@1px reads as "the backbone failed" when it
    actually means "patches are 14px apart".
    """

    def test_identity_warp_has_a_perfect_ceiling(self, pixel_probe):
        """No warp: every target lands exactly on a patch centre."""
        dense = torch.randn(1, 8, 4, 4)
        pair = (_feature_dict(dense), _feature_dict(dense))
        identity = {"homography": torch.eye(3, dtype=torch.float64), "size": (64, 64)}

        ceiling = pixel_probe.evaluate_ceiling([pair], [identity])
        assert ceiling["recall@1px"] == 1.0

    def test_ceiling_is_below_one_for_a_real_warp(self, pixel_probe):
        """A shifted target falls between patch centres and cannot be hit."""
        shift = torch.eye(3, dtype=torch.float64)
        shift[0, 2] = 7.0  # half a patch on a 64px image with a 4x4 grid
        dense = torch.randn(1, 8, 4, 4)
        pair = (_feature_dict(dense), _feature_dict(dense))

        ceiling = pixel_probe.evaluate_ceiling([pair], [{"homography": shift, "size": (64, 64)}])
        assert ceiling["recall@1px"] < 1.0

    def test_score_never_exceeds_its_ceiling(self, pixel_probe):
        """The invariant that makes the ceiling meaningful."""
        torch.manual_seed(0)
        warp = torch.eye(3, dtype=torch.float64)
        warp[0, 2] = 3.0
        dense_0 = torch.randn(1, 16, 8, 8)
        dense_1 = torch.randn(1, 16, 8, 8)
        pair = (_feature_dict(dense_0), _feature_dict(dense_1))
        geometry = [{"homography": warp, "size": (112, 112)}]

        scored = pixel_probe.evaluate([pair], geometry)
        ceiling = pixel_probe.evaluate_ceiling([pair], geometry)
        for threshold in (1, 2, 5, 10):
            key = f"recall@{threshold}px"
            assert scored[key] <= ceiling[key] + 1e-9, f"{key} beat its own ceiling"

    def test_ceiling_covers_only_the_kept_matches(self, pixel_probe):
        """The bug this guards: averaging over different populations.

        The ratio test discards most patches, and it keeps the distinctive
        ones — which are also easier to localise. A ceiling averaged over
        *every* patch is therefore not a bound at all, and a real Imagenette
        run reported 127% of it.
        """
        torch.manual_seed(0)
        # Half the grid is constant, so the ratio test rejects it wholesale.
        dense = torch.randn(1, 16, 8, 8)
        dense[:, :, 4:, :] = 1.0
        pair = (_feature_dict(dense), _feature_dict(dense.clone()))
        geometry = [{"homography": torch.eye(3, dtype=torch.float64), "size": (112, 112)}]

        kept, _ = pixel_probe.match(pair[0], pair[1])
        assert 0 < len(kept) < 64, "fixture must trigger a partial selection"

        scored = pixel_probe.evaluate([pair], geometry)
        ceiling = pixel_probe.evaluate_ceiling([pair], geometry)

        # Both averages must be over the same denominator.
        assert scored["num_matches"] == len(kept)
        for threshold in (1, 2, 5, 10):
            key = f"recall@{threshold}px"
            assert scored[key] <= ceiling[key] + 1e-9

    def test_ceiling_with_no_surviving_matches(self, pixel_probe):
        """Constant features: everything is rejected, so both sides report 0."""
        dense = torch.ones(1, 8, 4, 4)
        pair = (_feature_dict(dense), _feature_dict(dense.clone()))
        geometry = [{"homography": torch.eye(3, dtype=torch.float64), "size": (64, 64)}]

        assert pixel_probe.evaluate([pair], geometry)["num_matches"] == 0
        assert pixel_probe.evaluate_ceiling([pair], geometry)["recall@1px"] == 0.0

    def test_finer_grid_raises_the_ceiling(self, pixel_probe):
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

        assert pixel_probe.evaluate_ceiling([coarse], geometry)["recall@5px"] == 0.0
        # Not 1.0: patches near the right edge shift outside the grid entirely,
        # so their true target has no candidate near it. 26 of 28 columns.
        assert pixel_probe.evaluate_ceiling([fine], geometry)["recall@5px"] == pytest.approx(
            26 / 28
        )


class TestThresholdUnits:
    """Why pixels are the default, and why patch widths inverted a board.

    A patch width is a property of the *backbone*: 14px on DINOv2/14, 16px on
    CLIP ViT-B/16, 32px on ViT-B/32 or a ResNet stage. Scoring in patch widths
    asks each backbone to hit a different physical target and reports the
    answers under one metric name. On the real corpus that put `resnet18` first
    at 0.8927 and `dinov2_vits14` fourth at 0.7834; in pixels the same runs read
    0.0973 and 0.3049, and the order reverses.
    """

    def _pair_and_geometry(self, grid: int, size: int, shift: float):
        torch.manual_seed(0)
        dense = torch.randn(1, 16, grid, grid)
        warp = torch.eye(3, dtype=torch.float64)
        warp[0, 2] = shift
        return (
            [(_feature_dict(dense), _feature_dict(dense.clone()))],
            [{"homography": warp, "size": (size, size)}],
        )

    def test_default_unit_is_pixels(self, probe):
        """Changed in v0.6.1: the default is the unit that can rank backbones."""
        assert probe.threshold_units == "pixel"
        assert probe.thresholds == (1, 2, 5, 10)

    def test_metric_names_carry_the_unit(self, probe, patch_probe):
        pairs, geometry = self._pair_and_geometry(4, 64, 0.0)
        assert "recall@1px" in probe.evaluate(pairs, geometry)
        assert "recall@1p" in patch_probe.evaluate(pairs, geometry)

    def test_patch_units_hide_a_real_difference_in_precision(self, probe, patch_probe):
        """The bug, in one assertion.

        A coarse grid off by half a patch and a fine grid off by half a patch
        score *identically* in patch widths, while the coarse one is wrong by
        more than twice as many pixels. That is the inversion: a ResNet's 32px
        patch against DINOv2's 14px, scored as though the misses were equal.
        """
        coarse, coarse_geom = self._pair_and_geometry(7, 224, 16.0)
        fine, fine_geom = self._pair_and_geometry(16, 224, 7.0)

        in_patches = (
            patch_probe.evaluate(coarse, coarse_geom)["recall@1p"],
            patch_probe.evaluate(fine, fine_geom)["recall@1p"],
        )
        assert in_patches[0] == in_patches[1], "patch widths call these equally good"

        in_pixels = (
            probe.evaluate(coarse, coarse_geom)["recall@10px"],
            probe.evaluate(fine, fine_geom)["recall@10px"],
        )
        assert in_pixels[1] > in_pixels[0], "in pixels the finer grid is plainly better"

    def test_patch_units_remain_resolution_invariant(self, probe, patch_probe):
        """Not a bug, and why `patch` is kept rather than removed.

        The same grid over twice the resolution is the same *relative* sampling,
        and patch widths say so while pixels do not. A real question — just not
        the one a leaderboard row answers, so it is opt-in.
        """
        small, small_geom = self._pair_and_geometry(16, 112, 3.5)
        large, large_geom = self._pair_and_geometry(16, 224, 7.0)

        assert (
            patch_probe.evaluate(small, small_geom)["recall@1p"]
            == patch_probe.evaluate(large, large_geom)["recall@1p"]
        )
        assert (
            probe.evaluate(small, small_geom)["recall@5px"]
            != probe.evaluate(large, large_geom)["recall@5px"]
        )

    def test_the_ceiling_states_the_floor_rather_than_dividing_it_out(self, probe):
        """The honest handling of the quantisation limit.

        A coarse grid genuinely cannot be precise in pixels, and `ceiling_` says
        so — instead of normalising it away, which is what patch units did.
        """
        # 28px is exactly two 14px cells (the fine grid can place it) and half
        # a 56px cell (the worst case for the coarse one). A zero shift would
        # be perfectly representable on both and prove nothing.
        coarse, coarse_geom = self._pair_and_geometry(4, 224, 28.0)
        fine, fine_geom = self._pair_and_geometry(16, 224, 28.0)

        assert (
            probe.evaluate_ceiling(fine, fine_geom)["recall@5px"]
            > probe.evaluate_ceiling(coarse, coarse_geom)["recall@5px"]
        )

    def test_unit_is_recorded(self, probe, patch_probe):
        assert probe.describe()["task_params"]["threshold_units"] == "pixel"
        assert patch_probe.describe()["task_params"]["threshold_units"] == "patch"

    def test_the_two_units_are_never_ranked_together(self, probe, patch_probe):
        """A unit change is a protocol change, and the records already say so.

        `threshold_units` lives in `task_params`, which `comparability_key`
        includes wholesale — so no v0.6.0 patch-unit number can be silently
        ranked against a v0.6.1 pixel one. No special case was needed.
        """
        from visbench.results.leaderboard import comparability_key
        from visbench.results.schema import ResultRecord, utc_timestamp

        def record(task):
            return ResultRecord(
                backbone="b",
                backbone_key="k",
                task="correspondence",
                level="mid_level",
                dataset="d",
                split="val",
                pooling="mean",
                pooling_requested="mean",
                feature_mode="dense_only",
                metrics={},
                timestamp=utc_timestamp(),
                visbench_version="0",
                task_params=task.describe()["task_params"],
            )

        assert comparability_key(record(probe)) != comparability_key(record(patch_probe))


def test_patch_spacing_hand_computed():
    from visbench.tasks.mid_level.correspondence import patch_spacing

    assert patch_spacing((16, 16), (224, 224)) == 14.0
    assert patch_spacing((14, 14), (224, 224)) == 16.0


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


class TestMismatchedGeometry:
    """Pairs and geometry are matched by position, so lengths must agree.

    ``evaluate`` has always checked this explicitly. ``evaluate_ceiling`` never
    did: hand it ten pairs and nine geometries and it scored nine, and reported
    the number as if it covered the split. ``zip(strict=True)`` is what closes
    that, so the two entry points now agree about what is a valid call.
    """

    @pytest.fixture
    def pair_and_geometry(self):
        dense = torch.randn(1, 8, 4, 4)
        return (
            (_feature_dict(dense), _feature_dict(dense)),
            {"homography": torch.eye(3, dtype=torch.float64), "size": (64, 64)},
        )

    def test_evaluate_rejects_extra_pairs(self, probe, pair_and_geometry):
        """Caught by the explicit length check, which names both counts."""
        pair, identity = pair_and_geometry
        with pytest.raises(ValueError, match="2 feature pairs for 1 geometries"):
            probe.evaluate([pair, pair], [identity])

    def test_evaluate_rejects_extra_geometry(self, probe, pair_and_geometry):
        pair, identity = pair_and_geometry
        with pytest.raises(ValueError, match="1 feature pairs for 2 geometries"):
            probe.evaluate([pair], [identity, identity])

    def test_ceiling_rejects_extra_pairs(self, probe, pair_and_geometry):
        """Caught by strict=True; before it, this scored one pair of two."""
        pair, identity = pair_and_geometry
        with pytest.raises(ValueError, match="shorter"):
            probe.evaluate_ceiling([pair, pair], [identity])

    def test_ceiling_rejects_extra_geometry(self, probe, pair_and_geometry):
        pair, identity = pair_and_geometry
        with pytest.raises(ValueError, match="longer"):
            probe.evaluate_ceiling([pair], [identity, identity])

    def test_equal_lengths_still_score(self, probe, pair_and_geometry):
        """The guard must not cost the ordinary path."""
        pair, identity = pair_and_geometry
        assert probe.evaluate([pair, pair], [identity, identity])["recall@1px"] == 1.0
        assert probe.evaluate_ceiling([pair, pair], [identity, identity])["recall@1px"] == 1.0
