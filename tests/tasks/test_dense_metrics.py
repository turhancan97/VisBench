"""Depth metrics, against hand-computed values.

These follow probe3d's ``evaluate_depth`` (MIT; see NOTICE). Small differences
in masking and averaging convention move published depth numbers noticeably, so
each convention gets its own test rather than being implied by an end-to-end
score that would still look plausible if it were wrong.
"""

import pytest
import torch

from visbench.metrics.dense import NYU_CROP, depth_metrics, match_scale_and_shift


def constant(value: float, size: int = 4, batch: int = 1) -> torch.Tensor:
    return torch.full((batch, size, size), float(value))


# -- the easy cases anchor everything else ------------------------------------


def test_perfect_prediction_scores_perfectly():
    target = torch.rand(2, 8, 8) + 0.5
    metrics = depth_metrics(target.clone(), target)

    assert metrics["d1"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["abs_rel"] == pytest.approx(0.0)


def test_accepts_both_map_shapes():
    target = torch.rand(2, 8, 8) + 0.5
    flat = depth_metrics(target.clone(), target)
    channelled = depth_metrics(target.clone().unsqueeze(1), target.unsqueeze(1))
    assert flat == channelled


def test_rmse_is_the_root_mean_square_error():
    target = constant(4.0)
    pred = constant(5.0)
    assert depth_metrics(pred, target)["rmse"] == pytest.approx(1.0)


def test_abs_rel_is_relative_to_the_target():
    """|4 - 5| / 4 = 0.25."""
    assert depth_metrics(constant(5.0), constant(4.0))["abs_rel"] == pytest.approx(0.25)


# -- the delta thresholds -----------------------------------------------------


class TestThresholds:
    """delta_k is the fraction of pixels within a factor of 1.25**k, worst direction."""

    def test_ratio_just_inside_counts(self):
        metrics = depth_metrics(constant(1.24), constant(1.0))
        assert metrics["d1"] == pytest.approx(1.0)

    def test_ratio_just_outside_does_not(self):
        metrics = depth_metrics(constant(1.26), constant(1.0))
        assert metrics["d1"] == pytest.approx(0.0)
        assert metrics["d2"] == pytest.approx(1.0), "still inside 1.25 squared"

    def test_the_bound_is_strict(self):
        """Exactly 1.25 is out. probe3d uses `<`, and a boundary pixel counted
        differently shifts d1 on any dataset with quantised depth."""
        assert depth_metrics(constant(1.25), constant(1.0))["d1"] == pytest.approx(0.0)

    def test_under_prediction_is_penalised_the_same_as_over(self):
        """max(gt/pr, pr/gt): being half as deep is as wrong as twice as deep."""
        over = depth_metrics(constant(2.0), constant(1.0))
        under = depth_metrics(constant(1.0), constant(2.0))
        assert over["d1"] == under["d1"] == pytest.approx(0.0)
        assert over["d3"] == under["d3"] == pytest.approx(0.0)

    def test_fraction_is_over_pixels(self):
        target = torch.ones(1, 1, 4)
        pred = torch.tensor([[[1.0, 1.0, 1.0, 10.0]]])
        assert depth_metrics(pred, target)["d1"] == pytest.approx(0.75)


# -- masking ------------------------------------------------------------------


class TestValidPixels:
    """A pixel is valid where target > 0. Sensor depth maps are full of holes."""

    def test_invalid_pixels_are_excluded(self):
        target = torch.tensor([[[2.0, 0.0]]])
        exact = torch.tensor([[[2.0, 2.0]]])
        wild = torch.tensor([[[2.0, 999.0]]])
        assert depth_metrics(exact, target) == depth_metrics(wild, target)

    def test_a_fully_invalid_image_does_not_divide_by_zero(self):
        metrics = depth_metrics(torch.ones(1, 4, 4), torch.zeros(1, 4, 4))
        assert all(value == value for value in metrics.values()), "NaN in metrics"

    def test_negative_targets_are_invalid_too(self):
        target = torch.tensor([[[2.0, -1.0]]])
        assert depth_metrics(torch.tensor([[[2.0, 500.0]]]), target)["d1"] == pytest.approx(1.0)


class TestPerImageAveraging:
    """Each image contributes one number; images are weighted equally.

    Pooling every pixel of the split instead would weight images by how much
    valid depth they happen to contain, letting uneven hole coverage silently
    reweight the dataset.
    """

    def test_images_weigh_equally_despite_different_valid_counts(self):
        # Image 0: one valid pixel, perfect. Image 1: three valid, all wrong.
        target = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 0.0]]],
        )
        pred = torch.tensor(
            [[[1.0, 9.0, 9.0, 9.0]], [[9.0, 9.0, 9.0, 9.0]]],
        )
        # Per image: 1.0 and 0.0 -> mean 0.5. Pixel-pooled would give 1/4.
        assert depth_metrics(pred, target)["d1"] == pytest.approx(0.5)


# -- scale-invariant scoring --------------------------------------------------


class TestScaleInvariance:
    def test_a_scaled_prediction_is_recovered(self):
        target = torch.rand(2, 8, 8) + 1.0
        scaled = target * 3.7

        assert depth_metrics(scaled, target)["d1"] < 0.5
        assert depth_metrics(scaled, target, scale_invariant=True)["d1"] == pytest.approx(1.0)

    def test_a_shifted_prediction_is_recovered(self):
        target = torch.rand(2, 8, 8) + 1.0
        shifted = target + 2.5
        assert depth_metrics(shifted, target, scale_invariant=True)["rmse"] == pytest.approx(
            0.0, abs=1e-4
        )

    def test_it_does_not_rescue_a_wrong_arrangement(self):
        """The point of the metric: no affine fit saves a prediction whose
        relative depths are wrong."""
        target = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        scrambled = torch.tensor([[[3.0, 1.0, 4.0, 2.0]]])
        assert depth_metrics(scrambled, target, scale_invariant=True)["d1"] < 0.6

    def test_an_inverted_prediction_scores_perfectly(self):
        """Documenting a real property of the reference protocol, not endorsing it.

        The fit is unconstrained affine, so a negative scale is available and a
        depth map with near and far swapped is an exact solution. probe3d and
        MiDaS both allow this; VisBench keeps parity rather than adding a
        constraint that would make its numbers incomparable with theirs. It is
        a reason to prefer the default metric scoring for anything load-bearing.
        """
        target = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        inverted = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
        assert depth_metrics(inverted, target, scale_invariant=True)["d1"] == pytest.approx(1.0)

    def test_alignment_uses_only_valid_pixels(self):
        target = torch.tensor([[[2.0, 4.0, 0.0]]])
        pred = torch.tensor([[[1.0, 2.0, 1000.0]]])
        aligned = match_scale_and_shift(pred, target)
        assert aligned[0, 0, 0] == pytest.approx(2.0, abs=1e-4)
        assert aligned[0, 0, 1] == pytest.approx(4.0, abs=1e-4)

    def test_a_degenerate_image_is_left_alone(self):
        """A constant prediction gives a singular system; dividing by that
        determinant would produce inf rather than an honest bad score."""
        pred = torch.ones(1, 4, 4)
        target = torch.rand(1, 4, 4) + 1.0
        aligned = match_scale_and_shift(pred, target)
        assert torch.isfinite(aligned).all()
        assert torch.allclose(aligned, pred)

    def test_no_valid_target_is_left_alone(self):
        aligned = match_scale_and_shift(torch.rand(1, 4, 4), torch.zeros(1, 4, 4))
        assert torch.isfinite(aligned).all()


# -- the NYU crop -------------------------------------------------------------


class TestNyuCrop:
    def test_it_changes_the_score(self):
        """Offered because a number computed with it is not comparable to one
        computed without it."""
        target = torch.ones(1, 480, 640)
        pred = torch.ones(1, 480, 640)
        pred[:, :40, :] = 99.0  # wrong only in the region the crop discards

        assert depth_metrics(pred, target)["d1"] < 1.0
        assert depth_metrics(pred, target, nyu_crop=True)["d1"] == pytest.approx(1.0)

    def test_the_crop_bounds_are_probe3ds(self):
        assert NYU_CROP == (45, 471, 41, 601)

    def test_a_resized_map_is_refused(self):
        """The crop is defined in raw NYUv2 pixels; applying it to a 224 map
        would cut out a different region than every published number used."""
        with pytest.raises(ValueError, match="480x640"):
            depth_metrics(torch.ones(1, 224, 224), torch.ones(1, 224, 224), nyu_crop=True)


# -- input validation ---------------------------------------------------------


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="must match"):
        depth_metrics(torch.ones(1, 4, 4), torch.ones(1, 8, 8))


def test_multi_channel_prediction_is_refused():
    with pytest.raises(ValueError, match="one depth channel"):
        depth_metrics(torch.ones(1, 3, 4, 4), torch.ones(1, 3, 4, 4))


def test_unbatched_map_is_refused():
    with pytest.raises(ValueError, match=r"\(B, H, W\)"):
        depth_metrics(torch.ones(4, 4), torch.ones(4, 4))


def test_metrics_are_plain_floats():
    """BaseTask.evaluate returns a flat dict; a tensor here would break the
    JSON record downstream."""
    metrics = depth_metrics(torch.ones(1, 4, 4), torch.ones(1, 4, 4))
    assert all(isinstance(value, float) for value in metrics.values())


def test_reported_keys():
    assert set(depth_metrics(torch.ones(1, 4, 4), torch.ones(1, 4, 4))) == {
        "d1",
        "d2",
        "d3",
        "rmse",
        "abs_rel",
    }
