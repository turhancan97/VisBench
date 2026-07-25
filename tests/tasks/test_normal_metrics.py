"""Surface-normal metrics, against probe3d's ``evaluate_surface_norm``.

Angles are constructed exactly rather than sampled: with a target of
``(0, 0, 1)`` and a prediction of ``(sin t, 0, cos t)``, the angular error is
``t`` by construction, so every assertion below is about the definition rather
than about a tolerance.

The conventions these pin down — cosine so magnitude is ignored, validity from
zero-length targets, per-image before per-split — are each worth a couple of
degrees on a published number, which is more than the gap between many pairs of
backbones.
"""

import math

import pytest
import torch

from visbench.metrics.dense import surface_normal_metrics


def normals(*angles_deg, height=1, width=1):
    """A batch of ``(sin t, 0, cos t)`` maps, one image per angle given."""
    maps = []
    for angle in angles_deg:
        radians = math.radians(angle)
        vector = torch.tensor([math.sin(radians), 0.0, math.cos(radians)])
        maps.append(vector.view(3, 1, 1).expand(3, height, width).clone())
    return torch.stack(maps)


UP = normals(0.0)


class TestAngularError:
    def test_a_perfect_prediction_is_flawless(self):
        target = normals(0.0, 30.0, 60.0, height=4, width=4)
        metrics = surface_normal_metrics(target.clone(), target)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-3)
        assert metrics["mean"] == pytest.approx(0.0, abs=1e-3)
        assert metrics["median"] == pytest.approx(0.0, abs=1e-3)
        assert metrics["d1"] == metrics["d2"] == metrics["d3"] == 1.0

    @pytest.mark.parametrize("angle", [5.0, 11.0, 22.0, 45.0, 90.0, 179.0])
    def test_the_error_is_the_constructed_angle(self, angle):
        metrics = surface_normal_metrics(normals(angle, height=4, width=4), UP.expand(1, 3, 4, 4))
        assert metrics["mean"] == pytest.approx(angle, abs=1e-2)
        assert metrics["rmse"] == pytest.approx(angle, abs=1e-2)
        assert metrics["median"] == pytest.approx(angle, abs=1e-2)

    def test_an_exactly_matching_prediction_does_not_produce_nan(self):
        """``acos`` of a cosine that floating point nudged past 1. probe3d
        clamps for this reason, and one NaN pixel takes the whole split."""
        target = torch.nn.functional.normalize(torch.randn(4, 3, 8, 8), dim=1)
        metrics = surface_normal_metrics(target.clone() * 3.7, target)
        assert all(math.isfinite(value) for value in metrics.values())

    def test_opposite_normals_are_the_worst_case(self):
        metrics = surface_normal_metrics(-UP.expand(1, 3, 4, 4), UP.expand(1, 3, 4, 4))
        assert metrics["mean"] == pytest.approx(180.0, abs=1e-2)

    def test_magnitude_is_ignored(self):
        """The error is a *cosine*, so a head that never learned to normalise
        is scored on direction alone — as probe3d scores it."""
        target = torch.nn.functional.normalize(torch.randn(3, 3, 8, 8), dim=1)
        pred = torch.randn(3, 3, 8, 8)
        scaled = pred * torch.rand(3, 1, 8, 8).add(0.1) * 100
        assert surface_normal_metrics(pred, target) == pytest.approx(
            surface_normal_metrics(scaled, target), abs=1e-3
        )


class TestThresholds:
    def test_the_thresholds_are_probe3d_s(self):
        """11.25 / 22.5 / 30 degrees, strictly *below*."""
        target = UP.expand(1, 3, 4, 4)
        assert surface_normal_metrics(normals(11.0, height=4, width=4), target)["d1"] == 1.0
        assert surface_normal_metrics(normals(12.0, height=4, width=4), target)["d1"] == 0.0
        assert surface_normal_metrics(normals(12.0, height=4, width=4), target)["d2"] == 1.0
        assert surface_normal_metrics(normals(23.0, height=4, width=4), target)["d2"] == 0.0
        assert surface_normal_metrics(normals(23.0, height=4, width=4), target)["d3"] == 1.0
        assert surface_normal_metrics(normals(31.0, height=4, width=4), target)["d3"] == 0.0

    def test_they_are_fractions_not_percentages(self):
        target = torch.nn.functional.normalize(torch.randn(2, 3, 8, 8), dim=1)
        metrics = surface_normal_metrics(torch.randn(2, 3, 8, 8), target)
        assert all(0.0 <= metrics[f"d{k}"] <= 1.0 for k in (1, 2, 3))

    def test_the_thresholds_nest(self):
        target = torch.nn.functional.normalize(torch.randn(4, 3, 8, 8), dim=1)
        metrics = surface_normal_metrics(torch.randn(4, 3, 8, 8), target)
        assert metrics["d1"] <= metrics["d2"] <= metrics["d3"]


class TestValidity:
    def test_zero_length_targets_are_skipped(self):
        """The default mask. Every normal-map format writes (0, 0, 0) for
        unknown, which is the role a 0 plays in a depth map."""
        target = UP.expand(1, 3, 1, 4).clone()
        target[0, :, 0, 2:] = 0.0

        pred = normals(0.0, height=1, width=4).clone()
        pred[0, :, 0, 2:] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1)  # 90 deg wrong

        assert surface_normal_metrics(pred, target)["mean"] == pytest.approx(0.0, abs=1e-3)
        assert surface_normal_metrics(pred, target)["d1"] == 1.0

    def test_an_explicit_mask_wins(self):
        """probe3d masks with ``depth > 0``; pass that mask and get its numbers."""
        target = UP.expand(1, 3, 1, 4).clone()
        pred = normals(0.0, height=1, width=4).clone()
        pred[0, :, 0, 2:] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1)

        mask = torch.ones(1, 1, 4)
        mask[0, 0, 2:] = 0.0
        assert surface_normal_metrics(pred, target, valid=mask)["mean"] == pytest.approx(
            0.0, abs=1e-3
        )
        assert surface_normal_metrics(pred, target)["mean"] == pytest.approx(45.0, abs=1e-2)

    def test_a_four_dimensional_mask_is_accepted(self):
        target = torch.nn.functional.normalize(torch.randn(2, 3, 4, 4), dim=1)
        pred = torch.randn(2, 3, 4, 4)
        flat = torch.ones(2, 4, 4)
        assert surface_normal_metrics(pred, target, valid=flat) == pytest.approx(
            surface_normal_metrics(pred, target, valid=flat.unsqueeze(1))
        )

    def test_an_image_with_no_valid_pixels_scores_zero_not_infinity(self):
        """The median reads a sorted array padded with +inf; an empty image
        would otherwise take the whole split's average with it."""
        target = torch.zeros(2, 3, 4, 4)
        target[0] = UP.view(3, 1, 1).expand(3, 4, 4)
        metrics = surface_normal_metrics(
            normals(20.0, height=4, width=4).expand(2, 3, 4, 4), target
        )
        assert all(math.isfinite(value) for value in metrics.values())
        # One good image at 20 degrees, one empty contributing zero.
        assert metrics["mean"] == pytest.approx(10.0, abs=1e-2)

    def test_a_mismatched_mask_is_refused(self):
        with pytest.raises(ValueError, match="does not match target"):
            surface_normal_metrics(
                torch.randn(2, 3, 4, 4),
                torch.randn(2, 3, 4, 4),
                valid=torch.ones(2, 8, 8),
            )


class TestAveraging:
    def test_images_are_weighted_equally_regardless_of_valid_count(self):
        """Pooling every pixel of the split instead would let a dataset with
        uneven hole coverage silently reweight itself."""
        target = torch.zeros(2, 3, 1, 100)
        target[0, 2, 0, :1] = 1.0  # one valid pixel
        target[1, 2, 0, :] = 1.0  # a hundred

        pred = torch.zeros(2, 3, 1, 100)
        pred[0] = normals(60.0)[0].expand(3, 1, 100)
        pred[1] = normals(0.0)[0].expand(3, 1, 100)

        assert surface_normal_metrics(pred, target)["mean"] == pytest.approx(30.0, abs=1e-2)

    def test_the_median_matches_torch_for_an_odd_count(self):
        angles = [5.0, 40.0, 10.0, 80.0, 20.0]
        target = UP.view(3, 1, 1).expand(3, 1, len(angles)).unsqueeze(0)
        pred = torch.cat([normals(a)[0] for a in angles], dim=2).unsqueeze(0)
        expected = torch.tensor(angles).median().item()
        assert surface_normal_metrics(pred, target)["median"] == pytest.approx(expected, abs=1e-2)

    def test_the_median_straddles_two_samples_for_an_even_count(self):
        angles = [10.0, 20.0, 30.0, 40.0]
        target = UP.view(3, 1, 1).expand(3, 1, 4).unsqueeze(0)
        pred = torch.cat([normals(a)[0] for a in angles], dim=2).unsqueeze(0)
        assert surface_normal_metrics(pred, target)["median"] == pytest.approx(25.0, abs=1e-2)

    def test_the_median_counts_only_valid_pixels(self):
        """Invalid pixels sort to +inf, so they must not pull the middle up."""
        target = UP.view(3, 1, 1).expand(3, 1, 5).unsqueeze(0).clone()
        target[0, :, 0, 3:] = 0.0
        pred = torch.cat([normals(a)[0] for a in (10.0, 20.0, 30.0, 90.0, 90.0)], dim=2).unsqueeze(
            0
        )
        assert surface_normal_metrics(pred, target)["median"] == pytest.approx(20.0, abs=1e-2)

    def test_rmse_and_mean_differ_when_errors_are_uneven(self):
        """A sanity check that rmse is not quietly computing the mean: it is
        the quantity probe3d reports, and it punishes outliers harder."""
        angles = [0.0, 0.0, 0.0, 90.0]
        target = UP.view(3, 1, 1).expand(3, 1, 4).unsqueeze(0)
        pred = torch.cat([normals(a)[0] for a in angles], dim=2).unsqueeze(0)
        metrics = surface_normal_metrics(pred, target)
        assert metrics["mean"] == pytest.approx(22.5, abs=1e-2)
        assert metrics["rmse"] == pytest.approx(45.0, abs=1e-2)


class TestShapes:
    def test_an_uncertainty_channel_is_ignored(self):
        """probe3d slices ``[:, :3]`` so a kappa channel can pass through."""
        target = torch.nn.functional.normalize(torch.randn(2, 3, 4, 4), dim=1)
        three = torch.randn(2, 3, 4, 4)
        four = torch.cat([three, torch.randn(2, 1, 4, 4)], dim=1)
        assert surface_normal_metrics(four, target) == pytest.approx(
            surface_normal_metrics(three, target)
        )

    def test_a_non_three_channel_target_is_refused(self):
        with pytest.raises(ValueError, match=r"\(B, 3, H, W\) target"):
            surface_normal_metrics(torch.randn(1, 3, 4, 4), torch.randn(1, 1, 4, 4))

    def test_too_few_predicted_channels_are_refused(self):
        with pytest.raises(ValueError, match="C >= 3"):
            surface_normal_metrics(torch.randn(1, 2, 4, 4), torch.randn(1, 3, 4, 4))

    def test_a_resolution_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="Resize the prediction"):
            surface_normal_metrics(torch.randn(1, 3, 8, 8), torch.randn(1, 3, 4, 4))
