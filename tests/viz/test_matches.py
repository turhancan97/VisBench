"""The correspondence renderer, and the diagnostic it turns into a number."""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.tasks.mid_level.correspondence import CorrespondenceTask, patch_centers
from visbench.viz import draw_matches, error_coherence, render_match_panels, style_for
from visbench.viz.matches import _sample


def _details(source, target, expected):
    errors = (target - expected).norm(dim=1)
    return {
        "source": source,
        "target": target,
        "expected": expected,
        "errors_px": errors,
        "errors": errors,
    }


class TestErrorCoherence:
    """A number for the thing the panel exists to show.

    Measured on real ResNet-18 features over 224px homography pairs: 0.29 and
    0.40 for two correctly-scored pairs, 0.98 and 1.00 for the same pairs with
    the homography deliberately in the wrong pixel frame. These tests pin the
    two regimes apart on constructed vectors, so the claim is checked without
    downloading weights.
    """

    def test_a_uniform_offset_is_coherent(self):
        """The historical bug's shape: every match wrong the same way."""
        target = torch.rand(64, 2, dtype=torch.float64) * 200
        expected = target - torch.tensor([37.0, -12.0], dtype=torch.float64)
        assert error_coherence(_details(target, target, expected)) > 0.99

    def test_scattered_errors_are_incoherent(self):
        generator = torch.Generator().manual_seed(0)
        target = torch.rand(512, 2, generator=generator, dtype=torch.float64) * 200
        expected = target + torch.randn(512, 2, generator=generator, dtype=torch.float64)
        assert error_coherence(_details(target, target, expected)) < 0.2

    def test_the_two_regimes_are_far_apart(self):
        """The separation is the point: a median error cannot make this call.

        A weak backbone and a broken pipeline can produce the same median. Only
        the direction distribution tells them apart.
        """
        generator = torch.Generator().manual_seed(1)
        points = torch.rand(256, 2, generator=generator, dtype=torch.float64) * 200
        noise = torch.randn(256, 2, generator=generator, dtype=torch.float64) * 30

        broken = error_coherence(_details(points, points, points - torch.tensor([30.0, 30.0])))
        weak = error_coherence(_details(points, points, points + noise))
        assert broken - weak > 0.7

    def test_perfect_matches_have_no_direction_to_average(self):
        points = torch.rand(8, 2, dtype=torch.float64)
        assert error_coherence(_details(points, points, points.clone())) == 0.0

    def test_it_ignores_magnitude(self):
        """Directions only: one huge error must not outvote fifty small ones."""
        target = torch.zeros(50, 2, dtype=torch.float64)
        expected = torch.zeros(50, 2, dtype=torch.float64)
        expected[:, 0] = -1.0  # fifty tiny errors, all pointing +x
        expected[0, 0] = 1000.0  # one enormous error pointing -x
        assert error_coherence(_details(target, target, expected)) > 0.9


class TestDrawing:
    def test_neither_view_is_resized(self):
        """The same no-second-geometry rule the panel viewer keeps."""
        view_0 = Image.new("RGB", (64, 48), (10, 20, 30))
        view_1 = Image.new("RGB", (64, 48), (30, 20, 10))
        points = torch.tensor([[10.0, 10.0], [40.0, 30.0]], dtype=torch.float64)
        canvas = draw_matches(view_0, view_1, _details(points, points, points))
        assert canvas.height == 48
        assert canvas.width == 64 + 64 + 12  # two views plus the seam

    def test_a_match_inside_the_threshold_is_drawn_differently(self):
        view = Image.new("RGB", (32, 32), (0, 0, 0))
        point = torch.tensor([[16.0, 16.0]], dtype=torch.float64)

        near = draw_matches(view, view, _details(point, point, point), threshold=5.0)
        far = draw_matches(view, view, _details(point, point, point + 50.0), threshold=5.0)
        assert np.asarray(near).tolist() != np.asarray(far).tolist()

    def test_a_pair_with_no_matches_still_draws(self):
        """Legitimate: the ratio test can reject everything on a hard pair."""
        view = Image.new("RGB", (32, 32), (0, 0, 0))
        empty = torch.zeros(0, 2, dtype=torch.float64)
        canvas = draw_matches(view, view, _details(empty, empty, empty))
        assert canvas.size == (76, 32)

    def test_matches_are_sampled_evenly_not_taken_from_the_front(self):
        """Matches arrive sorted by similarity, so a prefix flatters the probe."""
        chosen = _sample(1000, 10).tolist()
        assert len(chosen) == 10
        assert chosen[0] == 0 and chosen[-1] == 999
        assert chosen == sorted(chosen)

    def test_everything_is_drawn_when_there_is_room(self):
        assert _sample(6, 40).tolist() == list(range(6))


class TestThroughTheTask:
    """match_details is the scorer's own call, not a second copy of it."""

    def _task_and_features(self):
        task = CorrespondenceTask()
        torch.manual_seed(0)
        features = torch.randn(8, 4, 4)
        return task, features

    def test_details_agree_with_the_scored_errors(self):
        """The panel and the number come from one code path, by construction."""
        task, features = self._task_and_features()
        geometry = {"homography": torch.eye(3, dtype=torch.float64), "size": (224, 224)}

        details = task.match_details(features, features, geometry)
        scored = task._pair_errors(features, features, geometry)
        assert torch.equal(details["errors"], scored)

    def test_an_identity_homography_matched_to_itself_is_exact(self):
        """Which is also the check that `expected` is not off by half a patch."""
        task, features = self._task_and_features()
        geometry = {"homography": torch.eye(3, dtype=torch.float64), "size": (224, 224)}

        details = task.match_details(features, features, geometry)
        assert len(details["errors_px"]) > 0
        assert float(details["errors_px"].max()) < 1e-9
        # And `expected` really is the homography applied to `source`.
        assert torch.allclose(details["expected"], details["source"])

    def test_the_points_are_patch_centres_in_the_working_frame(self):
        task, features = self._task_and_features()
        geometry = {"homography": torch.eye(3, dtype=torch.float64), "size": (224, 224)}
        details = task.match_details(features, features, geometry)

        centres = patch_centers((4, 4), (224, 224))
        for point in details["source"]:
            assert (centres - point).norm(dim=1).min() < 1e-9

    def test_geometry_without_a_homography_is_refused(self):
        task, features = self._task_and_features()
        with pytest.raises(KeyError, match="homography"):
            task.match_details(features, features, {"size": (224, 224)})

    def test_no_matches_returns_empty_arrays_of_the_right_shape(self):
        task, features = self._task_and_features()
        task.ratio_threshold = 0.0  # rejects everything
        geometry = {"homography": torch.eye(3, dtype=torch.float64), "size": (224, 224)}

        details = task.match_details(features, features, geometry)
        assert details["source"].shape == (0, 2)
        assert details["errors_px"].shape == (0,)


class TestTheStyleRow:
    def test_correspondence_is_drawable_but_not_a_panel(self):
        """It has a style so `show` lists it, and refuses the panel colouriser."""
        from visbench.viz import COMPOSITE_KINDS, target_to_rgb

        style = style_for("correspondence")
        assert style.kind == "matches"
        assert style.kind in COMPOSITE_KINDS
        with pytest.raises(ValueError, match="draw_matches"):
            target_to_rgb(torch.rand(4, 4), style)


class TestThePage:
    def test_one_row_per_pair_with_the_counts_in_the_label(self):
        class FakePairs:
            def __getitem__(self, index):
                view = Image.new("RGB", (32, 32), (index * 10, 0, 0))
                return view, view, {}

            def labels(self):
                return [{"homography": torch.eye(3, dtype=torch.float64), "size": (32, 32)}] * 2

        class FakeTask:
            def match_details(self, features_0, features_1, geometry):
                points = torch.tensor([[8.0, 8.0], [24.0, 24.0]], dtype=torch.float64)
                return _details(points, points, points)

        page = render_match_panels(FakePairs(), FakeTask(), [(None, None)] * 2, [0, 1])
        assert page.width == 200 + (32 + 12 + 32) + 8
