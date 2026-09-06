"""Colourisers: the display range, the palette, and where magenta lands."""

import numpy as np
import pytest
import torch

from visbench.viz import INVALID_RGB, DisplayRange, display_range, style_for, target_to_rgb
from visbench.viz.colour import _DEPTH_ANCHORS, voc_palette


def _rgb(target, probe, span=None):
    style = style_for(probe)
    if span is None and style.kind in ("magnitude", "depth"):
        valid = None if style.invalid is None else ~style.invalid(target)
        span = display_range(target, valid)
    return target_to_rgb(target, style, span)


class TestDisplayRange:
    def test_invalid_pixels_do_not_set_the_range(self):
        """Depth's holes are stored as 0, so including them pins every low end.

        The measured consequence is not subtle: with the holes in, a frame whose
        real depths span 4-5 m is drawn against 0-5 and lands in the top fifth of
        the ramp, flat, for every frame in the split.
        """
        target = torch.tensor([[0.0, 0.0], [4.0, 5.0]])
        valid = target > 0
        assert display_range(target, valid).low >= 4.0
        assert display_range(target, None).low == 0.0

    def test_nan_never_reaches_the_percentile(self):
        """A percentile over NaN is NaN, and the whole panel would go black."""
        target = torch.tensor([[float("nan"), 1.0], [2.0, 3.0]])
        span = display_range(target, ~torch.isnan(target))
        assert np.isfinite(span.low) and np.isfinite(span.high)

    def test_a_constant_frame_is_mid_grey_rather_than_a_division_by_zero(self):
        span = DisplayRange(3.0, 3.0)
        assert span.normalise(np.array([3.0])).tolist() == [0.5]

    def test_an_entirely_invalid_frame_still_has_a_range(self):
        target = torch.zeros(2, 2)
        assert display_range(target, torch.zeros(2, 2, dtype=torch.bool)) == DisplayRange(0.0, 1.0)

    def test_the_caption_states_the_unit(self):
        assert DisplayRange(0.41, 6.24).caption("m") == "0.41 to 6.24 m"


class TestWhereMagentaLands:
    """The four conventions, through the colouriser rather than the table."""

    def test_depth_draws_a_hole_as_invalid(self):
        rgb = _rgb(torch.tensor([[0.0, 1.0], [2.0, 3.0]]), "depth")
        assert tuple(rgb[0, 0]) == INVALID_RGB

    @pytest.mark.parametrize("probe", ["edge", "keypoints2d", "corner"])
    def test_a_magnitude_zero_is_drawn_as_data(self, probe):
        """The failure this guards renders, and looks like a target full of holes."""
        rgb = _rgb(torch.tensor([[0.0, 1.0], [2.0, 3.0]]), probe)
        assert tuple(rgb[0, 0]) != INVALID_RGB
        assert tuple(rgb[0, 0]) == (0, 0, 0)

    def test_occlusion_edge_draws_nan_as_invalid_and_zero_as_data(self):
        target = torch.tensor([[float("nan"), 0.0], [2.0, 3.0]])
        rgb = _rgb(target, "occlusion_edge")
        assert tuple(rgb[0, 0]) == INVALID_RGB
        assert tuple(rgb[0, 1]) != INVALID_RGB

    def test_segmentation_keeps_class_zero_and_drops_negatives(self):
        rgb = _rgb(torch.tensor([[-1.0, 0.0]]), "semantic_segmentation")
        assert tuple(rgb[0, 0]) == INVALID_RGB
        assert tuple(rgb[0, 1]) == (0, 0, 0)

    def test_a_zero_length_normal_is_invalid(self):
        target = torch.zeros(3, 1, 2)
        target[2, 0, 1] = 1.0
        rgb = _rgb(target, "surface_normal")
        assert tuple(rgb[0, 0]) == INVALID_RGB
        assert tuple(rgb[0, 1]) == (128, 128, 255)

    def test_no_colouriser_can_produce_magenta_by_itself(self):
        """Why magenta was chosen: it is never ambiguous with real data.

        Greyscale has no hue, ``(n + 1) / 2`` cannot reach it for a unit vector,
        and VOC's palette does not contain it.
        """
        assert INVALID_RGB not in {tuple(int(v) for v in c) for c in voc_palette()}
        grey = _rgb(torch.rand(16, 16), "edge")
        assert not (grey == np.array(INVALID_RGB)).all(axis=-1).any()

    def test_orientation_brightness_is_coherence_and_hue_is_the_angle(self):
        """A zero-coherence pixel reads as black, not as a confident colour;
        two different orientations at equal coherence read as different hues."""
        # column 0: coherence 0 -> black. column 1: coherence 1, orientation 0.
        # column 2: coherence 1, orientation 45 deg (2*theta = pi/2 -> sin term).
        target = torch.zeros(2, 1, 3)
        target[0, 0, 1] = 1.0
        target[1, 0, 2] = 1.0
        rgb = _rgb(target, "orientation")
        assert tuple(rgb[0, 0]) == (0, 0, 0)
        assert rgb[0, 1].max() > 200  # bright
        assert tuple(rgb[0, 1]) != tuple(rgb[0, 2])  # different orientation, different hue


class TestTheVOCPalette:
    def test_class_indices_are_used_as_indices(self):
        """The palette bug, pinned.

        ``convert("L")`` resolved VOC's palette and turned classes
        ``[0, 1, 15, 255]`` into ``[0, 38, 147, 220]``, which loads, trains and
        scores against labels that mean nothing. Four classes must give four
        colours, and class 15 must be the one a VOC reader recognises.
        """
        target = torch.tensor([[0.0, 1.0], [15.0, 20.0]])
        rgb = _rgb(target, "semantic_segmentation")
        assert len({tuple(int(v) for v in colour) for colour in rgb.reshape(-1, 3)}) == 4
        assert tuple(rgb[1, 0]) == tuple(voc_palette()[15]) == (192, 128, 128)

    def test_it_is_voc_s_own_colours(self):
        palette = voc_palette()
        assert tuple(palette[0]) == (0, 0, 0)
        assert tuple(palette[1]) == (128, 0, 0)
        assert tuple(palette[255]) == (224, 224, 192)


class TestSharedRange:
    def test_a_prediction_is_drawn_against_the_target_s_range(self):
        """The guard against scaling each panel to its own extremes.

        That implementation is the obvious one and it hides the most common way
        a regression head is wrong: a prediction uniformly half the target
        renders *identically* to a correct one, because both are stretched to
        fill the ramp.
        """
        target = torch.rand(16, 16) * 4 + 1
        style = style_for("edge")
        span = display_range(target)
        drawn = target_to_rgb(target, style, span).mean()
        halved = target_to_rgb(target * 0.5, style, span).mean()
        assert halved < drawn - 40

        # And the failure mode itself: rescaled independently, they match.
        independent = target_to_rgb(target * 0.5, style, display_range(target * 0.5)).mean()
        assert abs(independent - drawn) < 5

    def test_a_scalar_map_without_a_range_is_refused(self):
        """Rather than derived on the spot, which is what loses the comparison."""
        with pytest.raises(ValueError, match="display range"):
            target_to_rgb(torch.rand(4, 4), style_for("depth"))

    def test_a_box_target_is_not_a_panel(self):
        with pytest.raises(ValueError, match="draw_boxes"):
            target_to_rgb(torch.rand(4, 4), style_for("detection"))


class TestPredictionShapes:
    def test_the_uncertainty_channel_is_not_drawn(self):
        """probe3d's loss adds a kappa channel; folded in it would tint the panel."""
        from visbench.viz.panels import _as_target_form

        raw = torch.zeros(4, 2, 2)
        assert _as_target_form(raw, style_for("surface_normal")).shape == (3, 2, 2)

    def test_semantic_logits_become_class_indices(self):
        from visbench.viz.panels import _as_target_form

        logits = torch.zeros(3, 2, 2)
        logits[2] = 1.0
        drawn = _as_target_form(logits, style_for("semantic_segmentation"))
        assert drawn.shape == (2, 2)
        assert (drawn == 2).all()


class TestDepthRamp:
    """A distance is drawn as a ramp; a magnitude stays grey.

    The module docstring's case against a colour map — that its transitions
    read as edges in the data — is answered by luminance, not by preference,
    so it is asserted here rather than argued in a comment.
    """

    def _ramp(self):
        """The ramp itself, with no invalid pixel in it.

        Starts at 0.1 rather than 0 deliberately: depth's convention makes 0
        *invalid*, so a row starting there is painted magenta by the caller and
        every property below would be measured against the marker instead of
        the ramp. Which is how these tests first failed.
        """
        span = DisplayRange(0.1, 1.0)
        row = torch.linspace(0.1, 1.0, 256).reshape(1, 256)
        return _rgb(row, "depth", span)[0].astype(np.float64)

    @staticmethod
    def _luminance(rgb):
        return rgb @ np.array([0.2126, 0.7152, 0.0722])

    def test_luminance_increases_all_the_way_along_the_ramp(self):
        """Why it cannot manufacture a boundary greyscale does not have.

        The greyscale panel is recoverable as this one's luminance channel, so
        a ramp whose luminance never reverses adds hue to the picture without
        adding structure to it.
        """
        assert np.all(np.diff(self._luminance(self._ramp())) >= 0)
        assert self._luminance(self._ramp())[-1] > self._luminance(self._ramp())[0]

    def test_the_ramp_stays_far_from_the_invalid_marker(self):
        """Magenta must not merely be absent, it must not be approached.

        Viridis proper starts at (68, 1, 84) — hue 296 degrees, four from
        magenta's 300. Dropping that end is why the anchors are what they are,
        and an exact-inequality test would not have noticed it.
        """
        distances = np.linalg.norm(self._ramp() - np.array(INVALID_RGB, dtype=np.float64), axis=-1)
        assert distances.min() > 150.0

    def test_near_and_far_are_different_hues_not_just_different_greys(self):
        ramp = self._ramp()
        near, far = ramp[0], ramp[-1]
        assert abs(float(near[2]) - float(far[2])) > 40.0  # blue channel inverts
        assert float(near[2]) > float(near[0])  # near is blue-dominant
        assert float(far[0]) > float(far[2])  # far is warm

    def test_a_magnitude_is_still_grey(self):
        """The docstring's rule survives for the kind it was written about."""
        grey = _rgb(torch.linspace(0.1, 1.0, 64).reshape(1, 64), "edge", DisplayRange(0.1, 1.0))
        assert (grey[..., 0] == grey[..., 1]).all()
        assert (grey[..., 1] == grey[..., 2]).all()

    def test_depth_and_a_magnitude_no_longer_draw_the_same_pixels(self):
        """The regression this change is: they were identical before."""
        row = torch.linspace(0.1, 1.0, 64).reshape(1, 64)
        span = DisplayRange(0.1, 1.0)
        assert not np.array_equal(_rgb(row, "depth", span), _rgb(row, "edge", span))

    def test_a_prediction_is_still_drawn_against_the_target_s_range(self):
        """The ramp changes the lookup, never the scaling.

        A prediction at half the target's magnitude must not draw as a correct
        one, which is the property the shared `DisplayRange` exists for.
        """
        target = torch.linspace(0.5, 1.0, 64).reshape(1, 64)
        span = DisplayRange(0.5, 1.0)
        halved = _rgb(target * 0.5, "depth", span)
        assert not np.array_equal(halved, _rgb(target, "depth", span))
        # ... and scaling the prediction to its own extremes would hide it.
        assert np.array_equal(halved, _rgb(target * 0.5, "depth", span))

    def test_nan_goes_to_the_top_of_the_ramp_not_to_black(self):
        values = torch.tensor([[0.5, float("nan")]])
        rgb = _rgb(values, "depth", DisplayRange(0.5, 1.0))
        assert tuple(rgb[0, 1]) == tuple(np.asarray(_DEPTH_ANCHORS[-1], dtype=np.uint8))
