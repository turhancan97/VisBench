"""Colourisers: the display range, the palette, and where magenta lands."""

import numpy as np
import pytest
import torch

from visbench.viz import INVALID_RGB, DisplayRange, display_range, style_for, target_to_rgb
from visbench.viz.colour import voc_palette


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
        assert DisplayRange(0.41, 6.24).caption("m") == "0.41-6.24 m"


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
