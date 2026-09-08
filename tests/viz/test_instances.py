"""`draw_instances`: the overlay, and what its colours are allowed to mean.

Instance segmentation is the seventeenth probe and the first whose panel draws
*many* targets on one frame. Two properties are load-bearing and neither is
visible in a shape:

- the colours separate instances rather than naming classes, because two
  touching objects of the same class are exactly what this probe measures and
  `semantic_segmentation` cannot;
- no blended instance colour may read as `INVALID_RGB`, which every other panel
  in this package uses for "no ground truth here". The palette does not
  *contain* magenta, so the existing exactness test in `test_colour.py` passed
  before this module existed — but blending is a step no earlier colouriser
  had, and VOC's index 13 blends to something the eye calls magenta. That was
  found by rendering the gallery page and looking at it, which is how every
  gallery bug in this project has been found.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.viz import INVALID_RGB, style_for
from visbench.viz.panels import _instance_palette, draw_instances


def _frame(size: int = 8) -> Image.Image:
    """A mid-grey frame, so a blend is visible in either direction."""
    return Image.fromarray(np.full((size, size, 3), 128, dtype=np.uint8))


def _mask(size: int, rows: slice, cols: slice) -> torch.Tensor:
    mask = torch.zeros(size, size, dtype=torch.bool)
    mask[rows, cols] = True
    return mask


class TestThePalette:
    def test_no_instance_colour_can_read_as_the_invalid_marker(self):
        """The reason this module exists. Distance, not equality.

        Equality is what `test_colour.py` already asserts of the raw palette and
        it is not enough here: a colour 190 away in L1 blends over a pale pixel
        into something indistinguishable from the marker.
        """
        for colour in _instance_palette():
            distance = abs(colour.astype(int) - np.array(INVALID_RGB)).sum()
            assert distance > 255, (tuple(int(v) for v in colour), int(distance))

    def test_it_drops_vocs_index_13_specifically(self):
        """Named, so a future palette change that reintroduces it fails here."""
        assert (192, 0, 128) not in {tuple(int(v) for v in c) for c in _instance_palette()}

    def test_black_is_excluded_because_it_is_vocs_background(self):
        assert (0, 0, 0) not in {tuple(int(v) for v in c) for c in _instance_palette()}

    def test_there_are_enough_colours_for_a_crowded_frame(self):
        """VOC's densest val frame carries 20-odd instances."""
        assert len(_instance_palette()) >= 20


class TestTheOverlay:
    def test_two_instances_get_two_colours(self):
        """The property that makes this probe drawable at all."""
        size = 8
        masks = torch.stack(
            [_mask(size, slice(0, 4), slice(None)), _mask(size, slice(4, 8), slice(None))]
        )
        drawn = np.asarray(draw_instances(_frame(size), masks))
        assert tuple(drawn[0, 0]) != tuple(drawn[7, 0])

    def test_a_pixel_no_instance_covers_is_left_alone(self):
        """The image is evidence; only annotated pixels may be painted over."""
        size = 8
        masks = _mask(size, slice(0, 2), slice(0, 2)).unsqueeze(0)
        drawn = np.asarray(draw_instances(_frame(size), masks))
        assert tuple(drawn[7, 7]) == (128, 128, 128)
        assert tuple(drawn[0, 0]) != (128, 128, 128)

    def test_no_instances_returns_the_frame_unchanged(self):
        """An image where nothing was detected is ordinary, not an error."""
        size = 8
        empty = torch.zeros((0, size, size), dtype=torch.bool)
        drawn = np.asarray(draw_instances(_frame(size), empty))
        assert (drawn == 128).all()

    def test_the_void_region_is_drawn_in_the_marker_colour(self):
        """VOC's void is "no ground truth", which is what magenta means here."""
        size = 8
        masks = torch.zeros((0, size, size), dtype=torch.bool)
        ignore = _mask(size, slice(0, 1), slice(0, 1))
        drawn = np.asarray(draw_instances(_frame(size), masks, ignore=ignore))
        assert tuple(drawn[0, 0]) == INVALID_RGB

    def test_the_void_wins_over_an_instance_that_overlaps_it(self):
        """Drawn last, so an excluded pixel is never hidden by a mask."""
        size = 8
        masks = _mask(size, slice(None), slice(None)).unsqueeze(0)
        ignore = _mask(size, slice(0, 1), slice(0, 1))
        drawn = np.asarray(draw_instances(_frame(size), masks, ignore=ignore))
        assert tuple(drawn[0, 0]) == INVALID_RGB
        assert tuple(drawn[7, 7]) != INVALID_RGB

    def test_colours_cycle_rather_than_running_out(self):
        """A frame with more instances than colours still draws."""
        size = 4
        count = len(_instance_palette()) + 3
        masks = torch.stack([_mask(size, slice(0, 1), slice(0, 1)) for _ in range(count)])
        draw_instances(_frame(size), masks)  # must not raise


class TestItNeverReconcilesGeometry:
    def test_a_mask_of_the_wrong_size_is_refused(self):
        """The rule this whole module exists to keep: no resizing, ever.

        A viewer that resized to fit would make a misaligned pipeline look fine,
        which is worse than having no viewer.
        """
        masks = torch.zeros((1, 4, 4), dtype=torch.bool)
        with pytest.raises(ValueError, match="never resizes"):
            draw_instances(_frame(8), masks)

    def test_a_two_dimensional_mask_stack_is_refused(self):
        with pytest.raises(ValueError, match=r"\(N, H, W\)"):
            draw_instances(_frame(8), torch.zeros((8, 8), dtype=torch.bool))


class TestTheStyleRow:
    def test_the_note_says_the_colours_are_not_matched(self):
        """The one thing a reader could otherwise get wrong from the picture."""
        note = style_for("instance_segmentation").note
        assert "NOT matched" in note

    def test_it_is_a_composite_kind_so_target_to_rgb_refuses_it(self):
        """A dict target has no single colouriser; the refusal names the right one."""
        from visbench.viz.colour import target_to_rgb

        with pytest.raises(ValueError, match="draw_instances"):
            target_to_rgb(torch.zeros(4, 4), style_for("instance_segmentation"))
