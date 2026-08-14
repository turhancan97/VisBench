"""Page assembly, and the rule that the viewer applies no geometry of its own."""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.data.dense import DenseFolderDataset
from visbench.data.detection import DetectionFolderDataset
from visbench.viz import draw_boxes, render_panels, render_probe_panels


@pytest.fixture
def depth_split(tmp_path):
    """A non-square split, so a missed rescale is visible rather than a no-op."""
    root = tmp_path / "depth" / "val"
    (root / "images").mkdir(parents=True)
    (root / "depths").mkdir(parents=True)
    for index in range(2):
        pixels = np.zeros((150, 200, 3), dtype=np.uint8)
        pixels[40:90, 60:140] = (200, 40, 40)
        Image.fromarray(pixels).save(root / "images" / f"{index:02d}.png")
        target = np.zeros((150, 200), dtype=np.float32)
        target[40:90, 60:140] = 3.0 + index
        np.save(root / "depths" / f"{index:02d}.npy", target)
    return DenseFolderDataset(root, target_dir="depths", split="val", image_size=64)


@pytest.fixture
def box_split(tmp_path):
    root = tmp_path / "voc" / "val"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir(parents=True)
    Image.new("RGB", (200, 150), (20, 20, 20)).save(root / "JPEGImages" / "a.jpg")
    (root / "Annotations" / "a.xml").write_text(
        "<annotation><size><width>200</width><height>150</height><depth>3</depth></size>"
        "<object><name>person</name><difficult>0</difficult><bndbox>"
        "<xmin>61</xmin><ymin>41</ymin><xmax>140</xmax><ymax>90</ymax>"
        "</bndbox></object></annotation>"
    )
    return DetectionFolderDataset(
        root,
        image_dir="JPEGImages",
        annotation_dir="Annotations",
        split="val",
        image_size=64,
        include_difficult=True,
    )


class TestNoSecondGeometry:
    """The rule the whole package exists to keep.

    A viewer that resized for layout, re-read the source file, or re-cropped
    could make a misaligned pipeline look fine and a correct one look broken —
    which would make it worse than no viewer at all, since the pair being
    aligned or not is the entire evidence a panel carries.
    """

    def test_the_image_panel_is_the_dataset_s_own_pixels(self, depth_split):
        page = render_probe_panels(depth_split, "depth", [0])
        expected = np.asarray(depth_split[0][0])
        # The panel is pasted at a known offset; crop it back out and compare.
        drawn = np.asarray(page)[18 : 18 + expected.shape[0], 200 : 200 + expected.shape[1]]
        assert np.array_equal(drawn, expected)

    def test_the_target_panel_is_the_target_s_own_shape(self, depth_split):
        page = render_probe_panels(depth_split, "depth", [0])
        image, target = depth_split[0]
        assert target.shape == (image.height, image.width)
        # Two panels of equal width plus the gutter and one gap.
        assert page.width == 200 + 2 * image.width + 2 * 8

    def test_a_misaligned_target_is_visible_rather_than_averaged_away(self, depth_split):
        """What the viewer is for, stated as a test.

        The image's red block and the target's raised region cover the same
        pixels. Roll the target and the two panels disagree — which is exactly
        what a reader sees, and what neither the loss nor the metric would say.

        Compared by bounding box rather than pixel by pixel, because the image
        is resized **bicubic** and the target **nearest** — deliberately, since
        averaging a depth map across a discontinuity invents surfaces no sensor
        saw. That difference moves boundary pixels and nothing else, so the
        question a test can ask exactly is where each region *is*.
        """
        image, target = depth_split[0]

        def extent(mask):
            rows, columns = np.nonzero(mask)
            return rows.min(), rows.max(), columns.min(), columns.max()

        bright = extent(np.asarray(image)[..., 0] > 100)
        raised = extent(target.numpy() > 1.0)
        assert max(abs(a - b) for a, b in zip(bright, raised, strict=True)) <= 1

        shifted = extent(np.roll(target.numpy() > 1.0, 12, axis=1))
        assert max(abs(a - b) for a, b in zip(bright, shifted, strict=True)) >= 10


class TestBoxes:
    def test_boxes_land_where_the_dataset_put_them(self, box_split):
        """Drawn in the frame's own coordinates, so a missed rescale shows.

        The source is 200x150 and the crop is 64x64, so a box left in original
        pixels would fall outside the panel entirely.
        """
        image, annotation = box_split[0]
        boxes = annotation["boxes"]
        assert len(boxes) == 1
        assert float(boxes[0][2]) <= image.width and float(boxes[0][3]) <= image.height

        drawn = np.asarray(draw_boxes(image, boxes))
        plain = np.asarray(image)
        changed = np.argwhere((drawn != plain).any(axis=-1))
        top, left = changed.min(axis=0)
        bottom, right = changed.max(axis=0)
        # The drawn outline hugs the box: the 2px stroke and the caption above it
        # are the only slack.
        assert abs(left - float(boxes[0][0])) <= 3
        assert abs(right - float(boxes[0][2])) <= 3
        assert abs(bottom - float(boxes[0][3])) <= 3
        assert top <= float(boxes[0][1]) + 3

    def test_a_frame_that_lost_every_box_is_labelled_not_dropped(self, box_split):
        """A centre crop genuinely removes objects; that is not a parse failure."""
        page = render_probe_panels(box_split, "detection", [0])
        assert page.width > 0


class TestPage:
    def test_it_refuses_to_draw_nothing(self):
        with pytest.raises(ValueError, match="no frames"):
            render_panels([], ["image"])

    def test_panels_keep_their_own_size(self):
        rows = [("a", [np.zeros((10, 20, 3), np.uint8), np.zeros((10, 30, 3), np.uint8)])]
        page = render_panels(rows, ["image", "target"])
        assert page.width == 200 + 20 + 30 + 2 * 8

    def test_the_legend_names_the_invalid_colour_only_when_one_exists(self, depth_split):
        page = render_probe_panels(depth_split, "depth", [0])
        assert page.mode == "RGB"

    def test_a_prediction_column_appears_only_when_given(self, depth_split):
        image, _ = depth_split[0]
        without = render_probe_panels(depth_split, "depth", [0])
        with_prediction = render_probe_panels(
            depth_split, "depth", [0], torch.rand(1, 1, image.height, image.width)
        )
        assert with_prediction.width == without.width + image.width + 8
