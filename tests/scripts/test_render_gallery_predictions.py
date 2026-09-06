"""The four prediction-only gallery figures: what their rows and legend say.

`depth`, `surface_normal`, `keypoints2d` and `occlusion_edge` have no
redistributable ground truth, so their pages are `image | prediction` with a
footer saying why. Drawing one needs the network, the `[hub]` extra and a
published head, so for its whole life the only check on what it said was
looking at it -- and it shipped for three weeks captioning every row with a
bare index, on figures whose entire greyscale ramp is unreadable without the
range beside it. These tests call the two pure helpers that page is now built
from, so the fast suite covers what it claims.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from visbench.viz.styles import style_for

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "render_gallery.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("render_gallery", SCRIPT)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def image():
    return np.zeros((8, 8, 3), dtype=np.uint8)


class TestPredictionRow:
    def test_a_scalar_row_states_the_range_it_is_drawn_against(self, module, image):
        """The defect these figures shipped with, as a test.

        The span was computed to colour the panel and then discarded, so a
        `depth` page said nothing about whether bright meant near or far.
        """
        prediction = torch.linspace(0.5, 4.0, 64).reshape(1, 8, 8)
        label, panels, span = module.prediction_row("frame", image, prediction, style_for("depth"))

        assert span is not None
        assert label.split("\n")[0] == "frame"
        assert label.split("\n")[1].endswith(" m")
        assert len(panels) == 2

    def test_a_normal_map_is_not_asked_for_a_range(self, module, image):
        """(3, H, W) has no span, and asking for one is a shape error."""
        prediction = torch.rand(3, 8, 8)
        label, _, span = module.prediction_row(
            "frame", image, prediction, style_for("surface_normal")
        )

        assert span is None
        assert label == "frame"


class TestPredictionFooter:
    def test_it_states_the_feature_grid_and_the_size_it_is_stretched_to(self, module):
        """Without this the coarseness reads as a broken head, not a readout."""
        footer = module.prediction_footer("keypoints2d", "org/repo", (16, 16), ranged=True)

        assert "16x16" in footer
        assert f"{module.PREDICTION_SIZE}px" in footer
        assert "org/repo" in footer

    def test_the_grid_is_read_from_the_features_rather_than_assumed(self, module):
        assert "7x7" in module.prediction_footer("depth", "org/repo", (7, 7), ranged=True)

    def test_only_a_ranged_page_explains_whose_range_it_used(self, module):
        """A normals page has no range, so the clause would be a false legend."""
        assert "own range" in module.prediction_footer("depth", "org/repo", (16, 16), ranged=True)
        assert "own range" not in module.prediction_footer(
            "surface_normal", "org/repo", (16, 16), ranged=False
        )
