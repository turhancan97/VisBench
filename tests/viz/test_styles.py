"""The listed style table, and that it stays in step with the probe registry."""

import pytest
import torch

import visbench
from visbench.cli.datasets import showable_probes
from visbench.viz import TARGET_STYLES, UnknownTargetStyle, show_probes, style_for


class TestTheTable:
    def test_every_style_names_a_registered_probe(self):
        """A row for a probe that does not exist would never be reached."""
        assert set(TARGET_STYLES) <= set(visbench.list_probes())

    def test_the_cli_and_the_drawing_table_agree(self):
        """Two tables in two packages answering halves of one question.

        ``visbench.viz`` holds *how* a target is drawn and
        ``visbench.cli.datasets`` holds *which flags build it*. They are
        separate so the Python API needs no CLI — and that is exactly the
        arrangement in which one grows a row the other lacks, so a probe would
        become listable and undrawable, or drawable and unreachable.
        """
        assert show_probes() == showable_probes()

    def test_a_typo_raises_rather_than_defaulting(self):
        """The whole posture of this module, in one assertion.

        Falling back to "scalar map, mask the zeros" would be correct for depth
        and silently wrong for the four probes where 0 is a real reading.
        """
        with pytest.raises(UnknownTargetStyle, match="Unknown probe"):
            style_for("depht")

    def test_a_registered_probe_without_a_style_is_distinguished_from_a_typo(self, monkeypatch):
        """Reached by removing a row, because every probe now has one.

        Since 9c the table covers the whole registry, so this branch cannot fire
        against today's probes — and an untested branch that cannot fire is the
        QuickGELU failure waiting to happen. It is kept rather than deleted
        because it is the message a contributor sees when they add a probe and
        forget its style, which ``CONTRIBUTING.md`` tells them to expect.
        """
        without_depth = {k: v for k, v in TARGET_STYLES.items() if k != "depth"}
        monkeypatch.setattr("visbench.viz.styles.TARGET_STYLES", without_depth)

        with pytest.raises(UnknownTargetStyle, match="registered probe"):
            style_for("depth")


class TestTheFourConventions:
    """One test per validity convention, because there is no shared rule.

    These are the assertions that stop a future edit collapsing four
    conventions into one helper. Each one fails if the wrong rule is applied,
    and the wrong rule *renders* — it does not raise.
    """

    @pytest.mark.parametrize("probe", ["edge", "keypoints2d", "corner"])
    def test_a_real_zero_is_not_invalid(self, probe):
        """0 means "no edge here", a reading covering most of a frame."""
        style = style_for(probe)
        assert style.invalid is None

    def test_depth_marks_zero_invalid(self):
        style = style_for("depth")
        assert style.invalid is not None
        mask = style.invalid(torch.tensor([[0.0, 1.0]]))
        assert mask.tolist() == [[True, False]]

    def test_normals_mark_the_zero_vector_invalid_per_pixel(self):
        style = style_for("surface_normal")
        target = torch.zeros(3, 1, 2)
        target[2, 0, 1] = 1.0
        mask = style.invalid(target)
        assert mask.shape == (1, 2)
        assert mask.tolist() == [[True, False]]

    @pytest.mark.parametrize("probe", ["generic_segmentation", "semantic_segmentation"])
    def test_segmentation_marks_negatives_invalid_and_keeps_class_zero(self, probe):
        """Reusing depth's rule here would erase every background pixel."""
        style = style_for(probe)
        mask = style.invalid(torch.tensor([[-1.0, 0.0, 3.0]]))
        assert mask.tolist() == [[True, False, False]]

    def test_occlusion_edge_marks_nan_invalid(self):
        style = style_for("occlusion_edge")
        mask = style.invalid(torch.tensor([[float("nan"), 0.0]]))
        assert mask.tolist() == [[True, False]]
