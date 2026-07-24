"""Pluggable task heads.

Heads must be selectable per run and never hardcoded to one architecture
(CLAUDE.md, v0.2), so most of what matters here is the contract between a head
and the feature mode it is fed: mismatches should fail at construction, not as
a shape error partway through training.
"""

import pytest
import torch

from visbench.heads import DPTHead, LinearHead, build_head, get_head, list_heads
from visbench.heads.base import BaseHead, register_head
from visbench.types import FeatureMode

# -- the registry -------------------------------------------------------------


def test_both_required_heads_are_registered():
    """CLAUDE.md requires at minimum a linear probe and a DPT-style head."""
    assert set(list_heads()) >= {"linear", "dpt"}


def test_build_by_name():
    head = build_head("linear", in_channels=8, out_channels=2)
    assert isinstance(head, LinearHead)


def test_unknown_head_lists_the_known_ones():
    with pytest.raises(KeyError) as excinfo:
        get_head("transformer_decoder")
    assert "linear" in str(excinfo.value)


def test_duplicate_registration_raises():
    """A contributor's typo must not shadow the linear baseline."""
    with pytest.raises(ValueError, match="already registered"):

        @register_head("linear")
        class Impostor:
            pass


def test_a_contributor_can_add_a_head():
    """The documented extension point."""

    @register_head("test_only_head")
    class Mine(BaseHead):
        supported_feature_modes = (FeatureMode.DENSE_ONLY,)

        def forward(self, features):
            return features

    assert "test_only_head" in list_heads()
    from visbench.heads import base

    del base._HEADS["test_only_head"]


# -- feature-mode contract ----------------------------------------------------


def test_linear_rejects_plus_cls_at_construction_time():
    """The point of check_feature_mode: fail before training, not inside a conv."""
    with pytest.raises(ValueError, match="does not accept"):
        LinearHead.check_feature_mode(FeatureMode.DENSE_PLUS_CLS)


def test_linear_accepts_the_modes_it_declares():
    LinearHead.check_feature_mode(FeatureMode.DENSE_ONLY)
    LinearHead.check_feature_mode(FeatureMode.DENSE_CLS_BROADCAST)


def test_dpt_accepts_all_three():
    for mode in (
        FeatureMode.DENSE_ONLY,
        FeatureMode.DENSE_CLS_BROADCAST,
        FeatureMode.DENSE_PLUS_CLS,
    ):
        DPTHead.check_feature_mode(mode)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown feature_mode"):
        LinearHead.check_feature_mode("dense_and_vibes")


# -- linear head --------------------------------------------------------------


class TestLinearHead:
    def test_projects_and_upsamples(self):
        head = LinearHead(in_channels=32, out_channels=5, output_size=64)
        assert head(torch.rand(2, 32, 16, 16)).shape == (2, 5, 64, 64)

    def test_without_output_size_stays_on_the_grid(self):
        head = LinearHead(in_channels=32, out_channels=5)
        assert head(torch.rand(2, 32, 16, 16)).shape == (2, 5, 16, 16)

    def test_is_a_1x1_convolution(self):
        """The least expressive head available, on purpose.

        A dense-task headline number reported with this one is a difference
        between representations; anything deeper can compensate for a weak
        feature map.
        """
        head = LinearHead(in_channels=4, out_channels=3, bias=False)
        assert sum(p.numel() for p in head.parameters()) == 4 * 3

    def test_each_location_is_projected_independently(self):
        """No spatial mixing: changing one cell must not move its neighbours."""
        head = LinearHead(in_channels=4, out_channels=2)
        features = torch.zeros(1, 4, 3, 3)
        baseline = head(features)

        features[0, :, 1, 1] = 1.0
        changed = head(features)

        assert not torch.allclose(changed[0, :, 1, 1], baseline[0, :, 1, 1])
        assert torch.allclose(changed[0, :, 0, 0], baseline[0, :, 0, 0])

    def test_broadcast_mode_needs_the_widened_channel_count(self):
        """Under dense_cls_broadcast the head sees 2x the backbone's embed_dim."""
        head = LinearHead(in_channels=16, out_channels=1)
        with pytest.raises(ValueError, match="doubled"):
            head(torch.rand(1, 32, 8, 8))

    def test_a_cls_pair_is_refused_with_the_reason(self):
        head = LinearHead(in_channels=16, out_channels=1)
        with pytest.raises(TypeError, match="dense_plus_cls"):
            head((torch.rand(1, 16, 8, 8), torch.rand(1, 16)))

    def test_non_spatial_input_raises(self):
        head = LinearHead(in_channels=16, out_channels=1)
        with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
            head(torch.rand(1, 16, 8))

    def test_is_trainable(self):
        head = LinearHead(in_channels=8, out_channels=1, output_size=16)
        loss = head(torch.rand(2, 8, 4, 4)).mean()
        loss.backward()
        assert head.proj.weight.grad is not None


# -- DPT head -----------------------------------------------------------------


class TestDPTHead:
    def _stages(self, n=4, channels=32, size=16, batch=2):
        return [torch.rand(batch, channels, size, size) for _ in range(n)]

    def test_fuses_layers_into_one_prediction(self):
        head = DPTHead(in_channels=32, out_channels=1, num_layers=4, hidden_dim=16)
        assert head(self._stages()).shape[:2] == (2, 1)

    def test_output_size_is_honoured(self):
        head = DPTHead(in_channels=32, out_channels=3, num_layers=4, hidden_dim=16, output_size=64)
        assert head(self._stages()).shape == (2, 3, 64, 64)

    def test_single_tensor_is_refused(self):
        """A DPT fed one layer is not multiscale; duplicating it would report a
        single-layer result under a name that claims otherwise."""
        head = DPTHead(in_channels=32, out_channels=1, num_layers=4, hidden_dim=16)
        with pytest.raises(TypeError, match="multiscale"):
            head(torch.rand(2, 32, 16, 16))

    def test_wrong_layer_count_raises(self):
        head = DPTHead(in_channels=32, out_channels=1, num_layers=4, hidden_dim=16)
        with pytest.raises(ValueError, match="expects 4 layers, got 2"):
            head(self._stages(n=2))

    def test_per_layer_widths(self):
        """Backbone depths need not share a channel count."""
        head = DPTHead(in_channels=[16, 32, 64], out_channels=1, num_layers=3, hidden_dim=8)
        stages = [torch.rand(2, c, 8, 8) for c in (16, 32, 64)]
        assert head(stages).shape[:2] == (2, 1)

    def test_width_mismatch_names_the_layer(self):
        head = DPTHead(in_channels=[16, 32], out_channels=1, num_layers=2, hidden_dim=8)
        with pytest.raises(ValueError, match="Layer 1 has 64 channels"):
            head([torch.rand(2, 16, 8, 8), torch.rand(2, 64, 8, 8)])

    def test_in_channels_length_must_match_num_layers(self):
        with pytest.raises(ValueError, match="Got 2 in_channels for num_layers=4"):
            DPTHead(in_channels=[16, 32], out_channels=1, num_layers=4)

    def test_every_layer_influences_the_output(self):
        """If a stage could be dropped without changing anything, the head is
        not actually fusing it."""
        torch.manual_seed(0)
        head = DPTHead(in_channels=8, out_channels=1, num_layers=3, hidden_dim=8).eval()
        stages = self._stages(n=3, channels=8, size=8)

        with torch.no_grad():
            baseline = head(stages)
            for index in range(3):
                perturbed = list(stages)
                perturbed[index] = perturbed[index] + 1.0
                assert not torch.allclose(head(perturbed), baseline), f"layer {index} ignored"

    def test_cls_vector_is_used_when_requested(self):
        torch.manual_seed(0)
        head = DPTHead(
            in_channels=8, out_channels=1, num_layers=2, hidden_dim=8, use_cls=True
        ).eval()
        stages = self._stages(n=2, channels=8, size=8)

        with torch.no_grad():
            a = head((stages, torch.zeros(2, 8)))
            b = head((stages, torch.ones(2, 8)))
        assert not torch.allclose(a, b)

    def test_cls_vector_is_ignored_when_not_requested(self):
        """use_cls=False means dense_plus_cls degrades to dense_only, quietly
        but deliberately — the head simply has no projection for it."""
        torch.manual_seed(0)
        head = DPTHead(in_channels=8, out_channels=1, num_layers=2, hidden_dim=8).eval()
        stages = self._stages(n=2, channels=8, size=8)

        with torch.no_grad():
            a = head((stages, torch.zeros(2, 8)))
            b = head((stages, torch.ones(2, 8)))
        assert torch.allclose(a, b)

    def test_uses_no_normalisation_layer(self):
        """BatchNorm over a frozen backbone's features would make a probe's
        score depend on how the loader grouped images."""
        head = DPTHead(in_channels=8, out_channels=1, num_layers=2, hidden_dim=8)
        assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in head.modules())

    def test_is_trainable(self):
        head = DPTHead(in_channels=8, out_channels=1, num_layers=2, hidden_dim=8)
        head(self._stages(n=2, channels=8, size=8)).mean().backward()
        assert head.reassemble[0].weight.grad is not None

    def test_is_deeper_than_linear(self):
        """The reason both exist: they are not interchangeable in capacity."""
        linear = LinearHead(in_channels=32, out_channels=1)
        dpt = DPTHead(in_channels=32, out_channels=1, num_layers=4, hidden_dim=32)
        assert sum(p.numel() for p in dpt.parameters()) > 100 * sum(
            p.numel() for p in linear.parameters()
        )


def test_heads_accept_real_backbone_features(fake_vit):
    """End to end from a backbone, in the mode each head declares."""
    batch = torch.rand(2, 3, 64, 64)

    broadcast = fake_vit.extract_features(batch, feature_mode=FeatureMode.DENSE_CLS_BROADCAST)
    linear = LinearHead(in_channels=broadcast["dense"].shape[1], out_channels=1, output_size=64)
    assert linear(broadcast["dense"]).shape == (2, 1, 64, 64)

    plus_cls = fake_vit.extract_features(batch, feature_mode=FeatureMode.DENSE_PLUS_CLS)
    dpt = DPTHead(
        in_channels=plus_cls["dense"].shape[1],
        out_channels=1,
        num_layers=2,
        hidden_dim=16,
        use_cls=True,
        output_size=64,
    )
    stages = [plus_cls["dense"], plus_cls["dense"]]
    assert dpt((stages, plus_cls["cls"])).shape == (2, 1, 64, 64)
