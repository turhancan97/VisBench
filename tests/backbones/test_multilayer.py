"""Multi-layer feature extraction.

Declared in the interface from v0.1 and wired up in v0.2, once the single-layer
path was proven (CLAUDE.md, "Multi-layer extraction"). This is what a
multiscale head such as :class:`~visbench.heads.DPTHead` consumes, so most of
what matters here is that the layers a caller asks for are the layers they get,
in the order they asked, from a single forward pass.
"""

import pytest
import torch

from visbench.heads import DPTHead
from visbench.types import FeatureMode

IMAGE = torch.rand(2, 3, 64, 64)


# -- resolving the request ----------------------------------------------------


class TestResolveLayers:
    def test_none_is_the_last_layer(self, fake_vit):
        assert fake_vit.resolve_layers(None) == [fake_vit.num_layers - 1]

    def test_negative_indices_count_from_the_end(self, fake_vit):
        assert fake_vit.resolve_layers([-1]) == [11]
        assert fake_vit.resolve_layers([-4, -1]) == [8, 11]

    def test_minus_one_resolves_to_the_same_index_as_its_absolute_form(self, fake_vit):
        """Both must reach one cache entry, not two holding identical features."""
        assert fake_vit.resolve_layers([-1]) == fake_vit.resolve_layers([11]) == [11]

    def test_out_of_range_names_the_valid_span(self, fake_vit):
        with pytest.raises(ValueError, match=r"valid: 0\.\.11, or -1\.\.-12"):
            fake_vit.resolve_layers([12])

    def test_out_of_range_negative_is_caught_too(self, fake_vit):
        with pytest.raises(ValueError, match="out of range"):
            fake_vit.resolve_layers([-13])

    def test_descending_order_is_refused(self, fake_vit):
        """Order is part of the request: a head reads the first layer as coarsest."""
        with pytest.raises(ValueError, match="strictly increasing"):
            fake_vit.resolve_layers([11, 6])

    def test_duplicates_are_refused(self, fake_vit):
        """A pyramid built from one layer twice is not multiscale."""
        with pytest.raises(ValueError, match="strictly increasing"):
            fake_vit.resolve_layers([6, 6])

    def test_aliasing_duplicates_are_caught_after_resolution(self, fake_vit):
        """[11, -1] is the same layer written two ways."""
        with pytest.raises(ValueError, match="strictly increasing"):
            fake_vit.resolve_layers([11, -1])

    def test_empty_list_is_refused(self, fake_vit):
        with pytest.raises(ValueError, match="requests nothing"):
            fake_vit.resolve_layers([])

    def test_non_int_index_is_refused(self, fake_vit):
        with pytest.raises(TypeError, match="must be ints"):
            fake_vit.resolve_layers([1.5])

    def test_bare_int_is_refused(self, fake_vit):
        """layers=5 is a plausible typo for layers=[5]."""
        with pytest.raises(TypeError, match="list of ints"):
            fake_vit.resolve_layers(5)


# -- what extract_features returns --------------------------------------------


class TestMultiLayerOutput:
    def test_dense_layers_appears_only_when_asked(self, fake_vit):
        assert "dense_layers" not in fake_vit.extract_features(IMAGE)
        assert "dense_layers" in fake_vit.extract_features(IMAGE, layers=[3, 7])

    def test_one_map_per_requested_layer(self, fake_vit):
        features = fake_vit.extract_features(IMAGE, layers=[2, 5, 9])
        assert len(features["dense_layers"]) == 3
        assert features["layer_indices"] == [2, 5, 9]

    def test_layer_indices_are_resolved(self, fake_vit):
        """A record should say which depth was read, not a relative index."""
        assert fake_vit.extract_features(IMAGE, layers=[-2, -1])["layer_indices"] == [10, 11]

    def test_dense_is_the_last_requested_layer(self, fake_vit):
        features = fake_vit.extract_features(IMAGE, layers=[3, 7])
        assert torch.equal(features["dense"], features["dense_layers"][-1])

    def test_the_layers_differ_from_each_other(self, fake_vit):
        """Otherwise the whole path could be returning one layer repeatedly."""
        first, second = fake_vit.extract_features(IMAGE, layers=[3, 7])["dense_layers"]
        assert not torch.allclose(first, second)

    def test_a_single_layer_request_still_gives_a_list(self, fake_vit):
        """The type follows the shape of the request, not the count.

        A caller who narrows layers=[3, 7] to layers=[7] must not have the
        return type change underneath them.
        """
        features = fake_vit.extract_features(IMAGE, layers=[7])
        assert len(features["dense_layers"]) == 1

    def test_the_last_layer_matches_a_single_layer_call(self, fake_vit):
        """A multi-layer call is a superset of the single-layer one."""
        multi = fake_vit.extract_features(IMAGE, layers=[3, 11])
        single = fake_vit.extract_features(IMAGE, layers=[11])
        assert torch.equal(multi["dense"], single["dense"])
        assert torch.equal(multi["pooled"], single["pooled"])

    def test_default_call_is_unchanged_by_all_of_this(self, fake_vit):
        """layers=None must still mean exactly what it meant in v0.1."""
        default = fake_vit.extract_features(IMAGE)
        last = fake_vit.extract_features(IMAGE, layers=[fake_vit.num_layers - 1])
        assert torch.equal(default["dense"], last["dense"])
        assert torch.equal(default["pooled"], last["pooled"])

    def test_one_forward_pass_for_the_whole_request(self, fake_vit):
        """The entire reason layers is a list rather than a loop of calls."""
        before = fake_vit.call_count
        fake_vit.extract_features(IMAGE, layers=[1, 5, 9])
        assert fake_vit.call_count == before + 1

    def test_pooled_comes_from_the_last_layer(self, fake_vit):
        multi = fake_vit.extract_features(IMAGE, layers=[0, 4])
        single = fake_vit.extract_features(IMAGE, layers=[4])
        assert torch.equal(multi["pooled"], single["pooled"])


# -- feature modes apply per layer --------------------------------------------


class TestFeatureModesAcrossLayers:
    def test_broadcast_widens_every_layer(self, fake_vit):
        features = fake_vit.extract_features(
            IMAGE, layers=[3, 7], feature_mode=FeatureMode.DENSE_CLS_BROADCAST
        )
        for dense in features["dense_layers"]:
            assert dense.shape[1] == fake_vit.embed_dim * 2

    def test_plus_cls_returns_the_last_layers_cls(self, fake_vit):
        """`cls` describes `dense`, and `dense` is the last requested layer."""
        multi = fake_vit.extract_features(
            IMAGE, layers=[3, 7], feature_mode=FeatureMode.DENSE_PLUS_CLS
        )
        single = fake_vit.extract_features(
            IMAGE, layers=[7], feature_mode=FeatureMode.DENSE_PLUS_CLS
        )
        assert torch.equal(multi["cls"], single["cls"])

    def test_dense_only_leaves_the_width_alone(self, fake_vit):
        features = fake_vit.extract_features(IMAGE, layers=[3, 7])
        for dense in features["dense_layers"]:
            assert dense.shape[1] == fake_vit.embed_dim


# -- CNNs, where stages genuinely differ in shape -----------------------------


class TestCNNStages:
    def test_stages_differ_in_width_and_resolution(self, fake_cnn):
        """Unlike a ViT's blocks. This is why DPTHead takes per-layer widths."""
        maps = fake_cnn.extract_features(IMAGE, layers=[0, 1, 2])["dense_layers"]
        widths = [dense.shape[1] for dense in maps]
        grids = [tuple(dense.shape[-2:]) for dense in maps]
        assert len(set(widths)) == 3
        assert len(set(grids)) == 3

    def test_grid_hw_describes_the_last_layer(self, fake_cnn):
        features = fake_cnn.extract_features(IMAGE, layers=[0, 2])
        assert features["grid_hw"] == tuple(features["dense_layers"][-1].shape[-2:])

    def test_a_cnn_has_no_cls_to_broadcast(self, fake_cnn):
        with pytest.raises(ValueError):
            fake_cnn.extract_features(
                IMAGE, layers=[0, 2], feature_mode=FeatureMode.DENSE_CLS_BROADCAST
            )


# -- the reason any of this exists --------------------------------------------


def test_a_vit_pyramid_feeds_dpt(fake_vit):
    """End to end: extract several depths, fuse them into a dense prediction."""
    features = fake_vit.extract_features(IMAGE, layers=[2, 5, 8, 11])
    head = DPTHead(
        in_channels=fake_vit.embed_dim,
        out_channels=1,
        num_layers=4,
        hidden_dim=16,
        output_size=64,
    )
    assert head(features["dense_layers"]).shape == (2, 1, 64, 64)


def test_a_cnn_pyramid_feeds_dpt(fake_cnn):
    """The same head, per-layer widths, on stages of three different shapes."""
    features = fake_cnn.extract_features(IMAGE, layers=[0, 1, 2])
    head = DPTHead(
        in_channels=[dense.shape[1] for dense in features["dense_layers"]],
        out_channels=1,
        num_layers=3,
        hidden_dim=16,
        output_size=64,
    )
    assert head(features["dense_layers"]).shape == (2, 1, 64, 64)


def test_dpt_with_a_global_vector_from_the_deepest_layer(fake_vit):
    features = fake_vit.extract_features(
        IMAGE, layers=[5, 11], feature_mode=FeatureMode.DENSE_PLUS_CLS
    )
    head = DPTHead(
        in_channels=fake_vit.embed_dim,
        out_channels=1,
        num_layers=2,
        hidden_dim=16,
        use_cls=True,
        output_size=64,
    )
    output = head((features["dense_layers"], features["cls"]))
    assert output.shape == (2, 1, 64, 64)


def test_a_backbone_returning_the_wrong_layer_count_is_caught(fake_vit, monkeypatch):
    """The base class cannot verify *which* layers came back, but it can verify
    how many — a subclass silently dropping one would misalign every index."""
    monkeypatch.setattr(
        fake_vit, "_forward_features", lambda image, layers: [(torch.rand(2, 16, 8), None, (4, 4))]
    )
    with pytest.raises(RuntimeError, match="returned 1 layers for 3 requested"):
        fake_vit.extract_features(IMAGE, layers=[1, 2, 3])
