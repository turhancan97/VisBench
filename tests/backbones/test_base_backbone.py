"""Contract tests for BaseBackbone.

The contract, per CLAUDE.md ("Feature extraction design"):

* ``extract_features`` returns all three keys, with ``dense`` ``(B, C, H, W)``,
  ``pooled`` ``(B, C)`` and ``grid_hw`` matching ``dense.shape[-2:]``.
* ``pooling="default"`` resolves to CLS when ``has_cls_token``, else mean.
* ``pooling="cls"`` on a backbone without a CLS token raises rather than
  silently falling back.
* Parameters are frozen and the module is in eval mode after construction.
* ``layers`` with more than one entry raises until v0.2.
"""

import pytest
import torch

from visbench.types import FeatureMode, Pooling


def test_extract_features_returns_dense_and_pooled(fake_vit):
    batch = torch.rand(2, 3, 64, 64)
    features = fake_vit.extract_features(batch)

    assert set(features) == {"dense", "pooled", "grid_hw"}
    assert features["dense"].shape == (2, fake_vit.embed_dim, 4, 4)
    assert features["pooled"].shape == (2, fake_vit.embed_dim)
    assert features["grid_hw"] == (4, 4)
    assert tuple(features["dense"].shape[-2:]) == features["grid_hw"]


def test_cnn_and_vit_share_one_return_shape(fake_vit, fake_cnn):
    """The abstraction's whole claim: a caller cannot tell the families apart."""
    batch = torch.rand(2, 3, 64, 64)
    vit = fake_vit.extract_features(batch)
    cnn = fake_cnn.extract_features(batch)

    assert set(vit) == set(cnn)
    assert vit["dense"].ndim == cnn["dense"].ndim == 4
    assert vit["pooled"].shape == cnn["pooled"].shape


def test_default_pooling_follows_architecture(fake_vit, fake_cnn):
    assert fake_vit.default_pooling() == Pooling.CLS
    assert fake_cnn.default_pooling() == Pooling.MEAN

    batch = torch.rand(2, 3, 64, 64)
    assert torch.allclose(
        fake_vit.extract_features(batch, pooling=Pooling.DEFAULT)["pooled"],
        fake_vit.extract_features(batch, pooling=Pooling.CLS)["pooled"],
    )
    assert torch.allclose(
        fake_cnn.extract_features(batch, pooling=Pooling.DEFAULT)["pooled"],
        fake_cnn.extract_features(batch, pooling=Pooling.MEAN)["pooled"],
    )


def test_pooling_choice_changes_pooled_not_dense(fake_vit):
    """Pooling is the task's decision; it must not perturb the dense grid."""
    batch = torch.rand(2, 3, 64, 64)
    cls = fake_vit.extract_features(batch, pooling=Pooling.CLS)
    mean = fake_vit.extract_features(batch, pooling=Pooling.MEAN)

    assert torch.allclose(cls["dense"], mean["dense"])
    assert not torch.allclose(cls["pooled"], mean["pooled"])


def test_cls_pooling_without_cls_token_raises(fake_cnn):
    with pytest.raises(ValueError, match="no CLS token"):
        fake_cnn.extract_features(torch.rand(1, 3, 64, 64), pooling=Pooling.CLS)


def test_unknown_pooling_raises(fake_vit):
    with pytest.raises(ValueError, match="Unknown pooling"):
        fake_vit.extract_features(torch.rand(1, 3, 64, 64), pooling="average")


def test_backbone_is_frozen_and_eval(fake_vit):
    assert not fake_vit.training
    assert fake_vit.parameters(), "fixture must own a parameter for this to mean anything"
    assert all(not p.requires_grad for p in fake_vit.parameters())


def test_extract_features_does_not_build_a_graph(fake_vit):
    """Frozen means frozen: no autograd graph, so probes cannot leak gradients."""
    features = fake_vit.extract_features(torch.rand(1, 3, 64, 64))
    assert not features["dense"].requires_grad
    assert not features["pooled"].requires_grad


def test_single_layer_request_is_accepted(fake_vit):
    features = fake_vit.extract_features(torch.rand(1, 3, 64, 64), layers=[11])
    assert features["pooled"].shape == (1, fake_vit.embed_dim)


def test_unbatched_input_raises_with_a_hint(fake_vit):
    with pytest.raises(ValueError, match="unsqueeze"):
        fake_vit.extract_features(torch.rand(3, 64, 64))


def test_pil_input_raises_pointing_at_preprocess(fake_vit, solid_images):
    with pytest.raises(TypeError, match="preprocess"):
        fake_vit.extract_features(solid_images[0])


def test_forward_is_extract_features(fake_vit):
    batch = torch.rand(2, 3, 64, 64)
    assert torch.allclose(fake_vit(batch)["pooled"], fake_vit.extract_features(batch)["pooled"])


def test_preprocess_produces_an_extractable_batch(fake_vit, solid_images):
    batch = fake_vit.preprocess(solid_images)
    assert batch.shape == (4, 3, 64, 64)
    assert fake_vit.extract_features(batch)["pooled"].shape == (4, fake_vit.embed_dim)


def test_cache_key_tracks_resolution(fake_vit):
    """A cache key that ignores input size would serve mismatched features."""
    before = fake_vit.cache_key()
    fake_vit.image_size = 128
    assert fake_vit.cache_key() != before


class TestFeatureModes:
    """All three modes are callable from v0.1; only dense_only is used by v0.1 tasks."""

    def _dense_and_cls(self, backbone):
        features = backbone.extract_features(torch.rand(2, 3, 64, 64))
        return features["dense"], features["pooled"]

    def test_dense_only_passes_through(self, fake_vit):
        from visbench.backbones.pooling import apply_feature_mode

        dense, cls = self._dense_and_cls(fake_vit)
        assert apply_feature_mode(dense, cls, FeatureMode.DENSE_ONLY) is dense

    def test_broadcast_widens_channels(self, fake_vit):
        from visbench.backbones.pooling import apply_feature_mode

        dense, cls = self._dense_and_cls(fake_vit)
        out = apply_feature_mode(dense, cls, FeatureMode.DENSE_CLS_BROADCAST)
        assert out.shape == (2, dense.shape[1] + cls.shape[1], 4, 4)
        # Every spatial location carries the same CLS copy.
        assert torch.allclose(out[:, dense.shape[1] :, 0, 0], cls)
        assert torch.allclose(out[:, dense.shape[1] :, 3, 3], cls)

    def test_plus_cls_keeps_them_separate(self, fake_vit):
        from visbench.backbones.pooling import apply_feature_mode

        dense, cls = self._dense_and_cls(fake_vit)
        out_dense, out_cls = apply_feature_mode(dense, cls, FeatureMode.DENSE_PLUS_CLS)
        assert out_dense.shape == dense.shape
        assert out_cls.shape == cls.shape

    def test_unknown_mode_raises(self, fake_vit):
        from visbench.backbones.pooling import apply_feature_mode

        dense, cls = self._dense_and_cls(fake_vit)
        with pytest.raises(ValueError, match="Unknown feature mode"):
            apply_feature_mode(dense, cls, "dense_and_vibes")


def test_tokens_to_grid_rejects_unstripped_cls():
    """The failure this guard exists for: a leftover CLS token misaligns the grid."""
    from visbench.backbones.pooling import tokens_to_grid

    with pytest.raises(ValueError, match="CLS or register token"):
        tokens_to_grid(torch.rand(1, 17, 8), (4, 4))


class TestFeatureModeThroughExtractFeatures:
    """The seam heads plug into.

    All three modes existed and were tested from v0.1, but `extract_features`
    had no `feature_mode` parameter, so `apply_feature_mode` had zero callers
    and modes 2 and 3 were unreachable through the public API. A DPT head is
    exactly the consumer that wants `dense_plus_cls`.
    """

    def test_default_is_dense_only(self, fake_vit):
        features = fake_vit.extract_features(torch.rand(2, 3, 64, 64))
        assert features["dense"].shape == (2, fake_vit.embed_dim, 4, 4)
        assert "cls" not in features

    def test_broadcast_widens_the_channel_dim(self, fake_vit):
        features = fake_vit.extract_features(
            torch.rand(2, 3, 64, 64), feature_mode=FeatureMode.DENSE_CLS_BROADCAST
        )
        assert features["dense"].shape == (2, 2 * fake_vit.embed_dim, 4, 4)
        assert "cls" not in features

    def test_plus_cls_returns_the_vector_separately(self, fake_vit):
        features = fake_vit.extract_features(
            torch.rand(2, 3, 64, 64), feature_mode=FeatureMode.DENSE_PLUS_CLS
        )
        assert features["dense"].shape == (2, fake_vit.embed_dim, 4, 4)
        assert features["cls"].shape == (2, fake_vit.embed_dim)

    def test_pooled_is_unaffected_by_the_mode(self, fake_vit):
        """pooled answers a different question; a task may want both."""
        batch = torch.rand(2, 3, 64, 64)
        baseline = fake_vit.extract_features(batch)["pooled"]
        for mode in (FeatureMode.DENSE_CLS_BROADCAST, FeatureMode.DENSE_PLUS_CLS):
            assert torch.allclose(
                fake_vit.extract_features(batch, feature_mode=mode)["pooled"], baseline
            )

    def test_mean_pooling_composes_with_a_dense_mode(self, fake_vit):
        features = fake_vit.extract_features(
            torch.rand(2, 3, 64, 64),
            pooling=Pooling.MEAN,
            feature_mode=FeatureMode.DENSE_PLUS_CLS,
        )
        assert features["pooled"].shape == (2, fake_vit.embed_dim)
        assert "cls" in features

    def test_cnn_cannot_use_a_cls_mode(self, fake_cnn):
        for mode in (FeatureMode.DENSE_CLS_BROADCAST, FeatureMode.DENSE_PLUS_CLS):
            with pytest.raises(ValueError, match="needs a CLS token"):
                fake_cnn.extract_features(torch.rand(1, 3, 64, 64), feature_mode=mode)

    def test_unknown_mode_raises(self, fake_vit):
        with pytest.raises(ValueError, match="Unknown feature_mode"):
            fake_vit.extract_features(torch.rand(1, 3, 64, 64), feature_mode="dense_and_vibes")
