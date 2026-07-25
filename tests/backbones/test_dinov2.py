"""DINOv2 against the real checkpoint.

Every test here downloads weights from torch.hub, so the whole module is marked
``slow`` and is skipped by default:

    pytest -m slow          # run these
    pytest -m "not slow"    # default in CI

The fake-backbone tests in test_base_backbone.py cover the *contract*; these
cover the claims that only real weights can settle — that the token grid lines
up with the patch size, that CLS and mean pooling actually differ, and that
DINOv2 features separate images the way a probe would need them to.
"""

import pytest
import torch
from PIL import Image

import visbench
from visbench.types import Pooling

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def dinov2():
    return visbench.get_backbone("dinov2_vits14", device="cpu")


def test_builds_through_public_api(dinov2):
    assert dinov2.name == "dinov2_vits14"
    assert dinov2.embed_dim == 384
    assert dinov2.patch_size == 14


def test_frozen_and_eval(dinov2):
    assert not dinov2.training
    assert all(not p.requires_grad for p in dinov2.parameters())


def test_grid_matches_patch_size(dinov2, solid_images):
    batch = dinov2.preprocess(solid_images[:2])
    assert batch.shape == (2, 3, 224, 224)

    features = dinov2.extract_features(batch)
    expected = 224 // 14
    assert features["grid_hw"] == (expected, expected)
    assert features["dense"].shape == (2, 384, expected, expected)
    assert features["pooled"].shape == (2, 384)


def test_cls_and_mean_differ(dinov2, solid_images):
    batch = dinov2.preprocess(solid_images[:1])
    cls = dinov2.extract_features(batch, pooling=Pooling.CLS)["pooled"]
    mean = dinov2.extract_features(batch, pooling=Pooling.MEAN)["pooled"]
    assert not torch.allclose(cls, mean)


def test_default_pooling_is_cls(dinov2, solid_images):
    batch = dinov2.preprocess(solid_images[:1])
    assert torch.allclose(
        dinov2.extract_features(batch)["pooled"],
        dinov2.extract_features(batch, pooling=Pooling.CLS)["pooled"],
    )


def test_features_are_deterministic(dinov2, solid_images):
    """Eval mode with no dropout: two passes must agree bit for bit."""
    batch = dinov2.preprocess(solid_images[:1])
    assert torch.equal(
        dinov2.extract_features(batch)["pooled"],
        dinov2.extract_features(batch)["pooled"],
    )


def test_distinct_images_give_distinct_features(dinov2, solid_images):
    batch = dinov2.preprocess(solid_images)
    pooled = torch.nn.functional.normalize(dinov2.extract_features(batch)["pooled"], dim=1)
    similarity = pooled @ pooled.T
    off_diagonal = similarity[~torch.eye(len(solid_images), dtype=torch.bool)]
    assert off_diagonal.max() < 0.999, "four different images collapsed to one feature"


def test_non_multiple_resolution_raises(dinov2):
    with pytest.raises(ValueError, match="multiple of patch size"):
        dinov2.extract_features(torch.rand(1, 3, 100, 100))


def test_image_size_must_divide_patch_size():
    from visbench.backbones.dinov2 import DINOv2

    with pytest.raises(ValueError, match="not a multiple"):
        DINOv2(variant="dinov2_vits14", image_size=100)


def test_unknown_variant_raises():
    from visbench.backbones.dinov2 import DINOv2

    with pytest.raises(ValueError, match="Unknown DINOv2 variant"):
        DINOv2(variant="dinov2_vitg14")


def test_cache_key_distinguishes_variants():
    """vits14 and vitb14 share a class; their cached features must not mix."""
    small = visbench.get_backbone("dinov2_vits14", device="cpu")
    base = visbench.get_backbone("dinov2_vitb14", device="cpu")
    assert small.cache_key() != base.cache_key()


def test_hub_ref_is_pinned(dinov2):
    """Unpinned weights would let upstream change what a cached feature means."""
    from visbench.backbones.dinov2 import HUB_REF

    assert len(HUB_REF) == 40, "expected a commit SHA; upstream publishes no tags"
    assert dinov2.hub_ref == HUB_REF


def test_cache_key_includes_the_weights_ref(dinov2):
    from visbench.backbones.dinov2 import HUB_REF

    assert HUB_REF[:12] in dinov2.cache_key()


def test_cache_key_changes_with_the_ref(dinov2, monkeypatch):
    """Bumping HUB_REF must invalidate every existing cache entry."""
    before = dinov2.cache_key()
    monkeypatch.setattr(dinov2, "hub_ref", "0" * 40)
    assert dinov2.cache_key() != before


def test_local_checkpoint_gets_its_own_cache_key(tmp_path, dinov2):
    """Local weights are not the pinned hub weights and must not share entries."""
    path = tmp_path / "weights.pth"
    torch.save(dinov2.model.state_dict(), path)

    local = visbench.get_backbone("dinov2_vits14", device="cpu", checkpoint=path)
    assert local.cache_key() != dinov2.cache_key()
    assert "local-" in local.cache_key()

    # Same architecture and same weights, so the features must still match.
    batch = dinov2.preprocess([Image.new("RGB", (64, 64), (120, 60, 30))])
    assert torch.allclose(
        local.extract_features(batch)["pooled"],
        dinov2.extract_features(batch)["pooled"],
    )


def test_end_to_end_through_the_cache(tmp_path, dinov2, solid_images):
    """The v0.1 promise: one forward pass per image per backbone."""
    from visbench.cache import FeatureCache

    cache = FeatureCache(root=tmp_path / "cache")
    first = cache.extract_dataset(dinov2, solid_images, batch_size=2)
    assert cache.stats()["misses"] == 4

    second = cache.extract_dataset(dinov2, solid_images, batch_size=2)
    assert cache.stats()["hits"] == 4
    assert torch.equal(first["pooled"], second["pooled"])
    assert torch.equal(first["dense"], second["dense"])


@pytest.mark.slow
class TestIntermediateLayers:
    """get_intermediate_layers takes the whole index list — one forward pass."""

    def test_one_map_per_block(self, dinov2, solid_images):
        batch = dinov2.preprocess(solid_images[:2])
        features = dinov2.extract_features(batch, layers=[2, 5, 8, 11])
        assert features["layer_indices"] == [2, 5, 8, 11]
        assert all(
            dense.shape == (2, dinov2.embed_dim, 16, 16) for dense in features["dense_layers"]
        )

    def test_blocks_share_a_width_and_grid(self, dinov2, solid_images):
        """Unlike a CNN's stages. This is why a ViT can use a scalar in_channels."""
        maps = dinov2.extract_features(dinov2.preprocess(solid_images[:1]), layers=[0, 11])[
            "dense_layers"
        ]
        assert maps[0].shape == maps[1].shape

    def test_depths_are_genuinely_different_features(self, dinov2, solid_images):
        """A shallow block and a deep one must not come back equal."""
        maps = dinov2.extract_features(dinov2.preprocess(solid_images[:1]), layers=[1, 11])[
            "dense_layers"
        ]
        assert not torch.allclose(maps[0], maps[1])

    def test_the_last_block_matches_the_default_call(self, dinov2, solid_images):
        batch = dinov2.preprocess(solid_images[:2])
        default = dinov2.extract_features(batch)
        multi = dinov2.extract_features(batch, layers=[3, 11])
        assert torch.equal(default["dense"], multi["dense"])
        assert torch.equal(default["pooled"], multi["pooled"])

    def test_num_layers_matches_the_model(self, dinov2):
        assert dinov2.num_layers == len(dinov2.model.blocks) == 12

    def test_out_of_range_block_is_caught(self, dinov2, solid_images):
        with pytest.raises(ValueError, match="out of range"):
            dinov2.extract_features(dinov2.preprocess(solid_images[:1]), layers=[12])
