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
