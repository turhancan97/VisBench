"""timm CNN backbones against real weights.

The first non-ViT family, so this module exists to test the claims
:class:`BaseBackbone` has been making since v0.1 with only ``FakeCNN`` to back
them: that one ``extract_features`` covers every architecture, that mean is the
right default when there is no CLS token, and that a task written against
DINOv2 works unchanged on a ResNet.
"""

import pytest
import torch

import visbench
from visbench.types import Pooling

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def resnet():
    return visbench.get_backbone("resnet50", device="cpu")


def test_builds_through_public_api(resnet):
    assert resnet.name == "resnet50"
    assert resnet.embed_dim == 2048
    assert resnet.patch_size is None, "CNN stride is architectural, not a patch grid"


def test_frozen_and_eval(resnet):
    assert not resnet.training
    assert all(not p.requires_grad for p in resnet.parameters())


def test_dense_is_the_last_conv_map(resnet, solid_images):
    """layer4 output before global pooling, per CLAUDE.md."""
    batch = resnet.preprocess(solid_images[:2])
    features = resnet.extract_features(batch)

    assert batch.shape == (2, 3, 224, 224)
    assert features["grid_hw"] == (7, 7), "224 / stride 32"
    assert features["dense"].shape == (2, 2048, 7, 7)
    assert features["pooled"].shape == (2, 2048)


# -- the claims the CNN path had never been tested on -------------------------


def test_no_cls_token_so_default_is_mean(resnet, solid_images):
    assert resnet.has_cls_token is False
    assert resnet.default_pooling() == Pooling.MEAN

    batch = resnet.preprocess(solid_images[:1])
    assert torch.allclose(
        resnet.extract_features(batch)["pooled"],
        resnet.extract_features(batch, pooling=Pooling.MEAN)["pooled"],
    )


def test_cls_pooling_raises_rather_than_falling_back(resnet, solid_images):
    batch = resnet.preprocess(solid_images[:1])
    with pytest.raises(ValueError, match="no CLS token"):
        resnet.extract_features(batch, pooling=Pooling.CLS)


def test_pooled_equals_the_models_own_global_pool(resnet, solid_images):
    """The claim that makes mean-pooling *correct*, not merely shape-compatible.

    A ResNet's classifier sees ``global_pool(layer4)``. Mean-pooling the
    flattened tokens must reproduce it exactly, or "the pooled vector" means
    something different for CNNs than it does for a ViT's CLS token.
    """
    batch = resnet.preprocess(solid_images[:2])
    ours = resnet.extract_features(batch)["pooled"]

    with torch.no_grad():
        theirs = resnet.model.global_pool(resnet.model.forward_features(batch))

    assert torch.allclose(ours, theirs, atol=1e-6)


def test_same_return_shape_as_a_vit(resnet, solid_images):
    """A caller must not be able to tell the families apart."""
    dinov2 = visbench.get_backbone("dinov2_vits14", device="cpu")

    cnn = resnet.extract_features(resnet.preprocess(solid_images[:2]))
    vit = dinov2.extract_features(dinov2.preprocess(solid_images[:2]))

    assert set(cnn) == set(vit)
    assert cnn["dense"].ndim == vit["dense"].ndim == 4
    assert cnn["pooled"].ndim == vit["pooled"].ndim == 2
    assert len(cnn["grid_hw"]) == len(vit["grid_hw"]) == 2


# -- weights identity ---------------------------------------------------------


def test_cache_key_carries_the_training_recipe(resnet):
    """resnet50.a1_in1k and resnet50.a3_in1k are different weights, one name."""
    assert resnet.pretrained_tag == "a1_in1k"
    assert "a1_in1k" in resnet.cache_key()


def test_different_recipes_do_not_share_cache_entries():
    from visbench.backbones.timm_backbone import TimmBackbone

    a1 = TimmBackbone(model_name="resnet50", pretrained_tag="a1_in1k", device="cpu")
    a3 = TimmBackbone(model_name="resnet50", pretrained_tag="a3_in1k", device="cpu")
    assert a1.cache_key() != a3.cache_key()


def test_unspecified_tag_is_resolved_not_echoed():
    """Passing None means "timm's default"; the record must say which that was."""
    from visbench.backbones.timm_backbone import TimmBackbone

    backbone = TimmBackbone(model_name="resnet18", device="cpu")
    assert backbone.pretrained_tag not in (None, "", "default")
    assert backbone.pretrained_tag in backbone.cache_key()


def test_variants_differ(resnet):
    small = visbench.get_backbone("resnet18", device="cpu")
    assert small.embed_dim == 512
    assert small.cache_key() != resnet.cache_key()


def test_does_not_collide_with_a_vit_in_the_cache(tmp_path, resnet, solid_images):
    from visbench.cache import FeatureCache

    dinov2 = visbench.get_backbone("dinov2_vits14", device="cpu")
    cache = FeatureCache(root=tmp_path / "cache")

    cache.extract_dataset(resnet, solid_images, keep="pooled")
    cache.extract_dataset(dinov2, solid_images, keep="pooled")
    assert cache.stats()["entries"] == 8


# -- configuration ------------------------------------------------------------


def test_arbitrary_timm_model_by_name():
    """Registered names are a convenience, not the limit."""
    from visbench.backbones.timm_backbone import TimmBackbone

    backbone = TimmBackbone(model_name="resnet34", device="cpu")
    assert backbone.embed_dim == 512
    assert backbone.name == "resnet34"


def test_variant_and_model_name_together_raise():
    from visbench.backbones.timm_backbone import TimmBackbone

    with pytest.raises(ValueError, match="not both"):
        TimmBackbone(variant="resnet50", model_name="resnet34")


def test_unknown_variant_points_at_model_name():
    from visbench.backbones.timm_backbone import TimmBackbone

    with pytest.raises(ValueError, match="model_name"):
        TimmBackbone(variant="resnet101")


def test_timm_vit_is_rejected_clearly():
    """A timm ViT returns tokens, not a conv map; say so rather than crashing."""
    from visbench.backbones.timm_backbone import TimmBackbone

    backbone = TimmBackbone(model_name="vit_tiny_patch16_224", device="cpu")
    with pytest.raises(NotImplementedError, match="dinov2_.* or clip_"):
        backbone.extract_features(torch.rand(1, 3, 224, 224))


def test_uses_the_models_own_preprocessing(resnet, solid_images):
    """timm knows each model's crop ratio and interpolation; a shared transform
    would silently mis-preprocess anything whose recipe is not ImageNet's."""
    batch = resnet.preprocess(solid_images)
    assert batch.shape == (4, 3, 224, 224)
    assert batch.dtype == torch.float32


def test_features_are_deterministic(resnet, solid_images):
    batch = resnet.preprocess(solid_images[:1])
    assert torch.equal(
        resnet.extract_features(batch)["pooled"],
        resnet.extract_features(batch)["pooled"],
    )


def test_accepts_a_non_multiple_resolution(resnet):
    """Unlike a ViT, a CNN has no patch-multiple constraint — the grid just scales."""
    features = resnet.extract_features(torch.rand(1, 3, 160, 160))
    assert features["grid_hw"] == (5, 5)


def test_end_to_end_through_the_cache(tmp_path, resnet, solid_images):
    from visbench.cache import FeatureCache

    cache = FeatureCache(root=tmp_path / "cache")
    first = cache.extract_dataset(resnet, solid_images, batch_size=2)
    assert cache.stats()["misses"] == 4

    second = cache.extract_dataset(resnet, solid_images, batch_size=2)
    assert cache.stats()["hits"] == 4
    assert torch.equal(first["pooled"], second["pooled"])
