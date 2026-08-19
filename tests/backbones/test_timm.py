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


def test_an_unregistered_timm_vit_works_by_name():
    """timm ViTs were refused outright until v0.9; now they are read.

    The refusal was right for what the class then was: ``has_cls_token`` and
    ``patch_size`` were *class* attributes declaring "CNN" for everything, and
    once ``forward_intermediates`` is asked for NCHW a ViT's tokens are reshaped
    into a grid — from that point indistinguishable from a conv map, with the
    CLS token dropped and the record claiming there was none.

    What replaced it is reading both per instance, so any timm ViT is usable and
    describes itself honestly, not only the three registered here.
    """
    from visbench.backbones.timm_backbone import TimmBackbone

    backbone = TimmBackbone(model_name="vit_tiny_patch16_224", device="cpu")
    assert backbone.is_transformer is True
    assert backbone.has_cls_token is True
    assert backbone.patch_size == 16


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


@pytest.mark.slow
class TestResNetStages:
    """A CNN's stages differ in width and stride — the case a ViT never exercises."""

    def test_stage_shapes_differ(self, resnet):
        features = resnet.extract_features(torch.rand(2, 3, 224, 224), layers=[1, 2, 3, 4])
        maps = features["dense_layers"]
        assert [dense.shape[1] for dense in maps] == resnet.layer_channels([1, 2, 3, 4])
        assert [tuple(dense.shape[-2:]) for dense in maps] == [(56, 56), (28, 28), (14, 14), (7, 7)]

    def test_the_last_stage_matches_the_default_call(self, resnet):
        """Multi-layer extraction must not change what a single-layer run means."""
        default = resnet.extract_features(torch.rand(2, 3, 224, 224).mul(0).add(0.5))
        image = torch.rand(2, 3, 224, 224).mul(0).add(0.5)
        multi = resnet.extract_features(image, layers=[2, 4])
        assert torch.allclose(default["dense"], multi["dense"], atol=1e-6)
        assert torch.allclose(default["pooled"], multi["pooled"], atol=1e-6)

    def test_layer_channels_reports_real_widths(self, resnet):
        assert resnet.layer_channels([4]) == [resnet.embed_dim]

    def test_out_of_range_stage_is_caught(self, resnet):
        with pytest.raises(ValueError, match="out of range"):
            resnet.extract_features(torch.rand(1, 3, 224, 224), layers=[0, 9])

    def test_stages_feed_a_dpt_head(self, resnet):
        """The point of all of it: a ResNet pyramid into a multiscale head."""
        from visbench.heads import DPTHead

        features = resnet.extract_features(torch.rand(2, 3, 224, 224), layers=[1, 2, 3, 4])
        head = DPTHead(
            in_channels=resnet.layer_channels([1, 2, 3, 4]),
            out_channels=1,
            num_layers=4,
            hidden_dim=32,
            output_size=224,
        )
        assert head(features["dense_layers"]).shape == (2, 1, 224, 224)


# -- transformers, and the head-equality question -----------------------------
#
# This class refused timm ViTs outright until v0.9, on the grounds that
# `has_cls_token = False` would be a lie for one. These pin the claims that
# replaced the refusal.


@pytest.fixture(scope="module")
def mae():
    return visbench.get_backbone("mae_vitb16", device="cpu")


@pytest.fixture(scope="module")
def siglip():
    return visbench.get_backbone("siglip_vitb16", device="cpu")


@pytest.fixture(scope="module")
def supervised_vit():
    return visbench.get_backbone("supervised_vitb16", device="cpu")


@pytest.fixture(scope="module")
def dino_vit():
    return visbench.get_backbone("dino_vitb16", device="cpu")


def _normalisation(backbone):
    """``(mean, std)`` from the transform timm resolved for this checkpoint.

    Matched on the class *name* rather than by importing torchvision, which is
    a transitive dependency here and not a declared one.
    """
    for op in backbone._transform.transforms:
        if type(op).__name__ == "Normalize":
            return tuple(float(v) for v in op.mean), tuple(float(v) for v in op.std)
    raise AssertionError(f"{backbone} has no Normalize step in its transform")


class TestTransformers:
    def test_mae_reports_its_cls_token_and_patch_grid(self, mae):
        assert mae.has_cls_token is True
        assert mae.patch_size == 16
        assert mae.embed_dim == 768

    def test_siglip_gap_has_no_cls_token_but_still_has_a_patch_grid(self, siglip):
        """The case the old class attributes could not express.

        `has_cls_token = False` was right for this model and `patch_size = None`
        was wrong, and both were the same hard-coded pair.
        """
        assert siglip.has_cls_token is False
        assert siglip.patch_size == 16

    def test_the_model_decides_the_default_pooling(self, mae, siglip):
        """MAE pools by its CLS token, SigLIP-GAP by average — timm says so.

        Not inferred from `has_cls_token`, which is only a proxy: it happens to
        agree for both of these, and reading `global_pool` is what makes it a
        statement about the checkpoint rather than a coincidence.
        """
        assert mae.default_pooling() == Pooling.CLS
        assert siglip.default_pooling() == Pooling.MEAN

    def test_dense_features_are_the_patch_grid(self, mae, siglip, solid_images):
        for backbone in (mae, siglip):
            features = backbone.extract_features(backbone.preprocess(solid_images))
            assert features["grid_hw"] == (14, 14)
            assert features["dense"].shape[1] == 768

    def test_the_cls_token_is_kept_not_discarded(self, mae, solid_images):
        """The failure the old refusal existed to prevent."""
        features = mae.extract_features(mae.preprocess(solid_images), feature_mode="dense_plus_cls")
        assert features["cls"] is not None
        assert features["cls"].shape[-1] == 768

    def test_supervised_vit_is_mae_with_a_different_objective(
        self, mae, supervised_vit, solid_images
    ):
        """The claim the `supervised_vitb16` entry rests on, as a test.

        Both are `vit_base_patch16_224` pretrained on ImageNet-1k. Everything a
        record and a comparability group can see about the two is identical --
        architecture, width, depth, patch grid, prefix tokens, default pooling
        -- so the *only* variable between those two rows of any board is
        supervised labels against masked pixel reconstruction. That is what
        makes the pair worth measuring, and it is why the registered tag is
        `augreg_in1k`: a 21k recipe would move the pretraining data too.
        """
        assert supervised_vit.has_cls_token == mae.has_cls_token is True
        assert supervised_vit.patch_size == mae.patch_size == 16
        assert supervised_vit.embed_dim == mae.embed_dim == 768
        assert supervised_vit.default_pooling() == mae.default_pooling() == Pooling.CLS

        pixels = supervised_vit.preprocess(solid_images)
        assert (
            supervised_vit.extract_features(pixels)["grid_hw"]
            == mae.extract_features(mae.preprocess(solid_images))["grid_hw"]
            == (14, 14)
        )

    def test_supervised_vit_is_not_mae(self, mae, supervised_vit, solid_images):
        """Different weights, which nothing above would notice.

        Every structural assertion in the test above passes if both names
        resolve to the same checkpoint -- which is one typo in `_VARIANTS`
        away, and would put the same numbers on the board twice under two
        names. The cache key is what separates them, so assert on it and on
        the features it keys.
        """
        assert supervised_vit.cache_key() != mae.cache_key()

        pixels = supervised_vit.preprocess(solid_images)
        assert not torch.allclose(
            supervised_vit.extract_features(pixels)["pooled"],
            mae.extract_features(pixels)["pooled"],
            atol=1e-3,
        )

    def test_dino_completes_the_objective_family(self, mae, supervised_vit, dino_vit):
        """Three objectives on one architecture and one pretraining set.

        `supervised_vitb16` made the objective a controlled variable across one
        pair; this makes it three-valued -- supervised labels, masked pixel
        reconstruction, self-distillation -- which is what separates "trained
        with labels" from "trained toward semantics". Everything a record or a
        comparability group can see about the three is identical, so a board
        that ranks them differently is ranking objectives.
        """
        for other in (mae, supervised_vit):
            assert dino_vit.has_cls_token == other.has_cls_token is True
            assert dino_vit.patch_size == other.patch_size == 16
            assert dino_vit.embed_dim == other.embed_dim == 768
            assert dino_vit.default_pooling() == other.default_pooling() == Pooling.CLS

    def test_dino_holds_the_input_normalisation_fixed_against_mae(self, mae, dino_vit):
        """The half of the comparison that is *tighter* than the existing pair.

        Each checkpoint is preprocessed with the statistics it was trained
        under -- the only correct handling, and what `resolve_data_config`
        does -- so `supervised_vitb16` (mean/std 0.5) differs from MAE in its
        input normalisation as well as its objective. DINO and MAE share
        ImageNet statistics, so that pair varies the objective alone.

        Asserted on the transform rather than argued in a comment, because the
        claim is about two checkpoints' metadata and a future timm release is
        free to move it.
        """
        assert _normalisation(dino_vit) == _normalisation(mae)

    def test_dino_is_not_mae_or_the_supervised_vit(
        self, mae, supervised_vit, dino_vit, solid_images
    ):
        """Different weights, which nothing structural would notice.

        Every assertion in `test_dino_completes_the_objective_family` passes if
        two names resolve to one checkpoint, which is a typo away and would put
        the same numbers on a board twice under different names.
        """
        pixels = dino_vit.preprocess(solid_images)
        pooled = dino_vit.extract_features(pixels)["pooled"]

        for other in (mae, supervised_vit):
            assert dino_vit.cache_key() != other.cache_key()
            assert not torch.allclose(pooled, other.extract_features(pixels)["pooled"], atol=1e-3)

    def test_multi_layer_costs_one_forward_pass(self, mae, solid_images):
        features = mae.extract_features(mae.preprocess(solid_images), layers=[2, 5, 8, 11])
        assert len(features["dense_layers"]) == 4
        assert features["layer_indices"] == [2, 5, 8, 11]


class TestPooledMatchesTheModelsOwnHead:
    """Which backbones' pooled vector is what the model hands its classifier.

    The claim this module has made since v0.2, now that it covers seven models
    and one of them breaks it. Pinned in both directions so the exception
    cannot quietly become the rule, or be quietly fixed into one.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "resnet18",
            "resnet50",
            "mae_vitb16",
            "siglip_vitb16",
            "supervised_vitb16",
            "dino_vitb16",
        ],
    )
    def test_it_matches(self, name, solid_images):
        backbone = visbench.get_backbone(name, device="cpu")
        pixels = backbone.preprocess(solid_images)
        with torch.no_grad():
            theirs = backbone.model(pixels)
        assert torch.allclose(backbone.extract_features(pixels)["pooled"], theirs, atol=1e-4)

    def test_convnext_is_the_documented_exception(self, solid_images):
        """Its head is `avg -> LayerNorm2d`, so the model pools *then* norms.

        There is no way to satisfy both invariants: LayerNorm across channels
        does not commute with a spatial mean. The one kept is that `pooled` is
        always a reduction of `dense`, because that is what the cache stores and
        what every pooling task reduces.
        """
        backbone = visbench.get_backbone("convnext_base", device="cpu")
        pixels = backbone.preprocess(solid_images)
        features = backbone.extract_features(pixels)
        with torch.no_grad():
            theirs = backbone.model(pixels)

        assert not torch.allclose(features["pooled"], theirs, atol=1e-2)
        # The invariant that *is* kept: pooled is the mean of dense.
        assert torch.allclose(features["pooled"], features["dense"].flatten(2).mean(-1), atol=1e-5)
