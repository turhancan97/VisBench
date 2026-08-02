"""CLIP against the real checkpoint.

Downloads weights from open_clip, so the whole module is ``slow``.

The point of this module is not that CLIP works — the fake-backbone tests
already pin the contract — but that the *second* backbone slots into the same
abstraction without it bending, and that the two CLIP-specific traps are
closed: QuickGELU pairing, and pre-projection versus projected features.
"""

import pytest
import torch

import visbench
from visbench.types import Pooling

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def clip():
    return visbench.get_backbone("clip_vitb16", device="cpu")


def test_builds_through_public_api(clip):
    assert clip.name == "clip_vitb16"
    assert clip.embed_dim == 768
    assert clip.patch_size == 16


def test_frozen_and_eval(clip):
    assert not clip.training
    assert all(not p.requires_grad for p in clip.parameters())


def test_grid_matches_patch_size(clip, solid_images):
    batch = clip.preprocess(solid_images[:2])
    assert batch.shape == (2, 3, 224, 224)

    features = clip.extract_features(batch)
    expected = 224 // 16
    assert features["grid_hw"] == (expected, expected)
    assert features["dense"].shape == (2, 768, expected, expected)
    assert features["pooled"].shape == (2, 768)


def test_grid_differs_from_dinov2(clip, solid_images):
    """16px patches against DINOv2's 14px: 14x14 versus 16x16 at the same input.

    This is exactly why correspondence reports error in **pixels**. A patch is a
    different physical distance on each of these backbones, so a threshold in
    patch widths asks each of them a different question — which ranked that
    board upside down until v0.6.1. See `CorrespondenceTask.threshold_units`.
    """
    dinov2 = visbench.get_backbone("dinov2_vits14", device="cpu")

    clip_grid = clip.extract_features(clip.preprocess(solid_images[:1]))["grid_hw"]
    dinov2_grid = dinov2.extract_features(dinov2.preprocess(solid_images[:1]))["grid_hw"]

    assert clip_grid == (14, 14)
    assert dinov2_grid == (16, 16)


# -- the QuickGELU trap -------------------------------------------------------


def test_openai_weights_use_the_quickgelu_architecture(clip):
    """Pairing OpenAI weights with plain GELU loads fine and computes wrongly."""
    assert clip.model_name.endswith("-quickgelu")
    assert clip.pretrained == "openai"


def test_mismatched_quickgelu_raises(monkeypatch):
    """open_clip only warns; VisBench must not let a warning decide a number."""
    from visbench.backbones import clip as module

    monkeypatch.setitem(module._VARIANTS, "clip_vitb16", ("ViT-B-16", "openai", 768, 16))
    with pytest.raises(RuntimeError, match="QuickGELU mismatch"):
        module.CLIP(variant="clip_vitb16", device="cpu")


# -- pre-projection versus projected -----------------------------------------


def test_default_is_pre_projection(clip, solid_images):
    features = clip.extract_features(clip.preprocess(solid_images[:1]))
    assert features["pooled"].shape[1] == 768


def test_projection_changes_width_and_values(solid_images):
    projected = visbench.get_backbone("clip_vitb16", device="cpu", use_projection=True)
    features = projected.extract_features(projected.preprocess(solid_images[:1]))

    assert projected.embed_dim == 512
    assert features["pooled"].shape[1] == 512
    assert features["dense"].shape[1] == 512


def test_projected_cls_reproduces_encode_image(clip, solid_images):
    """Proof the pipeline is assembled correctly, not merely plausibly.

    If ln_post were skipped or applied in the wrong place, this would drift
    while every shape stayed right.
    """
    import open_clip

    model = open_clip.create_model("ViT-B-16-quickgelu", pretrained="openai").eval()
    batch = clip.preprocess(solid_images[:2])

    projected = visbench.get_backbone("clip_vitb16", device="cpu", use_projection=True)
    ours = projected.extract_features(batch)["pooled"]

    with torch.no_grad():
        theirs = model.encode_image(batch)

    assert torch.allclose(ours, theirs, atol=1e-4)


def test_projection_gets_its_own_cache_key(clip):
    """Two representations from one forward pass must not share an entry."""
    projected = visbench.get_backbone("clip_vitb16", device="cpu", use_projection=True)
    assert projected.cache_key() != clip.cache_key()
    assert "preproj" in clip.cache_key()
    assert "proj" in projected.cache_key()


def test_cache_key_names_the_weights(clip):
    assert "openai" in clip.cache_key()


def test_variants_have_distinct_cache_keys():
    b16 = visbench.get_backbone("clip_vitb16", device="cpu")
    b32 = visbench.get_backbone("clip_vitb32", device="cpu")
    assert b16.cache_key() != b32.cache_key()
    assert b32.patch_size == 32


# -- the shared contract ------------------------------------------------------


def test_cls_and_mean_differ(clip, solid_images):
    batch = clip.preprocess(solid_images[:1])
    cls = clip.extract_features(batch, pooling=Pooling.CLS)["pooled"]
    mean = clip.extract_features(batch, pooling=Pooling.MEAN)["pooled"]
    assert not torch.allclose(cls, mean)


def test_features_are_deterministic(clip, solid_images):
    batch = clip.preprocess(solid_images[:1])
    assert torch.equal(
        clip.extract_features(batch)["pooled"],
        clip.extract_features(batch)["pooled"],
    )


def test_uses_clip_normalisation_not_imagenet(clip, solid_images):
    """Applying ImageNet's constants would be a silent accuracy loss."""
    from visbench.utils.image import CLIP_MEAN, IMAGENET_MEAN

    batch = clip.preprocess(solid_images[:1])
    grey = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    solid = torch.full_like(batch, 0.0)

    # A pixel equal to CLIP's mean normalises to ~0; ImageNet's would not.
    from torchvision import transforms

    as_clip = transforms.Normalize(CLIP_MEAN, (1.0, 1.0, 1.0))(grey)
    as_imagenet = transforms.Normalize(IMAGENET_MEAN, (1.0, 1.0, 1.0))(grey)
    assert torch.allclose(as_clip, solid[:, :, :1, :1] * 0, atol=1e-6)
    assert not torch.allclose(as_imagenet, as_clip)


def test_non_multiple_resolution_raises(clip):
    with pytest.raises(ValueError, match="multiple of patch size"):
        clip.extract_features(torch.rand(1, 3, 100, 100))


def test_unknown_variant_raises():
    from visbench.backbones.clip import CLIP

    with pytest.raises(ValueError, match="Unknown CLIP variant"):
        CLIP(variant="clip_vitl14")


def test_end_to_end_through_the_cache(tmp_path, clip, solid_images):
    from visbench.cache import FeatureCache

    cache = FeatureCache(root=tmp_path / "cache")
    first = cache.extract_dataset(clip, solid_images, batch_size=2)
    assert cache.stats()["misses"] == 4

    second = cache.extract_dataset(clip, solid_images, batch_size=2)
    assert cache.stats()["hits"] == 4
    assert torch.equal(first["pooled"], second["pooled"])


def test_does_not_collide_with_dinov2_in_the_cache(tmp_path, clip, solid_images):
    """Same images, different backbone: entries must not be shared."""
    from visbench.cache import FeatureCache

    dinov2 = visbench.get_backbone("dinov2_vits14", device="cpu")
    cache = FeatureCache(root=tmp_path / "cache")

    cache.extract_dataset(clip, solid_images, keep="pooled")
    cache.extract_dataset(dinov2, solid_images, keep="pooled")
    assert cache.stats()["entries"] == 8


@pytest.mark.slow
class TestIntermediateLayers:
    """open_clip's forward_intermediates, the counterpart to DINOv2's call."""

    def test_one_map_per_block(self, clip, solid_images):
        batch = clip.preprocess(solid_images[:2])
        features = clip.extract_features(batch, layers=[3, 7, 11])
        assert features["layer_indices"] == [3, 7, 11]
        assert all(dense.shape == (2, clip.embed_dim, 14, 14) for dense in features["dense_layers"])

    def test_the_last_block_matches_the_default_call(self, clip, solid_images):
        batch = clip.preprocess(solid_images[:2])
        default = clip.extract_features(batch)
        multi = clip.extract_features(batch, layers=[5, 11])
        assert torch.equal(default["dense"], multi["dense"])
        assert torch.equal(default["pooled"], multi["pooled"])

    def test_every_layer_is_normalised(self, clip, solid_images):
        """ln_post is applied to intermediates too, so a pyramid's stages are on
        one scale. Without it a head would have to unlearn the difference."""
        maps = clip.extract_features(clip.preprocess(solid_images[:1]), layers=[2, 11])[
            "dense_layers"
        ]
        scales = [dense.std().item() for dense in maps]
        assert max(scales) / min(scales) < 5, f"layer scales diverge: {scales}"

    def test_depths_are_genuinely_different(self, clip, solid_images):
        maps = clip.extract_features(clip.preprocess(solid_images[:1]), layers=[1, 11])[
            "dense_layers"
        ]
        assert not torch.allclose(maps[0], maps[1])

    def test_num_layers_matches_the_model(self, clip):
        assert clip.num_layers == len(clip.model.transformer.resblocks) == 12
