"""Wrapping an arbitrary nn.Module.

No downloads: the point of this backbone is that it takes whatever the user
already has, so small hand-built modules exercise it exactly as a real one
would.

The theme throughout is refusing to guess. A custom backbone has no upstream
ref, no known normalisation and no declared patch grid, and every one of those
is a place where guessing produces plausible numbers that are wrong.
"""

import pytest
import torch
import torch.nn as nn
from PIL import Image

import visbench
from visbench.backbones.custom import CustomBackbone, hash_weights
from visbench.types import Pooling


def preprocess(image: Image.Image) -> torch.Tensor:
    array = torch.frombuffer(
        bytearray(image.convert("RGB").resize((32, 32)).tobytes()), dtype=torch.uint8
    )
    return array.view(32, 32, 3).permute(2, 0, 1).float() / 255


@pytest.fixture
def images():
    return [Image.new("RGB", (64, 64), (i * 60, 20, 20)) for i in range(4)]


class ConvNet(nn.Module):
    """Returns a (B, C, H, W) conv map — the unambiguous case."""

    def __init__(self, width: int = 12):
        super().__init__()
        self.conv = nn.Conv2d(3, width, kernel_size=8, stride=8)

    def forward(self, x):
        return self.conv(x)


class TokenNet(nn.Module):
    """Returns a (B, N, C) sequence, optionally with a leading CLS token."""

    def __init__(self, width: int = 12, patch: int = 8, cls: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(3, width, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.randn(1, 1, width)) if cls else None

    def forward(self, x):
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        if self.cls is None:
            return tokens
        return torch.cat([self.cls.expand(len(tokens), -1, -1), tokens], dim=1)


# -- conv map -----------------------------------------------------------------


def test_conv_map_is_understood(images):
    backbone = CustomBackbone(ConvNet(), preprocess=preprocess, device="cpu")
    features = backbone.extract_features(backbone.preprocess(images))

    assert features["grid_hw"] == (4, 4)
    assert features["dense"].shape == (4, 12, 4, 4)
    assert features["pooled"].shape == (4, 12)


def test_conv_map_has_no_cls_so_default_is_mean(images):
    backbone = CustomBackbone(ConvNet(), preprocess=preprocess, device="cpu")
    assert backbone.default_pooling() == Pooling.MEAN

    with pytest.raises(ValueError, match="no CLS token"):
        backbone.extract_features(backbone.preprocess(images), pooling=Pooling.CLS)


def test_embed_dim_is_filled_in_from_the_first_pass(images):
    """A record saying the feature width is 0 is worse than one saying 12."""
    backbone = CustomBackbone(ConvNet(), preprocess=preprocess, device="cpu")
    assert backbone.embed_dim == 0

    backbone.extract_features(backbone.preprocess(images))
    assert backbone.embed_dim == 12


# -- token sequences ----------------------------------------------------------


def test_tokens_with_cls(images):
    backbone = CustomBackbone(
        TokenNet(cls=True), preprocess=preprocess, has_cls_token=True, device="cpu"
    )
    features = backbone.extract_features(backbone.preprocess(images))

    assert features["grid_hw"] == (4, 4)
    assert features["dense"].shape == (4, 12, 4, 4), "CLS must be stripped before reshaping"
    assert backbone.default_pooling() == Pooling.CLS


def test_tokens_without_cls(images):
    backbone = CustomBackbone(
        TokenNet(cls=False), preprocess=preprocess, has_cls_token=False, device="cpu"
    )
    assert backbone.extract_features(backbone.preprocess(images))["grid_hw"] == (4, 4)


def test_wrong_cls_declaration_is_caught(images):
    """Declaring a CLS token that is not there shifts every patch by one.

    A square grid still comes out — 15 tokens is not square, so this one is
    caught; the patch_size path catches it exactly.
    """
    backbone = CustomBackbone(
        TokenNet(cls=False), preprocess=preprocess, has_cls_token=True, patch_size=8, device="cpu"
    )
    with pytest.raises(ValueError, match="Check has_cls_token"):
        backbone.extract_features(backbone.preprocess(images))


def test_patch_size_derives_the_grid_rather_than_guessing(images):
    backbone = CustomBackbone(
        TokenNet(cls=True),
        preprocess=preprocess,
        has_cls_token=True,
        patch_size=8,
        device="cpu",
    )
    assert backbone.extract_features(backbone.preprocess(images))["grid_hw"] == (4, 4)


def test_non_square_token_count_raises(images):
    """N alone does not determine a grid; say so rather than misalign it."""

    class Ragged(nn.Module):
        def forward(self, x):
            return torch.randn(len(x), 15, 8)

    backbone = CustomBackbone(Ragged(), preprocess=preprocess, device="cpu")
    with pytest.raises(ValueError, match="not a square grid"):
        backbone.extract_features(backbone.preprocess(images))


def test_non_square_input_with_assumed_grid_raises():
    """The dangerous case: a square token *count* from a non-square layout.

    32x128 at patch 8 is a 4x16 grid — 64 tokens, which is 8 squared. Guessing
    8x8 would succeed silently and put every patch in the wrong place, so the
    input shape is checked rather than trusted.
    """
    backbone = CustomBackbone(TokenNet(cls=False), preprocess=preprocess, device="cpu")
    with pytest.raises(ValueError, match="square token grid was assumed"):
        backbone.extract_features(torch.rand(1, 3, 32, 128))


# -- the escape hatch ---------------------------------------------------------


def test_feature_fn_takes_full_control(images):
    """For anything the shape rules cannot read."""

    class Odd(nn.Module):
        def forward(self, x):
            return {"weird": torch.randn(len(x), 16, 6)}

    def extract(module, batch):
        tokens = module(batch)["weird"]
        return tokens, None, (4, 4)

    backbone = CustomBackbone(Odd(), preprocess=preprocess, feature_fn=extract, device="cpu")
    assert backbone.extract_features(backbone.preprocess(images))["dense"].shape == (4, 6, 4, 4)


def test_unreadable_output_raises_pointing_at_feature_fn(images):
    class Odd(nn.Module):
        def forward(self, x):
            return {"weird": torch.randn(len(x), 16, 6)}

    backbone = CustomBackbone(Odd(), preprocess=preprocess, device="cpu")
    with pytest.raises(TypeError, match="feature_fn"):
        backbone.extract_features(backbone.preprocess(images))


def test_five_dimensional_output_raises(images):
    class Video(nn.Module):
        def forward(self, x):
            return torch.randn(len(x), 8, 2, 4, 4)

    backbone = CustomBackbone(Video(), preprocess=preprocess, device="cpu")
    with pytest.raises(ValueError, match="5D tensor"):
        backbone.extract_features(backbone.preprocess(images))


def test_tuple_output_takes_the_first_element(images):
    class Pair(nn.Module):
        def forward(self, x):
            return torch.randn(len(x), 6, 4, 4), "auxiliary"

    backbone = CustomBackbone(Pair(), preprocess=preprocess, device="cpu")
    assert backbone.extract_features(backbone.preprocess(images))["grid_hw"] == (4, 4)


# -- weights identity ---------------------------------------------------------


def test_weights_hash_is_stable():
    module = ConvNet()
    assert hash_weights(module) == hash_weights(module)


def test_different_weights_hash_differently():
    torch.manual_seed(0)
    a = ConvNet()
    torch.manual_seed(1)
    b = ConvNet()
    assert hash_weights(a) != hash_weights(b)


def test_fine_tuning_changes_the_cache_key():
    """The failure a custom backbone is most exposed to: no upstream ref, so
    a tuned checkpoint would otherwise reuse its parent's cached features."""
    module = ConvNet()
    before = CustomBackbone(module, preprocess=preprocess, device="cpu").cache_key()

    with torch.no_grad():
        module.conv.weight.add_(0.5)
    after = CustomBackbone(module, preprocess=preprocess, device="cpu").cache_key()

    assert before != after


def test_architecture_change_is_caught():
    narrow = CustomBackbone(ConvNet(width=8), preprocess=preprocess, device="cpu")
    wide = CustomBackbone(ConvNet(width=16), preprocess=preprocess, device="cpu")
    assert narrow.cache_key() != wide.cache_key()


def test_explicit_weights_id_is_respected():
    backbone = CustomBackbone(ConvNet(), preprocess=preprocess, weights_id="paper-v3", device="cpu")
    assert "paper-v3" in backbone.cache_key()


def test_name_reaches_the_cache_key():
    backbone = CustomBackbone(ConvNet(), preprocess=preprocess, name="mine", device="cpu")
    assert "mine" in backbone.cache_key()


# -- contract -----------------------------------------------------------------


def test_module_is_frozen_and_eval():
    backbone = CustomBackbone(ConvNet(), preprocess=preprocess, device="cpu")
    assert not backbone.training
    assert all(not p.requires_grad for p in backbone.parameters())


def test_non_module_raises():
    with pytest.raises(TypeError, match="nn.Module"):
        CustomBackbone(object(), preprocess=preprocess)


def test_missing_preprocess_raises():
    with pytest.raises(TypeError, match="normalisation"):
        CustomBackbone(ConvNet(), preprocess=None)


def test_works_with_a_task_and_the_cache(tmp_path, images):
    """The whole point: an arbitrary module probed by the built-in tasks."""
    from visbench.cache import FeatureCache

    backbone = CustomBackbone(ConvNet(), preprocess=preprocess, name="mine", device="cpu")
    cache = FeatureCache(root=tmp_path / "cache")

    features = cache.extract_dataset(backbone, images, keep="pooled")
    probe = visbench.get_probe("retrieval")
    metrics = probe.evaluate(features, [0, 0, 1, 1])

    assert "recall@1" in metrics
    assert cache.stats()["entries"] == 4


def test_registering_a_subclass_still_works():
    """The documented path for giving a custom backbone a registry name."""
    from visbench.backbones.base import BaseBackbone

    @visbench.register_backbone("test_only_backbone")
    class Mine(BaseBackbone):
        def __init__(self):
            super().__init__("cpu")
            self.name = "test_only_backbone"
            self._finalize()

        def _forward_features(self, image, layers):
            return [(torch.randn(len(image), 16, 4), None, (4, 4))]

        def preprocess(self, images):
            raise NotImplementedError

        def cache_key(self):
            return "test/only"

    assert "test_only_backbone" in visbench.list_backbones()
    # Clean up so the registry does not leak into other tests.
    from visbench import registry

    del registry._BACKBONES["test_only_backbone"]


class TestCustomMultiLayer:
    """VisBench cannot tap an arbitrary module's intermediates, so the user says how."""

    def test_a_plain_module_exposes_one_layer(self):
        backbone = CustomBackbone(ConvNet(), preprocess=preprocess, device="cpu")
        assert backbone.num_layers == 1

    def test_multi_layer_request_on_a_plain_module_is_refused(self, images):
        """Better than returning the final map several times, which would let a
        multiscale head report a single-layer result."""
        backbone = CustomBackbone(ConvNet(), preprocess=preprocess, device="cpu")
        with pytest.raises(ValueError, match="out of range"):
            backbone.extract_features(backbone.preprocess(images), layers=[0, 1])

    def test_promising_layers_without_a_function_is_refused(self):
        with pytest.raises(ValueError, match="layer_feature_fn"):
            CustomBackbone(ConvNet(), preprocess=preprocess, num_layers=3, device="cpu")

    def test_a_function_with_no_layers_to_serve_is_refused(self):
        """num_layers=1 means nothing could ever reach it."""
        with pytest.raises(ValueError, match="num_layers"):
            CustomBackbone(
                ConvNet(),
                preprocess=preprocess,
                layer_feature_fn=lambda module, image, layers: [],
                device="cpu",
            )

    def test_layer_feature_fn_serves_the_request(self, images):
        seen = []

        def layers_fn(module, image, layers):
            seen.append(list(layers))
            output = module(image)
            tokens = output.flatten(2).transpose(1, 2)
            return [(tokens * (index + 1), None, output.shape[-2:]) for index in layers]

        backbone = CustomBackbone(
            ConvNet(),
            preprocess=preprocess,
            layer_feature_fn=layers_fn,
            num_layers=4,
            device="cpu",
        )
        features = backbone.extract_features(backbone.preprocess(images), layers=[1, 3])

        assert seen == [[1, 3]], "resolved indices are handed through"
        assert features["layer_indices"] == [1, 3]
        assert len(features["dense_layers"]) == 2
        assert not torch.allclose(*features["dense_layers"])

    def test_a_short_return_is_caught(self, images):
        def layers_fn(module, image, layers):
            output = module(image)
            return [(output.flatten(2).transpose(1, 2), None, output.shape[-2:])]

        backbone = CustomBackbone(
            ConvNet(),
            preprocess=preprocess,
            layer_feature_fn=layers_fn,
            num_layers=4,
            device="cpu",
        )
        with pytest.raises(ValueError, match="one .* per requested index"):
            backbone.extract_features(backbone.preprocess(images), layers=[0, 2])

    def test_a_malformed_triple_names_the_layer(self, images):
        backbone = CustomBackbone(
            ConvNet(),
            preprocess=preprocess,
            layer_feature_fn=lambda module, image, layers: ["nonsense" for _ in layers],
            num_layers=4,
            device="cpu",
        )
        with pytest.raises(ValueError, match="layer 0"):
            backbone.extract_features(backbone.preprocess(images), layers=[0, 2])
