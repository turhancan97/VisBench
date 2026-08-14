"""Reading a timm model's own structure, without downloading one.

``tests/backbones/test_timm.py`` is entirely ``slow`` because every assertion
there needs real weights — and CI does not run ``-m slow``. The three decisions
:func:`describe_transformer` makes are exactly the kind CLAUDE.md says must have
a fast test: each produces a *silently wrong number* rather than an error.

- a wrong ``has_cls_token`` discards the CLS token while the record claims the
  model never had one, so the probe scores mean-pooled features under a name
  that says CLS;
- a wrong ``patch_size`` maps every token to the wrong pixel, which
  correspondence reports as a weak backbone rather than as a bug;
- an unrecognised ``global_pool`` silently pools some way the model does not.

So the logic is a module-level function over a stub rather than a method over a
downloaded checkpoint.
"""

from types import SimpleNamespace

import pytest

from visbench.backbones.timm_backbone import _POOL_TYPES, describe_transformer
from visbench.types import Pooling


def _model(prefix=1, patch=(16, 16), pool="token"):
    return SimpleNamespace(
        num_prefix_tokens=prefix,
        patch_embed=SimpleNamespace(patch_size=patch),
        global_pool=pool,
    )


class TestWhatItReads:
    def test_a_cls_token_is_detected(self):
        """MAE's shape: one prefix token, pooled by that token."""
        assert describe_transformer(_model(prefix=1, pool="token"), "m") == (True, 16, "token")

    def test_no_prefix_token_means_no_cls(self):
        """SigLIP-GAP's shape: no prefix tokens, pooled by average."""
        assert describe_transformer(_model(prefix=0, pool="avg"), "m") == (False, 16, "avg")

    def test_the_patch_size_comes_from_the_model(self):
        assert describe_transformer(_model(patch=(32, 32)), "m")[1] == 32

    def test_a_scalar_patch_size_is_accepted(self):
        assert describe_transformer(_model(patch=14), "m")[1] == 14


class TestWhatItRefuses:
    def test_a_learned_pooling_head_is_refused_by_name(self):
        """SigLIP's canonical `map` head is a trained module, not a reduction.

        It cannot be a pooling *mode*: the cache stores tokens, and `map` is an
        `AttentionPoolLatent` with weights of its own. The message says which
        sibling to use instead.
        """
        with pytest.raises(NotImplementedError, match="AttentionPoolLatent"):
            describe_transformer(_model(pool="map"), "vit_base_patch16_siglip_224")

    def test_registers_are_refused_rather_than_silently_dropped(self):
        """More than one prefix token means the extras have nowhere to go."""
        with pytest.raises(NotImplementedError, match="prefix tokens"):
            describe_transformer(_model(prefix=5), "vit_with_registers")

    def test_a_model_with_no_patch_embed_is_refused(self):
        """Without one there is no way to map a token back to a pixel."""
        model = SimpleNamespace(num_prefix_tokens=0, global_pool="avg")
        with pytest.raises(NotImplementedError, match="patch_embed"):
            describe_transformer(model, "something_odd")

    def test_an_unknown_pooling_is_refused_rather_than_defaulted(self):
        with pytest.raises(NotImplementedError, match="no mode for"):
            describe_transformer(_model(pool="something_new"), "m")


class TestThePoolingTable:
    def test_it_maps_onto_visbench_modes(self):
        assert _POOL_TYPES == {"token": Pooling.CLS, "avg": Pooling.MEAN, "": Pooling.MEAN}

    def test_map_is_deliberately_absent(self):
        """Guarding the guard: adding 'map' here would make the refusal vacuous."""
        assert "map" not in _POOL_TYPES

    @pytest.mark.parametrize(("pool", "expected"), [("token", "cls"), ("avg", "mean")])
    def test_the_model_decides_the_default(self, pool, expected):
        """`default` is what the model hands its own classifier, not a proxy.

        The base class infers it from `has_cls_token`, which is a good default
        and only a proxy: a ViT can carry a CLS token and still be trained to
        average. timm records which, so it is read.
        """
        _, _, pool_type = describe_transformer(_model(pool=pool), "m")
        assert _POOL_TYPES[pool_type] == expected
