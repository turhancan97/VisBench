"""Unfreezing the last N backbone blocks — the v0.3 mechanism (step 6a).

Every check here is in the **fast** suite on purpose. What fine-tuning can get
wrong is not a crash: a run that unfreezes nothing, or one whose gradients never
reach the backbone, trains exactly like a frozen probe and reports the number as
fine-tuned. That is the failure the CLIP QuickGELU guard shipped with for its
whole life — a check that existed, passed its own tests, and did nothing — and a
guard against it that only ran under ``-m slow`` would not run in CI at all.
"""

import pytest
import torch
from PIL import Image

from tests.conftest import BlockViT, FakeViT, ParameterlessViT


def images(count=2, size=64):
    return [Image.new("RGB", (size, size), (10 * i + 5, 20, 30)) for i in range(count)]


class TestRefusals:
    """Everything that must fail loudly rather than train the wrong thing."""

    def test_a_family_without_blocks_is_refused_by_name(self):
        """FakeViT has no ``_blocks``; the message must say so and name it.

        Step 6a covers DINOv2 only. A backbone that cannot be fine-tuned has to
        say which one it is, or the caller cannot tell whether they asked for
        the wrong thing or hit a gap.
        """
        with pytest.raises(NotImplementedError, match="not supported.*fake_vit"):
            FakeViT().unfreeze_last(2)

    def test_blocks_without_parameters_raise(self):
        """The guard this file exists for: 0 trainable parameters is not a
        fine-tuned run, it is a frozen one wearing the label."""
        with pytest.raises(RuntimeError, match="0 parameters trainable"):
            ParameterlessViT().unfreeze_last(2)

    @pytest.mark.parametrize("n", [0, -1])
    def test_n_below_one_raises(self, n):
        """``n=0`` is a frozen probe, and must not be spelled as a fine-tune."""
        with pytest.raises(ValueError, match="n >= 1"):
            BlockViT().unfreeze_last(n)

    def test_more_blocks_than_exist_raises_and_is_not_clamped(self):
        with pytest.raises(ValueError, match="has 4"):
            BlockViT(depth=4).unfreeze_last(5)

    def test_a_frozen_backbone_refuses_the_trainable_path(self):
        """Without an unfreeze, a graph-building pass would yield no gradients
        at all — so it is refused rather than silently training nothing."""
        backbone = BlockViT()
        batch = backbone.preprocess(images())
        with pytest.raises(RuntimeError, match="fully frozen"):
            backbone.extract_features_trainable(batch)


class TestUnfreezing:
    def test_only_the_last_n_blocks_become_trainable(self):
        backbone = BlockViT(depth=4)
        backbone.unfreeze_last(2)
        trainable = [any(p.requires_grad for p in block.parameters()) for block in backbone.blocks]
        assert trainable == [False, False, True, True]

    def test_it_returns_the_parameter_count(self):
        backbone = BlockViT(depth=4)
        expected = sum(p.numel() for p in backbone.blocks[2:].parameters())
        assert backbone.unfreeze_last(2) == expected

    def test_trainable_blocks_is_recorded(self):
        backbone = BlockViT()
        assert backbone.trainable_blocks == 0
        backbone.unfreeze_last(2)
        assert backbone.trainable_blocks == 2

    def test_the_backbone_stays_in_eval_mode(self):
        """Not an oversight — see BaseBackbone.unfreeze_last.

        Train mode would start BatchNorm updating and activate dropout, so a
        fine-tuned number would differ from its frozen baseline for two reasons
        at once and the record would show only one of them.
        """
        backbone = BlockViT()
        backbone.unfreeze_last(2)
        assert not backbone.training
        assert all(not block.training for block in backbone.blocks)


class TestGradients:
    """The mechanism actually carries a gradient — the part a no-op passes."""

    def test_gradients_reach_the_unfrozen_blocks_and_stop_at_the_frozen_ones(self):
        backbone = BlockViT(depth=4)
        backbone.unfreeze_last(2)

        features = backbone.extract_features_trainable(backbone.preprocess(images()))
        features["dense"].sum().backward()

        grads = [block.weight.grad for block in backbone.blocks]
        assert grads[0] is None and grads[1] is None, "frozen blocks received a gradient"
        assert all(g is not None and g.abs().sum() > 0 for g in grads[2:]), (
            "unfrozen blocks received no gradient"
        )

    def test_the_frozen_entry_point_still_detaches_after_an_unfreeze(self):
        """``extract_features`` must stay safe to cache even once the backbone
        can train, or a fine-tuning run would poison the frozen cache."""
        backbone = BlockViT()
        backbone.unfreeze_last(2)
        features = backbone.extract_features(backbone.preprocess(images()))
        assert not features["dense"].requires_grad
        assert not features["pooled"].requires_grad

    def test_trainable_parameters_lists_exactly_what_was_unfrozen(self):
        backbone = BlockViT(depth=4)
        backbone.unfreeze_last(1)
        listed = backbone.trainable_parameters()
        assert listed and all(p.requires_grad for p in listed)
        assert sum(p.numel() for p in listed) == sum(
            p.numel() for p in backbone.blocks[3].parameters()
        )


def test_a_real_forward_still_matches_after_unfreezing():
    """Unfreezing changes what can be optimised, never what is computed.

    If the two entry points disagreed numerically, a fine-tuned run's first
    step would start from different features than the frozen baseline measured,
    and the comparison the field exists to support would be invalid.
    """
    backbone = BlockViT()
    batch = backbone.preprocess(images())
    before = backbone.extract_features(batch)["dense"]
    backbone.unfreeze_last(2)
    after = backbone.extract_features(batch)["dense"]
    assert torch.equal(before, after)
