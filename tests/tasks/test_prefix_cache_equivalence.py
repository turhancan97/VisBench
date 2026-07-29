"""Prefix caching must change the clock and nothing else — step 6b.

Step 6a bypassed the feature cache because fine-tuned weights change at every
optimiser step. The blocks *below* the cut do not: they are frozen for the whole
run, so their output is as fixed as a frozen backbone's and can be cached. This
file is the evidence for that claim, which is otherwise exactly the kind of
optimisation that quietly returns something slightly different.

The equality asserted here is **exact**. A tolerance would be admitting the
prefix path is an approximation, and it is not — no parameter below the cut
requires grad, so the recomputing path never built a graph through those blocks
either. The only difference is where the tokens came from.
"""

import numpy as np
import pytest
import torch
from PIL import Image

import visbench
from tests.conftest import BlockViT
from visbench.cache import FeatureCache, PrefixCache
from visbench.data import DenseFolderDataset, load_label_map

NUM_CLASSES = 3


@pytest.fixture
def seg_dataset(tmp_path):
    """A tiny labelled split; class ``k`` fills the lower half of a flat frame."""

    def build(name, count, seed):
        root = tmp_path / name
        if root.exists():
            # Two runs over the *same* split is the point of several tests here,
            # and rebuilding it would give them different images to compare.
            return DenseFolderDataset(
                root, target_dir="labels", image_size=32, target_loader=load_label_map
            )
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir(parents=True)
        rng = np.random.RandomState(seed)
        for index in range(count):
            klass = int(rng.randint(1, NUM_CLASSES))
            level = np.uint8(40 + 60 * klass)
            Image.fromarray(np.full((32, 32, 3), level, dtype=np.uint8)).save(
                root / "images" / f"s{index:03d}.png"
            )
            label = np.zeros((32, 32), dtype=np.uint8)
            label[16:] = klass
            np.save(root / "labels" / f"s{index:03d}.npy", label)
        return DenseFolderDataset(
            root, target_dir="labels", image_size=32, target_loader=load_label_map
        )

    return build


def probe(**kwargs):
    return visbench.get_probe("semantic_segmentation", num_classes=NUM_CLASSES, **kwargs)


class TestTheBackboneSplit:
    def test_resuming_reproduces_the_whole_forward_pass_exactly(self):
        """The claim the whole step rests on."""
        backbone = BlockViT(depth=4)
        backbone.unfreeze_last(2)
        torch.manual_seed(0)
        image = torch.randn(3, 3, 64, 64)

        whole = backbone.extract_features_trainable(image, pooling="mean")
        prefix, grid_hw = backbone.forward_prefix(image)
        resumed = backbone.extract_features_from_prefix(prefix, grid_hw, pooling="mean")

        assert torch.equal(whole["dense"], resumed["dense"])
        assert torch.equal(whole["pooled"], resumed["pooled"])
        assert whole["grid_hw"] == resumed["grid_hw"]

    def test_gradients_still_reach_the_unfrozen_blocks(self):
        """A prefix that cut the graph too high would train nothing while
        reporting itself as fine-tuned."""
        backbone = BlockViT(depth=4)
        backbone.unfreeze_last(2)
        prefix, grid_hw = backbone.forward_prefix(torch.randn(2, 3, 64, 64))

        backbone.extract_features_from_prefix(prefix, grid_hw)["dense"].sum().backward()

        moved = [param.grad is not None for param in backbone.trainable_parameters()]
        assert moved and all(moved)
        assert all(param.grad is None for param in backbone.blocks[:2].parameters()), (
            "a frozen block below the cut received a gradient"
        )

    def test_the_prefix_carries_no_graph(self):
        """It is written to disk; a tensor holding a graph would pin the whole
        forward pass in memory for the lifetime of the entry."""
        backbone = BlockViT(depth=4)
        backbone.unfreeze_last(2)
        prefix, _ = backbone.forward_prefix(torch.randn(2, 3, 64, 64))
        assert not prefix.requires_grad
        assert prefix.grad_fn is None

    def test_layers_below_the_cut_are_refused(self):
        """A single activation cannot serve a depth beneath it. Refusing is the
        only honest answer — returning the cut's features under a shallower
        layer's name is a wrong number, not a slow one."""
        backbone = BlockViT(depth=4)
        backbone.unfreeze_last(2)
        prefix, grid_hw = backbone.forward_prefix(torch.randn(2, 3, 64, 64))
        with pytest.raises(ValueError, match="below the cut"):
            backbone.extract_features_from_prefix(prefix, grid_hw, layers=[0, 3])

    def test_can_use_prefix_cache_answers_before_the_run(self):
        backbone = BlockViT(depth=4)
        assert not backbone.can_use_prefix_cache(), "frozen backbone offered a prefix cache"
        backbone.unfreeze_last(2)
        assert backbone.can_use_prefix_cache() is True
        assert backbone.can_use_prefix_cache([2, 3]) is True
        assert backbone.can_use_prefix_cache([0, 1, 2, 3]) is False

    def test_an_unsupported_family_refuses_rather_than_approximating(self):
        from tests.conftest import FakeViT

        backbone = FakeViT()
        assert backbone.can_use_prefix_cache() is False
        with pytest.raises(NotImplementedError, match="Prefix caching is not supported"):
            backbone.forward_prefix(torch.randn(1, 3, 64, 64))

    def test_a_frozen_backbone_refuses_to_resume(self):
        """Resuming with nothing unfrozen trains no parameters while the record
        would still call the run fine-tuned."""
        backbone = BlockViT(depth=4)
        backbone.unfreeze_last(2)
        prefix, grid_hw = backbone.forward_prefix(torch.randn(2, 3, 64, 64))

        refrozen = BlockViT(depth=4)
        with pytest.raises(RuntimeError, match="fully frozen"):
            refrozen.extract_features_from_prefix(prefix, grid_hw)

    def test_an_image_passed_as_a_prefix_is_refused(self):
        """The two take tensors of different rank and mixing them up would
        otherwise fail deep inside a block."""
        backbone = BlockViT(depth=4)
        backbone.unfreeze_last(2)
        with pytest.raises(ValueError, match=r"\(B, tokens, dim\)"):
            backbone.extract_features_from_prefix(torch.randn(2, 3, 64, 64), (4, 4))


class TestThroughRun:
    """The metric is what a user reads, so the equivalence is asserted there."""

    def _run(self, seg_dataset, tmp_path, name, **kwargs):
        torch.manual_seed(0)
        return visbench.run(
            BlockViT(depth=4),
            probe(finetune_blocks=2, epochs=2, warmup_epochs=0, batch_size=2),
            seg_dataset("val", 4, seed=1),
            train_dataset=seg_dataset("train", 6, seed=0),
            cache=FeatureCache(root=tmp_path / name),
            **kwargs,
        )

    def test_prefix_cached_and_recomputed_runs_agree_to_the_digit(self, seg_dataset, tmp_path):
        cached = self._run(seg_dataset, tmp_path, "a", use_prefix_cache=True)
        recomputed = self._run(seg_dataset, tmp_path, "b", use_prefix_cache=False)

        assert cached.metrics == recomputed.metrics
        assert cached.record.finetune["prefix_cache"] is True
        assert recomputed.record.finetune["prefix_cache"] is False

    def test_the_second_run_reads_what_the_first_wrote(self, seg_dataset, tmp_path):
        """A cache nothing ever hits is the QuickGELU failure — it passes its
        own tests forever while doing nothing."""
        first = self._run(seg_dataset, tmp_path, "shared", use_prefix_cache=True)
        prefix = PrefixCache(root=tmp_path / "shared")
        written = prefix.stats()["entries"]
        assert written > 0, "the prefix cache was declared used but nothing was written"

        second = self._run(seg_dataset, tmp_path, "shared", use_prefix_cache=True)

        assert second.metrics == first.metrics
        assert prefix.stats()["entries"] == written, "a second run rewrote every entry"

    def test_it_never_writes_to_the_feature_cache(self, seg_dataset, tmp_path):
        """6a's guarantee, unchanged: fine-tuned *features* are still poison."""
        cache = FeatureCache(root=tmp_path / "c")
        visbench.run(
            BlockViT(depth=4),
            probe(finetune_blocks=2, epochs=1, warmup_epochs=0, batch_size=2),
            seg_dataset("val", 4, seed=1),
            train_dataset=seg_dataset("train", 4, seed=0),
            cache=cache,
        )
        assert cache.stats()["entries"] == 0

    def test_a_frozen_run_writes_no_prefixes(self, seg_dataset, tmp_path):
        """There is no cut on a frozen probe, so there is nothing to cache and
        an entry here would be keyed on a cut that never happened."""
        cache = FeatureCache(root=tmp_path / "d")
        visbench.run(
            BlockViT(depth=4),
            probe(epochs=1, warmup_epochs=0, batch_size=2),
            seg_dataset("val", 4, seed=1),
            train_dataset=seg_dataset("train", 4, seed=0),
            cache=cache,
        )
        assert PrefixCache(root=tmp_path / "d").stats()["entries"] == 0
        assert cache.stats()["entries"] > 0
