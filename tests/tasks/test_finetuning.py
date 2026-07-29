"""Fine-tuning through a dense probe and through ``run()`` — step 6a.

The backbone half of the mechanism is tested in
``tests/backbones/test_unfreezing.py``. What this file covers is the half that
decides whether a *number* is trustworthy:

* the cache is **bypassed**, not keyed differently — an entry written from
  fine-tuned weights is stale on arrival and indistinguishable from a frozen
  one, so it would be served to every later frozen run of that backbone;
* the record says a run was fine-tuned, so a frozen and a fine-tuned score
  cannot be averaged or ranked together by accident;
* the backbone's parameters actually move, which a no-op unfreeze would not
  reveal — it would train the head alone and score like a frozen probe.
"""

import numpy as np
import pytest
import torch
from PIL import Image

import visbench
from tests.conftest import BlockViT
from visbench.cache import FeatureCache
from visbench.data import DenseFolderDataset, load_label_map

NUM_CLASSES = 3


@pytest.fixture
def seg_dataset(tmp_path):
    """A tiny labelled split: class ``k`` fills the lower half of a flat frame."""

    def build(name, count, seed):
        root = tmp_path / name
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


class TestConstruction:
    def test_frozen_is_the_default(self):
        """Every v0.1 and v0.2 number was measured this way, and still is."""
        assert probe().finetune_blocks == 0
        assert probe().finetune() is None

    def test_a_backbone_lr_without_finetuning_raises(self):
        """It would be silently ignored, and the run would look configured."""
        with pytest.raises(ValueError, match="nothing in the backbone trains"):
            probe(backbone_lr=1e-6)

    def test_the_backbone_lr_defaults_well_below_the_head(self):
        """Pretrained weights at the head's rate are destroyed in one epoch,
        which shows up as a score *below* the frozen baseline, not an error."""
        task = probe(finetune_blocks=2)
        assert task.backbone_lr == task.lr / 100

    def test_negative_blocks_raise(self):
        with pytest.raises(ValueError, match="finetune_blocks must be >= 0"):
            probe(finetune_blocks=-1)

    def test_attaching_a_backbone_to_a_frozen_probe_raises(self):
        """A frozen probe reads cached features; holding a backbone would mean
        re-running it every epoch for nothing."""
        with pytest.raises(ValueError, match="finetune_blocks=0"):
            probe().attach_backbone(BlockViT())


class TestItRefusesExtractedFeatures:
    def test_fitting_on_features_while_finetuning_raises(self, seg_dataset):
        """Features are extracted with frozen weights, so they would not move
        as the backbone trains — training on them is silently a frozen probe."""
        task = probe(finetune_blocks=1, epochs=1, warmup_epochs=0, batch_size=2)
        backbone = BlockViT()
        task.attach_backbone(backbone)

        dataset = seg_dataset("d", 4, seed=0)
        features = FeatureCache(enabled=False).extract_dataset(
            backbone, dataset, keep="dense", pooling="mean"
        )
        with pytest.raises(TypeError, match="needs the image dataset"):
            task.fit(features, dataset.targets())

    def test_passing_targets_alongside_the_dataset_raises(self, seg_dataset):
        """The dataset already pairs each target with its own image, so a second
        source of supervision is one that would be silently ignored."""
        task = probe(finetune_blocks=1, epochs=1, warmup_epochs=0, batch_size=2)
        task.attach_backbone(BlockViT())
        dataset = seg_dataset("d", 4, seed=0)
        with pytest.raises(ValueError, match="two sources of truth"):
            task.fit(dataset, dataset.targets())


class TestTraining:
    def test_the_backbone_weights_actually_move(self, seg_dataset):
        """The check a no-op unfreeze cannot pass.

        Head-only training would leave every backbone weight identical and
        still produce a plausible score — which is exactly why this asserts on
        the weights rather than on the loss.
        """
        backbone = BlockViT(depth=4)
        task = probe(finetune_blocks=2, epochs=2, warmup_epochs=0, batch_size=2, lr=1e-2)
        task.attach_backbone(backbone)

        before = [block.weight.detach().clone() for block in backbone.blocks]
        task.fit(seg_dataset("train", 6, seed=0))
        after = [block.weight.detach() for block in backbone.blocks]

        assert torch.equal(before[0], after[0]), "a frozen block was modified"
        assert torch.equal(before[1], after[1]), "a frozen block was modified"
        assert not torch.equal(before[2], after[2]), "an unfrozen block did not move"
        assert not torch.equal(before[3], after[3]), "an unfrozen block did not move"

    def test_it_trains_and_scores_end_to_end(self, seg_dataset):
        backbone = BlockViT()
        task = probe(finetune_blocks=1, epochs=2, warmup_epochs=0, batch_size=2)
        task.attach_backbone(backbone)
        task.fit(seg_dataset("train", 6, seed=0))

        metrics = task.evaluate(seg_dataset("val", 4, seed=1))
        assert 0.0 <= metrics["miou"] <= 1.0
        assert task.train_loss is not None

    def test_predict_returns_one_map_per_image(self, seg_dataset):
        backbone = BlockViT()
        task = probe(finetune_blocks=1, epochs=1, warmup_epochs=0, batch_size=2)
        task.attach_backbone(backbone)
        task.fit(seg_dataset("train", 4, seed=0))

        predictions = task.predict(seg_dataset("val", 4, seed=1))
        assert predictions.shape == (4, NUM_CLASSES, 32, 32)


class TestThroughRun:
    def test_finetuning_writes_nothing_to_the_cache(self, seg_dataset, tmp_path):
        """The architectural guard of step 6a.

        Cache keys name the weights through ``cache_key()``, and fine-tuned
        weights differ at every optimiser step. An entry written here would be
        stale immediately *and* indistinguishable from a frozen one, so a later
        frozen run of the same backbone would silently read it.
        """
        cache = FeatureCache(root=tmp_path / "cache")
        visbench.run(
            BlockViT(),
            probe(finetune_blocks=1, epochs=1, warmup_epochs=0, batch_size=2),
            seg_dataset("val", 4, seed=1),
            train_dataset=seg_dataset("train", 4, seed=0),
            cache=cache,
        )
        assert cache.stats()["entries"] == 0

    def test_the_record_says_what_was_unfrozen(self, seg_dataset, tmp_path):
        backbone = BlockViT(depth=4)
        result = visbench.run(
            backbone,
            probe(finetune_blocks=2, epochs=1, warmup_epochs=0, batch_size=2),
            seg_dataset("val", 4, seed=1),
            train_dataset=seg_dataset("train", 4, seed=0),
            cache=FeatureCache(root=tmp_path / "cache"),
        )
        finetune = result.record.finetune
        assert finetune["blocks"] == 2
        assert finetune["trainable_params"] == sum(
            p.numel() for p in backbone.blocks[2:].parameters()
        )
        assert finetune["backbone_lr"] == result.probe.backbone_lr

    def test_a_frozen_run_records_none_and_still_caches(self, seg_dataset, tmp_path):
        """The frozen path is untouched by any of this — same record shape,
        same cache behaviour it has had since v0.1."""
        cache = FeatureCache(root=tmp_path / "cache")
        result = visbench.run(
            BlockViT(),
            probe(epochs=1, warmup_epochs=0, batch_size=2),
            seg_dataset("val", 4, seed=1),
            train_dataset=seg_dataset("train", 4, seed=0),
            cache=cache,
        )
        assert result.record.finetune is None
        assert cache.stats()["entries"] > 0
