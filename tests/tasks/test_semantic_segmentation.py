"""Semantic (multi-class) segmentation — the fourth dense task.

The training machinery is :class:`~visbench.tasks.dense_base.DenseTrainingTask`,
shared with depth, normals and binary segmentation and tested there. What is
specific to this task, and tested here:

* a **class-index** target, which is the first one in VisBench that is not a
  float measurement — the base coerces via ``target_dtype`` for exactly this;
* cross-entropy over logits, with ``_activate`` deliberately an identity;
* the two mIoU reductions being reported side by side under distinct names;
* ``num_classes`` having no default, because a wrong one does not raise.
"""

import numpy as np
import pytest
import torch
from PIL import Image

import visbench
from visbench.cache import FeatureCache
from visbench.data import DenseFolderDataset, load_label_map
from visbench.tasks.high_level.semantic_segmentation import (
    IGNORE_INDEX,
    SemanticSegmentationTask,
)

NUM_CLASSES = 4


@pytest.fixture
def label_folder(tmp_path):
    """Images whose brightness says which class fills the lower half.

    A constant-colour frame: background (0) on top, class ``k`` below. FakeViT
    projects the mean colour linearly and adds the token index, so a linear head
    can express "class k below a colour-dependent row". A probe that cannot
    learn this has a real problem, not a hard dataset.
    """

    def build(root, count, seed, ignore_border=False):
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
            if ignore_border:
                label[15:17] = 255
            np.save(root / "labels" / f"s{index:03d}.npy", label)
        return DenseFolderDataset(
            root,
            target_dir="labels",
            image_size=32,
            target_loader=load_label_map,
        )

    return build


@pytest.fixture
def features_and_targets(label_folder, tmp_path, fake_vit):
    dataset = label_folder(tmp_path / "data", 12, seed=0)
    cache = FeatureCache(root=tmp_path / "cache")
    features = cache.extract_dataset(fake_vit, dataset, keep="dense", pooling="mean")
    return features, dataset.targets()


def probe(**kwargs):
    return visbench.get_probe("semantic_segmentation", num_classes=NUM_CLASSES, **kwargs)


# -- construction --------------------------------------------------------------


class TestConstruction:
    def test_num_classes_is_required(self):
        """A default would silently size the head for someone else's dataset."""
        with pytest.raises(TypeError):
            visbench.get_probe("semantic_segmentation")

    def test_out_channels_follows_num_classes(self):
        assert probe().out_channels == NUM_CLASSES
        assert visbench.get_probe("semantic_segmentation", num_classes=21).out_channels == 21

    def test_one_class_is_refused_and_points_at_the_binary_task(self):
        with pytest.raises(ValueError, match="generic_segmentation"):
            visbench.get_probe("semantic_segmentation", num_classes=1)

    def test_it_is_high_level(self):
        """The counterpart to mid-level generic segmentation."""
        assert probe().level == "high_level"
        assert visbench.get_probe("generic_segmentation").level == "mid_level"

    def test_targets_are_long_not_float(self):
        """The one place a class-index target does not fit the base's path."""
        assert probe().target_dtype == torch.long
        assert visbench.get_probe("generic_segmentation").target_dtype == torch.float32


# -- loss ----------------------------------------------------------------------


class TestLoss:
    def test_a_confident_correct_prediction_costs_almost_nothing(self):
        target = torch.zeros(1, 1, 4, 4, dtype=torch.long)
        target[:, :, 2:] = 2
        logits = torch.full((1, NUM_CLASSES, 4, 4), -10.0)
        logits[0, 0, :2] = 10.0
        logits[0, 2, 2:] = 10.0

        assert probe()._loss(logits, target).item() < 0.01

    def test_chance_logits_cost_log_num_classes(self):
        """Uniform scores over 4 classes is ln(4) ≈ 1.386, the chance baseline."""
        target = torch.zeros(1, 1, 4, 4, dtype=torch.long)
        logits = torch.zeros(1, NUM_CLASSES, 4, 4)

        assert probe()._loss(logits, target).item() == pytest.approx(np.log(NUM_CLASSES), rel=1e-4)

    def test_ignored_pixels_do_not_contribute(self):
        """A wrong answer on an ignored pixel must cost exactly nothing."""
        target = torch.zeros(1, 1, 4, 4, dtype=torch.long)
        logits = torch.full((1, NUM_CLASSES, 4, 4), -10.0)
        logits[0, 0] = 10.0

        ignored = target.clone()
        ignored[:, :, 2:] = IGNORE_INDEX
        logits_wrong = logits.clone()
        logits_wrong[0, 0, 2:] = -10.0
        logits_wrong[0, 3, 2:] = 10.0

        assert probe()._loss(logits_wrong, ignored).item() == pytest.approx(
            probe()._loss(logits, target).item(), abs=1e-6
        )

    def test_a_head_of_the_wrong_width_is_caught(self):
        target = torch.zeros(1, 1, 4, 4, dtype=torch.long)
        with pytest.raises(ValueError, match="channels for 4 classes"):
            probe()._loss(torch.zeros(1, 7, 4, 4), target)


class TestActivate:
    def test_it_is_the_identity(self):
        """Cross-entropy needs logits, and _activate feeds the loss."""
        raw = torch.randn(2, NUM_CLASSES, 3, 3)
        assert torch.equal(probe()._activate(raw), raw)

    def test_predict_returns_scores_not_labels(self, features_and_targets):
        features, targets = features_and_targets
        task = probe(epochs=2)
        task.fit(features, targets)

        scores = task.predict(features)
        assert scores.shape[1] == NUM_CLASSES
        assert task.predict_labels(features).shape == (len(targets), 32, 32)

    def test_predict_labels_is_the_argmax_of_predict(self, features_and_targets):
        features, targets = features_and_targets
        task = probe(epochs=2)
        task.fit(features, targets)

        assert torch.equal(task.predict_labels(features), task.predict(features).argmax(dim=1))


# -- training ------------------------------------------------------------------


class TestTraining:
    def test_it_learns_something(self, features_and_targets):
        """Clear of both baselines: chance loss, and the all-background collapse.

        Half of every frame is background, so a probe that answers 0 everywhere
        scores IoU 0.5 on class 0 and 0 on the other three — mIoU 0.125. At the
        default ten-epoch schedule this fixture lands on exactly that, which is
        underfitting rather than a weak representation; ``train_loss`` is what
        separates the two. The plateau here is about 0.44, set by FakeViT's
        8-channel 4x4 grid upsampled to 32x32, not by the probe.
        """
        features, targets = features_and_targets
        visbench.utils.set_seed(0)
        task = probe(epochs=200, lr=2e-1)
        task.fit(features, targets)

        assert task.train_loss < 0.8 * np.log(NUM_CLASSES)
        assert task.evaluate(features, targets)["miou"] > 0.25

    def test_an_untrained_probe_refuses_to_score(self):
        with pytest.raises(RuntimeError):
            probe().evaluate({"dense": torch.randn(2, 8, 4, 4), "grid_hw": (4, 4)}, None)

    def test_ignored_pixels_survive_the_whole_pipeline(self, label_folder, tmp_path, fake_vit):
        """255 in the file must be -1 in the target and absent from the score."""
        dataset = label_folder(tmp_path / "d", 8, seed=1, ignore_border=True)
        targets = dataset.targets()
        assert (targets < 0).any(), "the 255 border should have become -1"

        cache = FeatureCache(root=tmp_path / "c")
        features = cache.extract_dataset(fake_vit, dataset, keep="dense", pooling="mean")
        task = probe(epochs=2)
        task.fit(features, targets)

        metrics = task.evaluate(features, targets)
        assert 0.0 <= metrics["miou"] <= 1.0


class TestEvaluate:
    def test_both_reductions_are_reported(self, features_and_targets):
        features, targets = features_and_targets
        task = probe(epochs=2)
        task.fit(features, targets)

        metrics = task.evaluate(features, targets)
        assert {"miou", "miou_per_image", "pixel_acc", "mean_acc"} <= set(metrics)

    def test_dataset_level_miou_is_not_the_batch_mean(self, features_and_targets):
        """Why evaluate is overridden: the ratio of sums is not the sum of ratios.

        With a batch size of 1 the per-image mean and the dataset-level number
        are computed over the same predictions, and on a multi-class split with
        classes missing from individual frames they do not coincide.
        """
        features, targets = features_and_targets
        visbench.utils.set_seed(0)
        task = probe(epochs=30, lr=5e-2, batch_size=1)
        task.fit(features, targets)

        metrics = task.evaluate(features, targets)
        assert metrics["miou"] != pytest.approx(metrics["miou_per_image"], abs=1e-9)

    def test_an_empty_split_is_refused(self, features_and_targets):
        features, targets = features_and_targets
        task = probe(epochs=2)
        task.fit(features, targets)

        empty = {"dense": features["dense"][:0], "grid_hw": features["grid_hw"]}
        with pytest.raises(ValueError, match="empty split"):
            task.evaluate(empty, targets[:0])


# -- the record ----------------------------------------------------------------


class TestDescribe:
    def test_it_does_not_claim_probe3d(self):
        """probe3d has no semantic segmentation task to borrow a protocol from."""
        assert probe().describe()["task_params"]["protocol"] == "visbench_semantic_seg"

    def test_it_records_the_class_count(self):
        """A number computed over 21 classes is not comparable to one over 150."""
        params = visbench.get_probe("semantic_segmentation", num_classes=21).describe()[
            "task_params"
        ]
        assert params["num_classes"] == 21

    def test_it_records_which_reductions_were_used(self):
        assert probe().describe()["task_params"]["miou_reduction"] == "dataset_and_per_image"

    def test_it_records_the_ignore_convention(self):
        """Negative, not 0 — 0 is background, a real class."""
        assert probe().describe()["task_params"]["ignore_index"] == IGNORE_INDEX
        assert IGNORE_INDEX < 0


class TestThroughRun:
    def test_it_runs_end_to_end_and_writes_a_record(self, label_folder, tmp_path, fake_vit):
        train = label_folder(tmp_path / "train", 8, seed=0)
        test = label_folder(tmp_path / "val", 6, seed=1)
        results = tmp_path / "results.jsonl"

        result = visbench.run(
            fake_vit,
            SemanticSegmentationTask(num_classes=NUM_CLASSES, epochs=2),
            test,
            train_dataset=train,
            cache=FeatureCache(root=tmp_path / "cache"),
            results=results,
        )

        assert result.record.task == "semantic_segmentation"
        assert result.record.level == "high_level"
        assert "miou" in result.metrics
        assert results.exists()
