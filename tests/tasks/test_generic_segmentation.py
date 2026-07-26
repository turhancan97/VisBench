"""Generic (binary) object segmentation — the third dense task.

The training machinery is :class:`~visbench.tasks.dense_base.DenseTrainingTask`,
shared with depth and normals and tested there. What is specific to this task,
and tested here: the loss, the probability output, and the fact that this is the
first target in VisBench where **0 is a real label** rather than a hole — so
validity shifts from ``> 0`` to ``>= 0`` and an unlabelled pixel is negative.
"""

import numpy as np
import pytest
import torch
from PIL import Image

import visbench
from visbench.cache import FeatureCache
from visbench.data import DenseFolderDataset, load_mask
from visbench.metrics.dense import binary_iou
from visbench.tasks.mid_level.generic_segmentation import GenericSegmentationTask, masked_bce_loss


@pytest.fixture
def mask_folder(tmp_path):
    """Images whose brightness says how much of the frame is foreground.

    A constant-colour image with the top ``k`` quarters masked. FakeViT projects
    the mean colour linearly and adds the token index, so a linear head can
    express "foreground below a colour-dependent row" exactly — a probe that
    cannot learn this has a real problem, not a hard dataset.
    """

    def build(root, count, seed):
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir(parents=True)
        rng = np.random.RandomState(seed)
        for index in range(count):
            quarters = int(rng.randint(1, 4))
            level = np.uint8(40 + 60 * quarters)
            Image.fromarray(np.full((32, 32, 3), level, dtype=np.uint8)).save(
                root / "images" / f"s{index:03d}.png"
            )
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[: 8 * quarters] = 255
            np.save(root / "masks" / f"s{index:03d}.npy", mask)
        return DenseFolderDataset(root, target_dir="masks", image_size=32, target_loader=load_mask)

    return build


@pytest.fixture
def features_and_targets(mask_folder, tmp_path, fake_vit):
    dataset = mask_folder(tmp_path / "data", 12, seed=0)
    cache = FeatureCache(root=tmp_path / "cache")
    features = cache.extract_dataset(fake_vit, dataset, keep="dense", pooling="mean")
    return features, dataset.targets()


# -- loss ---------------------------------------------------------------------


class TestMaskedBCELoss:
    def test_a_confident_correct_prediction_costs_almost_nothing(self):
        target = torch.zeros(1, 1, 4, 4)
        target[:, :, :2] = 1.0
        pred = target * 0.998 + 0.001
        assert masked_bce_loss(pred, target).item() < 0.01

    def test_a_confident_wrong_prediction_is_expensive_but_finite(self):
        """The clamp earns its place here: an exactly-saturated sigmoid on the
        wrong side of a label is ``log(0)``, and one inf poisons the batch."""
        target = torch.ones(1, 1, 4, 4)
        loss = masked_bce_loss(torch.zeros(1, 1, 4, 4), target)
        assert torch.isfinite(loss)
        assert loss.item() > 10.0

    def test_an_uncertain_prediction_costs_log_two(self):
        target = torch.zeros(1, 1, 4, 4)
        target[:, :, :2] = 1.0
        assert masked_bce_loss(torch.full((1, 1, 4, 4), 0.5), target).item() == pytest.approx(
            0.693, abs=1e-3
        )

    def test_a_saturated_prediction_stays_differentiable(self):
        target = torch.ones(1, 1, 4, 4)
        pred = torch.zeros(1, 1, 4, 4, requires_grad=True)
        masked_bce_loss(pred, target).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_negative_targets_are_skipped(self):
        target = torch.zeros(1, 1, 4, 4)
        target[:, :, 2:] = -1.0

        good = torch.full((1, 1, 4, 4), 0.01)
        wild = good.clone()
        wild[:, :, 2:] = 0.99

        assert masked_bce_loss(good, target).item() == pytest.approx(
            masked_bce_loss(wild, target).item(), abs=1e-6
        )

    def test_an_entirely_unlabelled_target_keeps_the_graph(self):
        """A bare zero would detach the head and silently skip the gradient."""
        pred = torch.rand(1, 1, 4, 4, requires_grad=True)
        loss = masked_bce_loss(pred, torch.full((1, 1, 4, 4), -1.0))
        loss.backward()
        assert loss.item() == 0.0
        assert pred.grad is not None

    def test_background_is_supervised_not_ignored(self):
        """The trap this task inherits from its neighbours: depth and normals
        both read 0 as "no ground truth", and reusing that here would throw away
        every background pixel and train the probe to say foreground."""
        zeros = torch.zeros(1, 1, 4, 4)
        assert masked_bce_loss(torch.full((1, 1, 4, 4), 0.99), zeros).item() > 1.0

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="must match"):
            masked_bce_loss(torch.rand(1, 1, 4, 4), torch.zeros(1, 1, 8, 8))


# -- the task -----------------------------------------------------------------


class TestGenericSegmentationTask:
    def test_it_is_registered(self):
        assert "generic_segmentation" in visbench.list_probes()
        assert isinstance(visbench.get_probe("generic_segmentation"), GenericSegmentationTask)

    def test_it_declares_what_it_needs(self):
        task = GenericSegmentationTask()
        assert task.uses_dense, "the cache must keep dense features for this"
        assert task.level == "mid_level", "figure-ground without semantics is mid-level"
        assert not task.zero_shot
        assert task.target_channels == 1
        assert task.out_channels == 1

    def test_fit_then_predict_shapes(self, features_and_targets):
        features, targets = features_and_targets
        task = GenericSegmentationTask(epochs=2, warmup_epochs=0, batch_size=4)
        task.fit(features, targets)
        assert task.predict(features).shape == (12, 1, 32, 32)

    def test_predictions_are_probabilities(self, features_and_targets):
        """Not a hard mask: a caller can threshold, overlay or calibrate them,
        and the metric applies the threshold itself."""
        features, targets = features_and_targets
        task = GenericSegmentationTask(epochs=2, warmup_epochs=0, batch_size=4)
        predicted = task.fit(features, targets).predict(features)
        assert predicted.min() >= 0.0 and predicted.max() <= 1.0
        assert not torch.isin(predicted, torch.tensor([0.0, 1.0])).all(), "these are not labels"

    def test_evaluate_returns_the_segmentation_metrics(self, features_and_targets):
        features, targets = features_and_targets
        task = GenericSegmentationTask(epochs=2, warmup_epochs=0, batch_size=4)
        metrics = task.fit(features, targets).evaluate(features, targets)
        assert set(metrics) == {"iou", "f1", "pixel_acc"}
        assert all(isinstance(value, float) for value in metrics.values())

    def test_it_learns_something(self, features_and_targets):
        """Beats both degenerate constant masks — all foreground and all
        background — which between them are what a probe that learned nothing
        collapses to. All-foreground is the one that bites: it scores 0.44 IoU
        here, because these frames are mostly object.

        It plateaus around 0.68 rather than at 1.0, and that is the fixture's
        ceiling rather than the probe's: FakeViT emits a 4x4 grid that a linear
        head resamples to 32x32, so the mask's edge can only ever land within a
        patch. Enough to separate learning from not learning, which is all this
        asserts.
        """
        features, targets = features_and_targets
        visbench.utils.set_seed(0)
        task = GenericSegmentationTask(epochs=200, batch_size=4, lr=1e-1)
        task.fit(features, targets)

        learned = task.evaluate(features, targets)["iou"]
        stacked = targets.unsqueeze(1) if targets.ndim == 3 else targets
        for constant in (torch.zeros_like(stacked), torch.ones_like(stacked)):
            assert learned > binary_iou(constant, stacked)["iou"]

    def test_training_loss_is_recorded_as_a_diagnostic(self, features_and_targets):
        features, targets = features_and_targets
        task = GenericSegmentationTask(epochs=3, warmup_epochs=0, batch_size=4)
        assert task.fit(features, targets).train_loss is not None

    def test_unfitted_predict_raises(self, features_and_targets):
        features, _ = features_and_targets
        with pytest.raises(RuntimeError, match="not been fitted"):
            GenericSegmentationTask().predict(features)

    def test_missing_targets_raise(self, features_and_targets):
        features, _ = features_and_targets
        with pytest.raises(ValueError, match="requires target masks"):
            GenericSegmentationTask(epochs=1, warmup_epochs=0).fit(features, None)

    def test_a_vector_target_is_refused(self, features_and_targets):
        """A normal map handed to the segmentation probe."""
        features, targets = features_and_targets
        vectors = targets.unsqueeze(1).repeat(1, 3, 1, 1)
        with pytest.raises(ValueError, match=r"\(N, H, W\) or \(N, 1, H, W\)"):
            GenericSegmentationTask(epochs=1, warmup_epochs=0).fit(features, vectors)

    def test_unlabelled_pixels_survive_to_the_metric(self, features_and_targets):
        """End to end, not just in the loss: marking every pixel unlabelled must
        leave nothing to score rather than scoring against zeros."""
        features, targets = features_and_targets
        task = GenericSegmentationTask(epochs=1, warmup_epochs=0, batch_size=4)
        task.fit(features, targets)
        ignored = torch.full_like(targets, -1.0)
        assert task.evaluate(features, ignored) == {"iou": 0.0, "f1": 0.0, "pixel_acc": 0.0}

    def test_the_dpt_head_trains_on_several_layers(self, tmp_path, fake_vit, mask_folder):
        dataset = mask_folder(tmp_path / "data", 8, seed=2)
        cache = FeatureCache(root=tmp_path / "cache")
        features = cache.extract_dataset(
            fake_vit, dataset, keep="dense", pooling="mean", layers=[2, 5, 8, 11]
        )
        task = GenericSegmentationTask(
            head="dpt", layers=[2, 5, 8, 11], epochs=2, warmup_epochs=0, batch_size=4, hidden_dim=8
        )
        task.fit(features, dataset.targets())
        assert task.predict(features).shape == (8, 1, 32, 32)


# -- provenance ---------------------------------------------------------------


class TestDescribe:
    def test_it_records_what_shaped_the_number(self):
        params = GenericSegmentationTask(head="dpt", layers=[3, 7], epochs=4).describe()[
            "task_params"
        ]
        assert params["head"] == "dpt"
        assert params["layers"] == [3, 7]
        assert params["epochs"] == 4
        assert params["threshold"] == 0.5

    def test_it_does_not_claim_probe3d(self):
        """Depth and normals reproduce that paper's protocol; this one has no
        counterpart there, and a record saying otherwise would be worse than no
        record. The optimiser schedule is still theirs, and still recorded."""
        params = GenericSegmentationTask().describe()["task_params"]
        assert params["protocol"] == "visbench_binary_seg"
        assert params["optimizer"] == "adamw"


# -- streaming and the runner -------------------------------------------------


class TestStreaming:
    def _streamed(self, cache, fake_vit, dataset):
        return cache.materialise(fake_vit, dataset, pooling="mean", targets=dataset.target)

    def test_fit_from_a_streaming_source(self, tmp_path, fake_vit, mask_folder):
        dataset = mask_folder(tmp_path / "data", 12, seed=0)
        cache = FeatureCache(root=tmp_path / "cache")
        task = GenericSegmentationTask(epochs=2, warmup_epochs=0, batch_size=4)

        task.fit(self._streamed(cache, fake_vit, dataset))
        assert task.train_loss is not None
        assert task.predict(self._streamed(cache, fake_vit, dataset)).shape == (12, 1, 32, 32)

    def test_evaluate_matches_the_in_memory_path(self, tmp_path, fake_vit, mask_folder):
        """Scoring batch by batch must give exactly the whole-split number: the
        metrics are per-image averages, so weighting each batch by its size
        recovers it — but only if the target pairing survived the loader."""
        dataset = mask_folder(tmp_path / "data", 12, seed=3)
        cache = FeatureCache(root=tmp_path / "cache")
        stacked = cache.extract_dataset(fake_vit, dataset, keep="dense", pooling="mean")
        targets = dataset.targets()

        visbench.utils.set_seed(0)
        task = GenericSegmentationTask(epochs=3, warmup_epochs=0, batch_size=5)
        task.fit(stacked, targets)

        assert task.evaluate(stacked, targets) == pytest.approx(
            task.evaluate(self._streamed(cache, fake_vit, dataset)), abs=1e-5
        )


class TestThroughRun:
    def test_a_segmentation_task_runs_end_to_end(self, tmp_path, fake_vit, mask_folder):
        train = mask_folder(tmp_path / "train", 8, seed=0)
        val = mask_folder(tmp_path / "val", 6, seed=1)
        results = tmp_path / "results.jsonl"

        outcome = visbench.run(
            fake_vit,
            "generic_segmentation",
            val,
            train_dataset=train,
            cache=FeatureCache(root=tmp_path / "cache"),
            results=results,
            epochs=2,
            warmup_epochs=0,
            batch_size=4,
        )

        assert set(outcome.metrics) == {"iou", "f1", "pixel_acc"}
        assert outcome.record.task == "generic_segmentation"
        assert outcome.record.level == "mid_level"
        assert results.read_text().strip(), "the run must have logged a record"
