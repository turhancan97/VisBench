"""Depth estimation — the first dense task.

Reproduces probe3d's configured protocol (bin prediction, scale-invariant plus
gradient loss, ten-epoch cosine schedule). The tests that matter most are the
ones about the *contract* between a task, its head and its features: a dense
probe that trains on misaligned or wrongly-shaped supervision still produces a
number, and the number is the only thing anyone looks at.
"""

import numpy as np
import pytest
import torch
from PIL import Image

import visbench
from visbench.cache import FeatureCache
from visbench.data import DenseFolderDataset
from visbench.tasks.mid_level.depth import DepthBinPrediction, DepthTask, depth_loss


@pytest.fixture
def depth_folder(tmp_path):
    """Images whose brightness determines their (constant) depth.

    Learnable from FakeViT, whose features are the mean colour projected — so a
    probe that fails here has a real problem, not a hard dataset.
    """

    def build(root, count, seed):
        (root / "images").mkdir(parents=True)
        (root / "depths").mkdir(parents=True)
        rng = np.random.RandomState(seed)
        for index in range(count):
            depth_value = 1.0 + 4.0 * rng.rand()
            depth = np.full((32, 32), depth_value, dtype=np.float32)
            brightness = np.uint8(depth_value / 5.0 * 255)
            Image.fromarray(np.full((32, 32, 3), brightness, dtype=np.uint8)).save(
                root / "images" / f"s{index:03d}.png"
            )
            np.save(root / "depths" / f"s{index:03d}.npy", depth)
        return DenseFolderDataset(root, image_size=32, max_target=10.0)

    return build


@pytest.fixture
def features_and_targets(depth_folder, tmp_path, fake_vit):
    dataset = depth_folder(tmp_path / "data", 12, seed=0)
    cache = FeatureCache(root=tmp_path / "cache")
    features = cache.extract_dataset(fake_vit, dataset, keep="dense", pooling="mean")
    return features, dataset.targets()


# -- bin prediction -----------------------------------------------------------


class TestDepthBinPrediction:
    """probe3d predicts a distribution over depths, then takes its expectation."""

    def test_output_shape(self):
        predict = DepthBinPrediction(n_bins=16, max_depth=10.0)
        assert predict(torch.rand(2, 16, 4, 4)).shape == (2, 1, 4, 4)

    def test_a_confident_bin_gives_that_depth(self):
        predict = DepthBinPrediction(min_depth=0.0001, max_depth=10.0, n_bins=11)
        scores = torch.zeros(1, 11, 1, 1)
        scores[0, 5] = 1000.0
        assert predict(scores)[0, 0, 0, 0].item() == pytest.approx(5.0, abs=0.1)

    def test_predictions_stay_inside_the_range(self):
        predict = DepthBinPrediction(min_depth=1.0, max_depth=4.0, n_bins=32)
        depth = predict(torch.randn(4, 32, 8, 8) * 50)
        assert depth.min() >= 1.0
        assert depth.max() <= 4.0

    def test_all_negative_scores_fall_back_to_uniform(self):
        """The reason probe3d adds 0.1 after the ReLU: without it this pixel
        divides by zero and NaN poisons the rest of the epoch."""
        predict = DepthBinPrediction(min_depth=0.0001, max_depth=10.0, n_bins=64)
        depth = predict(torch.full((1, 64, 2, 2), -5.0))
        assert torch.isfinite(depth).all()
        assert depth.mean().item() == pytest.approx(5.0, abs=0.2), "uniform means mid-range"

    def test_it_is_differentiable(self):
        predict = DepthBinPrediction(n_bins=8, max_depth=10.0)
        scores = torch.rand(1, 8, 2, 2, requires_grad=True)
        predict(scores).sum().backward()
        assert scores.grad is not None

    def test_wrong_bin_count_is_refused(self):
        predict = DepthBinPrediction(n_bins=16)
        with pytest.raises(ValueError, match=r"\(B, 16, H, W\)"):
            predict(torch.rand(1, 8, 4, 4))

    def test_invalid_range_is_refused(self):
        with pytest.raises(ValueError, match="min_depth < max_depth"):
            DepthBinPrediction(min_depth=5.0, max_depth=1.0)


# -- loss ---------------------------------------------------------------------


class TestDepthLoss:
    def test_a_perfect_prediction_costs_nothing(self):
        target = torch.rand(2, 1, 16, 16) + 1.0
        assert depth_loss(target.clone(), target).item() == pytest.approx(0.0, abs=1e-3)

    def test_the_scale_invariant_term_forgives_a_uniform_scale(self):
        """Its whole point: a prediction uniformly too deep is barely punished,
        one with the wrong relative arrangement is."""
        target = torch.rand(2, 1, 16, 16) + 1.0
        scaled = depth_loss(target * 2.0, target, weight_gradient=0.0).item()
        scrambled = depth_loss(target.flip(-1) * 2.0, target, weight_gradient=0.0).item()
        assert scaled < scrambled

    def test_the_gradient_term_punishes_a_blurred_edge(self):
        """The term that asks for edges in the right places, which is the
        mid-level structure a depth probe measures."""
        target = torch.ones(1, 1, 32, 32)
        target[..., 16:] = 4.0
        blurred = torch.nn.functional.avg_pool2d(target, 5, stride=1, padding=2)

        sharp = depth_loss(target.clone(), target, weight_si=0.0).item()
        assert depth_loss(blurred, target, weight_si=0.0).item() > sharp

    def test_invalid_pixels_are_ignored(self):
        target = torch.rand(1, 1, 8, 8) + 1.0
        holed = target.clone()
        holed[..., :4] = 0.0

        pred = target.clone()
        wild = target.clone()
        wild[..., :4] = 999.0
        assert depth_loss(pred, holed).item() == pytest.approx(depth_loss(wild, holed).item())

    def test_an_entirely_invalid_target_keeps_the_graph(self):
        """Returning a bare zero would detach the head and silently skip the
        batch's gradient."""
        pred = (torch.rand(1, 1, 8, 8) + 1.0).requires_grad_(True)
        depth_loss(pred, torch.zeros(1, 1, 8, 8)).backward()
        assert pred.grad is not None

    def test_it_does_not_mutate_the_target(self):
        """The reference implementation zeroes the caller's tensor in place."""
        target = torch.rand(1, 1, 8, 8) + 1.0
        before = target.clone()
        depth_loss(target.clone(), target)
        assert torch.equal(target, before)


# -- the task -----------------------------------------------------------------


class TestDepthTask:
    def test_it_is_registered(self):
        assert "depth" in visbench.list_probes()
        assert isinstance(visbench.get_probe("depth"), DepthTask)

    def test_it_declares_what_it_needs(self):
        task = DepthTask()
        assert task.uses_dense, "the cache must keep dense features for this"
        assert task.level == "mid_level"
        assert not task.zero_shot

    def test_fit_then_predict_shapes(self, features_and_targets):
        features, targets = features_and_targets
        task = DepthTask(epochs=2, warmup_epochs=0, batch_size=4, n_bins=16, max_depth=8.0).fit(
            features, targets
        )
        assert task.predict(features).shape == (12, 1, 32, 32)

    def test_predictions_are_at_the_target_resolution(self, features_and_targets):
        """The head upsamples from a 2x2 grid; scoring happens in target pixels."""
        features, targets = features_and_targets
        task = DepthTask(epochs=1, warmup_epochs=0, batch_size=4, n_bins=8).fit(features, targets)
        assert task.predict(features).shape[-2:] == targets.shape[-2:]

    def test_evaluate_returns_the_probe3d_metrics(self, features_and_targets):
        features, targets = features_and_targets
        task = DepthTask(epochs=2, warmup_epochs=0, batch_size=4, n_bins=16, max_depth=8.0).fit(
            features, targets
        )
        metrics = task.evaluate(features, targets)
        assert set(metrics) == {"d1", "d2", "d3", "rmse", "abs_rel"}
        assert all(isinstance(value, float) for value in metrics.values())

    def test_it_learns_something(self, features_and_targets):
        """Beats predicting the mean depth everywhere — the baseline any probe
        that has learned nothing at all would match."""
        features, targets = features_and_targets
        task = DepthTask(epochs=40, batch_size=4, n_bins=64, max_depth=8.0, lr=5e-3)
        task.fit(features, targets)

        from visbench.metrics.dense import depth_metrics

        constant = torch.full_like(targets, targets.mean().item())
        assert task.evaluate(features, targets)["rmse"] < depth_metrics(constant, targets)["rmse"]

    def test_training_loss_is_recorded_as_a_diagnostic(self, features_and_targets):
        features, targets = features_and_targets
        task = DepthTask(epochs=3, warmup_epochs=0, batch_size=4, n_bins=16).fit(features, targets)
        assert task.train_loss is not None and task.train_loss >= 0

    def test_unfitted_predict_raises(self, features_and_targets):
        features, _ = features_and_targets
        with pytest.raises(RuntimeError, match="not been fitted"):
            DepthTask().predict(features)

    def test_mismatched_counts_raise(self, features_and_targets):
        features, targets = features_and_targets
        with pytest.raises(ValueError, match="12 feature maps for 5 targets"):
            DepthTask(epochs=1, warmup_epochs=0).fit(features, targets[:5])

    def test_missing_targets_raise(self, features_and_targets):
        features, _ = features_and_targets
        with pytest.raises(ValueError, match="requires target maps"):
            DepthTask(epochs=1, warmup_epochs=0).fit(features, None)

    def test_pooled_only_features_are_refused(self, tmp_path, fake_vit, depth_folder):
        dataset = depth_folder(tmp_path / "data", 4, seed=1)
        cache = FeatureCache(root=tmp_path / "cache")
        pooled = cache.extract_dataset(fake_vit, dataset, keep="pooled", pooling="mean")
        with pytest.raises(KeyError, match="dense"):
            DepthTask(epochs=1, warmup_epochs=0).fit(pooled, dataset.targets())


# -- head selection -----------------------------------------------------------


class TestPluggableHead:
    def test_the_linear_head_is_the_default(self, features_and_targets):
        """The number to quote when comparing representations."""
        features, targets = features_and_targets
        task = DepthTask(epochs=1, warmup_epochs=0, batch_size=4, n_bins=8).fit(features, targets)
        assert type(task.head).__name__ == "LinearHead"

    def test_the_dpt_head_needs_layers(self, features_and_targets):
        """It refuses a single map rather than duplicating it, so a DPT run over
        one layer fails loudly instead of reporting a single-layer result."""
        features, targets = features_and_targets
        with pytest.raises(TypeError, match="multiscale"):
            DepthTask(head="dpt", epochs=1, warmup_epochs=0, batch_size=4, n_bins=8).fit(
                features, targets
            )

    def test_declared_layers_without_multilayer_features_raise(self, features_and_targets):
        features, targets = features_and_targets
        with pytest.raises(KeyError, match="layers="):
            DepthTask(layers=[1, 5], epochs=1, warmup_epochs=0).fit(features, targets)

    def test_the_dpt_head_trains_on_several_layers(self, tmp_path, fake_vit, depth_folder):
        dataset = depth_folder(tmp_path / "data", 8, seed=2)
        cache = FeatureCache(root=tmp_path / "cache")
        features = cache.extract_dataset(
            fake_vit, dataset, keep="dense", pooling="mean", layers=[2, 5, 8, 11]
        )
        task = DepthTask(
            head="dpt",
            layers=[2, 5, 8, 11],
            epochs=2,
            warmup_epochs=0,
            batch_size=4,
            n_bins=8,
            hidden_dim=8,
        )
        task.fit(features, dataset.targets())
        assert task.predict(features).shape == (8, 1, 32, 32)

    def test_an_unknown_head_names_the_known_ones(self, features_and_targets):
        features, targets = features_and_targets
        with pytest.raises(KeyError, match="linear"):
            DepthTask(head="transformer", epochs=1, warmup_epochs=0).fit(features, targets)

    def test_head_kwargs_reach_the_head(self, features_and_targets):
        features, targets = features_and_targets
        task = DepthTask(
            epochs=1, warmup_epochs=0, batch_size=4, n_bins=8, head_kwargs={"bias": False}
        )
        task.fit(features, targets)
        assert task.head.proj.bias is None


# -- guard rails --------------------------------------------------------------


class TestConfiguration:
    def test_warmup_must_fit_inside_training(self):
        with pytest.raises(ValueError, match="warmup_epochs"):
            DepthTask(epochs=2, warmup_epochs=2)

    def test_zero_epochs_is_refused(self):
        with pytest.raises(ValueError, match="epochs must be"):
            DepthTask(epochs=0)

    def test_empty_layers_is_refused(self):
        with pytest.raises(ValueError, match="requests nothing"):
            DepthTask(layers=[])

    def test_the_memory_ceiling_explains_itself(self, features_and_targets):
        """Dense features are held in RAM for training; 24k NYUv2 images at
        DINOv2-B would be ~19 GB. Better to refuse than to be OOM-killed."""
        features, targets = features_and_targets
        task = DepthTask(epochs=1, warmup_epochs=0)
        task.max_feature_elements = 10
        with pytest.raises(ValueError, match="GB ceiling"):
            task.fit(features, targets)

    def test_non_square_targets_are_refused(self, features_and_targets):
        features, targets = features_and_targets
        with pytest.raises(ValueError, match="square"):
            DepthTask(epochs=1, warmup_epochs=0).fit(features, targets[:, :16, :])


# -- provenance ---------------------------------------------------------------


def test_describe_records_the_protocol():
    """A dense number without its head, layers and schedule is not reproducible."""
    described = DepthTask(head="dpt", layers=[2, 5, 8, 11]).describe()
    params = described["task_params"]

    assert described["task"] == "depth"
    assert described["level"] == "mid_level"
    assert described["layers"] == [2, 5, 8, 11]
    assert params["head"] == "dpt"
    assert params["protocol"] == "probe3d"
    assert (params["epochs"], params["lr"], params["warmup_epochs"]) == (10, 5e-4, 1.5)
    assert (params["n_bins"], params["max_depth"]) == (256, 10.0)


def test_defaults_match_probe3ds_configuration():
    """configs/probe/depth_dpt.yaml + configs/optimizer/ten_epoch.yaml."""
    task = DepthTask()
    assert (task.min_depth, task.max_depth, task.n_bins) == (0.001, 10.0, 256)
    assert (task.epochs, task.lr, task.warmup_epochs, task.batch_size) == (10, 5e-4, 1.5, 8)
    assert task.hidden_dim == 512


# -- through the public entry point -------------------------------------------


class TestThroughRun:
    """visbench.run() has to carry a dense task's declared layers into extraction."""

    def test_a_dense_task_runs_end_to_end(self, tmp_path, fake_vit, depth_folder):
        train = depth_folder(tmp_path / "train", 8, seed=0)
        val = depth_folder(tmp_path / "val", 6, seed=1)
        results = tmp_path / "results.jsonl"

        outcome = visbench.run(
            fake_vit,
            "depth",
            val,
            train_dataset=train,
            cache=FeatureCache(root=tmp_path / "cache"),
            results=results,
            epochs=2,
            warmup_epochs=0,
            batch_size=4,
            n_bins=16,
            max_depth=8.0,
        )
        assert set(outcome.metrics) == {"d1", "d2", "d3", "rmse", "abs_rel"}
        assert results.exists()

    def test_the_record_carries_the_resolved_layers(self, tmp_path, fake_vit, depth_folder):
        """A record saying [-4, -1] does not name the layers that produced the
        number, and means two different things on a 12- and a 24-block ViT."""
        train = depth_folder(tmp_path / "train", 8, seed=0)
        val = depth_folder(tmp_path / "val", 4, seed=1)

        outcome = visbench.run(
            fake_vit,
            "depth",
            val,
            train_dataset=train,
            cache=FeatureCache(root=tmp_path / "cache"),
            head="dpt",
            layers=[-4, -1],
            epochs=2,
            warmup_epochs=0,
            batch_size=4,
            n_bins=8,
            hidden_dim=8,
        )
        assert outcome.record.layers == [8, 11]
        assert outcome.record.task_params["head"] == "dpt"

    def test_a_single_layer_task_records_no_layers(self, tmp_path, fake_vit, depth_folder):
        train = depth_folder(tmp_path / "train", 6, seed=0)
        val = depth_folder(tmp_path / "val", 4, seed=1)

        outcome = visbench.run(
            fake_vit,
            "depth",
            val,
            train_dataset=train,
            cache=FeatureCache(root=tmp_path / "cache"),
            epochs=2,
            warmup_epochs=0,
            batch_size=4,
            n_bins=8,
        )
        assert outcome.record.layers is None

    def test_dense_extraction_happens_once_per_split(self, tmp_path, fake_vit, depth_folder):
        """The cache still holds: two splits, two forward passes, no more."""
        train = depth_folder(tmp_path / "train", 8, seed=0)
        val = depth_folder(tmp_path / "val", 8, seed=1)

        visbench.run(
            fake_vit,
            "depth",
            val,
            train_dataset=train,
            cache=FeatureCache(root=tmp_path / "cache"),
            epochs=2,
            warmup_epochs=0,
            batch_size=8,
            n_bins=8,
        )
        assert fake_vit.call_count == 2


# -- streaming from disk ------------------------------------------------------


class TestStreaming:
    """Training from CachedFeatures, which is what lifts the memory ceiling."""

    def _streamed(self, cache, fake_vit, dataset):
        return cache.materialise(fake_vit, dataset, pooling="mean", targets=dataset.target)

    def test_fit_from_a_streaming_source(self, tmp_path, fake_vit, depth_folder):
        dataset = depth_folder(tmp_path / "data", 12, seed=0)
        cache = FeatureCache(root=tmp_path / "cache")
        task = DepthTask(epochs=2, warmup_epochs=0, batch_size=4, n_bins=16, max_depth=8.0)

        task.fit(self._streamed(cache, fake_vit, dataset))
        assert task.train_loss is not None
        assert task.predict(self._streamed(cache, fake_vit, dataset)).shape == (12, 1, 32, 32)

    def test_the_memory_ceiling_does_not_apply(self, tmp_path, fake_vit, depth_folder):
        """The whole point: a ceiling that stops the in-memory path must not
        stop the streaming one, because nothing is being stacked."""
        dataset = depth_folder(tmp_path / "data", 12, seed=0)
        cache = FeatureCache(root=tmp_path / "cache")
        stacked = cache.extract_dataset(fake_vit, dataset, keep="dense", pooling="mean")

        task = DepthTask(epochs=1, warmup_epochs=0, batch_size=4, n_bins=8)
        task.max_feature_elements = stacked["dense"].numel() - 1
        with pytest.raises(ValueError, match="materialise"):
            task.fit(stacked, dataset.targets())

        task.fit(self._streamed(cache, fake_vit, dataset))
        assert task.train_loss is not None

    def test_evaluate_matches_the_in_memory_path(self, tmp_path, fake_vit, depth_folder):
        """Scoring batch by batch must give exactly the whole-split number.

        The metrics are per-image averages, so weighting each batch by its size
        and dividing by the total is the same arithmetic — but only if nothing
        else differs, which is what this pins.
        """
        dataset = depth_folder(tmp_path / "data", 10, seed=3)
        cache = FeatureCache(root=tmp_path / "cache")
        stacked = cache.extract_dataset(fake_vit, dataset, keep="dense", pooling="mean")

        task = DepthTask(epochs=2, warmup_epochs=0, batch_size=4, n_bins=16, max_depth=8.0)
        task.fit(stacked, dataset.targets())

        in_memory = task.evaluate(stacked, dataset.targets())
        streamed = task.evaluate(self._streamed(cache, fake_vit, dataset))
        for name, value in in_memory.items():
            assert streamed[name] == pytest.approx(value, abs=1e-5), name

    def test_batch_size_does_not_change_the_score(self, tmp_path, fake_vit, depth_folder):
        dataset = depth_folder(tmp_path / "data", 10, seed=4)
        cache = FeatureCache(root=tmp_path / "cache")
        task = DepthTask(epochs=1, warmup_epochs=0, batch_size=5, n_bins=8, max_depth=8.0)
        task.fit(self._streamed(cache, fake_vit, dataset))

        wide = task.evaluate(self._streamed(cache, fake_vit, dataset))
        task.batch_size = 3
        narrow = task.evaluate(self._streamed(cache, fake_vit, dataset))
        for name, value in wide.items():
            assert narrow[name] == pytest.approx(value, abs=1e-5), name

    def test_multi_layer_streaming(self, tmp_path, fake_vit, depth_folder):
        dataset = depth_folder(tmp_path / "data", 8, seed=5)
        cache = FeatureCache(root=tmp_path / "cache")
        streamed = cache.materialise(
            fake_vit, dataset, pooling="mean", layers=[2, 5, 8, 11], targets=dataset.target
        )
        task = DepthTask(
            head="dpt",
            layers=[2, 5, 8, 11],
            epochs=2,
            warmup_epochs=0,
            batch_size=4,
            n_bins=8,
            hidden_dim=8,
        )
        task.fit(streamed)
        assert task.predict(streamed).shape == (8, 1, 32, 32)

    def test_targets_given_twice_is_refused(self, tmp_path, fake_vit, depth_folder):
        """Two sources of truth for the supervision; one of them would win
        silently."""
        dataset = depth_folder(tmp_path / "data", 6, seed=6)
        cache = FeatureCache(root=tmp_path / "cache")
        with pytest.raises(ValueError, match="two sources of truth"):
            DepthTask(epochs=1, warmup_epochs=0).fit(
                self._streamed(cache, fake_vit, dataset), dataset.targets()
            )

    def test_streaming_without_targets_says_how_to_supply_them(
        self, tmp_path, fake_vit, depth_folder
    ):
        dataset = depth_folder(tmp_path / "data", 6, seed=7)
        cache = FeatureCache(root=tmp_path / "cache")
        bare = cache.materialise(fake_vit, dataset, pooling="mean")
        with pytest.raises(ValueError, match="materialise"):
            DepthTask(epochs=1, warmup_epochs=0).fit(bare)

    def test_separate_targets_can_still_be_paired(self, tmp_path, fake_vit, depth_folder):
        dataset = depth_folder(tmp_path / "data", 8, seed=8)
        cache = FeatureCache(root=tmp_path / "cache")
        bare = cache.materialise(fake_vit, dataset, pooling="mean")

        task = DepthTask(epochs=1, warmup_epochs=0, batch_size=4, n_bins=8)
        task.fit(bare, dataset.targets())
        assert task.train_loss is not None

    def test_run_uses_the_streaming_path(self, tmp_path, fake_vit, depth_folder):
        """visbench.run() must not stack a dense split, features or targets."""
        from visbench.cache import CachedFeatures

        train = depth_folder(tmp_path / "train", 8, seed=0)
        val = depth_folder(tmp_path / "val", 6, seed=1)

        seen = {}
        original = DepthTask.fit

        def spy(self, features, labels=None):
            seen["features"] = type(features).__name__
            seen["labels"] = labels
            return original(self, features, labels)

        DepthTask.fit = spy
        try:
            visbench.run(
                fake_vit,
                "depth",
                val,
                train_dataset=train,
                cache=FeatureCache(root=tmp_path / "cache"),
                epochs=1,
                warmup_epochs=0,
                batch_size=4,
                n_bins=8,
            )
        finally:
            DepthTask.fit = original

        assert seen["features"] == CachedFeatures.__name__
        assert seen["labels"] is None, "targets should travel with the features"
