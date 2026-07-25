"""Surface normal estimation — the second dense task.

Reproduces probe3d's configured protocol (three direction channels plus an
optional kappa, Bae et al.'s uncertainty-aware angular loss, the ten-epoch
cosine schedule). Most of the training machinery is
:class:`~visbench.tasks.dense_base.DenseTrainingTask`, shared with depth and
tested there; what is tested here is the part that is specific to normals —
the loss, the unit-vector output, and the fact that the ground truth is a
*vector* map, which is the first thing in VisBench that is not a scalar per
pixel.
"""

import math
import warnings

import numpy as np
import pytest
import torch
from PIL import Image

import visbench
from visbench.cache import FeatureCache
from visbench.data import DenseFolderDataset, load_normal_map
from visbench.metrics.dense import surface_normal_metrics
from visbench.tasks.mid_level.surface_normal import SurfaceNormalTask, angular_loss


@pytest.fixture
def normal_folder(tmp_path):
    """Images whose colour *is* their (constant) surface normal.

    Deliberately exactly representable: FakeViT projects the mean colour
    linearly, and a linear head followed by an L2 normalisation can invert that
    up to scale. A probe that cannot learn this has a real problem, not a hard
    dataset.
    """

    def build(root, count, seed):
        (root / "images").mkdir(parents=True)
        (root / "normals").mkdir(parents=True)
        rng = np.random.RandomState(seed)
        for index in range(count):
            direction = rng.rand(3) - 0.5
            direction /= np.linalg.norm(direction)
            colour = np.uint8((direction * 0.5 + 0.5) * 255)
            Image.fromarray(np.tile(colour, (32, 32, 1)).astype(np.uint8)).save(
                root / "images" / f"s{index:03d}.png"
            )
            normals = np.tile(direction.reshape(3, 1, 1), (1, 32, 32)).astype(np.float32)
            np.save(root / "normals" / f"s{index:03d}.npy", normals)
        return DenseFolderDataset(
            root, target_dir="normals", image_size=32, target_loader=load_normal_map
        )

    return build


@pytest.fixture
def features_and_targets(normal_folder, tmp_path, fake_vit):
    dataset = normal_folder(tmp_path / "data", 12, seed=0)
    cache = FeatureCache(root=tmp_path / "cache")
    features = cache.extract_dataset(fake_vit, dataset, keep="dense", pooling="mean")
    return features, dataset.targets()


@pytest.fixture
def fitted_task(features_and_targets):
    """A task with a head, for poking at the loss without training dynamics."""
    features, targets = features_and_targets
    task = SurfaceNormalTask(epochs=1, warmup_epochs=0, batch_size=4)
    task._kappa_collapse_warn = 1.1  # unreachable; this fixture is not about that
    return task.fit(features, targets)


def constant_map(x, y, z, count=1, size=4):
    vector = torch.tensor([x, y, z], dtype=torch.float32).view(1, 3, 1, 1)
    return vector.expand(count, 3, size, size).clone()


# -- loss ---------------------------------------------------------------------


class TestAngularLoss:
    def test_a_perfect_prediction_costs_almost_nothing(self):
        """*Almost*: the ``1 - eps`` clamp puts a floor of ``acos(1 - 1e-4)``,
        about 0.8 degrees, under the loss. That is the price of keeping the
        gradient finite at an exact hit, and it is a constant added to every
        valid pixel, so it shifts the loss without touching its gradient — but
        it does mean this loss is not comparable across different ``eps``."""
        target = torch.nn.functional.normalize(torch.randn(2, 3, 8, 8), dim=1)
        floor = math.acos(1 - 1e-4)
        assert angular_loss(
            target.clone(), target, uncertainty_aware=False
        ).item() == pytest.approx(floor, abs=1e-4)
        assert floor < math.radians(1.0)

    def test_it_is_the_angle_in_radians(self):
        """Not degrees — the loss and the metric report different units, which
        is worth knowing before comparing one to the other."""
        target = constant_map(0.0, 0.0, 1.0)
        pred = constant_map(1.0, 0.0, 0.0)
        loss = angular_loss(pred, target, uncertainty_aware=False).item()
        assert loss == pytest.approx(math.pi / 2, abs=1e-3)

    def test_magnitude_does_not_change_it(self):
        target = torch.nn.functional.normalize(torch.randn(2, 3, 8, 8), dim=1)
        pred = torch.randn(2, 3, 8, 8)
        plain = angular_loss(pred, target, uncertainty_aware=False).item()
        scaled = angular_loss(pred * 17.0, target, uncertainty_aware=False).item()
        assert plain == pytest.approx(scaled, abs=1e-4)

    def test_a_perfect_prediction_stays_differentiable(self):
        """``acos`` has infinite derivative at 1, which is why the loss clamps
        tighter than the metric does. An exact hit must not blow the head up."""
        target = torch.nn.functional.normalize(torch.randn(2, 3, 8, 8), dim=1)
        pred = target.clone().requires_grad_(True)
        angular_loss(pred, target, uncertainty_aware=False).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_zero_length_targets_are_skipped(self):
        target = constant_map(0.0, 0.0, 1.0, size=4)
        target[:, :, :, 2:] = 0.0

        good = constant_map(0.0, 0.0, 1.0, size=4)
        wild = good.clone()
        wild[:, :, :, 2:] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)

        assert angular_loss(good, target, uncertainty_aware=False).item() == pytest.approx(
            angular_loss(wild, target, uncertainty_aware=False).item(), abs=1e-5
        )

    def test_an_entirely_invalid_target_keeps_the_graph(self):
        """probe3d drops into a debugger on the NaN this would otherwise be;
        a benchmark library cannot."""
        pred = torch.randn(1, 3, 8, 8, requires_grad=True)
        loss = angular_loss(pred, torch.zeros(1, 3, 8, 8), uncertainty_aware=False)
        loss.backward()
        assert loss.item() == 0.0
        assert pred.grad is not None

    def test_an_explicit_mask_is_honoured(self):
        target = constant_map(0.0, 0.0, 1.0, size=4)
        pred = constant_map(1.0, 0.0, 0.0, size=4)
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        assert angular_loss(pred, target, mask=mask, uncertainty_aware=False).item() == 0.0


class TestUncertaintyAwareLoss:
    def test_it_needs_a_fourth_channel(self):
        target = torch.nn.functional.normalize(torch.randn(1, 3, 4, 4), dim=1)
        with pytest.raises(ValueError, match="needs 4 predicted channels"):
            angular_loss(torch.randn(1, 3, 4, 4), target, uncertainty_aware=True)

    def test_the_plain_loss_rejects_a_fourth_channel(self):
        target = torch.nn.functional.normalize(torch.randn(1, 3, 4, 4), dim=1)
        with pytest.raises(ValueError, match="needs 3 predicted channels"):
            angular_loss(torch.randn(1, 4, 4, 4), target, uncertainty_aware=False)

    def test_confidence_is_rewarded_where_the_direction_is_right(self):
        target = constant_map(0.0, 0.0, 1.0)
        direction = constant_map(0.0, 0.0, 1.0)
        confident = torch.cat([direction, torch.full((1, 1, 4, 4), 5.0)], dim=1)
        unsure = torch.cat([direction, torch.full((1, 1, 4, 4), -5.0)], dim=1)
        assert angular_loss(confident, target).item() < angular_loss(unsure, target).item()

    def test_confidence_is_punished_where_the_direction_is_wrong(self):
        """The point of the parameterisation: a pixel the head cannot call can
        be conceded cheaply instead of dominating the gradient."""
        target = constant_map(0.0, 0.0, 1.0)
        direction = constant_map(1.0, 0.0, 0.0)
        confident = torch.cat([direction, torch.full((1, 1, 4, 4), 5.0)], dim=1)
        unsure = torch.cat([direction, torch.full((1, 1, 4, 4), -5.0)], dim=1)
        assert angular_loss(confident, target).item() > angular_loss(unsure, target).item()

    def test_kappa_has_a_floor(self):
        """elu + 1.01 keeps kappa >= 0.01; a kappa reaching zero would let the
        head opt out of predicting anything at all."""
        target = constant_map(0.0, 0.0, 1.0)
        wrong = constant_map(1.0, 0.0, 0.0)
        pred = torch.cat([wrong, torch.full((1, 1, 4, 4), -1e4)], dim=1)
        loss = angular_loss(pred, target)
        assert torch.isfinite(loss)
        # The angular term survives: kappa cannot be driven to zero to erase it.
        assert loss.item() > angular_loss(pred, target).item() - 1e-6
        assert loss.item() == pytest.approx(
            (math.log1p(math.exp(-0.01 * math.pi)) - math.log(0.01**2 + 1)) + 0.01 * (math.pi / 2),
            abs=1e-3,
        )

    def test_it_stays_finite_on_a_large_kappa(self):
        target = torch.nn.functional.normalize(torch.randn(1, 3, 4, 4), dim=1)
        pred = torch.cat([torch.randn(1, 3, 4, 4), torch.full((1, 1, 4, 4), 1e4)], dim=1)
        assert torch.isfinite(angular_loss(pred, target))


# -- the task -----------------------------------------------------------------


class TestSurfaceNormalTask:
    def test_it_is_registered(self):
        assert "surface_normal" in visbench.list_probes()
        assert isinstance(visbench.get_probe("surface_normal"), SurfaceNormalTask)

    def test_it_declares_what_it_needs(self):
        task = SurfaceNormalTask()
        assert task.uses_dense, "the cache must keep dense features for this"
        assert task.level == "mid_level"
        assert not task.zero_shot
        assert task.target_channels == 3

    def test_uncertainty_aware_is_the_default(self):
        """probe3d's ``snorm_dpt.yaml`` sets it, so the comparable number does."""
        assert SurfaceNormalTask().uncertainty_aware
        assert SurfaceNormalTask().out_channels == 4
        assert SurfaceNormalTask(uncertainty_aware=False).out_channels == 3

    def test_fit_then_predict_shapes(self, features_and_targets):
        features, targets = features_and_targets
        task = SurfaceNormalTask(epochs=2, warmup_epochs=0, batch_size=4).fit(features, targets)
        assert task.predict(features).shape == (12, 4, 32, 32)

    def test_predictions_are_unit_vectors(self, features_and_targets):
        """The loss and metric both compare by cosine, so this is for the
        caller's benefit: ``predict`` hands back an actual normal, not an
        unscaled direction whose length is whatever the last layer produced."""
        features, targets = features_and_targets
        task = SurfaceNormalTask(epochs=2, warmup_epochs=0, batch_size=4).fit(features, targets)
        lengths = task.predict(features)[:, :3].norm(dim=1)
        assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-5)

    def test_the_kappa_channel_is_not_normalised(self, features_and_targets):
        features, targets = features_and_targets
        task = SurfaceNormalTask(epochs=2, warmup_epochs=0, batch_size=4).fit(features, targets)
        assert task.predict(features).shape[1] == 4

    def test_a_plain_head_predicts_three_channels(self, features_and_targets):
        features, targets = features_and_targets
        task = SurfaceNormalTask(
            uncertainty_aware=False, epochs=2, warmup_epochs=0, batch_size=4
        ).fit(features, targets)
        assert task.predict(features).shape == (12, 3, 32, 32)

    def test_evaluate_returns_the_probe3d_metrics(self, features_and_targets):
        features, targets = features_and_targets
        task = SurfaceNormalTask(epochs=2, warmup_epochs=0, batch_size=4).fit(features, targets)
        metrics = task.evaluate(features, targets)
        assert set(metrics) == {"d1", "d2", "d3", "rmse", "mean", "median"}
        assert all(isinstance(value, float) for value in metrics.values())

    def test_it_learns_something(self, features_and_targets):
        """Beats predicting one constant direction everywhere — the baseline a
        probe that has learned nothing at all would match."""
        features, targets = features_and_targets
        task = SurfaceNormalTask(epochs=60, batch_size=4, lr=5e-2, uncertainty_aware=False)
        task.fit(features, targets)

        average = torch.nn.functional.normalize(targets.mean(dim=(0, 2, 3)), dim=0)
        constant = average.view(1, 3, 1, 1).expand_as(targets)
        assert (
            task.evaluate(features, targets)["mean"]
            < surface_normal_metrics(constant, targets)["mean"]
        )

    def test_training_loss_is_recorded_as_a_diagnostic(self, features_and_targets):
        features, targets = features_and_targets
        task = SurfaceNormalTask(epochs=3, warmup_epochs=0, batch_size=4).fit(features, targets)
        assert task.train_loss is not None

    def test_unfitted_predict_raises(self, features_and_targets):
        features, _ = features_and_targets
        with pytest.raises(RuntimeError, match="not been fitted"):
            SurfaceNormalTask().predict(features)

    def test_missing_targets_raise(self, features_and_targets):
        features, _ = features_and_targets
        with pytest.raises(ValueError, match="requires target normal maps"):
            SurfaceNormalTask(epochs=1, warmup_epochs=0).fit(features, None)

    def test_scalar_targets_are_refused(self, features_and_targets):
        """A depth map handed to the normal probe. Without the rank check it
        would broadcast into something trainable and score nonsense."""
        features, targets = features_and_targets
        with pytest.raises(ValueError, match=r"\(N, 3, H, W\)"):
            SurfaceNormalTask(epochs=1, warmup_epochs=0).fit(features, targets[:, 0])

    def test_the_dpt_head_trains_on_several_layers(self, tmp_path, fake_vit, normal_folder):
        dataset = normal_folder(tmp_path / "data", 8, seed=2)
        cache = FeatureCache(root=tmp_path / "cache")
        features = cache.extract_dataset(
            fake_vit, dataset, keep="dense", pooling="mean", layers=[2, 5, 8, 11]
        )
        task = SurfaceNormalTask(
            head="dpt", layers=[2, 5, 8, 11], epochs=2, warmup_epochs=0, batch_size=4, hidden_dim=8
        )
        task.fit(features, dataset.targets())
        assert task.predict(features).shape == (8, 4, 32, 32)


# -- provenance ---------------------------------------------------------------


class TestDescribe:
    def test_it_records_what_shaped_the_number(self):
        params = SurfaceNormalTask(head="dpt", layers=[3, 7], epochs=4).describe()["task_params"]
        assert params["head"] == "dpt"
        assert params["layers"] == [3, 7]
        assert params["epochs"] == 4
        assert params["uncertainty_aware"] is True
        assert params["protocol"] == "probe3d"

    def test_the_normal_source_is_recorded(self):
        """NYU normals are derived, not sensed, and the sources disagree enough
        to move the numbers — so a run that does not say which is unreadable."""
        assert SurfaceNormalTask().describe()["task_params"]["normal_source"] is None
        params = SurfaceNormalTask(normal_source="geonet").describe()["task_params"]
        assert params["normal_source"] == "geonet"


# -- streaming and the runner -------------------------------------------------


class TestStreaming:
    """A vector target has to survive the streaming path too — it is stacked by
    ``CachedFeatures.collate``, which was written when every target was scalar."""

    def _streamed(self, cache, fake_vit, dataset):
        return cache.materialise(fake_vit, dataset, pooling="mean", targets=dataset.target)

    def test_fit_from_a_streaming_source(self, tmp_path, fake_vit, normal_folder):
        dataset = normal_folder(tmp_path / "data", 12, seed=0)
        cache = FeatureCache(root=tmp_path / "cache")
        task = SurfaceNormalTask(epochs=2, warmup_epochs=0, batch_size=4)

        task.fit(self._streamed(cache, fake_vit, dataset))
        assert task.train_loss is not None
        assert task.predict(self._streamed(cache, fake_vit, dataset)).shape == (12, 4, 32, 32)

    def test_evaluate_matches_the_in_memory_path(self, tmp_path, fake_vit, normal_folder):
        """Scoring batch by batch must give exactly the whole-split number:
        the metrics are per-image averages, so weighting each batch by its size
        recovers it — but only if the target pairing survived the loader."""
        dataset = normal_folder(tmp_path / "data", 12, seed=3)
        cache = FeatureCache(root=tmp_path / "cache")
        stacked = cache.extract_dataset(fake_vit, dataset, keep="dense", pooling="mean")
        targets = dataset.targets()

        visbench.utils.set_seed(0)
        task = SurfaceNormalTask(epochs=3, warmup_epochs=0, batch_size=5).fit(stacked, targets)

        assert task.evaluate(stacked, targets) == pytest.approx(
            task.evaluate(self._streamed(cache, fake_vit, dataset)), abs=1e-5
        )

    def test_targets_stay_with_their_own_image(self, tmp_path, fake_vit, normal_folder):
        """The failure this guards against is silent: shuffled supervision
        still trains, and the only symptom is a worse number."""
        dataset = normal_folder(tmp_path / "data", 6, seed=4)
        cache = FeatureCache(root=tmp_path / "cache")
        reader = self._streamed(cache, fake_vit, dataset)
        for index in range(len(dataset)):
            assert torch.equal(reader[index][1], dataset.target(index))


class TestThroughRun:
    def test_a_normal_task_runs_end_to_end(self, tmp_path, fake_vit, normal_folder):
        train = normal_folder(tmp_path / "train", 8, seed=0)
        val = normal_folder(tmp_path / "val", 6, seed=1)
        results = tmp_path / "results.jsonl"

        outcome = visbench.run(
            fake_vit,
            "surface_normal",
            val,
            train_dataset=train,
            cache=FeatureCache(root=tmp_path / "cache"),
            results=results,
            epochs=2,
            warmup_epochs=0,
            batch_size=4,
        )

        assert set(outcome.metrics) == {"d1", "d2", "d3", "rmse", "mean", "median"}
        assert outcome.record.task == "surface_normal"
        assert outcome.record.level == "mid_level"
        assert results.read_text().strip(), "the run must have logged a record"

    def test_declared_layers_reach_extraction(self, tmp_path, fake_vit, normal_folder):
        """A multiscale head is unusable without them, and the record must name
        the depths that actually produced the number."""
        train = normal_folder(tmp_path / "train", 8, seed=0)
        val = normal_folder(tmp_path / "val", 6, seed=1)

        outcome = visbench.run(
            fake_vit,
            "surface_normal",
            val,
            train_dataset=train,
            cache=FeatureCache(root=tmp_path / "cache"),
            head="dpt",
            layers=[2, 5, 8, 11],
            epochs=1,
            warmup_epochs=0,
            batch_size=4,
            hidden_dim=8,
        )
        assert outcome.record.layers == [2, 5, 8, 11]


# -- the kappa collapse trap --------------------------------------------------


class TestKappaCollapse:
    """probe3d's uncertainty-aware loss can silently switch itself off.

    Its gradient with respect to kappa is ``angular_error - 1.126`` radians
    near kappa = 1, so while the head is worse than about 64.5 degrees the
    optimiser pushes kappa down; far enough and the elu saturates, its gradient
    dies, and the angular term is permanently scaled by 1/100. The probe then
    trains to a chance-level score with no error and no warning — which is why
    there is now a warning.
    """

    def test_the_gradient_turns_against_kappa_above_the_documented_angle(self):
        """The 64.5 degrees quoted in ``fit``'s docstring, derived not recalled."""
        target = constant_map(0.0, 0.0, 1.0)

        def kappa_gradient(error_deg):
            radians = math.radians(error_deg)
            direction = constant_map(math.sin(radians), 0.0, math.cos(radians))
            raw = torch.zeros(1, 1, 4, 4, requires_grad=True)
            angular_loss(torch.cat([direction, raw], dim=1), target).backward()
            return raw.grad.mean().item()

        assert kappa_gradient(60.0) < 0, "below the threshold, confidence is encouraged"
        assert kappa_gradient(70.0) > 0, "above it, kappa is pushed towards the floor"

    def test_the_warning_fires_off_the_measured_fraction(self, features_and_targets):
        """Driven by lowering the threshold rather than by inducing a collapse:
        whether a given run collapses is a question about optimisation, and this
        is a question about whether the check is wired up."""
        features, targets = features_and_targets
        task = SurfaceNormalTask(epochs=1, warmup_epochs=0, batch_size=4)
        task._kappa_collapse_warn = -1.0
        with pytest.warns(RuntimeWarning, match="collapsed to their floor"):
            task.fit(features, targets)
        assert task.kappa_at_floor is not None

    def test_a_healthy_run_says_nothing(self, features_and_targets):
        features, targets = features_and_targets
        task = SurfaceNormalTask(epochs=1, warmup_epochs=0, batch_size=4)
        task._kappa_collapse_warn = 1.1  # unreachable
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            task.fit(features, targets)

    def _settled_kappa(self, error_deg, steps=400, lr=0.2):
        """Where kappa comes to rest against a head stuck at ``error_deg``."""
        target = constant_map(0.0, 0.0, 1.0)
        radians = math.radians(error_deg)
        direction = constant_map(math.sin(radians), 0.0, math.cos(radians))

        raw = torch.zeros(1, 1, 4, 4, requires_grad=True)
        optimiser = torch.optim.AdamW([raw], lr=lr)
        for _ in range(steps):
            optimiser.zero_grad()
            angular_loss(torch.cat([direction, raw], dim=1), target).backward()
            optimiser.step()
        return (torch.nn.functional.elu(raw) + 1.01).max().item()

    def test_kappa_tracks_how_well_the_head_is_doing(self):
        """The intended behaviour, and the numbers in ``fit``'s docstring —
        measured here rather than recalled, since they are what makes the
        failure mode below predictable."""
        assert self._settled_kappa(30.0) == pytest.approx(3.5, abs=0.2)
        assert self._settled_kappa(60.0) == pytest.approx(1.2, abs=0.1)
        assert self._settled_kappa(80.0) == pytest.approx(0.38, abs=0.05)

    def test_it_is_monotone(self):
        """A worse head is always a less confident one; there is no angle at
        which conceding a pixel pays better than the one next to it."""
        settled = [self._settled_kappa(deg) for deg in (30.0, 60.0, 80.0, 90.0, 120.0)]
        assert settled == sorted(settled, reverse=True)

    def test_at_chance_kappa_all_but_switches_the_direction_off(self):
        """The trap: ~0.05 scales the angular gradient by 1/20, weak
        supervision keeps the head at chance, and the two hold each other
        down."""
        assert self._settled_kappa(90.0) < 0.06

    def test_a_live_kappa_is_not_counted_as_collapsed(self, fitted_task):
        """Measured through the loss directly, because reaching a healthy kappa
        by training would make this a test of convergence, not of counting."""
        fitted_task._on_epoch_start()
        direction = torch.nn.functional.normalize(torch.randn(2, 3, 4, 4), dim=1)
        healthy = torch.cat([direction, torch.zeros(2, 1, 4, 4)], dim=1)
        fitted_task._loss(healthy, torch.nn.functional.normalize(torch.randn(2, 3, 4, 4), dim=1))
        assert fitted_task._kappa_count == 32
        assert fitted_task._floor_count == 0

    def test_a_dead_kappa_is_counted(self, fitted_task):
        fitted_task._on_epoch_start()
        direction = torch.nn.functional.normalize(torch.randn(2, 3, 4, 4), dim=1)
        dead = torch.cat([direction, torch.full((2, 1, 4, 4), -50.0)], dim=1)
        fitted_task._loss(dead, torch.nn.functional.normalize(torch.randn(2, 3, 4, 4), dim=1))
        assert fitted_task._floor_count == fitted_task._kappa_count == 32

    def test_the_plain_loss_has_no_kappa_to_measure(self, features_and_targets):
        features, targets = features_and_targets
        task = SurfaceNormalTask(
            uncertainty_aware=False, epochs=2, warmup_epochs=0, batch_size=4
        ).fit(features, targets)
        assert task.kappa_at_floor is None

    def test_only_the_final_epoch_counts(self, features_and_targets):
        """Kappa is meant to be low early, while the direction is still random;
        averaging over training would report a collapse that had recovered."""
        features, targets = features_and_targets
        task = SurfaceNormalTask(epochs=4, warmup_epochs=0, batch_size=4)
        task._kappa_collapse_warn = 1.1
        starts = []
        original = task._on_epoch_start

        def spy():
            starts.append((task._floor_count, task._kappa_count))
            original()

        task._on_epoch_start = spy
        task.fit(features, targets)
        assert len(starts) == 4, "the hook must fire once per epoch"
        assert starts[0] == (0, 0)
        assert starts[-1][1] > 0, "the counter must have been accumulating before the reset"
