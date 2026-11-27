"""The dense standardiser (#172): per-channel, train-split statistics, opt-in.

24b's scale control showed a linear dense board reads a backbone's feature
*scale*: multiplying two columns' features by a constant -- which removes nothing
a linear head could use -- reversed their order on three boards, because AdamW
at a fixed learning rate is not scale-invariant. The property this file exists
to pin is the one that fixes that: **with ``standardize=True``, a constant
rescaling of the features cannot move a prediction.** Everything else here keeps
the switch honest -- off by default and invisible in the record when off,
statistics from the training split only, carried through a saved artifact, and
refused where it cannot mean anything.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.cache import FeatureCache
from visbench.data import DenseFolderDataset, load_mask
from visbench.hub import load_probe, save_probe
from visbench.results.leaderboard import comparability_key
from visbench.results.schema import ResultRecord
from visbench.tasks.high_level.scene_parsing import SceneParsingTask
from visbench.tasks.high_level.semantic_segmentation import SemanticSegmentationTask
from visbench.tasks.low_level.corner import CornerTask
from visbench.tasks.low_level.edge import EdgeTask
from visbench.tasks.low_level.keypoints import Keypoint2DTask
from visbench.tasks.low_level.orientation import OrientationTask
from visbench.tasks.mid_level.depth import DepthTask
from visbench.tasks.mid_level.generic_segmentation import GenericSegmentationTask
from visbench.tasks.mid_level.occlusion_edge import OcclusionEdgeTask
from visbench.tasks.mid_level.surface_normal import SurfaceNormalTask
from visbench.utils import set_seed

#: The ten probes whose boards #172 is about, each constructible with defaults.
TEN = {
    "depth": lambda **kw: DepthTask(**kw),
    "surface_normal": lambda **kw: SurfaceNormalTask(**kw),
    "generic_segmentation": lambda **kw: GenericSegmentationTask(**kw),
    "semantic_segmentation": lambda **kw: SemanticSegmentationTask(num_classes=21, **kw),
    "scene_parsing": lambda **kw: SceneParsingTask(**kw),
    "edge": lambda **kw: EdgeTask(**kw),
    "keypoints2d": lambda **kw: Keypoint2DTask(**kw),
    "occlusion_edge": lambda **kw: OcclusionEdgeTask(**kw),
    "corner": lambda **kw: CornerTask(**kw),
    "orientation": lambda **kw: OrientationTask(**kw),
}


def _features(count: int = 12, channels: int = 6, grid: int = 4, seed: int = 0):
    """Random dense features with an awkward, uneven per-channel scale.

    The target is a fixed linear read-out of the features, so a linear head can
    express it exactly; the channel scales span four orders of magnitude, which
    is the situation the standardiser exists for.
    """
    g = torch.Generator().manual_seed(seed)
    scales = torch.logspace(-2, 2, channels).view(1, -1, 1, 1)
    dense = torch.randn(count, channels, grid, grid, generator=g) * scales + 3.0
    weights = torch.randn(channels, generator=g)
    logits = ((dense - 3.0) / scales * weights.view(1, -1, 1, 1)).sum(dim=1)
    targets = (logits > 0).float()
    return dense, targets


def _fit(dense, targets, **kwargs):
    set_seed(0)
    task = GenericSegmentationTask(epochs=3, batch_size=4, device="cpu", **kwargs)
    return task.fit({"dense": dense}, targets)


# -- the property ---------------------------------------------------------------


@pytest.mark.parametrize("factor", [8.66, 0.1155, 1000.0])
def test_a_constant_rescaling_cannot_move_a_standardised_prediction(factor):
    """24b's control, as a unit test: the scaled arm must equal the unscaled one."""
    dense, targets = _features()
    plain = _fit(dense, targets, standardize=True)
    scaled = _fit(dense * factor, targets, standardize=True)
    torch.testing.assert_close(
        plain.predict({"dense": dense}),
        scaled.predict({"dense": dense * factor}),
        rtol=1e-4,
        atol=1e-5,
    )


def test_without_it_the_same_rescaling_does_move_the_prediction():
    """Proves the test above is not vacuous: the raw path is scale-sensitive."""
    dense, targets = _features()
    plain = _fit(dense, targets)
    scaled = _fit(dense * 8.66, targets)
    assert not torch.allclose(
        plain.predict({"dense": dense}), scaled.predict({"dense": dense * 8.66}), atol=1e-3
    )


# -- the statistics ---------------------------------------------------------------


def test_statistics_are_per_channel_and_come_from_the_training_split():
    dense, targets = _features()
    task = _fit(dense, targets, standardize=True)
    assert task._means is not None and task._stds is not None
    assert len(task._means) == 1
    expected_mean = dense.double().mean(dim=(0, 2, 3)).float().view(1, -1, 1, 1)
    expected_std = dense.double().std(dim=(0, 2, 3), unbiased=False).float().view(1, -1, 1, 1)
    torch.testing.assert_close(task._means[0], expected_mean)
    torch.testing.assert_close(task._stds[0], expected_std, rtol=1e-5, atol=1e-6)


def test_evaluation_reuses_the_training_statistics():
    """Recomputing them on the evaluation split would leak it into the probe."""
    dense, targets = _features()
    task = _fit(dense, targets, standardize=True)
    before = [m.clone() for m in task._means]
    task.evaluate({"dense": dense + 50.0}, targets)
    task.predict({"dense": dense * 3.0})
    for kept, original in zip(task._means, before, strict=True):
        assert torch.equal(kept, original)


def test_a_constant_channel_does_not_divide_by_zero():
    dense, targets = _features()
    dense[:, 0] = 7.0
    task = _fit(dense, targets, standardize=True)
    assert torch.isfinite(task.predict({"dense": dense})).all()


def test_multi_layer_features_get_one_pair_of_statistics_per_layer():
    dense, targets = _features()
    layers = [dense, dense[:, :3] * 10.0]
    set_seed(0)
    task = GenericSegmentationTask(
        head="dpt",
        layers=[0, 1],
        hidden_dim=8,
        epochs=2,
        batch_size=4,
        device="cpu",
        standardize=True,
    ).fit({"dense_layers": layers}, targets)
    assert len(task._means) == 2
    assert task._means[0].shape == (1, 6, 1, 1)
    assert task._means[1].shape == (1, 3, 1, 1)


# -- off by default, and invisible when off ----------------------------------------


def test_off_by_default_and_the_record_is_unchanged():
    """Old records lack the key; writing ``standardize: False`` would split every
    new default run from the published boards it reproduces."""
    for name, build in TEN.items():
        params = build().describe()["task_params"]
        assert "standardize" not in params, name
        assert not params["protocol"].endswith("+std"), name


def test_on_and_off_land_in_different_comparability_groups():
    """A standardised number must never share a ranking with a raw one."""
    dense, targets = _features()

    def record(task) -> ResultRecord:
        described = task.describe()
        return ResultRecord(
            backbone="b",
            backbone_key="k",
            task=described["task"],
            level=described["level"],
            dataset="d",
            split="val",
            pooling="mean",
            feature_mode="dense_only",
            metrics={"iou": 0.5},
            task_params=described["task_params"],
            timestamp="2026-11-27T00:00:00Z",
            visbench_version="0.0.0",
        )

    off = record(_fit(dense, targets))
    on = record(_fit(dense, targets, standardize=True))
    assert comparability_key(off) != comparability_key(on)


def test_a_default_run_writes_exactly_the_keys_the_published_records_carry():
    """The corpus is the reference: a new default run must group with it."""
    import json
    from pathlib import Path

    corpus = Path(__file__).resolve().parents[2] / "results" / "corpus" / "visbench.jsonl"
    published = {
        frozenset(json.loads(line)["task_params"])
        for line in corpus.read_text().splitlines()
        if json.loads(line)["task"] == "generic_segmentation"
    }
    dense, targets = _features()
    assert frozenset(_fit(dense, targets).describe()["task_params"]) in published


@pytest.mark.parametrize("name", sorted(TEN))
def test_every_one_of_the_ten_protocols_gains_the_suffix_only_when_on(name):
    off = TEN[name]().describe()["task_params"]
    on = TEN[name](standardize=True).describe()["task_params"]
    assert on["protocol"] == off["protocol"] + "+std"
    assert on["standardize"] is True


def test_scene_parsing_keeps_its_own_protocol():
    assert SceneParsingTask().describe()["task_params"]["protocol"] == "visbench_scene_parsing"
    assert (
        SceneParsingTask(standardize=True).describe()["task_params"]["protocol"]
        == "visbench_scene_parsing+std"
    )


# -- refusals -----------------------------------------------------------------------


def test_refused_with_fine_tuning():
    with pytest.raises(ValueError, match="finetune_blocks"):
        GenericSegmentationTask(standardize=True, finetune_blocks=2)


def test_a_standardised_probe_without_statistics_refuses_to_predict():
    dense, targets = _features()
    task = _fit(dense, targets, standardize=True)
    task._means = task._stds = None
    with pytest.raises(RuntimeError, match="no statistics"):
        task.predict({"dense": dense})


# -- the artifact ---------------------------------------------------------------------


@pytest.fixture
def real_features(tmp_path, fake_vit):
    """Features from a fake backbone, so the probe can be saved against it."""
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "masks").mkdir(parents=True)
    rng = np.random.RandomState(0)
    for index in range(8):
        quarters = int(rng.randint(1, 4))
        level = np.uint8(40 + 60 * quarters)
        Image.fromarray(np.full((32, 32, 3), level, dtype=np.uint8)).save(
            tmp_path / "data" / "images" / f"s{index:03d}.png"
        )
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[: 8 * quarters] = 255
        np.save(tmp_path / "data" / "masks" / f"s{index:03d}.npy", mask)
    dataset = DenseFolderDataset(
        tmp_path / "data", target_dir="masks", image_size=32, target_loader=load_mask
    )
    features = FeatureCache(root=tmp_path / "cache").extract_dataset(
        fake_vit, dataset, keep="dense", pooling="mean"
    )
    return features, dataset.targets()


def test_the_statistics_travel_with_the_head(real_features, fake_vit, tmp_path):
    features, targets = real_features
    set_seed(0)
    task = GenericSegmentationTask(epochs=2, device="cpu", standardize=True).fit(features, targets)
    before = task.predict(features)
    save_probe(task, tmp_path / "probe.pt", backbone=fake_vit)

    loaded = load_probe(tmp_path / "probe.pt", backbone=fake_vit)
    assert loaded.standardize
    # Close rather than equal: the reloaded probe runs on the default device,
    # which is a GPU on the cluster and the CPU on CI.
    torch.testing.assert_close(before, loaded.predict(features).cpu(), rtol=1e-5, atol=1e-6)


def test_dropping_the_statistics_would_change_the_answer(real_features):
    """Proves the test above is not vacuous: the statistics decide the answer."""
    features, targets = real_features
    set_seed(0)
    task = GenericSegmentationTask(epochs=2, device="cpu", standardize=True).fit(features, targets)
    with_them = task.predict(features)
    task._means = [torch.zeros_like(m) for m in task._means]
    task._stds = [torch.ones_like(s) for s in task._stds]
    assert not torch.allclose(with_them, task.predict(features), atol=1e-4)


def test_a_probe_without_the_flag_carries_no_state(real_features):
    features, targets = real_features
    task = GenericSegmentationTask(epochs=2, device="cpu").fit(features, targets)
    assert task.probe_state() == {}


def test_loading_refuses_a_standardised_probe_with_no_statistics():
    task = GenericSegmentationTask(standardize=True)
    with pytest.raises(ValueError, match="no mean/std"):
        task.load_probe_state({})


def test_loading_refuses_a_malformed_state():
    task = GenericSegmentationTask()
    with pytest.raises(ValueError, match="Unexpected probe state"):
        task.load_probe_state({"mean_0": torch.zeros(1)})
