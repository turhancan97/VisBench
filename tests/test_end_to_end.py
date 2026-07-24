"""The full v0.1 path on a small local image folder.

This is what build step 3 exists to prove: folder -> backbone -> cache -> task
-> metrics -> structured record, with nothing stubbed in between.

The fake-backbone version runs by default. The same flow against real DINOv2
weights is marked ``slow``.
"""

import time

import pytest
import torch
from PIL import Image

import visbench
from visbench.cache import FeatureCache
from visbench.data import ImageFolderDataset
from visbench.results import ResultRecord, ResultWriter, read_records
from visbench.results.schema import utc_timestamp
from visbench.utils import set_seed


@pytest.fixture
def image_folder(tmp_path):
    """Two classes of four images, separable by colour alone.

    Deliberately trivial: this test proves the *path* works, not that any
    backbone is good. A test that could fail because a model underperforms
    would be a bad regression signal.
    """
    root = tmp_path / "tiny"
    palette = {"red": (200, 30, 30), "blue": (30, 30, 200)}
    for class_name, colour in palette.items():
        directory = root / class_name
        directory.mkdir(parents=True)
        for i in range(4):
            jitter = tuple(min(255, c + i * 5) for c in colour)
            Image.new("RGB", (64, 64), jitter).save(directory / f"{i}.png")
    return root


@pytest.fixture
def split_folders(tmp_path):
    """root/train/<class>/… and root/val/<class>/… — two independent datasets.

    No splitting machinery: a train/test split is two ImageFolderDatasets, so
    each half gets its own fingerprint in its own record.
    """
    root = tmp_path / "split"
    palette = {"red": (200, 30, 30), "blue": (30, 30, 200), "green": (30, 200, 30)}
    # Disjoint jitter ranges. Overlapping them would make some val images
    # byte-identical to train ones — leakage, and the content-addressed cache
    # would correctly collapse them into shared entries.
    for split, offsets in [("train", range(6)), ("val", range(10, 13))]:
        for class_name, colour in palette.items():
            directory = root / split / class_name
            directory.mkdir(parents=True)
            for i in offsets:
                jitter = tuple(min(255, c + i * 4) for c in colour)
                Image.new("RGB", (64, 64), jitter).save(directory / f"{i:02d}.png")
    return root / "train", root / "val"


def run_probe(backbone, dataset, cache, probe, seed=0):
    """Extract, evaluate, and build the record — the shape a CLI would take.

    Lives in the test rather than the library on purpose: the library helper
    that replaces it should be designed against a trained task as well as a
    zero-shot one, which arrives at build step 4.
    """
    used_seed = set_seed(seed)
    started = time.perf_counter()

    features = cache.extract_dataset(backbone, dataset, pooling=probe.pooling, keep="pooled")
    metrics = probe.fit(features).evaluate(features, dataset.labels())

    described = {**dataset.describe(), **probe.describe()}
    return metrics, ResultRecord(
        backbone=backbone.name,
        backbone_key=backbone.cache_key(),
        task=described["task"],
        level=described["level"],
        dataset=described["dataset"],
        split=described["split"],
        dataset_size=described["dataset_size"],
        dataset_fingerprint=described["dataset_fingerprint"],
        pooling=described["pooling"],
        feature_mode=described["feature_mode"],
        metrics=metrics,
        timestamp=utc_timestamp(),
        visbench_version=visbench.__version__,
        seed=used_seed,
        duration_seconds=time.perf_counter() - started,
    )


def test_folder_to_record(tmp_path, image_folder, fake_vit):
    dataset = ImageFolderDataset(image_folder, split="val")
    cache = FeatureCache(root=tmp_path / "cache")
    probe = visbench.get_probe("retrieval", topk=(1, 3))

    metrics, record = run_probe(fake_vit, dataset, cache, probe)

    assert set(metrics) == {"recall@1", "recall@3", "mAP"}
    assert all(isinstance(v, float) for v in metrics.values())

    path = tmp_path / "results.jsonl"
    with ResultWriter(path) as writer:
        writer.write(record)

    (loaded,) = read_records(path)
    assert loaded == record
    assert loaded.dataset == "tiny"
    assert loaded.split == "val"
    assert loaded.task == "retrieval"
    assert loaded.backbone_key == fake_vit.cache_key()
    assert loaded.dataset_size == 8
    assert loaded.dataset_fingerprint
    assert loaded.seed == 0
    assert loaded.duration_seconds > 0


def test_changing_the_data_changes_the_record(tmp_path, image_folder, fake_vit):
    """Two runs over different images must not produce identical-looking records."""
    cache = FeatureCache(root=tmp_path / "cache")
    probe = visbench.get_probe("retrieval")

    _, before = run_probe(fake_vit, ImageFolderDataset(image_folder), cache, probe)

    Image.new("RGB", (64, 64), (10, 200, 10)).save(image_folder / "red" / "extra.png")
    _, after = run_probe(fake_vit, ImageFolderDataset(image_folder), cache, probe)

    assert before.dataset == after.dataset, "same folder name, as the scenario requires"
    assert before.dataset_fingerprint != after.dataset_fingerprint
    assert (before.dataset_size, after.dataset_size) == (8, 9)


def test_second_run_reuses_the_cache(tmp_path, image_folder, fake_vit):
    """The v0.1 promise, over the real data path rather than a list of PILs."""
    dataset = ImageFolderDataset(image_folder)
    cache = FeatureCache(root=tmp_path / "cache")
    probe = visbench.get_probe("retrieval")

    first, _ = run_probe(fake_vit, dataset, cache, probe)
    assert fake_vit.call_count == 1

    second, _ = run_probe(fake_vit, dataset, cache, probe)
    assert fake_vit.call_count == 1, "second run re-ran the backbone"
    assert first == second


def test_labels_line_up_with_features(tmp_path, image_folder, fake_vit):
    """Sorted file order is what keeps cached features matched to labels."""
    dataset = ImageFolderDataset(image_folder)
    cache = FeatureCache(root=tmp_path / "cache")

    features = cache.extract_dataset(fake_vit, dataset, keep="pooled")
    assert len(features["pooled"]) == len(dataset.labels()) == 8
    assert dataset.labels() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_changing_pooling_changes_the_record(tmp_path, image_folder, fake_vit):
    """pooling is in the record because without it the metrics are not reproducible."""
    from visbench.types import Pooling

    dataset = ImageFolderDataset(image_folder)
    cache = FeatureCache(root=tmp_path / "cache")

    _, cls_record = run_probe(
        fake_vit, dataset, cache, visbench.get_probe("retrieval", pooling=Pooling.CLS)
    )
    _, mean_record = run_probe(
        fake_vit, dataset, cache, visbench.get_probe("retrieval", pooling=Pooling.MEAN)
    )

    assert cls_record.pooling == "cls"
    assert mean_record.pooling == "mean"
    assert fake_vit.call_count == 2, "different pooling must re-extract"


def run_trained_probe(backbone, train_dataset, test_dataset, cache, probe, seed=0):
    """Fit on one split, score on another — the shape retrieval never exercises."""
    used_seed = set_seed(seed)
    started = time.perf_counter()

    train_features = cache.extract_dataset(
        backbone, train_dataset, pooling=probe.pooling, keep="pooled"
    )
    test_features = cache.extract_dataset(
        backbone, test_dataset, pooling=probe.pooling, keep="pooled"
    )

    probe.fit(train_features, train_dataset.labels())
    metrics = probe.evaluate(test_features, test_dataset.labels())

    described = {**test_dataset.describe(), **probe.describe()}
    return metrics, ResultRecord(
        backbone=backbone.name,
        backbone_key=backbone.cache_key(),
        task=described["task"],
        level=described["level"],
        dataset=described["dataset"],
        split=described["split"],
        dataset_size=described["dataset_size"],
        dataset_fingerprint=described["dataset_fingerprint"],
        pooling=described["pooling"],
        feature_mode=described["feature_mode"],
        task_params=described["task_params"],
        metrics=metrics,
        timestamp=utc_timestamp(),
        visbench_version=visbench.__version__,
        seed=used_seed,
        duration_seconds=time.perf_counter() - started,
    )


def test_classification_train_test_to_record(tmp_path, split_folders, fake_vit):
    train_root, val_root = split_folders
    train = ImageFolderDataset(train_root, split="train")
    test = ImageFolderDataset(val_root, split="val")
    cache = FeatureCache(root=tmp_path / "cache")
    probe = visbench.get_probe("classification", device="cpu")

    metrics, record = run_trained_probe(fake_vit, train, test, cache, probe)

    assert "top1" in metrics
    assert record.task == "classification"
    assert record.split == "val", "the record must describe the split that was scored"
    assert record.dataset_size == 9
    assert record.seed == 0
    assert record.task_params["optimizer"] == "adamw"

    path = tmp_path / "results.jsonl"
    with ResultWriter(path) as writer:
        writer.write(record)
    (loaded,) = read_records(path)
    assert loaded.task_params == record.task_params


def test_train_and_test_features_are_cached_separately(tmp_path, split_folders, fake_vit):
    """Different images, so no train feature may be served for a test image."""
    train_root, val_root = split_folders
    train = ImageFolderDataset(train_root, split="train")
    test = ImageFolderDataset(val_root, split="val")
    cache = FeatureCache(root=tmp_path / "cache")

    cache.extract_dataset(fake_vit, train, keep="pooled")
    cache.extract_dataset(fake_vit, test, keep="pooled")

    assert cache.stats()["entries"] == len(train) + len(test)


def test_the_two_splits_have_different_fingerprints(split_folders):
    train_root, val_root = split_folders
    train = ImageFolderDataset(train_root, split="train")
    test = ImageFolderDataset(val_root, split="val")
    assert train.fingerprint() != test.fingerprint()


@pytest.mark.slow
def test_classification_with_real_dinov2(tmp_path, split_folders):
    """Colour-separable classes, so a converged linear probe should be exact."""
    train_root, val_root = split_folders
    backbone = visbench.get_backbone("dinov2_vits14", device="cpu")
    cache = FeatureCache(root=tmp_path / "cache")
    probe = visbench.get_probe("classification", device="cpu")

    metrics, record = run_trained_probe(
        backbone,
        ImageFolderDataset(train_root, split="train"),
        ImageFolderDataset(val_root, split="val"),
        cache,
        probe,
    )

    assert probe.train_top1 == 1.0, "the probe did not converge on the training split"
    assert metrics["top1"] == 1.0
    assert record.backbone == "dinov2_vits14"


@pytest.mark.slow
def test_folder_to_record_with_real_dinov2(tmp_path, image_folder):
    """Same flow, real weights. Colour-separable classes, so retrieval is exact."""
    backbone = visbench.get_backbone("dinov2_vits14", device="cpu")
    dataset = ImageFolderDataset(image_folder, split="val")
    cache = FeatureCache(root=tmp_path / "cache")
    probe = visbench.get_probe("retrieval")

    metrics, record = run_probe(backbone, dataset, cache, probe)

    assert metrics["recall@1"] == 1.0
    assert metrics["mAP"] == pytest.approx(1.0)
    assert record.backbone == "dinov2_vits14"
    assert "7764ea0f912e" in record.backbone_key

    torch.manual_seed(0)
    assert cache.stats()["entries"] == 8
