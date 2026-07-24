"""The full v0.1 path on a small local image folder.

This is what build step 3 exists to prove: folder -> backbone -> cache -> task
-> metrics -> structured record, with nothing stubbed in between.

The fake-backbone version runs by default. The same flow against real DINOv2
weights is marked ``slow``.
"""

import pytest
import torch
from PIL import Image

import visbench
from visbench.cache import FeatureCache
from visbench.data import ImageFolderDataset
from visbench.results import ResultRecord, ResultWriter, read_records
from visbench.results.schema import utc_timestamp


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


def run_probe(backbone, dataset, cache, probe):
    """Extract, evaluate, and build the record — the shape a CLI would take."""
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
        pooling=described["pooling"],
        feature_mode=described["feature_mode"],
        metrics=metrics,
        timestamp=utc_timestamp(),
        visbench_version=visbench.__version__,
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
