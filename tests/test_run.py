"""The one-call runner.

Most of what this needs to get right is not the happy path — it is refusing
combinations that would produce a result disagreeing with the caller's intent,
and recording what actually ran rather than what was asked for.
"""

import json

import pytest
from PIL import Image

import visbench
from visbench.cache import FeatureCache
from visbench.data import ImageFolderDataset
from visbench.results import read_records
from visbench.types import Pooling


@pytest.fixture
def splits(tmp_path):
    """root/train and root/val, three colour-separable classes."""
    root = tmp_path / "tiny"
    palette = {"red": (200, 30, 30), "blue": (30, 30, 200), "green": (30, 200, 30)}
    for split, offsets in [("train", range(6)), ("val", range(10, 13))]:
        for name, colour in palette.items():
            directory = root / split / name
            directory.mkdir(parents=True)
            for i in offsets:
                jitter = tuple(min(255, c + i * 4) for c in colour)
                Image.new("RGB", (64, 64), jitter).save(directory / f"{i:02d}.png")
    return (
        ImageFolderDataset(root / "train", split="train"),
        ImageFolderDataset(root / "val", split="val"),
    )


@pytest.fixture
def cache(tmp_path):
    return FeatureCache(root=tmp_path / "cache")


# -- zero-shot ----------------------------------------------------------------


def test_zero_shot_run(fake_vit, splits, cache):
    _, val = splits
    result = visbench.run(fake_vit, "retrieval", val, cache=cache)

    assert set(result.metrics) >= {"recall@1", "mAP"}
    assert result.record.task == "retrieval"
    assert result.record.split == "val"
    assert result.record.dataset_size == 9


def test_trained_run(fake_vit, splits, cache):
    train, val = splits
    result = visbench.run(
        fake_vit, "classification", val, train_dataset=train, cache=cache, device="cpu"
    )

    assert "top1" in result.metrics
    assert result.record.task_params["optimizer"] == "adamw"
    # The scored split is the one described, not the trained one.
    assert result.record.split == "val"
    assert result.record.dataset_size == 9


def test_fitted_probe_is_returned(fake_vit, splits, cache):
    """train_top1 diagnoses underfitting, so it is not a result but is needed."""
    train, val = splits
    result = visbench.run(
        fake_vit, "classification", val, train_dataset=train, cache=cache, device="cpu"
    )
    assert result.probe.train_top1 is not None


# -- refusing combinations that would silently disagree -----------------------


def test_zero_shot_with_train_dataset_raises(fake_vit, splits, cache):
    """Ignoring it would make the caller's intent and the result disagree."""
    train, val = splits
    with pytest.raises(ValueError, match="zero-shot and has nothing to fit"):
        visbench.run(fake_vit, "retrieval", val, train_dataset=train, cache=cache)


def test_trained_without_train_dataset_raises(fake_vit, splits, cache):
    _, val = splits
    with pytest.raises(ValueError, match="needs train_dataset"):
        visbench.run(fake_vit, "classification", val, cache=cache, device="cpu")


def test_task_kwargs_on_an_instance_raises(fake_vit, splits, cache):
    """Silently dropping them would give settings the caller never got."""
    _, val = splits
    probe = visbench.get_probe("retrieval")
    with pytest.raises(TypeError, match="only apply when"):
        visbench.run(fake_vit, probe, val, cache=cache, topk=(1, 3))


# -- what gets recorded -------------------------------------------------------


def test_pooling_is_recorded_resolved_not_deferred(fake_vit, splits, cache):
    """ "default" means CLS on a ViT and mean on a CNN.

    A record carrying the literal word does not say what produced the number,
    and two such records across architectures would compare different
    representations under one name.
    """
    _, val = splits
    probe = visbench.get_probe("retrieval")
    assert probe.pooling == Pooling.DEFAULT

    result = visbench.run(fake_vit, probe, val, cache=cache)
    assert result.record.pooling == "cls"


def test_explicit_pooling_passes_through(fake_vit, splits, cache):
    _, val = splits
    result = visbench.run(fake_vit, "retrieval", val, cache=cache, pooling=Pooling.MEAN)
    assert result.record.pooling == "mean"


def test_seed_and_duration_are_recorded(fake_vit, splits, cache):
    _, val = splits
    result = visbench.run(fake_vit, "retrieval", val, cache=cache, seed=7)
    assert result.record.seed == 7
    assert result.record.duration_seconds > 0


def test_notes_reach_the_record(fake_vit, splits, cache):
    _, val = splits
    result = visbench.run(fake_vit, "retrieval", val, cache=cache, notes="sanity check")
    assert result.record.notes == "sanity check"


def test_results_file_is_written(tmp_path, fake_vit, splits, cache):
    _, val = splits
    path = tmp_path / "out.jsonl"
    result = visbench.run(fake_vit, "retrieval", val, cache=cache, results=path)

    (loaded,) = read_records(path)
    assert loaded == result.record
    json.loads(path.read_text().strip())


def test_no_results_path_writes_nothing(tmp_path, fake_vit, splits, cache, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, val = splits
    visbench.run(fake_vit, "retrieval", val, cache=cache)
    assert not (tmp_path / "results").exists()


def test_appends_across_runs(tmp_path, fake_vit, splits, cache):
    _, val = splits
    path = tmp_path / "out.jsonl"
    visbench.run(fake_vit, "retrieval", val, cache=cache, results=path)
    visbench.run(fake_vit, "retrieval", val, cache=cache, results=path)
    assert len(read_records(path)) == 2


# -- extraction -------------------------------------------------------------


def test_pooled_task_does_not_store_dense(fake_vit, splits, cache, tmp_path):
    """Dense features are ~250x the size; a retrieval run must not write them."""
    import torch

    _, val = splits
    visbench.run(fake_vit, "retrieval", val, cache=cache)

    entry = torch.load(next((tmp_path / "cache").rglob("*.pt")), weights_only=True)
    assert "pooled" in entry
    assert "dense" not in entry


def test_second_run_reuses_the_cache(fake_vit, splits, cache):
    _, val = splits
    visbench.run(fake_vit, "retrieval", val, cache=cache)
    assert fake_vit.call_count == 1

    visbench.run(fake_vit, "retrieval", val, cache=cache)
    assert fake_vit.call_count == 1


def test_task_kwargs_are_forwarded(fake_vit, splits, cache):
    _, val = splits
    result = visbench.run(fake_vit, "retrieval", val, cache=cache, topk=(1, 3))
    assert "recall@3" in result.metrics
    assert "recall@5" not in result.metrics


def test_repr_is_readable(fake_vit, splits, cache):
    _, val = splits
    text = repr(visbench.run(fake_vit, "retrieval", val, cache=cache))
    assert "fake_vit" in text and "retrieval" in text
