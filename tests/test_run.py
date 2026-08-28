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


def test_the_request_is_recorded_beside_the_resolution(fake_vit, splits, cache):
    """Schema v7. A field that is declared and never filled is dead weight.

    This codebase has shipped that failure before — the CLIP QuickGELU guard
    passed its own tests for a year while checking a phrase open_clip never
    emitted. `pooling_requested` only earns its schema bump if `run()` actually
    writes it.
    """
    _, val = splits
    result = visbench.run(fake_vit, "retrieval", val, cache=cache)
    assert result.record.pooling == "cls"
    assert result.record.pooling_requested == "default"


def test_a_vit_and_a_cnn_asked_for_the_same_thing(fake_vit, fake_cnn, splits, cache):
    """The resolution differs by architecture; the request does not.

    This is the whole of schema v7, end to end rather than on a hand-built
    record: without `pooling_requested` these two runs are unrankable against
    each other, which is what a leaderboard exists to do.
    """
    _, val = splits
    vit = visbench.run(fake_vit, "retrieval", val, cache=cache).record
    cnn = visbench.run(fake_cnn, "retrieval", val, cache=cache).record

    assert (vit.pooling, cnn.pooling) == ("cls", "mean")
    assert vit.pooling_requested == cnn.pooling_requested == "default"

    from visbench.results.leaderboard import comparability_key

    assert comparability_key(vit) == comparability_key(cnn)


def test_a_trained_run_records_how_the_fit_went(fake_vit, splits, cache):
    """Schema v8. The diagnostic has to reach the *record*, not only a log line.

    That was the gap: every trained probe computed `train_loss`, printed it and
    dropped it, so a corpus of trained runs could not answer whether a low score
    was an underfitting probe -- which understates a backbone -- or a weak
    representation. Those are opposite conclusions from the same number.
    """
    train, val = splits
    result = visbench.run(
        fake_vit, "classification", val, train_dataset=train, cache=cache, device="cpu"
    )

    assert result.record.training is not None
    assert set(result.record.training) == {"train_loss", "train_top1"}
    assert result.record.training["train_top1"] == result.probe.train_top1
    # It describes the fit, not the evaluation, so it must not be in `metrics` --
    # where every leaderboard code path would meet it and could only refuse it.
    assert "train_loss" not in result.metrics
    assert result.record.schema_version == 8


def test_a_zero_shot_run_records_no_training(fake_vit, splits, cache):
    """None, like `finetune`: there is no fit to describe."""
    _, val = splits
    result = visbench.run(fake_vit, "retrieval", val, cache=cache)
    assert result.record.training is None


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


# -- pairwise: correspondence through run() (step 5j) --------------------------


@pytest.fixture
def pair_split(tmp_path):
    """A flat folder of six images, as a homography pair dataset."""
    from visbench.data import HomographyPairDataset

    root = tmp_path / "pairs"
    root.mkdir()
    for index in range(6):
        shade = 20 + 40 * index
        Image.new("RGB", (64, 64), (shade, 255 - shade, 90)).save(root / f"{index:02d}.png")
    return HomographyPairDataset(root, split="test", image_size=64)


class TestPairwiseRuns:
    """``run()`` covers correspondence, which it did not until step 5j.

    The failure this guards against is not a crash. Extracting one view per
    pair, or pairing view 0 of one image with view 1 of the next, produces a
    correspondence number computed against the wrong partner — which trains
    nothing, raises nothing, and simply reports a worse backbone.
    """

    def test_it_runs_and_scores(self, fake_vit, pair_split, cache):
        result = visbench.run(fake_vit, "correspondence", pair_split, cache=cache)
        assert result.record.task == "correspondence"
        assert result.record.dataset_size == 6
        assert "num_matches" in result.metrics

    def test_it_extracts_both_views_of_every_pair(self, fake_vit, pair_split, cache):
        """Twelve views for six pairs. Half of them would be a silent half-run."""
        visbench.run(fake_vit, "correspondence", pair_split, cache=cache, batch_size=1)
        assert fake_vit.call_count == 12

    def test_the_ceiling_is_recorded_beside_the_score(self, fake_vit, pair_split, cache):
        """CLAUDE.md's rule, now true for run() and not only for the example."""
        metrics = visbench.run(fake_vit, "correspondence", pair_split, cache=cache).metrics
        assert "recall@5px" in metrics and "ceiling_recall@5px" in metrics
        assert metrics["recall@5px"] <= metrics["ceiling_recall@5px"] + 1e-9

    def test_max_warp_reaches_the_record(self, fake_vit, pair_split, cache):
        """Two runs at different warps are not comparable, and now say so."""
        record = visbench.run(fake_vit, "correspondence", pair_split, cache=cache).record
        assert record.dataset_params["max_warp"] == pair_split.max_warp
        assert record.dataset_params["image_size"] == 64

    def test_a_second_run_reuses_every_cached_view(self, fake_vit, pair_split, cache):
        """view_identity finally has a caller: no image is decoded twice."""
        visbench.run(fake_vit, "correspondence", pair_split, cache=cache)
        calls = fake_vit.call_count
        visbench.run(fake_vit, "correspondence", pair_split, cache=cache)
        assert fake_vit.call_count == calls

    def test_a_plain_image_dataset_is_refused(self, fake_vit, splits, cache):
        _, val = splits
        with pytest.raises(TypeError, match="image pairs plus geometry"):
            visbench.run(fake_vit, "correspondence", val, cache=cache)


def test_dataset_params_carries_what_the_record_has_no_field_for(fake_vit, splits, cache):
    _, val = splits
    record = visbench.run(fake_vit, "retrieval", val, cache=cache).record
    assert record.dataset_params == {"num_classes": 3}
    # And never duplicates a field that does exist.
    assert "dataset" not in record.dataset_params
