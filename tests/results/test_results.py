"""The structured result record and its JSONL writer.

One schema from the first task, not retrofitted later — so these tests are
mostly about the guarantees a future leaderboard will depend on: identical keys
on every record, additive-only evolution, and a file that survives a crashed
run.
"""

import json

import pytest

from visbench.results import SCHEMA_VERSION, ResultRecord, ResultWriter, read_records
from visbench.results.schema import utc_timestamp


def make_record(**overrides) -> ResultRecord:
    payload = dict(
        backbone="dinov2_vitb14",
        backbone_key="dinov2/dinov2_vitb14/224/7764ea0f912e",
        task="retrieval",
        level="high_level",
        dataset="tiny",
        split="val",
        pooling="cls",
        feature_mode="dense_only",
        metrics={"recall@1": 0.5, "mAP": 0.25},
        timestamp=utc_timestamp(),
        visbench_version="0.1.0.dev0",
    )
    payload.update(overrides)
    return ResultRecord(**payload)


def test_round_trip(tmp_path):
    path = tmp_path / "r.jsonl"
    record = make_record()

    with ResultWriter(path) as writer:
        writer.write(record)

    assert read_records(path) == [record]


def test_none_fields_are_retained(tmp_path):
    """Every record must have identical keys, or tabular loading gets painful."""
    payload = make_record().to_dict()
    assert payload["seed"] is None
    assert payload["layer"] is None
    assert set(payload) == set(make_record(seed=7, layer=11).to_dict())


def test_appends_rather_than_overwrites(tmp_path):
    path = tmp_path / "r.jsonl"
    with ResultWriter(path) as writer:
        writer.write(make_record(task="retrieval"))
    with ResultWriter(path) as writer:
        writer.write(make_record(task="classification"))

    assert [r.task for r in read_records(path)] == ["retrieval", "classification"]


def test_one_record_per_line(tmp_path):
    path = tmp_path / "r.jsonl"
    with ResultWriter(path) as writer:
        for _ in range(3):
            writer.write(make_record())

    assert len(path.read_text().strip().splitlines()) == 3


def test_partial_file_is_still_readable(tmp_path):
    """A crashed run keeps every completed result."""
    path = tmp_path / "r.jsonl"
    writer = ResultWriter(path)
    writer.write(make_record())
    # No close(): flush + fsync on write is what makes this survive.

    assert len(read_records(path)) == 1


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "r.jsonl"
    with ResultWriter(path) as writer:
        writer.write(make_record())
    with open(path, "a") as handle:
        handle.write("\n\n")

    assert len(read_records(path)) == 1


def test_parent_directories_are_created(tmp_path):
    path = tmp_path / "deep" / "nested" / "r.jsonl"
    with ResultWriter(path) as writer:
        writer.write(make_record())
    assert path.exists()


# -- schema guarantees -------------------------------------------------------


def test_schema_version_is_stamped():
    assert make_record().to_dict()["schema_version"] == SCHEMA_VERSION


def test_future_schema_version_is_rejected():
    """A newer file needs a newer VisBench; guessing would corrupt a leaderboard."""
    payload = make_record().to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="Unsupported schema_version"):
        ResultRecord.from_dict(payload)


def test_older_schema_version_is_still_readable():
    """Additive-only means a v1 record is a v2 record missing two fields.

    Rejecting it would throw away exactly the history this library exists to
    accumulate.
    """
    v1 = {
        "backbone": "dinov2_vitb14",
        "backbone_key": "dinov2/dinov2_vitb14/224",
        "task": "retrieval",
        "level": "high_level",
        "dataset": "tiny",
        "split": "val",
        "pooling": "cls",
        "feature_mode": "dense_only",
        "metrics": {"recall@1": 0.5},
        "timestamp": "2026-07-24T09:15:04+00:00",
        "visbench_version": "0.1.0.dev0",
        "schema_version": 1,
        "layer": None,
        "seed": None,
        "duration_seconds": None,
        "notes": None,
    }

    record = ResultRecord.from_dict(v1)
    assert record.task == "retrieval"
    assert record.dataset_fingerprint is None
    assert record.dataset_size is None
    # Provenance is preserved: this record really was written by a v1 VisBench.
    assert record.schema_version == 1


def test_a_v1_record_predates_the_layers_field():
    """The reason `layers` was added rather than widening `layer`'s type: a
    record written before multi-layer extraction existed still parses."""
    v1 = {
        "backbone": "dinov2_vitb14",
        "backbone_key": "dinov2/dinov2_vitb14/224",
        "task": "retrieval",
        "level": "high_level",
        "dataset": "tiny",
        "split": "val",
        "pooling": "cls",
        "feature_mode": "dense_only",
        "metrics": {"recall@1": 0.5},
        "timestamp": "2026-07-24T09:15:04+00:00",
        "visbench_version": "0.1.0.dev0",
        "schema_version": 1,
        "layer": None,
    }
    record = ResultRecord.from_dict(v1)
    assert record.layers is None
    assert record.layer is None


def test_layers_round_trips():
    record = make_record(layers=[2, 5, 8, 11])
    assert ResultRecord.from_dict(record.to_dict()).layers == [2, 5, 8, 11]


def test_non_integer_schema_version_is_rejected():
    payload = make_record().to_dict()
    payload["schema_version"] = "2"

    with pytest.raises(ValueError, match="must be an int"):
        ResultRecord.from_dict(payload)


def test_new_records_carry_dataset_identity():
    payload = make_record(dataset_size=8, dataset_fingerprint="abc123").to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["dataset_size"] == 8
    assert payload["dataset_fingerprint"] == "abc123"


def test_task_params_round_trip():
    """A trained probe's hyperparameters must survive the file, or the number
    it sits next to is not reproducible."""
    params = {"optimizer": "adamw", "lr": 0.01, "epochs": 200}
    record = make_record(task="classification", task_params=params)
    assert ResultRecord.from_dict(record.to_dict()).task_params == params


def test_task_params_defaults_to_empty():
    """Zero-shot tasks have no hyperparameters; the field must not be None."""
    assert make_record().task_params == {}


def test_unknown_field_is_rejected():
    payload = make_record().to_dict()
    payload["accuracy_but_typoed"] = 1.0

    with pytest.raises(ValueError, match="Unknown fields"):
        ResultRecord.from_dict(payload)


def test_missing_required_field_is_rejected():
    payload = make_record().to_dict()
    del payload["backbone_key"]

    with pytest.raises(ValueError, match="Missing required fields"):
        ResultRecord.from_dict(payload)


def test_metrics_are_coerced_to_float():
    """evaluate() is the one field a task fills freely; a tensor must not
    make the whole results file unreadable."""
    import torch

    record = make_record(metrics={"recall@1": torch.tensor(0.5)})
    assert record.to_dict()["metrics"]["recall@1"] == 0.5
    json.dumps(record.to_dict())


def test_records_are_json_object_per_line(tmp_path):
    path = tmp_path / "r.jsonl"
    with ResultWriter(path) as writer:
        writer.write(make_record())

    line = path.read_text().strip()
    assert json.loads(line)["task"] == "retrieval"


def test_corrupt_line_names_the_line_number(tmp_path):
    path = tmp_path / "r.jsonl"
    with ResultWriter(path) as writer:
        writer.write(make_record())
    with open(path, "a") as handle:
        handle.write("{not json}\n")

    with pytest.raises(ValueError, match=":2 is not valid JSON"):
        read_records(path)


def test_timestamp_is_utc():
    """A leaderboard aggregating machines cannot order local timestamps."""
    assert utc_timestamp().endswith("+00:00")


# -- schema v5: dataset_params (step 5j) --------------------------------------


class TestDatasetParams:
    """The dataset's counterpart to ``task_params``.

    Before v5 a correspondence run's ``max_warp`` and a dense split's
    ``image_size`` were recorded nowhere. They changed the fingerprint, so two
    such runs were distinguishable — but only as "not the same data", with
    nothing saying how they differed, which is not enough to reproduce either.
    """

    def test_it_defaults_to_empty(self):
        record = make_record()
        assert record.dataset_params == {}

    def test_it_round_trips(self):
        record = make_record(dataset_params={"max_warp": 0.2, "image_size": 224})
        assert ResultRecord.from_dict(record.to_dict()).dataset_params == {
            "max_warp": 0.2,
            "image_size": 224,
        }

    def test_a_v4_record_still_reads(self):
        """Additive-only: the field a v4 file predates comes back empty, and
        refusing the file would throw away the history worth accumulating."""
        payload = make_record().to_dict()
        payload["schema_version"] = 4
        del payload["dataset_params"]
        assert ResultRecord.from_dict(payload).dataset_params == {}

    def test_the_version_moved(self):
        assert SCHEMA_VERSION == 8


class TestTraining:
    """``training`` says how the *fit* went, not how the evaluation scored (v8).

    The field exists because a low score has two opposite readings -- an
    underfitting probe, which understates a backbone, or a representation that
    genuinely does not carry the answer -- and every trained probe already
    computed the number that tells them apart, then dropped it before the
    record. A corpus of 156 trained runs could not answer the question.
    """

    def test_a_probe_that_trains_nothing_records_none(self):
        """Same convention as ``finetune``: None, not an empty dict.

        There is no fit to describe, which is a different statement from
        'trained and reported nothing about it'.
        """
        assert make_record().training is None

    def test_a_v7_record_still_reads(self):
        """Additive-only: a file written before v8 comes back with None here."""
        payload = make_record().to_dict()
        payload["schema_version"] = 7
        del payload["training"]
        record = ResultRecord.from_dict(payload)
        assert record.training is None
        assert record.schema_version == 7

    def test_it_round_trips(self):
        summary = {"train_loss": 0.0001, "train_top1": 1.0}
        record = make_record(training=summary)
        assert ResultRecord.from_dict(record.to_dict()).training == summary

    def test_it_is_not_merged_into_metrics(self):
        """The whole point of a separate field: `metrics` is the eval split.

        Merging them would put a training number in front of every leaderboard
        code path that reads `metrics`, where the only correct thing to do with
        it is refuse to rank it.
        """
        record = make_record(training={"train_loss": 0.5})
        assert "train_loss" not in record.metrics


class TestFinetune:
    """``finetune`` separates a fine-tuned number from a frozen one (v6)."""

    def test_a_frozen_run_records_none(self):
        """The default, and what every v0.1/v0.2 record carries by absence."""
        assert make_record().finetune is None

    def test_a_v5_record_still_reads(self):
        """Additive-only: a file written before the field parses as frozen,
        which is what it was — not as 'unknown'."""
        payload = make_record().to_dict()
        payload["schema_version"] = 5
        del payload["finetune"]
        assert ResultRecord.from_dict(payload).finetune is None

    def test_a_finetuned_run_survives_a_round_trip(self):
        record = make_record()
        record.finetune = {"blocks": 2, "backbone_lr": 5e-6, "trainable_params": 1_774_080}
        assert ResultRecord.from_dict(record.to_dict()).finetune == record.finetune
