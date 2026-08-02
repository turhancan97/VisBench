"""Serialising a trained probe, and refusing to load it onto the wrong features.

A head is only meaningful against the exact representation it was fitted on, and
the ways of getting that wrong are mostly *shape-compatible* — they load, they
run, and they score. Every test here pins one of those.
"""

import warnings

import pytest
import torch
import torch.nn as nn

import visbench
from visbench.hub import (
    ARTIFACT_VERSION,
    IncompatibleProbe,
    load_probe,
    probe_metadata,
    save_probe,
)
from visbench.tasks.high_level.classification import ClassificationTask


@pytest.fixture
def pooled(fake_vit):
    torch.manual_seed(0)
    return {"pooled": torch.randn(40, fake_vit.embed_dim)}


@pytest.fixture
def labels():
    torch.manual_seed(1)
    return torch.randint(0, 3, (40,))


@pytest.fixture
def fitted(pooled, labels):
    return ClassificationTask(epochs=3).fit(pooled, labels)


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_a_loaded_probe_predicts_what_the_saved_one_did(fitted, fake_vit, pooled, tmp_path):
    """The only claim that makes an artifact worth distributing."""
    before = fitted.predict(pooled)
    save_probe(fitted, tmp_path / "probe.pt", backbone=fake_vit)
    loaded = load_probe(tmp_path / "probe.pt", backbone=fake_vit)
    assert torch.equal(before, loaded.predict(pooled))


def test_the_standardiser_travels_with_the_weights(fake_vit, pooled, labels, tmp_path):
    """`_mean`/`_std` live outside the head and decide the answer.

    A linear layer fitted on ``(x - mean) / std`` is meaningless applied to raw
    features. It loads, it has the right shape, and it is wrong — so the tensors
    have to be in the artifact, not reconstructed or assumed.
    """
    task = ClassificationTask(epochs=3, standardize=True).fit(pooled, labels)
    before = task.predict(pooled)
    save_probe(task, tmp_path / "probe.pt", backbone=fake_vit)

    loaded = load_probe(tmp_path / "probe.pt", backbone=fake_vit)
    assert loaded.standardize
    assert torch.equal(before, loaded.predict(pooled))


def test_dropping_the_standardiser_would_change_the_answer(fake_vit, pooled, labels, tmp_path):
    """Proves the test above is not vacuous.

    If standardising made no difference, carrying it would be optional and the
    guard would pass whether or not the tensors were saved.
    """
    task = ClassificationTask(epochs=3, standardize=True).fit(pooled, labels)
    save_probe(task, tmp_path / "probe.pt", backbone=fake_vit)
    loaded = load_probe(tmp_path / "probe.pt", backbone=fake_vit)

    with_it = loaded.predict(pooled)
    loaded.standardize = False
    assert not torch.equal(with_it, loaded.predict(pooled))


def test_a_probe_with_no_standardiser_carries_none(fitted, fake_vit):
    assert fitted.probe_state() == {}
    assert probe_metadata(fitted, fake_vit)["task"] == "classification"


# --------------------------------------------------------------------------
# What must be refused
# --------------------------------------------------------------------------


def test_a_different_backbone_is_refused(fitted, fake_vit, fake_cnn, tmp_path):
    save_probe(fitted, tmp_path / "probe.pt", backbone=fake_vit)
    with pytest.raises(IncompatibleProbe, match="does not match"):
        load_probe(tmp_path / "probe.pt", backbone=fake_cnn)


def test_the_same_backbone_pooled_differently_is_refused(fitted, fake_vit, tmp_path):
    """The failure this whole module exists for.

    ``cls`` and ``mean`` produce the same rank and the same width on one
    backbone, so a head fitted on one and fed the other raises nothing at all.
    It simply scores against a representation it has never seen.
    """
    save_probe(fitted, tmp_path / "probe.pt", backbone=fake_vit)

    other = visbench.get_probe("classification")
    other.pooling = "mean"
    with pytest.raises(IncompatibleProbe, match="pooling"):
        load_probe(tmp_path / "probe.pt", backbone=fake_vit, task=other)


def test_different_weights_alone_are_refused(fitted, fake_vit, tmp_path):
    """The fine-tuning case, and the one no other check catches.

    A fine-tuned DINOv2-S and its parent share a name, a width, a pooling rule,
    a feature mode and a depth. `cache_key` is the only thing that differs, and
    the features differ completely. Written because dropping `backbone_key` from
    IDENTITY_FIELDS left every other test in this file passing — the
    cross-backbone test was being caught by the pooling check instead.
    """
    save_probe(fitted, tmp_path / "probe.pt", backbone=fake_vit)

    retrained = type(fake_vit)()
    retrained.cache_key = lambda: fake_vit.cache_key() + "/finetuned"
    assert retrained.default_pooling() == fake_vit.default_pooling()

    with pytest.raises(IncompatibleProbe, match="backbone_key"):
        load_probe(tmp_path / "probe.pt", backbone=retrained)


def test_a_different_layer_is_refused(fitted, fake_vit, tmp_path):
    """Right shape, wrong depth."""
    save_probe(fitted, tmp_path / "probe.pt", backbone=fake_vit)

    other = visbench.get_probe("classification")
    other.layers = [5]
    with pytest.raises(IncompatibleProbe, match="layers"):
        load_probe(tmp_path / "probe.pt", backbone=fake_vit, task=other)


def test_a_different_feature_mode_is_refused(fitted, fake_vit, tmp_path):
    save_probe(fitted, tmp_path / "probe.pt", backbone=fake_vit)

    other = visbench.get_probe("classification")
    other.feature_mode = "dense_cls_broadcast"
    with pytest.raises(IncompatibleProbe, match="feature_mode"):
        load_probe(tmp_path / "probe.pt", backbone=fake_vit, task=other)


def test_strict_false_warns_and_loads(fitted, fake_vit, fake_cnn, tmp_path):
    """Deliberate transfer is a legitimate experiment; silence is not."""
    save_probe(fitted, tmp_path / "probe.pt", backbone=fake_vit)
    with pytest.warns(RuntimeWarning, match="does not match"):
        loaded = load_probe(tmp_path / "probe.pt", backbone=fake_cnn, strict=False)
    assert loaded.head is not None


def test_a_matching_load_does_not_warn(fitted, fake_vit, tmp_path):
    save_probe(fitted, tmp_path / "probe.pt", backbone=fake_vit)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_probe(tmp_path / "probe.pt", backbone=fake_vit)


def test_an_unfitted_probe_is_refused(fake_vit, tmp_path):
    """An untrained head cannot be told apart from one that learned nothing."""
    task = visbench.get_probe("classification")
    with pytest.raises(ValueError, match="has not been fitted"):
        save_probe(task, tmp_path / "probe.pt", backbone=fake_vit)


def test_a_zero_shot_probe_is_refused(fake_vit, tmp_path):
    """Retrieval trains nothing, so there is nothing to distribute."""
    task = visbench.get_probe("retrieval")
    with pytest.raises(ValueError, match="zero-shot"):
        save_probe(task, tmp_path / "probe.pt", backbone=fake_vit)


def test_a_newer_artifact_version_is_refused(fitted, fake_vit, tmp_path):
    path = tmp_path / "probe.pt"
    save_probe(fitted, path, backbone=fake_vit)
    payload = torch.load(path, weights_only=True)
    payload["meta"]["artifact_version"] = ARTIFACT_VERSION + 1
    torch.save(payload, path)

    with pytest.raises(IncompatibleProbe, match="newer than this VisBench"):
        load_probe(path, backbone=fake_vit)


def test_unexpected_probe_state_is_refused_not_dropped(fitted, fake_vit, tmp_path):
    """A probe told to restore state it does not understand runs without it."""
    task = visbench.get_probe("depth")
    with pytest.raises(ValueError, match="no probe state to restore"):
        task.load_probe_state({"mystery": torch.zeros(1)})


def test_a_standardising_probe_refuses_an_artifact_without_one(fake_vit, pooled, labels):
    task = ClassificationTask(epochs=1, standardize=True).fit(pooled, labels)
    task._mean = None
    with pytest.raises(ValueError, match="no mean/std"):
        task.load_probe_state({})


# --------------------------------------------------------------------------
# The payload itself
# --------------------------------------------------------------------------


def test_the_artifact_loads_under_weights_only(fitted, fake_vit, tmp_path):
    """Step 6e-5 fetches these from a hub.

    An unrestricted ``torch.load`` on a downloaded file is arbitrary code
    execution, so nothing may enter the payload that needs unpickling. This
    fails the moment someone puts an object in the metadata.
    """
    path = tmp_path / "probe.pt"
    save_probe(fitted, path, backbone=fake_vit)
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"meta", "head_spec", "head_state", "probe_state"}


def test_the_identity_is_recorded_beside_the_weights(fitted, fake_vit, tmp_path):
    path = tmp_path / "probe.pt"
    save_probe(fitted, path, backbone=fake_vit, notes="a note")
    meta = torch.load(path, weights_only=True)["meta"]

    assert meta["backbone"] == fake_vit.name
    assert meta["backbone_key"] == fake_vit.cache_key()
    assert meta["pooling"] == "cls"  # resolved, not the literal "default"
    assert meta["pooling_requested"] == "default"
    assert meta["notes"] == "a note"


def test_pooling_is_recorded_resolved(fitted, fake_vit, fake_cnn):
    """Same reason the result record does it: "default" is not an answer."""
    assert probe_metadata(fitted, fake_vit)["pooling"] == "cls"
    assert probe_metadata(fitted, fake_cnn)["pooling"] == "mean"


def test_saving_creates_missing_parents(fitted, fake_vit, tmp_path):
    path = tmp_path / "nested" / "deeper" / "probe.pt"
    assert save_probe(fitted, path, backbone=fake_vit).exists()


# --------------------------------------------------------------------------
# Dense probes, which build a registered head rather than a bare Linear
# --------------------------------------------------------------------------


def test_a_dense_probe_round_trips(fake_vit, tmp_path):
    """The head recipe has to be captured where the shapes are known.

    `in_channels` and `output_size` are measured from the first batch of
    features, so nothing outside ``_build_head`` knows them — a save that
    reconstructed them afterwards would be guessing.
    """
    from visbench.tasks.mid_level.generic_segmentation import GenericSegmentationTask

    torch.manual_seed(0)
    features = {"dense": torch.randn(6, fake_vit.embed_dim, 8, 8)}
    targets = (torch.rand(6, 1, 16, 16) > 0.5).float()

    task = GenericSegmentationTask(epochs=1, warmup_epochs=0, batch_size=2).fit(features, targets)
    spec = task.head_spec()
    assert spec["kind"] == "registered"
    assert spec["kwargs"]["in_channels"] == fake_vit.embed_dim

    before = task.predict(features)
    save_probe(task, tmp_path / "seg.pt", backbone=fake_vit)

    from visbench.tasks.mid_level.generic_segmentation import (
        GenericSegmentationTask as Fresh,
    )

    loaded = load_probe(
        tmp_path / "seg.pt", backbone=fake_vit, task=Fresh(epochs=1, warmup_epochs=0, batch_size=2)
    )
    assert torch.allclose(before, loaded.predict(features))


def test_the_rebuilt_head_is_the_registered_class(fitted, fake_vit, tmp_path):
    save_probe(fitted, tmp_path / "probe.pt", backbone=fake_vit)
    loaded = load_probe(tmp_path / "probe.pt", backbone=fake_vit)
    assert isinstance(loaded.head, nn.Linear)
