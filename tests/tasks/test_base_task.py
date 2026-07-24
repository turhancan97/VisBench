"""Contract tests for BaseTask.

* ``evaluate`` returns a flat dict of floats — no nesting, no tensors, since
  the result schema depends on it.
* ``fit`` is a no-op returning ``self`` for zero-shot tasks.
* A task requests pooling from the backbone; the backbone never chooses.
* ``evaluate`` prints nothing — the caller writes the structured record.
"""

import pytest
import torch

import visbench
from visbench.tasks.base import BaseTask
from visbench.types import Pooling


@pytest.fixture
def retrieval():
    return visbench.get_probe("retrieval")


@pytest.fixture
def clustered_features():
    """Three tight clusters of four, so retrieval has an obvious right answer."""
    torch.manual_seed(0)
    centres = torch.eye(3) * 10
    features = torch.cat([centres[i].repeat(4, 1) for i in range(3)])
    features = features + torch.randn_like(features) * 0.01
    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    return features, labels


# -- the flat-dict contract the result schema depends on ---------------------


def test_evaluate_returns_flat_float_dict(retrieval, clustered_features):
    features, labels = clustered_features
    metrics = retrieval.evaluate(features, labels)

    assert isinstance(metrics, dict)
    assert metrics, "evaluate returned no metrics"
    for key, value in metrics.items():
        assert isinstance(key, str)
        assert isinstance(value, float), f"{key} is {type(value).__name__}, not float"
        assert not isinstance(value, torch.Tensor)


def test_metrics_survive_json_serialisation(retrieval, clustered_features):
    """The record writer will json.dumps these; a tensor would fail there, not here."""
    import json

    features, labels = clustered_features
    json.dumps(retrieval.evaluate(features, labels))


def test_evaluate_does_not_print(retrieval, clustered_features, capsys):
    features, labels = clustered_features
    retrieval.evaluate(features, labels)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# -- zero-shot fit -----------------------------------------------------------


def test_zero_shot_fit_is_noop(retrieval, clustered_features):
    features, labels = clustered_features
    before = retrieval.evaluate(features, labels)

    assert retrieval.fit(features, labels) is retrieval
    assert retrieval.evaluate(features, labels) == before


def test_fit_is_chainable(retrieval, clustered_features):
    features, labels = clustered_features
    assert retrieval.fit(features).evaluate(features, labels)


def test_non_zero_shot_task_must_implement_fit():
    class Trained(BaseTask):
        zero_shot = False

        def predict(self, features):
            return None

        def evaluate(self, features, labels=None):
            return {}

    with pytest.raises(NotImplementedError, match="must implement fit"):
        Trained().fit(torch.rand(2, 4))


# -- the task drives pooling -------------------------------------------------


def test_task_drives_pooling_choice(fake_vit, solid_images, tmp_path):
    """The backbone executes whatever the task asks for and holds no opinion."""
    from visbench.cache import FeatureCache

    requested = []
    original = fake_vit.extract_features

    def spy(image, pooling=Pooling.DEFAULT, layers=None, **kwargs):
        requested.append(pooling)
        return original(image, pooling=pooling, layers=layers, **kwargs)

    fake_vit.extract_features = spy

    probe = visbench.get_probe("retrieval", pooling=Pooling.MEAN)
    cache = FeatureCache(root=tmp_path / "c")
    cache.extract_dataset(fake_vit, solid_images, pooling=probe.pooling)

    assert requested == [Pooling.MEAN]


def test_pooling_is_recorded_in_describe():
    probe = visbench.get_probe("retrieval", pooling=Pooling.CLS)
    described = probe.describe()

    assert described["pooling"] == Pooling.CLS
    assert described["task"] == "retrieval"
    assert described["level"] == "high_level"
    assert described["zero_shot"] is True


def test_requires_labels_is_not_zero_shot(retrieval):
    """Retrieval needs no training labels but cannot be scored without them."""
    assert retrieval.zero_shot is True
    assert retrieval.requires_labels() is True


# -- input coercion ----------------------------------------------------------


def test_accepts_a_feature_dict(retrieval, clustered_features):
    features, labels = clustered_features
    from_dict = retrieval.evaluate({"pooled": features, "grid_hw": (4, 4)}, labels)
    from_tensor = retrieval.evaluate(features, labels)
    assert from_dict == from_tensor


def test_dense_only_dict_points_at_the_fix(retrieval, clustered_features):
    _, labels = clustered_features
    with pytest.raises(KeyError, match="keep='both'"):
        retrieval.evaluate({"dense": torch.rand(12, 8, 4, 4), "grid_hw": (4, 4)}, labels)


def test_missing_labels_raises(retrieval, clustered_features):
    features, _ = clustered_features
    with pytest.raises(ValueError, match="requires labels"):
        retrieval.evaluate(features)


def test_unlabeled_dataset_labels_raise(retrieval, clustered_features):
    features, _ = clustered_features
    with pytest.raises(ValueError, match="labeled=False"):
        retrieval.evaluate(features, [None] * 12)


def test_label_count_mismatch_raises(retrieval, clustered_features):
    features, labels = clustered_features
    with pytest.raises(ValueError, match="features for"):
        retrieval.evaluate(features, labels[:5])
