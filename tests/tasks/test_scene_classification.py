"""Scene classification — a thin subclass of the object-classification probe.

The mechanics (fitting, seeding, standardiser round-trip) are the parent's and
are covered by ``test_classification.py``. What this module pins is the part
that is *not* inherited: the distinct name, level and protocol string that keep
the two boards apart, and that the subclass still fits and evaluates.
"""

import pytest
import torch

import visbench
from visbench.cli.datasets import showable_probes, supported_probes
from visbench.results.render import HEADLINE_METRICS
from visbench.utils import set_seed
from visbench.viz import show_probes


@pytest.fixture
def separable():
    set_seed(0)
    features = torch.randn(400, 16)
    weights = torch.randn(16, 4)
    labels = (features @ weights).argmax(dim=1)
    return features[:300], labels[:300], features[300:], labels[300:]


@pytest.fixture
def probe():
    return visbench.get_probe("scene_classification", device="cpu")


def test_it_is_a_registered_probe():
    assert "scene_classification" in visbench.list_probes()


def test_it_is_wired_through_every_fixed_table():
    """A new probe must appear in all four sets or the show/CLI tests fail."""
    assert "scene_classification" in supported_probes()
    assert "scene_classification" in showable_probes()
    assert "scene_classification" in show_probes()
    assert "scene_classification" in HEADLINE_METRICS


def test_identity_is_distinct_from_object_classification(probe):
    assert probe.name == "scene_classification"
    assert probe.level == "high_level"
    described = probe.describe()
    assert described["task"] == "scene_classification"
    assert described["task_params"]["protocol"] == "visbench_scene_linear_probe"
    # The object-classification probe carries no protocol; the string is what
    # says in the record which kind of number this is.
    assert "protocol" not in visbench.get_probe("classification").describe()["task_params"]


def test_headline_metric_is_top1():
    assert HEADLINE_METRICS["scene_classification"] == "top1"


def test_learns_a_separable_problem(probe, separable):
    train_x, train_y, test_x, test_y = separable
    set_seed(0)
    probe.fit(train_x, train_y)
    metrics = probe.evaluate(test_x, test_y)
    assert metrics["top1"] > 0.8


def test_unfitted_probe_refuses_to_predict(probe):
    with pytest.raises(RuntimeError, match="not been fitted"):
        probe.predict(torch.randn(4, 16))
