"""Fine-grained recognition — a thin subclass of the object-classification probe.

The mechanics (fitting, seeding, standardiser round-trip) are the parent's and
are covered by ``test_classification.py``. What this module pins is the part
that is *not* inherited: the distinct name, level and protocol string that keep
the boards apart, and that the subclass still fits and evaluates.

It also pins the three-way separation, which two probes cannot show: the object,
scene and fine-grained probes share an implementation and must nonetheless never
share a comparability group.
"""

import pytest
import torch

import visbench
from visbench.cli.datasets import showable_probes, supported_probes
from visbench.results.render import CAVEATS, HEADLINE_METRICS
from visbench.utils import set_seed
from visbench.viz import show_probes

PROBE = "fine_grained_classification"


@pytest.fixture
def separable():
    set_seed(0)
    features = torch.randn(400, 16)
    weights = torch.randn(16, 4)
    labels = (features @ weights).argmax(dim=1)
    return features[:300], labels[:300], features[300:], labels[300:]


@pytest.fixture
def probe():
    return visbench.get_probe(PROBE, device="cpu")


def test_it_is_a_registered_probe():
    assert PROBE in visbench.list_probes()


def test_it_is_wired_through_every_fixed_table():
    """A new probe must appear in all four sets or the show/CLI tests fail."""
    assert PROBE in supported_probes()
    assert PROBE in showable_probes()
    assert PROBE in show_probes()
    assert PROBE in HEADLINE_METRICS


def test_identity_is_distinct_from_object_classification(probe):
    assert probe.name == PROBE
    assert probe.level == "high_level"
    described = probe.describe()
    assert described["task"] == PROBE
    assert described["task_params"]["protocol"] == "visbench_fine_grained_linear_probe"
    # The object-classification probe carries no protocol; the string is what
    # says in the record which kind of number this is.
    assert "protocol" not in visbench.get_probe("classification").describe()["task_params"]


def test_the_three_linear_probes_have_three_distinct_identities():
    """One implementation, three questions — and no two may collapse into one board."""
    names = {
        visbench.get_probe(name, device="cpu").name
        for name in ("classification", "scene_classification", PROBE)
    }
    assert names == {"classification", "scene_classification", PROBE}
    protocols = {
        visbench.get_probe(name, device="cpu").describe()["task_params"].get("protocol")
        for name in ("classification", "scene_classification", PROBE)
    }
    assert len(protocols) == 3


def test_headline_metric_is_top1():
    assert HEADLINE_METRICS[PROBE] == "top1"


def test_the_imagenet_bird_confound_is_stated_on_the_board():
    """The confound is a property of the protocol, so it travels with the table."""
    assert PROBE in CAVEATS
    assert "ImageNet-1k" in CAVEATS[PROBE]
    # A low score here is an underfitting probe more often than a weak backbone,
    # and the board is where a reader will be when they need to know that.
    assert "train_top1" in CAVEATS[PROBE]


def test_learns_a_separable_problem(probe, separable):
    train_x, train_y, test_x, test_y = separable
    set_seed(0)
    probe.fit(train_x, train_y)
    metrics = probe.evaluate(test_x, test_y)
    assert metrics["top1"] > 0.8


def test_unfitted_probe_refuses_to_predict(probe):
    with pytest.raises(RuntimeError, match="not been fitted"):
        probe.predict(torch.randn(4, 16))
