"""The fourth question on one linear-probe implementation, kept distinct.

`classification`, `scene_classification`, `fine_grained_classification` and
`vehicle_classification` share every line of `ClassificationTask`. The failure
mode is therefore not a wrong number but two of them **collapsing into one
board**: `comparability_key` groups by task name, dataset and fingerprint, and
`board_for` refuses a task carrying more than one group. So the identities are
pinned here, as they are for the other three.

The probe also carries a claim its siblings do not: its split is **VisBench's
own**, because the Stanford Cars copy on this machine is not the official
8,144/8,041 one. That is stated in the docstring and on the docs page, and the
staging script is what makes it reproducible; this file pins the identity, and
`tests/scripts/test_stage_cars_split.py` pins the split.
"""

from __future__ import annotations

import pytest

from visbench import get_probe, list_probes
from visbench.tasks.high_level.vehicle_classification import VehicleClassificationTask

FAMILY = (
    "classification",
    "scene_classification",
    "fine_grained_classification",
    "vehicle_classification",
)


def test_the_probe_is_registered_and_listed():
    assert "vehicle_classification" in list_probes()
    assert isinstance(get_probe("vehicle_classification"), VehicleClassificationTask)


def test_it_reports_its_own_name_rather_than_its_parents():
    """`ClassificationTask.__init__` hardcodes `classification`; this must move."""
    assert get_probe("vehicle_classification").name == "vehicle_classification"
    assert get_probe("vehicle_classification").level == "high_level"


def test_every_member_of_the_family_reports_a_distinct_identity():
    names = [get_probe(name).name for name in FAMILY]
    assert names == list(FAMILY), "two probes on this implementation share a name"


def test_every_member_carries_a_distinct_protocol_or_none():
    """The object board carries no protocol; the other three carry their own.

    A shared string would be a record claiming one of these numbers is
    comparable with another, which is the one thing the `protocol` field exists
    to prevent.
    """
    protocols = {name: get_probe(name).describe()["task_params"].get("protocol") for name in FAMILY}
    assert protocols["classification"] is None
    stated = [p for name, p in protocols.items() if name != "classification"]
    assert len(set(stated)) == len(stated), protocols
    assert protocols["vehicle_classification"] == "visbench_vehicle_linear_probe"


@pytest.mark.parametrize("name", FAMILY)
def test_the_family_shares_one_implementation(name):
    """If these ever stop sharing a class, the four-questions claim needs rewriting."""
    from visbench.tasks.high_level.classification import ClassificationTask

    assert isinstance(get_probe(name), ClassificationTask)
