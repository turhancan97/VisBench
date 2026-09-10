"""The pre-flight asks whether re-running a board would rejoin it.

A board is re-run by appending records beside the ones they supersede, which
only works while the new records stay *comparable* with the old. When they do
not, the failure is not a wrong number: `render_tables.py::board_for` refuses a
task with more than one comparability group, so the board vanishes from
`LEADERBOARD.md` instead of disagreeing with it.

The script itself needs the datasets on disk, so what is tested here is the part
that does not: that it can recover every board's command from `build_corpus.sh`
rather than carrying its own copy of the flags, and that it reads the runner's
own field set rather than a copy of it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "preflight_regroup.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("preflight_regroup", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_it_reads_the_runners_own_field_set(script):
    """Not a copy. Whatever a dataset describes beyond these lands in
    `dataset_params`, which the comparability key groups on -- so a set short by
    one name would build a different dict here than the run does, and the check
    would clear a board that splits."""
    from visbench.runner import _RECORD_FIELDS

    assert script._RECORD_FIELDS is _RECORD_FIELDS


@pytest.mark.parametrize(
    "probe",
    [
        "classification",
        "scene_classification",
        "fine_grained_classification",
        "semantic_segmentation",
        "generic_segmentation",
        "detection",
        "depth",
        "surface_normal",
        "edge",
        "corner",
        "orientation",
        "instance_segmentation",
    ],
)
def test_every_board_command_comes_out_of_build_corpus(script, probe):
    """The flags must not be duplicated here: a second copy is free to drift
    from the one that produced the corpus, and then the check verifies a command
    nobody runs. `DRY_RUN=1` needs no dataset, which is what makes this fast."""
    try:
        argv = script.command_for(probe)
    except script.NoCommand as error:
        # `corner` and `orientation` read a pinned frame set that a script
        # stages, and `generic_segmentation` binarised VOC masks. Absent, the
        # builder skips them by design -- what must hold is that the skip is
        # reported as a skip rather than as a missing command.
        pytest.skip(f"{probe} needs staged data: {error}")
    assert argv[0] == "run"
    assert argv[1] == probe
    assert "--backbone" in argv and script.BACKBONE in argv
    assert "--results" in argv
