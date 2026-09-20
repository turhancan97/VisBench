"""The scene-noise study is void unless its baseline reproduced the board.

`results/controls/scene_noise.json` (20d) perturbs `scene_classification`'s
cached features by a known amount and reports how far the score moves. That
number only says something about the *published* board if the unperturbed run
reproduced the committed sweep's seed-0 row — and the first version of the
script did not, because it built the backbone before seeding where `run()` seeds
first (`visbench/runner.py`). Every recorded field matched and the baseline was
6e-3 away, which is the same size as the disagreement the study exists to
explain.

So the gate is recorded in the study file and pinned here. A future re-run that
lands on a neighbouring configuration cannot be committed as though it measured
this board.

There is no equivalent for `pose_noise.json` because that study has no board to
reproduce: its runs perturb features and are compared against *each other*.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results" / "controls" / "scene_noise.json"


@pytest.fixture(scope="module")
def study() -> dict:
    if not STUDY.is_file():
        pytest.skip(f"no study at {STUDY}")
    return json.loads(STUDY.read_text(encoding="utf-8"))


def test_every_backbone_reproduced_its_board_row(study):
    """The gate, as the script recorded it: delta 0.0 against the sweep's seed 0."""
    assert study["backbones"], "the study holds no backbones"
    for row in study["backbones"]:
        assert row["gate_ok"], (
            f"{row['backbone']}'s unperturbed baseline missed the sweep's seed-0 row by "
            f"{row['gate_delta']:.2e}, so its curve describes some other configuration"
        )
        assert row["gate_delta"] == 0.0


def test_the_study_spans_a_thousand_fold_range(study):
    """A trigger is told from a dose by the *range*, so one magnitude proves nothing."""
    magnitudes = study["magnitudes"]
    assert max(magnitudes) / min(magnitudes) >= 1000, magnitudes
    for row in study["backbones"]:
        assert [entry["magnitude"] for entry in row["curve"]] == magnitudes


def test_the_study_records_the_schedule_it_ran(study):
    """The classification schedule, not the dense one -- 200 epochs at 1e-2."""
    assert study["schedule"] == {"epochs": 200, "lr": 1e-2, "batch_size": 256}
    assert study["limit"] == 100
