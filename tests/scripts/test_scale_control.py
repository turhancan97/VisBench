"""The feature-scale control (24b): its matrix, its arithmetic, and its gate.

Three scripts and one array define it -- `run_scale_control.py` (the arms),
`slurm/scale_control.sbatch` (the matrix), `analyse_scale_control.py` (the
reading) and `measure_feature_scale.py` (the cross-corpus table) -- which is the
shape that left the corpus array a probe short for a release (10b). So the
array is read here and compared against the scripts.

The committed control is also gated here, in the fast suite, the way the seed
sweeps' seed-0 rows are: its x1 arm must reproduce the published cell, or its
scaled arm describes some adjacent configuration and every number quoted from
it is about the wrong board.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest

import visbench

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SBATCH = ROOT / "slurm" / "scale_control.sbatch"
CONTROL = ROOT / "results" / "controls" / "scale_control.jsonl"
SCALE_TABLE = ROOT / "results" / "controls" / "feature_scale.json"


def _load(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run():
    return _load("run_scale_control")


@pytest.fixture(scope="module")
def analyse():
    return _load("analyse_scale_control")


@pytest.fixture(scope="module")
def measure():
    return _load("measure_feature_scale")


def _array(name: str) -> list[str]:
    text = SBATCH.read_text(encoding="utf-8")
    body = re.search(rf"^{name}=\(([^)]*)\)", text, re.MULTILINE)
    assert body, f"no {name}=( ... ) in {SBATCH.name}"
    return body.group(1).replace("\\", " ").split()


# -- the matrix --------------------------------------------------------------


def test_the_array_arms_are_the_script_arms(run, analyse):
    arms = [variant for variant, _ in run.ARMS]
    assert _array("ARM_LIST") == arms
    assert tuple(arms) == analyse.ARMS


def test_the_array_probes_are_linear_dense_boards(measure):
    probes = _array("PROBES")
    assert set(probes) <= set(visbench.list_probes())
    assert set(probes) <= set(measure.LINEAR_DENSE), (
        "the control varies what a LinearHead reads; a board with another head "
        "or pooled features is a different question"
    )


def test_the_arms_are_each_others_scale(run):
    """x8.66 and x1/8.66: each column moved onto the other's measured scale."""
    (_, up), (_, down) = run.ARMS
    assert up * down == pytest.approx(1.0)
    assert up > 1.0


def test_importing_the_script_registers_nothing(run):
    """An alias is registered only when a run asks for one, never at import."""
    assert not [n for n in visbench.list_backbones() if re.search(r"_x[0-9.]+$", n)]


def test_the_array_never_targets_the_node_it_was_moved_off():
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --exclude=dgx2" in text


# -- the arithmetic ----------------------------------------------------------


def test_an_alias_name_splits_into_column_and_scale(analyse):
    assert analyse.split_arm("convnext_base_x0.1155") == ("convnext_base", 0.1155)
    assert analyse.split_arm("clip_convnext_base_x8.66") == ("clip_convnext_base", 8.66)
    assert analyse.split_arm("clip_convnext_base_x1") == ("clip_convnext_base", 1.0)


def test_a_name_that_is_not_an_arm_is_refused(analyse):
    with pytest.raises(ValueError):
        analyse.split_arm("resnet50_x2")


@pytest.mark.parametrize(
    ("higher", "value", "expected"),
    [(True, 0.95, 1), (True, 0.55, 3), (True, 0.05, 4), (False, 0.05, 1), (False, 0.95, 4)],
)
def test_place_ranks_against_the_other_cells_only(analyse, higher, value, expected):
    cells = {"a": 0.9, "b": 0.6, "c": 0.1, "me": 0.5}
    assert analyse.place(cells, "me", value, higher=higher) == expected


def test_partial_correlation_is_one_when_x_is_y(measure):
    """r_xy = 1 and r_xz = r_zy, so the formula collapses to (1-r²)/(1-r²)."""
    x = {k: float(i) for i, k in enumerate("abcdefg")}
    z = {"a": 2.0, "b": 0.0, "c": 5.0, "d": 1.0, "e": 6.0, "f": 3.0, "g": 4.0}
    assert measure.partial_spearman(x, dict(x), z) == pytest.approx(1.0)


# -- the committed control ---------------------------------------------------


def _records() -> list[dict]:
    return [json.loads(line) for line in CONTROL.read_text().splitlines() if line.strip()]


def test_the_control_is_the_whole_matrix(run):
    probes = _array("PROBES")
    seen = {(r["task"], r["backbone"]) for r in _records()}
    for probe in probes:
        for variant, scale in run.ARMS:
            assert (probe, f"{variant}_x{scale:.4g}") in seen
            assert (probe, f"{variant}_x1") in seen
    assert len(_records()) == len(probes) * len(run.ARMS) * 2


def test_the_gate_reproduces_every_published_cell(analyse):
    """The x1 arm through the alias path against the corpus, per board's floor."""
    sweeps = _load_seed_tolerances()
    corpus = analyse._records(analyse.CORPUS)
    rows = analyse.analyse(corpus, analyse._records(CONTROL))
    assert rows
    for row in rows:
        assert row["gate"] is not None, (row["task"], row["backbone"])
        tolerance = max(sweeps.get(row["task"], 0.0), 1e-6)
        delta = abs(row["gate"] - row["published"])
        assert delta <= tolerance, (
            f"{row['task']} / {row['backbone']} x1 reads {row['gate']} against the "
            f"corpus's {row['published']} (delta {delta:.1e}); the control is not "
            "measuring this board"
        )


def _load_seed_tolerances() -> dict[str, float]:
    spec = importlib.util.spec_from_file_location(
        "_seed_sweep_tolerances", ROOT / "tests" / "results" / "test_seed_sweeps.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SEED_ZERO_TOLERANCE


def test_the_three_reversals_the_prose_quotes_are_in_the_control(analyse):
    """CORPUS_FINDINGS.md says the pair reverses on exactly these boards."""
    rows = analyse.analyse(analyse._records(analyse.CORPUS), analyse._records(CONTROL))
    by = {(r["task"], r["backbone"]): r for r in rows}
    for task in ("generic_segmentation", "semantic_segmentation", "depth"):
        clip, sup = by[(task, "clip_convnext_base")], by[(task, "convnext_base")]
        published_order = clip["published"] < sup["published"]
        swapped_order = clip["scaled"] < sup["scaled"]
        assert published_order and not swapped_order, task


# -- the cross-corpus table --------------------------------------------------


def test_the_scale_table_covers_every_column(measure):
    rows = json.loads(SCALE_TABLE.read_text())
    names = {row["backbone"] for row in rows}
    sys.path.insert(0, str(SCRIPTS))
    assert names == set(measure.board_backbones())


def test_the_quoted_spread_is_the_table_value():
    rows = json.loads(SCALE_TABLE.read_text())
    spread = max(r["std"] for r in rows) / min(r["std"] for r in rows)
    quoted = f"{spread:.1f}x"
    assert math.isfinite(spread)
    for path in ("CORPUS_FINDINGS.md", "results/controls/README.md", "CLAUDE.md"):
        text = (ROOT / path).read_text(encoding="utf-8")
        assert quoted in text, f"{path} should quote the scale spread as {quoted}"
