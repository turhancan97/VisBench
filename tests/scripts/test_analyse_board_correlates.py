"""The correlate script duplicates two tables, so both are pinned here.

It copies ``HEADLINE_METRICS`` and re-states which metrics are errors, so that
it runs without importing visbench -- and therefore without torch, which is
what lets it run on a login node where the project venv does not resolve. A
copy that drifts would rank a board on a metric nobody chose, or rank an error
metric upside down and print the result as a finding.

These are the same class of check as ``tests/test_readme.py``: nothing executes
the duplicated table, so nothing else would catch it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from visbench.results.leaderboard import metric_direction
from visbench.results.render import HEADLINE_METRICS

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyse_board_correlates.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyse_board_correlates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load()


def test_the_copied_headline_table_matches_the_real_one(script):
    """The one that actually breaks: a probe added with a new headline metric."""
    assert script.HEADLINE_METRICS == HEADLINE_METRICS


def test_lower_is_better_agrees_with_metric_direction(script):
    """Restating a direction is the error that reads as a finding.

    Checked against ``metric_direction`` rather than a second literal, so the
    source of truth is the same function the leaderboard ranks with.
    """
    for task, metric in HEADLINE_METRICS.items():
        expected = metric_direction(metric) == "lower"
        assert (task in script.LOWER_IS_BETTER) is expected, (
            f"{task}'s headline metric {metric!r} is "
            f"{metric_direction(metric)}-is-better, which LOWER_IS_BETTER disagrees with"
        )


def test_every_backbone_in_the_corpus_has_a_structure_entry(script):
    """A new column reaches the corpus before it reaches the structural table.

    Correlating against a structure that does not exist would drop the backbone
    from every coefficient silently, so the script exits 1 -- and this fails
    first, in the fast suite, rather than after someone reads a stale number.
    """
    boards = script.load_boards(script.CORPUS)
    seen = {backbone for board in boards.values() for backbone in board}
    assert seen <= set(script.STRUCTURE), (
        f"no STRUCTURE entry for {', '.join(sorted(seen - set(script.STRUCTURE)))}"
    )


def test_spearman_on_the_cases_with_known_answers(script):
    """Perfect agreement, perfect inversion, and the sign in between."""
    same = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert script.spearman(same, same) == pytest.approx(1.0)
    assert script.spearman(same, {"a": 1.0, "b": 2.0, "c": 3.0}) == pytest.approx(-1.0)
    assert script.spearman(same, {"a": 3.0, "b": 1.0, "c": 2.0}) == pytest.approx(0.5)


def test_spearman_correlates_only_the_shared_backbones(script):
    """A half-filled board shrinks the comparison rather than raising.

    The corpus is grown one backbone column at a time, so a board that is
    missing a row is a normal intermediate state, not an error.
    """
    left = {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0}
    right = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert script.spearman(left, right) == pytest.approx(1.0)


def test_spearman_refuses_a_sample_too_small_to_mean_anything(script):
    with pytest.raises(ValueError, match="at least 3"):
        script.spearman({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0})


def test_error_metrics_are_oriented_before_ranking(script):
    """surface_normal's headline is angular error, so the board must be negated.

    Without this the normals board ranks upside down, and every correlation
    drawn from it flips sign -- which looks like a result about geometry.
    """
    boards = script.load_boards(script.CORPUS)
    normals = boards["surface_normal"]
    best = max(normals, key=lambda k: normals[k])
    assert normals[best] < 0, "an error metric should have been negated"
    assert best == "mae_vitb16", "mae_vitb16 has the lowest mean angular error in the corpus"
