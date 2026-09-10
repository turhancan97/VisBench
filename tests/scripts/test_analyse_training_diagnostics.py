"""The training-diagnostics script duplicates three tables, and interprets one.

Like ``analyse_board_correlates.py`` it copies ``HEADLINE_METRICS`` and restates
which metrics are errors, so that it runs without importing visbench -- and so
without torch, which is what lets it run on a login node. A drifted copy reads a
board on a metric nobody chose, or reports "the better fit scores better"
upside down and prints it as a finding.

The rest of this file is about the one judgement the script makes: whether a
``train_loss`` is an unconverged head or a weaker backbone fitting slightly
worse. The first version asked whether the worst fit was above the median by
more than half the board's loss spread, which is true of the worst fit by
construction -- so it fired on ``edge``, where all twelve heads converged
inside a 0.046 span. Both directions are pinned below, because a criterion that
cannot fail to fire and one that can never fire are the same defect.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

import visbench
from visbench.results.leaderboard import metric_direction
from visbench.results.render import HEADLINE_METRICS

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyse_training_diagnostics.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyse_training_diagnostics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load()


def _board(losses, scores, *, top1=None, has_training=True, metric="iou"):
    """A synthetic board: ``{backbone: record}``, newest-wins already resolved."""
    board = {}
    for index, (loss, score) in enumerate(zip(losses, scores, strict=True)):
        training = None
        if has_training:
            training = {"train_loss": loss}
            if top1 is not None:
                training["train_top1"] = top1[index]
        board[f"backbone{index:02d}"] = {
            "metrics": {metric: score},
            "training": training,
            "visbench_version": "0.17.0",
        }
    return board


def _report(script, task, board):
    out = io.StringIO()
    verdict = script.report_board(task, board, out)
    return verdict, out.getvalue()


# -- the duplicated tables --------------------------------------------------


def test_the_copied_headline_table_matches_the_real_one(script):
    """The one that actually breaks: a probe added with a new headline metric."""
    assert script.HEADLINE_METRICS == HEADLINE_METRICS


def test_lower_is_better_agrees_with_metric_direction(script):
    """Restating a direction is the error that reads as a finding."""
    for task, metric in HEADLINE_METRICS.items():
        expected = metric_direction(metric) == "lower"
        assert (task in script.LOWER_IS_BETTER) is expected, (
            f"{task}'s headline metric {metric!r} is "
            f"{metric_direction(metric)}-is-better, which LOWER_IS_BETTER disagrees with"
        )


def test_the_zero_shot_set_matches_the_probes_own_declaration(script):
    """`training: None` is correct for these and a gap for everything else, so
    the script must not decide which is which from its own literal alone."""
    required = {
        "semantic_segmentation": {"num_classes": 3},
        "detection": {"num_classes": 3},
        "instance_segmentation": {"num_classes": 3},
    }
    declared = {
        name
        for name in visbench.list_probes()
        if visbench.get_probe(name, **required.get(name, {})).zero_shot
    }
    assert script.ZERO_SHOT == declared


# -- the judgement ----------------------------------------------------------


def test_a_converged_board_with_a_real_ordering_is_not_suspect(script):
    """The regression. Twelve heads that all converged, the weakest fitting
    slightly worse, is the ordinary case -- and is what the first version of
    this rule flagged, because it compared the worst fit against a spread the
    worst fit defines."""
    losses = [0.50 + 0.004 * i for i in range(12)]
    scores = [0.80 - 0.02 * i for i in range(12)]
    verdict, text = _report(script, "generic_segmentation", _board(losses, scores))
    assert verdict["verdict"] == "ok", text
    assert "SUSPECT" not in text


def test_a_head_that_did_not_converge_is_flagged(script):
    """The other direction: one loss away from the distribution the rest form."""
    losses = [0.50 + 0.004 * i for i in range(11)] + [2.0]
    scores = [0.80 - 0.02 * i for i in range(11)] + [0.05]
    verdict, text = _report(script, "generic_segmentation", _board(losses, scores))
    assert verdict["verdict"] == "suspect", text
    assert verdict["suspect"] == ["backbone11"]
    assert "above this board's median" in text and "MAD)" in text


def test_a_saturated_fit_is_reported_as_generalisation(script):
    """Every `train_top1` at 1.0 means the board's spread cannot be underfitting
    -- the CUB case, where the last-placed backbone also fits perfectly."""
    verdict, text = _report(
        script,
        "classification",
        _board(
            [0.01] * 12,
            [0.9 - 0.03 * i for i in range(12)],
            top1=[1.0] * 12,
            metric="top1",
        ),
    )
    assert verdict["verdict"] == "saturated", text
    assert "SATURATED" in text


def test_a_pre_v8_board_says_it_cannot_answer_rather_than_guessing(script):
    """`training: null` is the state the re-run existed to clear, so it must be
    reported as a limit of the board and never as a fit of zero."""
    verdict, text = _report(
        script,
        "generic_segmentation",
        _board([0.0] * 12, [0.5] * 12, has_training=False),
    )
    assert verdict["verdict"] == "unanswerable"
    assert "CANNOT answer" in text


def test_a_zero_shot_board_is_not_diagnosed_at_all(script):
    verdict, text = _report(script, "retrieval", _board([0.0] * 3, [0.5] * 3, has_training=False))
    assert verdict["verdict"] == "zero_shot"
    assert "fits nothing" in text


@pytest.mark.parametrize("task", ["generic_segmentation", "surface_normal"])
def test_the_fit_correlation_reads_the_same_way_on_both_polarities(script, task):
    """`surface_normal`'s headline metric is an angular error, so a better score
    is a *smaller* number. A board where the better fit scores better must print
    a positive rho either way, or the coefficient is a finding about nothing."""
    losses = [0.50 + 0.02 * i for i in range(6)]
    scores = [0.80 - 0.05 * i for i in range(6)]
    if task in script.LOWER_IS_BETTER:
        scores = [-score for score in scores]
    board = _board(losses, scores, metric=script.HEADLINE_METRICS[task])
    verdict, text = _report(script, task, board)
    assert verdict["rho"] > 0.9, text


def test_a_board_missing_its_headline_metric_names_the_key(script):
    """How the copied table drifts in practice: `occlusion_edge` emits
    `occlusion_edge_correlation`, and reading it as `edge_correlation` used to
    raise a TypeError from inside a format string -- naming neither the board
    nor the key."""
    verdict, text = _report(
        script, "generic_segmentation", _board([0.5] * 4, [0.5] * 4, metric="not_iou")
    )
    assert verdict["verdict"] == "unscored"
    assert "carry no 'iou'" in text
    assert "not_iou" in text


def test_a_saturated_board_is_never_also_suspect(script):
    """The object classification board: every `train_top1` is 1.0000 and
    `mae_vitb16`'s train_loss is 0.0061 against a median of 0.0000. Flagging it
    as an unconverged head contradicts the saturation line printed one line
    above -- a head at train_top1 1.0 fitted everything it was given."""
    losses = [0.0] * 11 + [0.0061]
    scores = [0.99 - 0.001 * i for i in range(11)] + [0.958]
    verdict, text = _report(
        script, "classification", _board(losses, scores, top1=[1.0] * 12, metric="top1")
    )
    assert verdict["verdict"] == "saturated", text
    assert "SUSPECT" not in text


def test_a_board_with_identical_losses_gets_no_outlier_test(script):
    """No spread, nothing to be an outlier against -- and `mad == 0` would make
    the multiple a division by zero."""
    verdict, text = _report(
        script, "generic_segmentation", _board([0.5] * 12, [0.9 - 0.01 * i for i in range(12)])
    )
    assert verdict["verdict"] == "ok", text
    assert "SUSPECT" not in text


def test_an_absurd_mad_multiple_is_reported_as_a_magnitude(script):
    """A flat-but-not-identical board can put an outlier thousands of MAD out,
    and "4905.6 MAD" reads as precision that is not there. The absolute excess
    leads; the multiple is rounded down to its magnitude."""
    losses = [1e-9 * i for i in range(11)] + [0.006]
    scores = [0.9 - 0.001 * i for i in range(12)]
    verdict, text = _report(script, "generic_segmentation", _board(losses, scores))
    assert verdict["verdict"] == "suspect", text
    assert "0.0060 above this board's median" in text
    assert ">" in text and "MAD" in text
