"""The 25b-1 learning-rate sweep's pure parts: the held-out slice and the rule.

The GPU half (`scripts/run_lr_sweep.py`'s runs) cannot run here. What can is
what decides the outcome: which images are held out, and which rate the rule
picks from a table of scores -- so both are pinned.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from analyse_lr_sweep import BAR_TOLERANCE, choose, rate_of  # noqa: E402
from run_lr_sweep import PUBLISHED_LR, RATES, arms, holdout_indices  # noqa: E402


class TestHoldout:
    def test_the_slice_is_fixed(self):
        assert holdout_indices(795) == holdout_indices(795)

    def test_the_halves_partition_the_split(self):
        fit, held = holdout_indices(600)
        assert not set(fit) & set(held)
        assert sorted(fit + held) == list(range(600))
        assert len(held) == 60

    def test_a_different_seed_is_a_different_slice(self):
        assert holdout_indices(600, seed=1) != holdout_indices(600)

    def test_a_split_too_small_to_hold_out_is_refused(self):
        with pytest.raises(ValueError):
            holdout_indices(1)


def test_one_raw_arm_at_the_published_rate_and_every_standardised_rate():
    planned = arms()
    assert planned[0] == ("raw", PUBLISHED_LR, False)
    assert [lr for _, lr, std in planned if std] == list(RATES)


class TestRule:
    def test_picks_the_best_mean_over_backbones(self):
        table = {
            "raw": {"a": 0.50, "b": 0.50},
            "std_lr0.001": {"a": 0.52, "b": 0.50},
            "std_lr0.002": {"a": 0.60, "b": 0.40},
            "std_lr0.005": {"a": 0.55, "b": 0.53},
        }
        result = choose(table, higher=True)
        assert result["chosen"] == 0.005
        assert result["verdict"] == "OK"

    def test_lower_is_better_boards_are_read_the_right_way_round(self):
        table = {
            "raw": {"a": 38.0, "b": 38.0},
            "std_lr0.001": {"a": 52.0, "b": 56.0},
            "std_lr0.005": {"a": 39.0, "b": 38.5},
        }
        assert choose(table, higher=False)["chosen"] == 0.005

    def test_a_tie_goes_to_the_smaller_rate(self):
        table = {"raw": {"a": 0.5}, "std_lr0.002": {"a": 0.6}, "std_lr0.001": {"a": 0.6}}
        assert choose(table, higher=True)["chosen"] == 0.001

    def test_a_board_that_stays_clearly_worse_than_raw_is_stopped(self):
        table = {"raw": {"a": 38.0}, "std_lr0.001": {"a": 52.0}}
        assert choose(table, higher=False)["verdict"] == "STOP"

    def test_a_shortfall_inside_the_tolerance_passes(self):
        raw = 0.5
        table = {"raw": {"a": raw}, "std_lr0.001": {"a": raw * (1 - BAR_TOLERANCE / 2)}}
        assert choose(table, higher=True)["verdict"] == "OK"

    def test_an_arm_missing_a_backbone_cannot_win(self):
        table = {
            "raw": {"a": 0.5, "b": 0.5},
            "std_lr0.001": {"a": 0.55, "b": 0.55},
            "std_lr0.002": {"a": 0.99},
        }
        assert choose(table, higher=True)["chosen"] == 0.001

    def test_rate_labels_parse(self):
        assert rate_of("std_lr0.002") == 0.002
        assert rate_of("raw") is None


def test_the_committed_sweep_reproduces_the_chosen_rates():
    """The table 25b-2 reads is recomputed from the records, never carried."""
    from analyse_lr_sweep import CHOSEN_RATES, CONTROL, STOPPED, _records, collect

    from visbench.results.leaderboard import metric_direction
    from visbench.results.render import HEADLINE_METRICS

    table = collect(_records(CONTROL))
    chosen, stopped = {}, []
    for task, by_arm in table.items():
        higher = metric_direction(HEADLINE_METRICS[task]) == "higher"
        result = choose(by_arm, higher)
        if result["verdict"] == "STOP":
            stopped.append(task)
        else:
            chosen[task] = result["chosen"]
    assert chosen == CHOSEN_RATES
    assert tuple(sorted(stopped)) == STOPPED


def test_no_chosen_rate_sits_on_the_edge_of_its_grid():
    """A choice at the grid's top was a limit of the search; the second pass
    extended it, and every chosen rate is now interior."""
    from analyse_lr_sweep import CHOSEN_RATES, CONTROL, _records, collect

    table = collect(_records(CONTROL))
    for task, rate in CHOSEN_RATES.items():
        swept = sorted(r for r in (rate_of(arm) for arm in table[task]) if r is not None)
        assert swept[0] < rate < swept[-1], task
