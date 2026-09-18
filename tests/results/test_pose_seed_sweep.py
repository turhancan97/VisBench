"""The pose seed sweep is only meaningful if its seed-0 rows are the board.

`results/controls/pose_seeds.jsonl` re-fits all thirteen corpus backbones at
five seeds so `relative_pose`'s tie list can be derived from paired differences
instead of a gap threshold (20a). Two things about it need a guard, and neither
is covered by `test_controls_stay_out_of_the_corpus.py`'s globbed checks.

**It is the second file that could merge invisibly.** Every other control
differs from its board in something `comparability_key` reads -- a head, a
layer set, a backbone name -- so it forms its own group. This one differs only
in `seed`, which the key does not read at all, so a merge would put sixty-five
rankable rows in the published board's own group and `latest_per_backbone`
would hand the board to whichever seed was written last.

**And the sweep is void unless its seed-0 rows reproduce the corpus.** Every
published pose cell was run at the default seed 0. If the sweep's seed-0 rows
disagree, it was measured against some adjacent configuration -- a different
cache, a moved default, a changed pair set -- and the other four seeds describe
that configuration rather than this board. The check is free and it is the only
thing standing between "five seeds of the published probe" and "five seeds of
something else", so it is pinned here rather than left to whoever re-runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visbench.results.leaderboard import comparability_key
from visbench.results.schema import ResultRecord

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "results" / "controls" / "pose_seeds.jsonl"
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
TASK = "relative_pose"
METRIC = "rotation_error_deg"


def _records(path: Path, task: str = TASK) -> list[ResultRecord]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if raw.get("task") == task:
            records.append(ResultRecord(**raw))
    return records


@pytest.fixture(scope="module")
def sweep() -> list[ResultRecord]:
    if not SWEEP.is_file():
        pytest.skip(f"no seed sweep at {SWEEP}")
    return _records(SWEEP)


@pytest.fixture(scope="module")
def board() -> dict[str, ResultRecord]:
    """The published cell per backbone, newest wins, as `latest_per_backbone`."""
    cells: dict[str, ResultRecord] = {}
    for record in _records(CORPUS):
        cells[record.backbone] = record
    return cells


def test_every_seed_zero_row_reproduces_its_published_cell(sweep, board):
    """The gate. A disagreement here voids every other seed in the file."""
    checked = 0
    for record in sweep:
        if record.seed != 0:
            continue
        published = board.get(record.backbone)
        assert published is not None, f"{record.backbone} is not on the published board"
        assert record.metrics[METRIC] == published.metrics[METRIC], (
            f"{record.backbone} seed 0 reads {record.metrics[METRIC]} against the "
            f"corpus's {published.metrics[METRIC]}. The sweep is not measuring this "
            "board, so its other seeds describe some other configuration."
        )
        checked += 1
    assert checked == len(board), (
        f"only {checked} of {len(board)} backbones have a seed-0 row; the sweep "
        "must cover the whole board or the tie list it supports has holes"
    )


def test_the_sweep_shares_the_boards_comparability_group(sweep, board):
    """The reason this file is dangerous, asserted rather than described.

    If this ever fails, something started distinguishing these records from the
    board -- which would make them *safe* to merge and would also mean the sweep
    is no longer re-fitting the published configuration. Either way the docstring
    above and `results/controls/README.md` need rewriting, so fail loudly.
    """
    for record in sweep:
        published = board.get(record.backbone)
        assert published is not None
        assert comparability_key(record) == comparability_key(published), (
            f"{record.backbone} seed {record.seed} no longer groups with the "
            "published cell; the sweep has stopped reproducing the board's "
            "configuration."
        )


def test_the_sweep_covers_several_seeds_per_backbone(sweep):
    """A one-seed-per-backbone file would pass everything above and say nothing."""
    seeds: dict[str, set[int]] = {}
    for record in sweep:
        seeds.setdefault(record.backbone, set()).add(record.seed)
    assert seeds, "no sweep records"
    for backbone, present in seeds.items():
        assert len(present) >= 3, (
            f"{backbone} has {len(present)} seed(s); a paired difference over "
            "fewer than three is not a measurement"
        )
    assert len({frozenset(s) for s in seeds.values()}) == 1, (
        "backbones were run at different seed sets, so the paired differences "
        f"are not paired: { ({b: sorted(s) for b, s in seeds.items()}) }"
    )
