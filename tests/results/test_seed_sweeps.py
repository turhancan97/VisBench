"""A seed sweep is only meaningful if its seed-0 rows are the board it claims.

`results/controls/seeds/<probe>.jsonl` re-fits a published board's backbones at
several seeds, so its adjacent pairs can be ordered from paired differences
instead of from a gap (20b). `test_pose_seed_sweep.py` guards the first such
file, written before there was a second; this covers every one of them, and the
reasoning is the same in both places.

**These files could merge invisibly.** Every other control differs from its
board in something `comparability_key` reads -- a head, a layer set, a backbone
name -- so it forms its own group and could at worst be ignored. A sweep differs
only in `seed`, which the key does not read at all, so merging one would put
rankable rows inside the published board's own group and `latest_per_backbone`
would hand the board to whichever seed was written last.

**And a sweep is void unless its seed-0 rows reproduce the corpus.** Every
published cell was run at the default seed 0. If a sweep's seed-0 rows disagree,
it was measured against some adjacent configuration -- a different cache, a
moved default, a changed split -- and its other seeds describe *that*
configuration rather than this board. The check is free, so it is pinned here
rather than left to whoever re-runs the sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visbench.results.leaderboard import comparability_key, group_comparable, latest_per_backbone
from visbench.results.render import HEADLINE_METRICS
from visbench.results.schema import ResultRecord
from visbench.results.writer import iter_records

ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = ROOT / "results" / "controls" / "seeds"
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"

#: How closely a seed-0 row must reproduce its published cell, per probe.
#:
#: These are **measured reproducibility floors**, not room made for a sweep that
#: missed. Every value here sits just above the worst delta observed while the
#: run reported a `train_loss` identical to its published cell's -- same fit,
#: moved score, so what moved is the metric. A sweep that re-fitted a slightly
#: different configuration also lands a few decimals away, and the fit is the
#: only thing that tells the two apart, which is what schema v8 is for.
#:
#: They fall into three groups, and the group is the reason rather than the
#: size:
#:
#:   * **exact** -- `classification`, `fine_grained_classification` and
#:     `relative_pose` reproduce bit for bit on all thirteen rows.
#:   * **float32 reduction order**, 1e-7 to 3e-5: every dense board. The
#:     ordering of a sum over patches or batches is not fixed, and
#:     `CORPUS_FINDINGS.md` already puts these at ~1e-7 relative.
#:   * **the metric is not a smooth function of the prediction**, 1e-3 to 3e-2.
#:     `detection` and `instance_segmentation` score a *ranking*, so a near-tie
#:     between two boxes flips and the whole AP curve moves; `orientation`'s
#:     angular error is ill-conditioned, which `CORPUS_FINDINGS.md` records from
#:     the run that added its ceilings. These are the boards this project
#:     already says to quote to three decimals.
SEED_ZERO_TOLERANCE: dict[str, float] = {
    # dense boards: float32 reduction order
    "scene_parsing": 1e-6,
    "corner": 1e-6,
    "keypoints2d": 1e-6,
    "edge": 1e-6,
    "generic_segmentation": 2e-6,
    "occlusion_edge": 1e-5,
    "surface_normal": 1e-5,
    "semantic_segmentation": 1e-4,
    "depth": 1e-4,
    # metrics that are not smooth in the prediction
    "instance_segmentation": 2e-3,
    "detection": 2e-3,
    "orientation": 5e-2,
}

SWEEPS = sorted(SWEEP_DIR.glob("*.jsonl")) if SWEEP_DIR.is_dir() else []


def _records(path: Path) -> list[ResultRecord]:
    return list(iter_records(path))


def _board(task: str) -> dict[str, ResultRecord]:
    """The published cell per backbone, grouped as `LEADERBOARD.md` groups them."""
    groups = group_comparable([r for r in iter_records(CORPUS) if r.task == task])
    assert len(groups) == 1, f"{task} has {len(groups)} comparability groups in the corpus"
    (group,) = groups.values()
    return {cell.backbone: cell for cell in latest_per_backbone(group)}


@pytest.fixture(scope="module")
def sweeps() -> dict[Path, list[ResultRecord]]:
    if not SWEEPS:
        pytest.skip(f"no seed sweeps under {SWEEP_DIR}")
    return {path: _records(path) for path in SWEEPS}


def test_a_sweep_holds_one_probe():
    """A file named for a probe that holds two would make every check below lie."""
    if not SWEEPS:
        pytest.skip(f"no seed sweeps under {SWEEP_DIR}")
    for path in SWEEPS:
        tasks = {record.task for record in _records(path)}
        assert tasks == {path.stem}, f"{path.name} holds {sorted(tasks)}"


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.stem)
def test_every_seed_zero_row_reproduces_its_published_cell(path):
    """The gate. A disagreement here voids every other seed in the file."""
    task = path.stem
    board = _board(task)
    metric = HEADLINE_METRICS[task]
    tolerance = SEED_ZERO_TOLERANCE.get(task, 0.0)

    checked = 0
    for record in _records(path):
        if record.seed != 0:
            continue
        published = board.get(record.backbone)
        assert published is not None, f"{record.backbone} is not on the published {task} board"
        delta = abs(record.metrics[metric] - published.metrics[metric])
        assert delta <= tolerance, (
            f"{task} / {record.backbone} seed 0 reads {record.metrics[metric]} against "
            f"the corpus's {published.metrics[metric]} (delta {delta:.2e}). The sweep is "
            "not measuring this board, so its other seeds describe some other "
            "configuration."
        )
        checked += 1
    assert checked == len(board), (
        f"only {checked} of {len(board)} backbones have a seed-0 row; a sweep must "
        "cover the whole board or the separability it supports has holes"
    )


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.stem)
def test_a_sweep_shares_its_boards_comparability_group(path):
    """The reason these files are dangerous, asserted rather than described.

    If this fails, something started distinguishing a sweep's records from the
    board -- which would make them *safe* to merge and would also mean the sweep
    has stopped re-fitting the published configuration. Either way this file and
    `results/controls/README.md` need rewriting, so fail loudly.
    """
    board = _board(path.stem)
    for record in _records(path):
        published = board.get(record.backbone)
        assert published is not None
        assert comparability_key(record) == comparability_key(published), (
            f"{path.stem} / {record.backbone} seed {record.seed} no longer groups with "
            "the published cell; the sweep has stopped reproducing the board's "
            "configuration."
        )


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.stem)
def test_a_sweep_is_paired_across_the_same_seeds(path):
    """Paired differences need the same seeds on both rows, not merely several."""
    seeds: dict[str, set[int]] = {}
    for record in _records(path):
        seeds.setdefault(record.backbone, set()).add(record.seed)
    assert seeds, f"no records in {path.name}"
    for backbone, present in seeds.items():
        assert len(present) >= 3, (
            f"{path.stem} / {backbone} has {len(present)} seed(s); a paired "
            "difference over fewer than three is not a measurement"
        )
    assert len({frozenset(s) for s in seeds.values()}) == 1, (
        "backbones were run at different seed sets, so the differences are not "
        f"paired: { ({b: sorted(s) for b, s in seeds.items()}) }"
    )


def test_no_sweep_record_is_in_the_corpus(sweeps):
    """The merge this whole file exists to prevent, checked directly."""
    corpus = {(r.task, r.backbone, r.timestamp) for r in iter_records(CORPUS)}
    for path, records in sweeps.items():
        shared = [
            (r.backbone, r.seed) for r in records if (r.task, r.backbone, r.timestamp) in corpus
        ]
        assert not shared, (
            f"{path.name}: {len(shared)} row(s) are already in the corpus: {shared[:3]}"
        )
