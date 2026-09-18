#!/usr/bin/env python
"""Which of a board's adjacent pairs are actually ordered? (20b)

    python scripts/analyse_seeds.py corner
    python scripts/analyse_seeds.py --sweep results/controls/seeds/detection.jsonl

`scripts/analyse_pose_seeds.py` asked this of one board and found that two of
the five pairs `relative_pose`'s published tie list names come out the other way
round across seeds. Nothing about that reasoning was specific to pose: every
trained board here is one seed of a fit whose run-to-run scatter nobody has
measured, and the gap between two rows is one draw of it.

So this is that script with the probe taken out of it. It reads the metric and
its direction from the same tables `LEADERBOARD.md` ranks with
(`HEADLINE_METRICS`, `metric_direction`) rather than naming one, because a
script that hard-codes "lower is better" silently inverts every verdict on the
first board where it is not.

**Three questions, in this order.**

*Does the sweep measure this board?* Every corpus cell was run at the default
seed 0, so every seed-0 row here must reproduce its published value **to that
board's own reproducibility floor** -- exactly for most, and `--tolerance` for
one that has a measured floor above zero. The check is free and the rest is
void without it: a disagreement means the sweep re-fitted some adjacent
configuration -- a moved default, a different cache, a changed split -- and its
other seeds describe that instead.

The two cases are told apart by the **fit**, not by the size of the gap, which
is what schema v8's `training` block exists for. A re-run whose `train_loss`
matches the published one to the digit fitted the same head on the same
features, so a score that still moves is the *metric* being non-deterministic --
`detection`'s is, because average precision is a ranking and near-ties flip. A
re-run whose `train_loss` also moved is a different configuration, and nothing
below applies to the published board.

*How big is the noise, per row?* A standard deviation and a range per backbone.
A range over three draws is itself unstable (19b), which is why the sweeps are
run at five and why the standard deviation is printed beside it.

*Which adjacent pairs are separable?* Not a gap against a noise figure. The same
seeds were run for both rows, so the **paired difference** is available, and it
answers something a threshold cannot express: a pair can fail to separate *and*
lean the other way. `relative_pose` had two of those, at gaps of 0.04 and 0.12.

**And one statistic that says whether a gap could ever have carried the answer:**
how much of the seed-to-seed variance is *common-mode*, i.e. a seed making every
backbone look good at once. Common-mode noise cancels in a difference, so a
board where most of it is common-mode is one where gaps are more trustworthy
than the per-row spread suggests. On `relative_pose` only 11% was, which is why
no function of the gap could carry that board's answer.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from visbench.results.leaderboard import (
    group_comparable,
    latest_per_backbone,
    metric_direction,
)
from visbench.results.render import HEADLINE_METRICS
from visbench.results.schema import ResultRecord
from visbench.results.writer import iter_records

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
SWEEP_DIR = ROOT / "results" / "controls" / "seeds"

#: Two-sided 95% critical value for Student's t at df = 4, i.e. five seeds.
#: Quoted rather than computed so the small sample is visible in the source: at
#: n=5 this test has little power, and a pair it cannot separate is often a pair
#: nobody measured enough times rather than a pair that is genuinely level.
T_CRITICAL = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def load(path: Path, task: str | None = None) -> list[ResultRecord]:
    if not path.is_file():
        print(f"no records at {path}", file=sys.stderr)
        return []
    return [r for r in iter_records(path) if task is None or r.task == task]


def board_cells(records: list[ResultRecord]) -> dict[str, ResultRecord]:
    """The published cell per backbone, grouped and de-duplicated as the board is.

    Read through `group_comparable` rather than by taking the newest line per
    backbone, because a task with two comparability groups is exactly the case
    `board_for` refuses to render -- and a sweep compared against a silently
    merged pair of groups would report disagreements that are really two
    different experiments.
    """
    groups = group_comparable(records)
    if not groups:
        return {}
    if len(groups) > 1:
        print(
            f"!!! {len(groups)} comparability groups for this task; a board is one group",
            file=sys.stderr,
        )
        return {}
    (group,) = groups.values()
    return {cell.backbone: cell for cell in latest_per_backbone(group)}


def by_seed(records: list[ResultRecord], metric: str) -> dict[str, dict[int, float]]:
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    for record in records:
        scores[record.backbone][record.seed] = record.metrics[metric]
    return scores


def check_seed_zero(
    published: dict[str, ResultRecord],
    seed_zero: dict[str, ResultRecord],
    metric: str,
    *,
    tolerance: float,
) -> bool:
    """The gate: does the sweep's seed 0 reproduce the published cell?

    Reports the fit beside the score, because the two ways this fails look
    identical in the number alone: a matching `train_loss` under a moved score
    is a non-deterministic metric, and a moved `train_loss` is a different
    configuration.
    """
    print("=== does the sweep measure this board? seed 0 against the corpus ===")
    ok = True
    worst = 0.0
    for backbone in sorted(seed_zero):
        cell = published.get(backbone)
        if cell is None:
            continue
        delta = seed_zero[backbone].metrics[metric] - cell.metrics[metric]
        worst = max(worst, abs(delta))
        same_fit = _same_fit(seed_zero[backbone], cell)
        within = abs(delta) <= tolerance
        ok = ok and within
        if within:
            note = ""
        else:
            note = "   <-- same fit, metric moved" if same_fit else "   <-- DIFFERENT FIT"
        print(
            f"  {backbone:20s} {seed_zero[backbone].metrics[metric]:10.4f}"
            f"  delta {delta:+.2e}  fit {'==' if same_fit else '!='}{note}"
        )
    missing = sorted(set(published) - set(seed_zero))
    if missing:
        ok = False
        print(f"  !!! no seed-0 row for {missing}")
    print(f"  -> worst |delta| {worst:.2e} against a tolerance of {tolerance:.2e}")
    print(f"  -> {'the sweep reproduces this board' if ok else 'NOT REPRODUCING within tolerance'}")
    return ok


#: How close two `train_loss` values have to be to mean "the same fit".
#:
#: Not exact equality, which is the obvious choice and is wrong: a seed-0 re-fit
#: that reproduces its published *score* bit for bit still moves `train_loss`,
#: because a loss is a mean over batches and float32 addition is not associative.
#: Measured across the pose and classification sweeps, that movement runs from
#: 1e-11 relative up to **6e-5**, the largest on a loss already down at 2e-07
#: where float32 has few digits left. Three significant figures sits well above
#: all of it and far below a changed configuration, which moves a loss by
#: percent. `abs_tol` covers the saturated case, where the published loss is
#: exactly 0.0 and no relative tolerance means anything.
FIT_REL_TOL = 1e-3
FIT_ABS_TOL = 1e-6


def _same_fit(a: ResultRecord, b: ResultRecord) -> bool:
    """Did the two runs fit the same head? `None` on either side cannot say."""
    if not a.training or not b.training:
        return False
    keys = set(a.training) & set(b.training)
    return bool(keys) and all(
        math.isclose(a.training[k], b.training[k], rel_tol=FIT_REL_TOL, abs_tol=FIT_ABS_TOL)
        for k in keys
    )


def spread(scores: dict[str, dict[int, float]]) -> list[float]:
    print("\n=== the noise, per row ===")
    # Significant figures rather than decimals: these boards span degrees of
    # angular error and a top-1 already at 0.99, where four decimals round the
    # quantity being measured away entirely.
    print(
        f"  {'backbone':20s} {'mean':>12s} {'sd':>11s} {'range':>11s} {'min':>12s} {'max':>12s}  n"
    )
    sds, rngs = [], []
    for backbone in sorted(scores, key=lambda b: statistics.mean(scores[b].values())):
        values = list(scores[backbone].values())
        if len(values) < 2:
            continue
        sd = statistics.stdev(values)
        sds.append(sd)
        rngs.append(max(values) - min(values))
        print(
            f"  {backbone:20s} {statistics.mean(values):12.5g} {sd:11.3g}"
            f" {max(values) - min(values):11.3g} {min(values):12.5g} {max(values):12.5g}"
            f"  {len(values)}"
        )
    if sds:
        print(
            f"\n  standard deviation: median {statistics.median(sds):.3g},"
            f" min {min(sds):.3g}, max {max(sds):.3g}"
        )
        print(
            f"  range:              median {statistics.median(rngs):.3g},"
            f" min {min(rngs):.3g}, max {max(rngs):.3g}"
        )
    return sds


def common_mode(scores: dict[str, dict[int, float]]) -> float | None:
    """What fraction of the seed variance moves every row together.

    A seed that makes the whole board look good cancels in any difference
    between two rows, so a board whose noise is mostly common-mode is one where
    a gap is more trustworthy than the per-row spread implies. Computed as the
    variance of the per-seed means over the mean of the per-backbone variances,
    which is the fraction it would contribute if rows were otherwise
    independent.
    """
    seeds = sorted(set.intersection(*(set(v) for v in scores.values()))) if scores else []
    rows = [b for b in scores if len(scores[b]) > 1]
    if len(seeds) < 2 or len(rows) < 2:
        return None
    centred = {b: [scores[b][s] - statistics.mean(scores[b].values()) for s in seeds] for b in rows}
    per_seed_mean = [statistics.mean(centred[b][i] for b in rows) for i in range(len(seeds))]
    shared = statistics.pvariance(per_seed_mean)
    total = statistics.mean(statistics.pvariance(centred[b]) for b in rows)
    return shared / total if total else None


def separability(
    cells: dict[str, float], scores: dict[str, dict[int, float]], *, higher_is_better: bool
) -> None:
    """Pairwise and paired: does the better row actually stay better?

    Not a gap against a noise figure. The same seeds were run for both rows, so
    the **paired difference** is available, and it is the statistic that answers
    the question a tie list is trying to ask. It also answers one a threshold
    cannot: a pair can fail to separate *and* lean the other way.
    """
    print("\n=== which adjacent pairs are separable? ===")
    print(
        "  (paired by seed: the same fits for both rows;"
        f" {'higher' if higher_is_better else 'lower'} is better)"
    )
    print(f"  {'pair':46s} {'gap':>11s} {'mean d':>11s} {'sd d':>10s} {'t':>7s}  wins  verdict")
    order = sorted(cells, key=lambda b: cells[b], reverse=higher_is_better)
    sign = 1.0 if higher_is_better else -1.0
    for better, worse in zip(order, order[1:], strict=False):
        if better not in scores or worse not in scores:
            continue
        shared = sorted(set(scores[better]) & set(scores[worse]))
        if len(shared) < 2:
            continue
        # Positive means the board's own order held on that seed, whichever
        # direction the metric runs.
        diffs = [sign * (scores[better][s] - scores[worse][s]) for s in shared]
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        t = mean / (sd / len(diffs) ** 0.5) if sd else float("inf")
        critical = T_CRITICAL.get(len(diffs), 2.365)
        if t <= -critical:
            verdict = "REVERSED"
        elif t >= critical:
            verdict = "ordered"
        else:
            verdict = "tied"
        print(
            f"  {better + ' vs ' + worse:46s} {abs(cells[better] - cells[worse]):11.4g}"
            f" {mean:11.4g} {sd:10.3g} {t:7.2f}  {sum(d > 0 for d in diffs)}/{len(diffs)}"
            f"  {verdict}"
        )
    print(
        f"\n  'ordered'/'REVERSED' is |t| >= {T_CRITICAL.get(5)} at five seeds"
        " (two-sided 95%); everything else is 'tied'."
    )
    print(
        "  REVERSED means the board's own order is the minority outcome: the pair"
        " is not\n  a coin flip, it leans the other way, which a gap threshold"
        " cannot say."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probe", nargs="?", help="probe name; inferred from --sweep if omitted")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--sweep", type=Path, help="default: results/controls/seeds/<probe>.jsonl")
    parser.add_argument("--metric", help="default: this probe's headline metric")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="how far a seed-0 row may sit from its published cell; this board's "
        "measured reproducibility floor, not a number chosen to make the gate pass",
    )
    args = parser.parse_args(argv)

    if not args.probe and not args.sweep:
        parser.error("name a probe, or point --sweep at a sweep file")
    sweep_path = args.sweep or SWEEP_DIR / f"{args.probe}.jsonl"

    records = load(sweep_path)
    if not records:
        return 1
    tasks = {r.task for r in records}
    if len(tasks) > 1:
        print(f"!!! {sweep_path} holds several probes: {sorted(tasks)}", file=sys.stderr)
        return 1
    task = args.probe or tasks.pop()
    records = [r for r in records if r.task == task]
    if not records:
        print(f"!!! no {task} records in {sweep_path}", file=sys.stderr)
        return 1

    metric = args.metric or HEADLINE_METRICS.get(task)
    if metric is None:
        print(f"!!! no headline metric listed for {task}", file=sys.stderr)
        return 1
    higher_is_better = metric_direction(metric) == "higher"

    cell_records = board_cells(load(args.corpus, task))
    cells = {name: cell.metrics[metric] for name, cell in cell_records.items()}
    if not cells:
        print(f"!!! no published {task} board in {args.corpus}", file=sys.stderr)
        return 1

    print(f"{task}: {metric} ({'higher' if higher_is_better else 'lower'} is better)")
    print(f"  board {args.corpus}\n  sweep {sweep_path}\n")

    scores = by_seed(records, metric)
    seed_zero = {r.backbone: r for r in records if r.seed == 0}
    reproduced = check_seed_zero(cell_records, seed_zero, metric, tolerance=args.tolerance)
    spread(scores)
    fraction = common_mode(scores)
    if fraction is not None:
        print(f"\n  common-mode: {fraction:.0%} of the seed variance moves every row together")
        print("  (it cancels in a difference, so the rest is what a gap has to survive)")
    separability(cells, scores, higher_is_better=higher_is_better)

    if not reproduced:
        print(
            "\n!!! seed 0 does not reproduce the corpus -- do not quote the above", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
