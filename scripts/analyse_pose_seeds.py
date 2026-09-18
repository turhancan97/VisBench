#!/usr/bin/env python
"""What is the `relative_pose` board's noise, measured over every row? (20a)

    python scripts/analyse_pose_seeds.py

`relative_pose` is quoted to whole degrees and five of its adjacent pairs are
called ties. That rule was derived in 16a-3 from **one** re-run, and 19b, which
re-derived it properly, could only reach two backbones -- which disagreed by 3x
(0.72 and 2.23 degrees of seed-to-seed range). So the published tie list rests
on two rows, one of which may be an outlier, and the statistic it rests on is a
range over three draws, which 19b showed is unstable enough to have caused the
problem in the first place.

`scripts/build_pose_seed_sweep.sh` re-fits all thirteen backbones at five seeds
in the published configuration. This reads those records and answers three
things:

**Does the sweep measure this board?** Every corpus pose cell was run at seed 0,
so every seed-0 row here must reproduce its published value exactly. That check
comes first and the rest is void without it.

**How big is the noise, per row and overall?** A standard deviation and a range
per backbone, so the board can state its scatter from a measurement over every
row it applies to rather than from two.

**Which adjacent pairs are actually separable?** Not by comparing a gap against
a noise figure -- that needs a distributional assumption nothing here has earned
-- but **directly and pairwise**: the same five seeds were run for both rows, so
count how often the higher-ranked backbone actually wins. Five out of five (or
zero) is an ordering; anything else is a tie, whatever the gap looks like. That
is the assumption-free version of the question the tie list is trying to answer.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRIC = "rotation_error_deg"
CORPUS = Path("results/corpus/visbench.jsonl")
SWEEP = Path("results/controls/pose_seeds.jsonl")


def load(path: Path, task: str = "relative_pose") -> list[dict]:
    if not path.is_file():
        print(f"no records at {path}", file=sys.stderr)
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("task") == task:
            records.append(record)
    return records


def published(records: list[dict]) -> dict[str, float]:
    """The board: one cell per backbone, newest wins, as `latest_per_backbone`."""
    cells: dict[str, float] = {}
    for record in records:
        cells[record["backbone"]] = record["metrics"][METRIC]
    return cells


def by_seed(records: list[dict]) -> dict[str, dict[int, float]]:
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    for record in records:
        scores[record["backbone"]][record["seed"]] = record["metrics"][METRIC]
    return scores


def check_seed_zero(board: dict[str, float], scores: dict[str, dict[int, float]]) -> bool:
    """The gate: a seed-0 row that disagrees means the sweep is not this board."""
    print("=== does the sweep measure this board? seed 0 against the corpus ===")
    ok = True
    for backbone in sorted(scores):
        if 0 not in scores[backbone] or backbone not in board:
            continue
        delta = scores[backbone][0] - board[backbone]
        flag = "" if delta == 0.0 else "   <-- DISAGREES"
        ok = ok and delta == 0.0
        print(f"  {backbone:20s} {scores[backbone][0]:8.4f}  delta {delta:+.2e}{flag}")
    print(f"  -> {'all reproduce exactly' if ok else 'NOT REPRODUCING, everything below is void'}")
    return ok


def spread(scores: dict[str, dict[int, float]]) -> list[tuple]:
    print("\n=== the noise, per row ===")
    print(f"  {'backbone':20s} {'mean':>8s} {'sd':>7s} {'range':>7s} {'min':>8s} {'max':>8s}  n")
    rows = []
    for backbone in sorted(scores, key=lambda b: statistics.mean(scores[b].values())):
        values = list(scores[backbone].values())
        if len(values) < 2:
            continue
        sd = statistics.stdev(values)
        rng = max(values) - min(values)
        rows.append((backbone, statistics.mean(values), sd, rng, len(values)))
        print(
            f"  {backbone:20s} {statistics.mean(values):8.4f} {sd:7.4f} {rng:7.4f}"
            f" {min(values):8.4f} {max(values):8.4f}  {len(values)}"
        )
    if rows:
        sds = [r[2] for r in rows]
        rngs = [r[3] for r in rows]
        print(
            f"\n  standard deviation: median {statistics.median(sds):.4f},"
            f" min {min(sds):.4f}, max {max(sds):.4f}"
        )
        print(
            f"  range:              median {statistics.median(rngs):.4f},"
            f" min {min(rngs):.4f}, max {max(rngs):.4f}"
        )
    return rows


#: Two-sided 95% critical value for Student's t at df = 4, i.e. five seeds.
#: Quoted rather than computed so the small sample is visible in the source: at
#: n=5 this test has little power, and a pair it cannot separate is often a pair
#: nobody measured enough times rather than a pair that is genuinely level.
T_CRITICAL_N5 = 2.776


def separability(board: dict[str, float], scores: dict[str, dict[int, float]]) -> None:
    """Pairwise and paired: does the better row actually stay better?

    Not a gap against a noise figure. The same seeds were run for both rows, so
    the **paired difference** is available, and it is the statistic that answers
    the question the tie list is asking. It also answers one the tie list cannot
    ask: a pair can fail to separate *and* lean the other way, which a gap
    threshold has no way to express.
    """
    print("\n=== which adjacent pairs are separable? ===")
    print("  (paired by seed: the same fits for both rows, lower is better)")
    print(f"  {'pair':46s} {'gap':>6s} {'mean d':>7s} {'sd d':>6s} {'t':>6s}  wins  verdict")
    order = sorted(board, key=lambda b: board[b])
    for better, worse in zip(order, order[1:], strict=False):
        if better not in scores or worse not in scores:
            continue
        shared = sorted(set(scores[better]) & set(scores[worse]))
        if len(shared) < 2:
            continue
        diffs = [scores[worse][s] - scores[better][s] for s in shared]
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        t = mean / (sd / len(diffs) ** 0.5) if sd else float("inf")
        wins = sum(d > 0 for d in diffs)

        if t <= -T_CRITICAL_N5:
            verdict = "REVERSED"
        elif t >= T_CRITICAL_N5:
            verdict = "ordered"
        else:
            verdict = "tied"
        print(
            f"  {better + ' vs ' + worse:46s} {board[worse] - board[better]:6.2f}"
            f" {mean:7.3f} {sd:6.3f} {t:6.2f}  {wins}/{len(diffs)}  {verdict}"
        )
    print(
        f"\n  'ordered'/'REVERSED' is |t| >= {T_CRITICAL_N5} (two-sided 95%, df=4);"
        " everything else is 'tied'."
    )
    print(
        "  REVERSED means the board's own order is the minority outcome: the pair"
        " is not\n  a coin flip, it leans the other way, which a gap threshold"
        " cannot say."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--sweep", type=Path, default=SWEEP)
    args = parser.parse_args(argv)

    board = published(load(args.corpus))
    records = load(args.sweep)
    if not board or not records:
        return 1
    scores = by_seed(records)

    reproduced = check_seed_zero(board, scores)
    spread(scores)
    separability(board, scores)
    if not reproduced:
        print(
            "\n!!! seed 0 does not reproduce the corpus -- do not quote the above", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
