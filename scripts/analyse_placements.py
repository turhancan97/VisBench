#!/usr/bin/env python
"""Is a board's first place a fact, or a coin flip? (22a)

    python scripts/analyse_placements.py
    python scripts/analyse_placements.py --verbose

`scripts/analyse_seeds.py` answers this one board at a time, for every adjacent
pair. This asks the question the *prose* actually makes: `CORPUS_FINDINGS.md`
and the probe pages say things like "MAE leads edge" and "first on five boards
and last on four", and a count of first places treats every first place as the
same event. It is not. On `edge` the leader is ahead of second place by
**t = +0.11** across five shared seeds, which is as close to a coin flip as this
corpus gets; on `surface_normal` the same backbone leads at **t = +75**.

So this walks every board in the corpus and reports whether its **first** and
**last** placements survive a paired re-fit -- the same statistic
`analyse_seeds.py` uses, restricted to the two placements the prose quotes.

**Three kinds of board, and conflating them is the mistake this guards
against.**

*Swept and trained.* The verdict comes from `results/controls/seeds/<probe>.jsonl`:
five seeds, the same seeds for both rows, tested as a paired difference.

*Zero-shot* (`retrieval`, `correspondence`, `similarity`). These fit no head, so
a re-run is bit-identical and the ordering is **exact** -- there is no
separability question to ask. Reporting them as "not measured" would understate
them and folding them in as "separable" would overstate the evidence, so they
are named.

*Held out* (`scene_classification`). Its sweep does not reproduce its board
(20c) because that board **amplifies** a disturbance too small to identify
(20d), so it lives outside `results/controls/seeds/`. 20d also showed the
board's separability verdict *survives* the disagreement -- the two
configurations differ on 2 of 78 pairs, both already tied -- so the sweep is
read here, and flagged, rather than dropped. Dropping it would silently remove a
board from a count that the prose states over all of them.

**The asymmetry this found, and the reason to look at last place at all.**
First places are much safer than last ones: 17 of 20 against 13 of 20 at the
time of writing. Scores compress at the bottom of most of these boards -- a
weak backbone is near the floor and so are the next two -- while a leader more
often has room. So "last on N boards" is the weaker half of a claim that reads
as though both halves were equally solid, and one published last place is
outright **reversed** (`relative_pose`, t = -3.06, which is one of the three
reversals 20a already documented -- a check on this script rather than a new
finding).

**Never read a `t` here as a score.** It says how repeatable a placement is, not
how good a backbone is, and it is not comparable across boards whose metrics
have different scales of noise.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
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
POSE_SWEEP = ROOT / "results" / "controls" / "pose_seeds.jsonl"

#: The board whose sweep is deliberately not under `seeds/`, and why it is still
#: read here. See the module docstring.
HELD_OUT = {
    "scene_classification": ROOT / "results" / "controls" / "scene_classification_seeds.jsonl"
}

#: Probes that fit no head. A re-run is bit-identical, so their orderings carry
#: no seed noise at all -- listed rather than detected, because "this board has
#: no sweep" and "this board needs no sweep" are different statements and only
#: one of them is a gap in the evidence.
ZERO_SHOT = ("retrieval", "correspondence", "similarity")

#: Two-sided 95% critical value for Student's t at df = 4, i.e. five seeds.
#: The same constant `analyse_seeds.py` uses; a placement is ORDERED at +t,
#: REVERSED at -t, and TIED between.
T_CRITICAL = 2.776


def sweeps() -> dict[str, Path]:
    """Every committed sweep, including the pose one and the held-out board."""
    found = {path.stem: path for path in SWEEP_DIR.glob("*.jsonl")} if SWEEP_DIR.is_dir() else {}
    if POSE_SWEEP.is_file():
        found["relative_pose"] = POSE_SWEEP
    found.update({task: path for task, path in HELD_OUT.items() if path.is_file()})
    return found


def scores_by_seed(path: Path, task: str, metric: str) -> dict[str, dict[int, float]]:
    """`{backbone: {seed: score}}` for one swept board."""
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for record in iter_records(path):
        if record.task == task:
            out[record.backbone][record.seed] = record.metrics[metric]
    return dict(out)


def paired_t(
    scores: dict[str, dict[int, float]], better: str, worse: str, *, higher_is_better: bool
) -> float | None:
    """The paired-difference t of `better` over `worse`, across their shared seeds.

    Signed so that a positive value always means the board's own order held,
    whichever direction the metric runs -- the sign convention is the one thing
    here that silently inverts every verdict if it is dropped.
    """
    shared = sorted(set(scores.get(better, {})) & set(scores.get(worse, {})))
    if len(shared) < 2:
        return None
    sign = 1.0 if higher_is_better else -1.0
    diffs = [sign * (scores[better][seed] - scores[worse][seed]) for seed in shared]
    deviation = statistics.stdev(diffs)
    if deviation == 0.0:
        return float("inf")
    return statistics.mean(diffs) / (deviation / len(diffs) ** 0.5)


def verdict(value: float | None) -> str:
    if value is None:
        return "unmeasured"
    if value >= T_CRITICAL:
        return "ordered"
    if value <= -T_CRITICAL:
        return "REVERSED"
    return "tied"


def board_cells(corpus: list[ResultRecord], task: str, metric: str) -> dict[str, float]:
    """The published cell per backbone, grouped as `LEADERBOARD.md` groups them."""
    (group,) = group_comparable([r for r in corpus if r.task == task]).values()
    return {cell.backbone: cell.metrics[metric] for cell in latest_per_backbone(group)}


def audit() -> list[dict]:
    """One row per board: who is first and last, and whether either is repeatable."""
    corpus = list(iter_records(CORPUS))
    available = sweeps()
    rows = []
    for task in sorted({record.task for record in corpus}):
        metric = HEADLINE_METRICS[task]
        higher = metric_direction(metric) == "higher"
        cells = board_cells(corpus, task, metric)
        order = sorted(cells, key=lambda b: cells[b], reverse=higher)
        row = {
            "task": task,
            "metric": metric,
            "first": order[0],
            "last": order[-1],
            "runner_up": order[1],
            "second_last": order[-2],
            "spread": abs(cells[order[0]] - cells[order[-1]]),
            "kind": "zero-shot"
            if task in ZERO_SHOT
            else ("held-out" if task in HELD_OUT else "swept"),
        }
        if task in ZERO_SHOT:
            # No head is fitted, so a re-run is bit-identical: exact, not merely
            # separable. `t` is meaningless here rather than large.
            row |= {
                "first_t": None,
                "last_t": None,
                "first": order[0],
                "first_verdict": "exact",
                "last_verdict": "exact",
            }
        else:
            scores = scores_by_seed(available[task], task, metric)
            first_t = paired_t(scores, order[0], order[1], higher_is_better=higher)
            last_t = paired_t(scores, order[-2], order[-1], higher_is_better=higher)
            row |= {
                "first_t": first_t,
                "last_t": last_t,
                "first_verdict": verdict(first_t),
                "last_verdict": verdict(last_t),
            }
        rows.append(row)
    return rows


def _fmt(value: float | None) -> str:
    return "     --" if value is None else f"{value:+7.2f}"


def report(rows: list[dict], *, verbose: bool) -> None:
    held = {row["task"] for row in rows if row["kind"] == "held-out"}
    print("=== per board: is the placement repeatable? ===")
    print(f"  {'board':30s} {'first':20s} {'t':>7s} {'verdict':9s}  {'last':20s} {'t':>7s} verdict")
    for row in sorted(rows, key=lambda r: r["task"]):
        mark = " *" if row["task"] in held else "  "
        print(
            f"{mark}{row['task']:30s} {row['first']:20s} {_fmt(row['first_t'])} "
            f"{row['first_verdict']:9s}  {row['last']:20s} {_fmt(row['last_t'])} "
            f"{row['last_verdict']}"
        )
    if held:
        print(
            f"\n  * read from a HELD-OUT sweep ({', '.join(sorted(held))}); "
            "see the module docstring"
        )

    for placement in ("first", "last"):
        total = len(rows)
        solid = sum(1 for r in rows if r[f"{placement}_verdict"] in ("ordered", "exact"))
        print(f"\n  {placement} places that are repeatable: {solid} of {total}")
        for row in sorted(rows, key=lambda r: r["task"]):
            if row[f"{placement}_verdict"] not in ("ordered", "exact"):
                other = row["runner_up"] if placement == "first" else row["second_last"]
                print(
                    f"    {row[f'{placement}_verdict']:8s} {row['task']:30s} "
                    f"{row[placement]} vs {other}  t={_fmt(row[f'{placement}_t'])}"
                )

    print("\n=== per backbone: placements, and how many survive a re-fit ===")
    print("  A count of first places is only as good as the separability of each one.")
    firsts: Counter[str] = Counter()
    firsts_solid: Counter[str] = Counter()
    lasts: Counter[str] = Counter()
    lasts_solid: Counter[str] = Counter()
    for row in rows:
        firsts[row["first"]] += 1
        lasts[row["last"]] += 1
        if row["first_verdict"] in ("ordered", "exact"):
            firsts_solid[row["first"]] += 1
        if row["last_verdict"] in ("ordered", "exact"):
            lasts_solid[row["last"]] += 1
    print(
        f"\n  {'backbone':22s} {'first':>6s} {'of which solid':>15s} "
        f"{'last':>6s} {'of which solid':>15s}"
    )
    for backbone in sorted(set(firsts) | set(lasts), key=lambda b: (-firsts[b], -lasts[b], b)):
        print(
            f"  {backbone:22s} {firsts[backbone]:6d} {firsts_solid[backbone]:15d} "
            f"{lasts[backbone]:6d} {lasts_solid[backbone]:15d}"
        )

    if verbose:
        print("\n=== board spreads, for context ===")
        for row in sorted(rows, key=lambda r: -r["spread"]):
            print(f"  {row['task']:30s} {row['metric']:18s} spread {row['spread']:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true", help="also print board spreads")
    args = parser.parse_args(argv)

    if not CORPUS.is_file():
        print(f"no corpus at {CORPUS}", file=sys.stderr)
        return 1
    report(audit(), verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
