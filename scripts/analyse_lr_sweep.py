#!/usr/bin/env python
"""Read the 25b-1 learning-rate sweep: one rate per board, and whether it clears the bar.

    python scripts/analyse_lr_sweep.py
    python scripts/analyse_lr_sweep.py --control results/controls/parts/lr_sweep

`scripts/run_lr_sweep.py` fits each of the ten dense boards on 90% of its
training split and scores the held-out 10%, for four backbones, at six rates
with ``--standardize`` and once raw at the published rate. This script applies
the maintainer's rule:

- **The chosen rate** for a board is the standardised rate with the best
  *mean* held-out headline metric over the four backbones (direction-aware;
  a tie goes to the smaller rate). One rate per board, never per backbone.
- **The bar** is the raw arm on the same slice. A board whose chosen rate still
  scores clearly worse than raw -- its mean below the raw mean by more than
  :data:`BAR_TOLERANCE` of the raw mean's magnitude -- is flagged ``STOP``: it
  is reported to the maintainer rather than re-run standardised.

Every number here is on a held-out slice of a *training* split, so none is a
board value and none may be quoted as one.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from visbench.results.leaderboard import metric_direction
from visbench.results.render import HEADLINE_METRICS
from visbench.results.schema import ResultRecord
from visbench.results.writer import iter_records

ROOT = Path(__file__).resolve().parent.parent
CONTROL = ROOT / "results" / "controls" / "lr_sweep.jsonl"

#: The rates this sweep chose (25b-1, 2026-11-27), and the board the stop rule
#: held back. 25b-2 re-runs the nine boards at these rates; a test recomputes
#: them from the committed sweep so the two can never disagree.
CHOSEN_RATES = {
    "corner": 1e-2,
    "depth": 2e-2,
    "edge": 5e-3,
    "generic_segmentation": 1e-2,
    "keypoints2d": 1e-2,
    "occlusion_edge": 2e-3,
    "orientation": 5e-3,
    "scene_parsing": 2e-3,
    "semantic_segmentation": 5e-3,
}
STOPPED = ("surface_normal",)

#: How far, relative to the raw mean, the chosen rate may fall short of raw
#: before a board is flagged. 1% of the score's magnitude.
BAR_TOLERANCE = 0.01

_ARM = re.compile(r"arm=(\S+)")


def arm_of(record: ResultRecord) -> str:
    """The arm label `run_lr_sweep.py` wrote into the record's notes."""
    match = _ARM.search(record.notes or "")
    if match is None:
        raise ValueError(f"record for {record.task}/{record.backbone} carries no arm label")
    return match.group(1)


def rate_of(arm: str) -> float | None:
    """``"std_lr0.002"`` -> 0.002; ``"raw"`` -> None."""
    return float(arm.removeprefix("std_lr")) if arm.startswith("std_lr") else None


def collect(records: list[ResultRecord]) -> dict[str, dict[str, dict[str, float]]]:
    """``{task: {arm: {backbone: headline}}}``."""
    table: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        table[record.task][arm_of(record)][record.backbone] = record.metrics[
            HEADLINE_METRICS[record.task]
        ]
    return table


def choose(by_arm: dict[str, dict[str, float]], higher: bool) -> dict:
    """The chosen standardised rate for one board, and its verdict against raw.

    Only arms that every backbone completed are eligible, so a missing run
    cannot win by averaging over fewer columns.
    """
    backbones = set(by_arm.get("raw", {}))
    for scores in by_arm.values():
        backbones |= set(scores)
    complete = {
        arm: statistics.fmean(scores.values())
        for arm, scores in by_arm.items()
        if set(scores) == backbones
    }
    rates = {arm: rate_of(arm) for arm in complete if rate_of(arm) is not None}
    if not rates or "raw" not in complete:
        return {"chosen": None, "verdict": "INCOMPLETE", "means": complete}

    def key(arm: str) -> tuple[float, float]:
        score = complete[arm]
        return (-score if higher else score, rates[arm])

    best = min(rates, key=key)
    raw = complete["raw"]
    shortfall = (raw - complete[best]) if higher else (complete[best] - raw)
    verdict = "STOP" if shortfall > BAR_TOLERANCE * abs(raw) else "OK"
    return {
        "chosen": rates[best],
        "chosen_mean": complete[best],
        "raw_mean": raw,
        "shortfall": shortfall,
        "verdict": verdict,
        "means": complete,
    }


def _records(path: Path) -> list[ResultRecord]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    return [record for file in files for record in iter_records(file)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control", type=Path, default=CONTROL)
    args = parser.parse_args(argv)
    if not args.control.exists():
        print(f"no sweep at {args.control}", file=sys.stderr)
        return 1

    table = collect(_records(args.control))
    for task in sorted(table):
        metric = HEADLINE_METRICS[task]
        higher = metric_direction(metric) == "higher"
        result = choose(table[task], higher)
        print(f"\n{task}  ({metric}, {'higher' if higher else 'lower'} is better)")
        for arm, mean in sorted(
            result["means"].items(), key=lambda kv: (rate_of(kv[0]) or 0.0, kv[0])
        ):
            mark = " <- chosen" if rate_of(arm) == result["chosen"] else ""
            print(f"  {arm:14s} mean {mean:10.4f}{mark}")
        if result["chosen"] is None:
            print("  verdict: INCOMPLETE")
        else:
            print(
                f"  verdict: {result['verdict']}  (chosen lr {result['chosen']:g}, "
                f"shortfall vs raw {result['shortfall']:+.4f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
