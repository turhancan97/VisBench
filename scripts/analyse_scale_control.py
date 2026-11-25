#!/usr/bin/env python
"""Read the feature-scale control (24b): does a dense board rank the representation or its scale?

    python scripts/analyse_scale_control.py
    python scripts/analyse_scale_control.py --control results/controls/parts/scale

`scripts/run_scale_control.py` refits seven linear dense boards with
`clip_convnext_base`'s features multiplied up to `convnext_base`'s scale and
`convnext_base`'s multiplied down to `clip_convnext_base`'s. A constant factor
removes nothing a linear head could use, so a score that moves under it moved
because of how the fit reached its answer, not because of what the features
carry.

**Three numbers per arm, and which pair answers what.**

    published  the corpus cell
    gate (x1)  the same alias path at scale 1 -- a fresh extraction under its
               own cache key, so it must reproduce `published`, or the arm
               below describes some adjacent configuration
    scaled     the arm itself

`scaled - gate` is the effect, read against the same path so nothing but the
scale differs. It is quoted in units of that column's **seed standard
deviation** from `results/controls/seeds/`, because a shift is only a finding
if it is larger than what re-running the same fit with another seed does. The
last column places the scaled cell on the published board with every other
cell left as it is.

**Never a score.** A scaled backbone is a model nobody would choose; nothing
here may reach the corpus or a generated table.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from visbench.results.leaderboard import group_comparable, latest_per_backbone, metric_direction
from visbench.results.render import HEADLINE_METRICS
from visbench.results.schema import ResultRecord
from visbench.results.writer import iter_records

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
SWEEP_DIR = ROOT / "results" / "controls" / "seeds"
CONTROL = ROOT / "results" / "controls" / "scale_control.jsonl"

#: The two columns, in the order the table prints them.
ARMS = ("clip_convnext_base", "convnext_base")


def _records(path: Path) -> list[ResultRecord]:
    """Every record in a file, or in every ``*.jsonl`` of a directory."""
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    return [record for file in files for record in iter_records(file)]


def split_arm(name: str) -> tuple[str, float]:
    """``"convnext_base_x0.1155"`` -> ``("convnext_base", 0.1155)``."""
    base, _, scale = name.rpartition("_x")
    if base not in ARMS:
        raise ValueError(f"{name!r} is not a scale-control alias of {ARMS}")
    return base, float(scale)


def board(corpus: list[ResultRecord], task: str) -> dict[str, float]:
    """The published cell per backbone, grouped as `LEADERBOARD.md` groups them."""
    metric = HEADLINE_METRICS[task]
    (group,) = group_comparable([r for r in corpus if r.task == task]).values()
    return {cell.backbone: cell.metrics[metric] for cell in latest_per_backbone(group)}


def seed_sd(task: str, backbone: str) -> float | None:
    """That column's standard deviation over its swept seeds, if a sweep exists."""
    path = SWEEP_DIR / f"{task}.jsonl"
    if not path.is_file():
        return None
    metric = HEADLINE_METRICS[task]
    values = [r.metrics[metric] for r in iter_records(path) if r.backbone == backbone]
    return statistics.stdev(values) if len(values) >= 2 else None


def place(cells: dict[str, float], backbone: str, value: float, *, higher: bool) -> int:
    """1-based rank of `value` for `backbone` with every other cell unchanged."""
    others = [v for name, v in cells.items() if name != backbone]
    return 1 + sum((v > value) if higher else (v < value) for v in others)


def analyse(corpus: list[ResultRecord], control: list[ResultRecord]) -> list[dict]:
    """One row per (board, arm) present in the control."""
    found: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    for record in control:
        base, scale = split_arm(record.backbone)
        found[(record.task, base)][scale] = record.metrics[HEADLINE_METRICS[record.task]]

    rows = []
    for (task, base), by_scale in sorted(
        found.items(), key=lambda kv: (kv[0][0], ARMS.index(kv[0][1]))
    ):
        cells = board(corpus, task)
        higher = metric_direction(HEADLINE_METRICS[task]) == "higher"
        gate = by_scale.get(1.0)
        scaled = next((v for k, v in by_scale.items() if k != 1.0), None)
        sd = seed_sd(task, base)
        shift = None if gate is None or scaled is None else scaled - gate
        rows.append(
            {
                "task": task,
                "backbone": base,
                "metric": HEADLINE_METRICS[task],
                "higher": higher,
                "published": cells[base],
                "gate": gate,
                "scaled": scaled,
                "shift": shift,
                "seed_sd": sd,
                "shift_in_sd": None if shift is None or not sd else shift / sd,
                "rank": place(cells, base, cells[base], higher=higher),
                "rank_scaled": None
                if scaled is None
                else place(cells, base, scaled, higher=higher),
                "board_size": len(cells),
            }
        )
    return rows


def _fmt(value: float | None, spec: str = ".4f", width: int = 9) -> str:
    """Right-aligned to `width`, with ``--`` for a value the control lacks."""
    return ("--" if value is None else format(value, spec)).rjust(width)


def report(rows: list[dict]) -> None:
    print(
        f"  {'board':22s} {'column':19s} {'published':>10s} {'gate x1':>9s} {'d(gate)':>9s} "
        f"{'scaled':>9s} {'shift':>9s} {'/seed sd':>9s} {'rank':>11s}"
    )
    for row in rows:
        gate_delta = None if row["gate"] is None else row["gate"] - row["published"]
        rank = f"{row['rank']} -> {_fmt(row['rank_scaled'], 'd', 0)}/{row['board_size']}"
        print(
            f"  {row['task']:22s} {row['backbone']:19s} {row['published']:10.4f} "
            f"{_fmt(row['gate'])} {_fmt(gate_delta, '+.1e')} "
            f"{_fmt(row['scaled'])} {_fmt(row['shift'], '+.4f')} "
            f"{_fmt(row['shift_in_sd'], '+.1f')} {rank:>11s}"
        )
    print(
        "\n  shift = scaled - gate, signed in the metric's own units (check its direction);"
        "\n  /seed sd = that shift over the column's seed spread on the swept board."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control", type=Path, default=CONTROL)
    args = parser.parse_args(argv)
    if not args.control.exists():
        print(f"no control at {args.control}", file=sys.stderr)
        return 1
    report(analyse(_records(CORPUS), _records(args.control)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
