#!/usr/bin/env python
"""Pull every number the video shows out of the committed corpus and controls.

    python video/extract_data.py          # writes video/data.json

The scenes read `data.json` and nothing else, so a number on screen traces to
`results/corpus/visbench.jsonl` or `results/controls/` through this one file --
the same rule the docs follow: prose drifts, generated values do not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_board_correlates import load_boards, load_levels  # noqa: E402

from visbench.results.leaderboard import metric_direction  # noqa: E402
from visbench.results.render import HEADLINE_METRICS  # noqa: E402

CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
SCALE = ROOT / "results" / "controls" / "scale_control.jsonl"
FEATURE_SCALE = ROOT / "results" / "controls" / "feature_scale.json"
SEEDS = ROOT / "results" / "controls" / "seeds"
OUT = Path(__file__).resolve().parent / "data.json"


def published(task: str) -> dict[str, float]:
    """The board in the metric's own units (load_boards negates lower-is-better)."""
    board = load_boards(CORPUS)[task]
    sign = 1.0 if metric_direction(HEADLINE_METRICS[task]) == "higher" else -1.0
    return {name: sign * value for name, value in board.items()}


def ranks(task: str) -> dict[str, int]:
    board = load_boards(CORPUS)[task]  # higher is better on every board here
    order = sorted(board, key=board.get, reverse=True)
    return {name: order.index(name) + 1 for name in board}


def seed_rows(task: str, names: tuple[str, ...]) -> dict[str, list[float]]:
    metric = HEADLINE_METRICS[task]
    rows: dict[str, list[tuple[int, float]]] = {n: [] for n in names}
    for line in (SEEDS / f"{task}.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["backbone"] in rows:
            rows[record["backbone"]].append((record["seed"], record["metrics"][metric]))
    return {n: [v for _, v in sorted(r)] for n, r in rows.items()}


def main() -> None:
    boards = load_boards(CORPUS)
    levels = load_levels(CORPUS)

    scale = {}
    for line in SCALE.read_text().splitlines():
        record = json.loads(line)
        base, _, factor = record["backbone"].rpartition("_x")
        scale.setdefault(record["task"], {})[f"{base}@{factor}"] = record["metrics"][
            HEADLINE_METRICS[record["task"]]
        ]

    data = {
        "backbones": sorted(boards["classification"]),
        "levels": {task: levels[task] for task in boards},
        "ranks": {task: ranks(task) for task in boards},
        "boards": {
            task: published(task)
            for task in ("generic_segmentation", "semantic_segmentation", "depth")
        },
        "metric": {task: HEADLINE_METRICS[task] for task in boards},
        "scale_control": scale,
        "feature_std": {
            row["backbone"]: row["std"] for row in json.loads(FEATURE_SCALE.read_text())
        },
        # a tied pair and an ordered pair, for the seed-sweep scene
        "seeds_corner": seed_rows(
            "corner", ("dino_vitb8", "dinov2_vitb14", "dino_vitb16", "mae_vitb16")
        ),
    }
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(
        f"wrote {OUT.relative_to(ROOT)}: {len(data['backbones'])} backbones, {len(boards)} boards"
    )


if __name__ == "__main__":
    main()
