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


T_CRITICAL_N5 = 2.776  # two-sided 95% at five seeds, as analyse_seeds.py uses


def sweep_pairs() -> dict[str, list[dict]]:
    """Every adjacent pair on every swept board: gap, paired t, verdict, seeds.

    The pair-level form of `tests/results/test_sweep_totals.py::_verdicts`, read
    against the same boards (`pinned_to_keys`), so the totals the video shows are
    the ones that test pins -- `main()` asserts they agree.
    """
    import statistics

    from visbench.results.leaderboard import (
        group_comparable,
        latest_per_backbone,
        pinned_to_keys,
    )
    from visbench.results.writer import iter_records

    corpus = list(iter_records(CORPUS))
    sweeps = {path.stem: path for path in SEEDS.glob("*.jsonl")}
    sweeps["relative_pose"] = ROOT / "results" / "controls" / "pose_seeds.jsonl"
    out: dict[str, list[dict]] = {}
    for task, path in sorted(sweeps.items()):
        metric = HEADLINE_METRICS[task]
        higher = metric_direction(metric) == "higher"
        sweep = list(iter_records(path))
        board = pinned_to_keys([r for r in corpus if r.task == task], sweep)
        (group,) = group_comparable(board).values()
        cells = {c.backbone: c.metrics[metric] for c in latest_per_backbone(group)}
        scores: dict[str, dict[int, float]] = {}
        for record in sweep:
            scores.setdefault(record.backbone, {})[record.seed] = record.metrics[metric]
        order = sorted(cells, key=lambda b: cells[b], reverse=higher)
        sign = 1.0 if higher else -1.0
        pairs = []
        for better, worse in zip(order, order[1:], strict=False):
            shared = sorted(set(scores[better]) & set(scores[worse]))
            diffs = [sign * (scores[better][k] - scores[worse][k]) for k in shared]
            sd = statistics.stdev(diffs)
            t = statistics.mean(diffs) / (sd / len(diffs) ** 0.5) if sd else float("inf")
            verdict = (
                "ordered" if t >= T_CRITICAL_N5 else "reversed" if t <= -T_CRITICAL_N5 else "tied"
            )
            pairs.append(
                {
                    "better": better,
                    "worse": worse,
                    "gap": abs(cells[better] - cells[worse]),
                    "t": t,
                    "verdict": verdict,
                    "seeds": {
                        better: [scores[better][k] for k in shared],
                        worse: [scores[worse][k] for k in shared],
                    },
                }
            )
        out[task] = pairs
    return out


def threshold_example(pairs: dict[str, list[dict]]) -> dict:
    """An ordered pair and a *visibly* tied pair, the tie with the wider gap.

    The scene's point is that the wider gap can be the coin flip, so the tie has
    to look like one on screen: its rows trade places on 2 or 3 of the 5 seeds,
    and the ordered pair wins all 5. Picks the board with the largest ratio of
    tied gap to ordered gap. Boards with a coarser reading rule are skipped --
    `detection`/`instance_segmentation` read to three decimals, `orientation`
    to two, `relative_pose` to whole degrees (CLAUDE.md) -- and so is the
    saturated `classification` board, whose gaps are in the fourth decimal.
    """
    skip = {"detection", "instance_segmentation", "orientation", "relative_pose", "classification"}
    best = None
    for task, rows in pairs.items():
        if task in skip:
            continue
        sign = 1.0 if metric_direction(HEADLINE_METRICS[task]) == "higher" else -1.0

        def wins(p: dict, sign: float = sign) -> int:
            a, b = p["seeds"][p["better"]], p["seeds"][p["worse"]]
            return sum(sign * (x - y) > 0 for x, y in zip(a, b, strict=True))

        ordered = [p for p in rows if p["verdict"] == "ordered" and wins(p) == 5]
        tied = [p for p in rows if p["verdict"] == "tied" and wins(p) in (2, 3)]
        if not ordered or not tied:
            continue
        small = min(ordered, key=lambda p: p["gap"])
        large = max(tied, key=lambda p: p["gap"])
        ratio = large["gap"] / small["gap"]
        if best is None or ratio > best["ratio"]:
            best = {
                "task": task,
                "ratio": ratio,
                "ordered": {**small, "wins": wins(small)},
                "tied": {**large, "wins": wins(large)},
            }
    assert best is not None
    best["metric"] = HEADLINE_METRICS[best["task"]]
    return best


def crop_fix() -> dict:
    """The timm double-crop fix (22c/23a): leaders before and after, and its size.

    "Before" is each board as it stood with the cropped timm cells, which the
    append-only corpus still holds under their pre-23a keys; "after" is the
    published board. The fourteenth column arrived after the fix, so it is left
    out of both: the comparison is the thirteen columns the fix re-measured.
    """
    from visbench.results.leaderboard import SUPERSEDED_KEYS
    from visbench.results.writer import iter_records

    records = [r for r in iter_records(CORPUS) if r.backbone != "clip_convnext_base"]
    fixed_keys = set(SUPERSEDED_KEYS.values())
    changes = []
    for task in sorted({r.task for r in records}):
        metric = HEADLINE_METRICS[task]
        higher = metric_direction(metric) == "higher"

        def leader(rows: list, metric: str = metric, higher: bool = higher) -> str:
            newest = {}
            for row in rows:  # file order is append order, so the last one wins
                newest[row.backbone] = row.metrics[metric]
            pick = max if higher else min
            return pick(newest, key=newest.get)

        rows = [r for r in records if r.task == task]
        before = leader([r for r in rows if r.backbone_key not in fixed_keys])
        after = leader(rows)
        if before != after:
            changes.append({"task": task, "before": before, "after": after})
    rerun = sum(r.backbone_key in fixed_keys for r in records)
    return {"changes": changes, "rerun_cells": rerun, "columns": len(SUPERSEDED_KEYS)}


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
    pairs = sweep_pairs()
    flat = [p for rows in pairs.values() for p in rows]
    data["sweep_totals"] = {
        "boards": len(pairs),
        "pairs": len(flat),
        **{v: sum(p["verdict"] == v for p in flat) for v in ("ordered", "tied", "reversed")},
    }
    # Boards where the largest unordered gap exceeds the smallest ordered one,
    # so no threshold on the gap could sort their pairs.
    thresholdless = 0
    for rows in pairs.values():
        ordered = [p["gap"] for p in rows if p["verdict"] == "ordered"]
        unordered = [p["gap"] for p in rows if p["verdict"] != "ordered"]
        thresholdless += bool(ordered and unordered and max(unordered) > min(ordered))
    data["sweep_totals"]["thresholdless"] = thresholdless
    expected = {
        "boards": 16,
        "pairs": 208,
        "ordered": 152,
        "tied": 56,
        "reversed": 0,
        "thresholdless": 12,
    }
    assert data["sweep_totals"] == expected, (data["sweep_totals"], "re-read the totals test")
    data["threshold_example"] = threshold_example(pairs)

    # The two high-level clusters (CORPUS_FINDINGS.md): within and across.
    from analyse_board_correlates import agreement

    rho = agreement(boards)
    image_level = ("classification", "retrieval")
    localised = (
        "detection",
        "semantic_segmentation",
        "scene_classification",
        "fine_grained_classification",
    )

    def r(a: str, b: str) -> float:
        return rho[tuple(sorted((a, b)))]

    data["clusters"] = {
        "image_level": list(image_level),
        "localised": list(localised),
        "within_image": r(*image_level),
        "within_localised_min": min(
            r(a, b) for i, a in enumerate(localised) for b in localised[i + 1 :]
        ),
        "across_max": max(r(a, b) for a in image_level for b in localised),
    }
    data["mean_rank"] = {
        name: sum(data["ranks"][t][name] for t in data["ranks"]) / len(data["ranks"])
        for name in data["backbones"]
    }
    data["crop_fix"] = crop_fix()
    # What the crop scene's narration says; fail here rather than in the voice.
    assert data["crop_fix"]["rerun_cells"] == 180, data["crop_fix"]["rerun_cells"]
    assert data["crop_fix"]["columns"] == 9
    assert len(data["crop_fix"]["changes"]) == 4
    assert {c["after"] for c in data["crop_fix"]["changes"]} == {"dino_vitb8"}
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(
        f"wrote {OUT.relative_to(ROOT)}: {len(data['backbones'])} backbones, {len(boards)} boards"
    )


if __name__ == "__main__":
    main()
