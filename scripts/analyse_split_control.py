#!/usr/bin/env python3
"""Read the split control: does `detection` rank like geometry on the instance
probe's images?

    python scripts/analyse_split_control.py

**The question.** 14a-4 found `instance_segmentation` ranking with the
mid-level geometry boards rather than its own high-level tier, and showed the
mask branch is not responsible: the `box_map_50` half of the same runs agrees
with the mask half at +0.986 and is topped by `occlusion_edge` too. So the box
half alone ranks with geometry while `detection` -- one implementation, one
matcher, one metric -- sits at +0.804 with `semantic_segmentation`. What differs
is the data.

**What is compared.** Two control configs of `detection` over VOC's
``ImageSets/Segmentation`` -- the instance probe's own 1464/1449 images:

    A  full      1464 train / 1449 val
    B  limit600  the corpus detection board's own --limit 600

against the two boards already in the corpus:

    C  detection             Main stems, --limit 600   (published)
    D  instance_segmentation Segmentation stems, full  (14a-4)

    A vs C  different images AND size, everything else equal -> is it the data?
    A vs B  same images, different size                      -> "how many"?
    A vs D  same images and size; differs only in box provenance (VOC's XML
            against boxes derived from the instance mask) and the mask branch
            riding alongside                                 -> the probe itself?

Those exhaust the difference between C and D, so whichever pair moves the
ranking is the answer -- and a null everywhere localises it to box provenance,
the one thing no pair varies alone.

**Never merged with the corpus.** `comparability_key` groups on dataset name and
fingerprint, so a `task=detection` record over a different split does not join
the published board, it makes that board unrenderable. These live in
`results/controls/` where nothing feeds a generated table.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
CONTROL = ROOT / "results" / "controls" / "detection_split.jsonl"

#: Headline metric per board, copied from `visbench.results.render`. A copy
#: rather than an import for `analyse_board_correlates.py`'s stated reason --
#: these scripts must read a fixed table even if the library's moves -- and a
#: test pins the two equal.
HEADLINE_METRICS: dict[str, str] = {
    "classification": "top1",
    "scene_classification": "top1",
    "fine_grained_classification": "top1",
    "retrieval": "mAP",
    "correspondence": "recall@5px",
    "similarity": "accuracy",
    "semantic_segmentation": "miou",
    "generic_segmentation": "iou",
    "detection": "map_50",
    "instance_segmentation": "mask_map_50",
    "depth": "d1",
    "surface_normal": "mean",
    "edge": "edge_correlation",
    "keypoints2d": "keypoint_correlation",
    "corner": "corner_correlation",
    "orientation": "orientation_error",
    "occlusion_edge": "occlusion_edge_correlation",
}

#: Boards whose headline metric is an error, so a lower number is better.
LOWER_IS_BETTER = {"surface_normal", "orientation"}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation with midranks, so a tie cannot fabricate an ordering."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in ra) ** 0.5
    var_b = sum((y - mean_b) ** 2 for y in rb) ** 0.5
    return cov / (var_a * var_b) if var_a and var_b else float("nan")


def agreement(left: dict[str, float], right: dict[str, float]) -> float:
    """Spearman over the backbones both sides hold, sign-corrected.

    An error metric is negated so that every reported rho means "ranks
    backbones the same way", never "the numbers move together" -- the mistake
    `METRIC_DIRECTIONS` exists to prevent one board over.
    """
    shared = sorted(set(left) & set(right))
    return spearman([left[b] for b in shared], [right[b] for b in shared])


def corpus_boards() -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Newest record per (task, backbone), as score-by-backbone per board."""
    latest: dict[tuple[str, str], dict] = {}
    for record in load(CORPUS):
        key = (record["task"], record["backbone"])
        current = latest.get(key)
        if current is None or record["timestamp"] >= current["timestamp"]:
            latest[key] = record

    boards: dict[str, dict[str, float]] = defaultdict(dict)
    levels: dict[str, str] = {}
    for (task, backbone), record in latest.items():
        value = record["metrics"][HEADLINE_METRICS[task]]
        # Negated here rather than at every comparison, so a caller cannot
        # forget and read an error board upside down.
        boards[task][backbone] = -value if task in LOWER_IS_BETTER else value
        levels[task] = record["level"]
    return dict(boards), levels


def control_configs() -> dict[str, dict[str, float]]:
    """The control's two configs, told apart by their training-set size.

    `dataset_params` carries no "config" field -- the runs differ only in
    `--limit`, which reaches the record as `dataset_size`, the scored split's
    own length. So the configs are recovered from what the run actually read
    rather than from a label this script would have to trust.

    `dataset` is a *string* (the loader's name), not a dict -- checked against a
    real record rather than assumed, which is what caught the first draft
    reading `record["dataset"]["num_samples"]`.
    """
    configs: dict[str, dict[str, float]] = defaultdict(dict)
    for record in load(CONTROL):
        name = "limit600" if record["dataset_size"] <= 600 else "full"
        configs[name][record["backbone"]] = record["metrics"]["map_50"]
    return dict(configs)


def tier_means(
    board: dict[str, float],
    boards: dict[str, dict[str, float]],
    levels: dict[str, str],
    skip: set[str],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for task, rows in boards.items():
        if task in skip:
            continue
        grouped[levels[task]].append(agreement(board, rows))
    return {tier: sum(v) / len(v) for tier, v in sorted(grouped.items())}


def report_one(
    label: str,
    board: dict[str, float],
    boards: dict[str, dict[str, float]],
    levels: dict[str, str],
    skip: set[str],
) -> None:
    print(f"\n--- {label}  (n={len(board)}) ---")
    rows = sorted(
        ((agreement(board, r), t) for t, r in boards.items() if t not in skip),
        reverse=True,
    )
    print("  strongest partners:")
    for rho, task in rows[:4]:
        print(f"    {rho:+.3f}  {task:26s} {levels[task]}")
    print("  weakest:")
    for rho, task in rows[-2:]:
        print(f"    {rho:+.3f}  {task:26s} {levels[task]}")
    means = tier_means(board, boards, levels, skip)
    for tier, mean in means.items():
        print(f"  mean vs {tier:11s} {mean:+.3f}")


def main() -> int:
    boards, levels = corpus_boards()
    configs = control_configs()
    if not configs:
        print(f"No control records at {CONTROL} -- run scripts/build_split_control.sh")
        return 1

    print("=" * 78)
    print("THE SPLIT CONTROL -- `detection` on the instance probe's images")
    print("=" * 78)
    for name, rows in sorted(configs.items()):
        print(f"  {name:9s} {len(rows):2d} backbones")

    # Each control config is `detection`, so both published detection-family
    # boards are excluded from its partner list: comparing a board with itself
    # or with its own re-run says nothing about clustering.
    skip = {"detection", "instance_segmentation"}
    for name, rows in sorted(configs.items()):
        report_one(f"detection / segmentation split / {name}", rows, boards, levels, skip)

    print(f"\n{'=' * 78}\nTHE THREE COMPARISONS\n{'=' * 78}")
    published = boards.get("detection", {})
    instance = boards.get("instance_segmentation", {})
    full = configs.get("full", {})
    limited = configs.get("limit600", {})

    def show(name: str, left: dict[str, float], right: dict[str, float], what: str) -> None:
        if not left or not right:
            print(f"  {name}: missing a side")
            return
        print(f"  {name:34s} rho {agreement(left, right):+.3f}   {what}")

    show("A(full) vs C(published board)", full, published, "different images AND size")
    show("A(full) vs B(limit600)", full, limited, "same images, different size")
    show("A(full) vs D(instance board)", full, instance, "same images and size")
    show("B(limit600) vs C(published)", limited, published, "same size, different images")
    show("B(limit600) vs D(instance)", limited, instance, "different size, same images")

    print("\n  For reference, from the corpus:")
    for pair in (
        ("detection", "semantic_segmentation"),
        ("detection", "occlusion_edge"),
        ("instance_segmentation", "semantic_segmentation"),
        ("instance_segmentation", "occlusion_edge"),
    ):
        a, b = pair
        if a in boards and b in boards:
            print(f"    {a} / {b}: {agreement(boards[a], boards[b]):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
