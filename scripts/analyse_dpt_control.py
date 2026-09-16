#!/usr/bin/env python
"""Read the DPT-head controls: how far a decoder gets, and whether it reorders.

    scripts/analyse_dpt_control.py                # both groups
    scripts/analyse_dpt_control.py --group cnn

`DenseTrainingTask.evaluate_oracle` models a **linear** head exactly --
``LinearHead`` is a 1x1 convolution per patch plus a bilinear upsample, which is
literally what the oracle computes. Whether it also bounds a progressive decoder
was an open question until v0.15.0, which answered it on two backbones: no. This
script reads the widened controls, which ask the two questions two backbones
could not.

**Why the two groups are read separately and must not be merged.** A ViT's
twelve blocks all share one grid, so ``--layers 2 5 8 11`` hands DPT four maps
at the *same* resolution and any gain is decoding rather than finer input --
exceeding the oracle there means structure was placed *within* a patch. A CNN's
stages are at 56/28/14/7, so DPT reads genuinely finer input than the grid the
oracle pools to, and exceeding it is the expected outcome rather than a
surprising one. ``layers`` is in ``comparability_key`` anyway, so they are
separate groups on disk.

Nothing here is a metric, nothing is ranked and nothing is written back. It
reads the two control files plus the corpus, and prints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
CONTROLS = {
    "vit": ROOT / "results" / "controls" / "dpt_head.jsonl",
    "cnn": ROOT / "results" / "controls" / "dpt_head_cnn.jsonl",
}

#: Mirrors ``render.HEADLINE_METRICS`` for the five probes with an oracle, and
#: says which way each one reads. `orientation_error` is **degrees of error**:
#: a naive score/ceiling ratio there reads 145% for the *worse* of two results,
#: which is why the direction is carried rather than assumed.
HEADLINE = {
    "edge": ("edge_correlation", "higher"),
    "keypoints2d": ("keypoint_correlation", "higher"),
    "occlusion_edge": ("occlusion_edge_correlation", "higher"),
    "corner": ("corner_correlation", "higher"),
    "orientation": ("orientation_error", "lower"),
}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Newest record per (task, backbone).

    The control files are append-only for the corpus's reason: a re-run is a new
    line beside the old one, never a replacement, so the file records what ran.
    """
    out: dict[tuple[str, str], dict] = {}
    for record in records:
        key = (record["task"], record["backbone"])
        current = out.get(key)
        if current is None or record["timestamp"] >= current["timestamp"]:
            out[key] = record
    return out


def fraction(score: float, ceiling: float, direction: str) -> float:
    """What fraction of the oracle a head reached, as a number where 1.0 means
    'level with it' whichever way the metric reads."""
    return ceiling / score if direction == "lower" else score / ceiling


def spearman(a: list[float], b: list[float]) -> float:
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


def order(rows: dict[str, float], direction: str) -> list[str]:
    return sorted(rows, key=lambda name: rows[name], reverse=direction == "higher")


def report(group: str, linear: dict, dpt: dict) -> None:
    print(f"\n{'=' * 82}\n{group.upper()} group -- {CONTROLS[group].name}\n{'=' * 82}")

    backbones = sorted({b for (_, b) in dpt})
    if not backbones:
        print("no records")
        return
    print(f"{len(backbones)} backbones: {', '.join(backbones)}")

    fractions: list[float] = []
    exceeded: list[tuple[str, str]] = []
    reversals: list[str] = []
    shifted: list[tuple[str, str, float, float]] = []

    for task, (metric, direction) in HEADLINE.items():
        rows = {b: r for (t, b), r in dpt.items() if t == task}
        if not rows:
            continue
        print(f"\n-- {task}  ({metric}, {direction} is better)")
        print(
            f"   {'backbone':18s} {'linear':>9s} {'DPT':>9s} {'DPT orc':>10s} "
            f"{'lin/own':>8s} {'DPT/own':>8s} {'gain':>7s}"
        )

        lin_scores: dict[str, float] = {}
        dpt_scores: dict[str, float] = {}
        for backbone in sorted(rows):
            record = rows[backbone]
            score = record["metrics"][metric]
            ceiling = record["metrics"][f"ceiling_{metric}"]
            base = linear.get((task, backbone))
            if base is None:
                print(f"   {backbone:18s} (no linear record)")
                continue
            lin = base["metrics"][metric]

            # EACH RECORD'S OWN CEILING, never one shared between them.
            #
            # For a ViT the two are bit-identical: all twelve blocks share one
            # grid, so `_grid_of` returns the same number whether one layer was
            # requested or four. For a CNN they are NOT. `_grid_of` takes the
            # *finest* requested map -- documented, and right, since a DPT head
            # is bounded by its finest input -- so a linear run reading the last
            # stage gets a 7x7 oracle while a DPT run reading stages 1-4 gets a
            # 56x56 one. Dividing both scores by the DPT run's ceiling was this
            # script's first draft, and it understated the linear head by
            # measuring it against a bottleneck that head never had.
            lin_ceiling = base["metrics"].get(f"ceiling_{metric}")
            lin_frac = fraction(lin, lin_ceiling, direction) if lin_ceiling else float("nan")
            dpt_frac = fraction(score, ceiling, direction)
            fractions.append(dpt_frac)
            if dpt_frac > 1.0:
                exceeded.append((task, backbone))
            lin_scores[backbone] = lin
            dpt_scores[backbone] = score

            moved = "*" if not lin_ceiling or abs(lin_ceiling - ceiling) > 1e-9 else " "
            if moved == "*":
                shifted.append((task, backbone, lin_ceiling or float("nan"), ceiling))
            print(
                f"   {backbone:18s} {lin:9.4f} {score:9.4f} {ceiling:9.4f}{moved} "
                f"{lin_frac * 100:7.1f}% {dpt_frac * 100:7.1f}% "
                f"{fraction(score, lin, direction):6.2f}x"
            )

        shared = sorted(set(lin_scores) & set(dpt_scores))
        if len(shared) >= 3:
            rho = spearman([lin_scores[b] for b in shared], [dpt_scores[b] for b in shared])
            lin_order = order({b: lin_scores[b] for b in shared}, direction)
            dpt_order = order({b: dpt_scores[b] for b in shared}, direction)
            same_top = lin_order[0] == dpt_order[0]
            print(
                f"   rho(linear, DPT) = {rho:+.3f} over {len(shared)} backbones; "
                f"leader {'unchanged' if same_top else f'{lin_order[0]} -> {dpt_order[0]}'}"
            )
            if not same_top:
                reversals.append(task)
            # Where the two orders disagree, which a single rho hides.
            swaps = sum(
                1
                for i in range(len(shared))
                for j in range(i + 1, len(shared))
                if lin_order.index(dpt_order[i]) > lin_order.index(dpt_order[j])
            )
            print(f"   discordant pairs: {swaps}/{len(shared) * (len(shared) - 1) // 2}")

    if not fractions:
        return
    ordered = sorted(fractions)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    print(f"\n-- {group} summary over {len(fractions)} cells")
    print(
        f"   DPT as a fraction of its OWN oracle: min {min(ordered) * 100:.1f}%  "
        f"median {median * 100:.1f}%  max {max(ordered) * 100:.1f}%"
    )
    print(
        f"   exceeded its own oracle: {len(exceeded)}/{len(fractions)}"
        f"{'  ' + str(sorted({b for _, b in exceeded})) if exceeded else ''}"
    )
    print(f"   leader changed on: {reversals or 'no board'}")
    if shifted:
        print(
            f"   * the oracle MOVED between the two runs in {len(shifted)} of "
            f"{len(fractions)} cells, e.g. {shifted[0][1]}/{shifted[0][0]}: "
            f"{shifted[0][2]:.4f} -> {shifted[0][3]:.4f}"
        )
        print("     `_grid_of` takes the FINEST requested map, so a multi-stage CNN's")
        print("     DPT run changes the bottleneck as well as the head. `lin/own` and")
        print("     `DPT/own` are then not two readings of one scale, and `gain` -- the")
        print("     DPT score over the linear one -- is the comparable column.")


#: Token counts, mirroring ``analyse_board_correlates.STRUCTURE``. Copied for
#: the reason that file copies ``HEADLINE_METRICS``: a script that reaches into
#: another script's namespace breaks when either moves, and the duplication is
#: pinned by a test rather than trusted.
TOKENS = {
    "clip_vitb16": 196,
    "clip_vitb32": 49,
    "dino_vitb16": 196,
    "dino_vitb8": 784,
    "dinov2_vitb14": 256,
    "dinov2_vits14": 256,
    "mae_vitb16": 196,
    "sam_vitb16": 196,
    "siglip_vitb16": 196,
    "supervised_vitb16": 196,
}


def report_grid(linear: dict, dpt: dict) -> None:
    """Does the DPT board track the feature grid harder than the linear one?

    The `dino_vitb8` board measured this on **one sibling pair** -- same
    objective, data, width and depth, 784 tokens against 196 -- and found that a
    4x finer grid moves the published linear boards by a rounding error while
    moving the DPT rows by one to two orders of magnitude more. This asks the
    same question of all ten ViTs at once, over data already committed, so the
    claim does not rest on a single pair.

    The prediction, if resolution is real but invisible to a linear readout: the
    DPT scores should correlate with token count *more strongly* than the linear
    scores do, probe by probe.
    """
    vits = sorted({b for (_, b) in dpt})
    tokens = [float(TOKENS[b]) for b in vits]
    print(f"\n-- does the grid correlation survive the head? ({len(vits)} ViTs)")
    print(f"   token counts: {sorted(set(int(v) for v in tokens))}")
    print(f"   {'probe':16s} {'linear':>8s} {'DPT':>8s} {'change':>8s}")

    totals = [0.0, 0.0]
    stronger = 0
    recovered: list[float] = []
    changes: list[float] = []
    for task, (metric, direction) in HEADLINE.items():
        sign = 1.0 if direction == "higher" else -1.0
        lin_scores = [sign * linear[(task, b)]["metrics"][metric] for b in vits]
        dpt_scores = [sign * dpt[(task, b)]["metrics"][metric] for b in vits]
        rho_lin = spearman(tokens, lin_scores)
        rho_dpt = spearman(tokens, dpt_scores)
        totals[0] += rho_lin
        totals[1] += rho_dpt
        stronger += rho_dpt > rho_lin
        shares = [
            fraction(
                linear[(task, b)]["metrics"][metric],
                linear[(task, b)]["metrics"]["ceiling_" + metric],
                direction,
            )
            for b in vits
        ]
        recovered.append(sum(shares) / len(shares))
        changes.append(rho_dpt - rho_lin)
        print(f"   {task:16s} {rho_lin:+8.3f} {rho_dpt:+8.3f} {rho_dpt - rho_lin:+8.3f}")

    n = len(HEADLINE)
    print(
        f"   {'mean':16s} {totals[0] / n:+8.3f} {totals[1] / n:+8.3f} "
        f"{(totals[1] - totals[0]) / n:+8.3f}"
    )
    print(f"   the DPT board tracks the grid more strongly on {stronger} of {n} probes")

    print("\n   ...and it is hidden exactly where a linear head recovers least:")
    print(f"   {'probe':16s} {'linear recovers':>16s} {'rho change':>12s}")
    for task, share, change in zip(HEADLINE, recovered, changes, strict=True):
        print(f"   {task:16s} {share:15.1%} {change:+12.3f}")
    print(
        f"   Spearman(recovered, change) = {spearman(recovered, changes):+.3f} over "
        f"{n} probes -- n={n}, so this is a mechanism that fits rather than one "
        "that is established."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["vit", "cnn", "both"], default="both")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="also ask whether the DPT board tracks the feature grid harder "
        "than the linear one (ViTs only -- a CNN's DPT run moves the oracle too)",
    )
    args = parser.parse_args()

    linear = latest([r for r in load(CORPUS) if r["task"] in HEADLINE])
    groups = ["vit", "cnn"] if args.group == "both" else [args.group]
    for group in groups:
        report(group, linear, latest(load(CONTROLS[group])))

    if args.grid:
        # ViTs only. A CNN's DPT run reads finer input than its linear one, so
        # the two are not two readings of one grid and the comparison would be
        # between different bottlenecks rather than between two heads.
        report_grid(linear, latest(load(CONTROLS["vit"])))

    print(
        "\nRead the two groups separately. A ViT's four blocks share one grid, so "
        "beating the oracle there means sub-patch structure; a CNN's stages do not, "
        "so it reads finer input than the oracle's grid and beating it is expected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
