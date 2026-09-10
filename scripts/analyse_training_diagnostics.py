#!/usr/bin/env python
"""Ask, of every board, whether its low scorers UNDERFITTED or simply cannot.

A low probe score has two opposite readings. An unconverged head *understates*
a backbone; a converged head that still scores badly says the representation
does not carry the answer. Schema v8's ``training`` block is what separates
them -- and it is a diagnostic, never a ranking, so nothing in
``LEADERBOARD.md`` reads it and this script is the only thing that does.

    scripts/analyse_training_diagnostics.py                  # every board
    scripts/analyse_training_diagnostics.py --board depth
    scripts/analyse_training_diagnostics.py --controls        # + results/controls/

What it prints per board, over the backbones ``latest_per_backbone`` resolves:

* each backbone's headline score beside its ``train_loss`` (and ``train_top1``
  where the probe reports one), ordered by score;
* Spearman rho between the fit and the score, which says whether the board's
  ordering tracks how well each head fitted its own training data;
* a **saturated** verdict when every ``train_top1`` is at 1.0, because then the
  board's whole spread is generalisation and no part of it is underfitting;
* a **suspect** verdict for a backbone whose ``train_loss`` is an *outlier*
  against the rest of its board -- above the median by more than three median
  absolute deviations -- which is the signature of one head failing to
  converge. A weaker representation fitting slightly worse is the ordinary
  case and must not trip it: on ``edge`` the whole board's losses span 0.046,
  and every backbone there converged. The test is skipped on a saturated board
  and on one whose losses are too flat for a multiple of their spread to mean
  anything -- both exclusions come from boards that tripped it.

  The first draft of that rule asked whether the worst fit was worse than the
  median *by more than half the board's loss spread*, which is true of the
  worst fit by construction -- it flagged ``edge`` and would have flagged every
  board with a finite spread. A criterion that cannot fail to fire is not a
  criterion; it is a tautology wearing a threshold.

``train_loss`` is comparable **within** a board and meaningless across boards:
each is a different loss on a different target. Nothing here compares two
boards' losses, and neither should a reader.

**The flag's one systematic confound, on a dense board, is feature
resolution.** A head reading a coarse grid has less to fit with, so it settles
at a higher training loss without anything having failed to converge -- and
resolution is already the strongest correlate of every dense board. Run against
``results/controls/dpt_head.jsonl`` the flag picks out ``clip_vitb32`` on
**all five** low-level boards, which is the nine ViTs' coarsest grid (7x7 at
224px against 14x14 for every other ViT there) rather than five separate
failures. So read a flag as "check the grid, then check the schedule", which is
why it is worded as something to re-check rather than a verdict.

Records that predate schema v8 carry ``training: null`` and are reported as
such rather than skipped, since "this board cannot answer the question" is the
finding that motivated the re-run in the first place.

Stdlib only, and no visbench import -- so it runs on a login node where the
project venv does not resolve. The two duplicated tables are pinned by
``tests/scripts/test_analyse_training_diagnostics.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
CONTROLS = ROOT / "results" / "controls"

#: The metric each board is ranked on, mirroring ``render.HEADLINE_METRICS``.
#:
#: Duplicated for the reason ``analyse_board_correlates.py`` duplicates it: this
#: script must run without torch importable. A test pins the two together.
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

#: Boards whose headline metric is an error, so a *lower* value ranks higher.
LOWER_IS_BETTER = {"surface_normal", "orientation"}

#: The three probes that fit nothing, so ``training: None`` is correct for them
#: rather than a gap. Stated rather than inferred: "no training block" and "a
#: training block nobody filled in" are the two cases this script separates.
ZERO_SHOT = {"retrieval", "correspondence", "similarity"}

#: How many median absolute deviations above a board's median ``train_loss``
#: counts as a head that did not converge, rather than a weaker backbone that
#: fitted slightly worse. Three is conventional for an MAD outlier and is
#: deliberately loose: this flags a run to be read, it does not reject one.
OUTLIER_MADS = 3.0


def load_records(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_per_backbone(records: list[dict]) -> dict[str, dict[str, dict]]:
    """``{task: {backbone: record}}``, newest record winning.

    Mirrors ``leaderboard.latest_per_backbone``: the corpus is append-only, so a
    re-run appends beside the record it supersedes and the newest is the one a
    board shows.
    """
    boards: dict[str, dict[str, dict]] = {}
    for record in sorted(records, key=lambda r: r["timestamp"]):
        boards.setdefault(record["task"], {})[record["backbone"]] = record
    return boards


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, ties averaged. ``None`` when either side is constant."""
    if len(xs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            mean_rank = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                out[order[position]] = mean_rank
            index = stop + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def report_board(task: str, board: dict[str, dict], out) -> dict:
    """Print one board's fit diagnostics and return what was concluded."""
    metric = HEADLINE_METRICS[task]
    rows = []
    for backbone, record in board.items():
        training = record.get("training")
        rows.append(
            {
                "backbone": backbone,
                "score": record["metrics"].get(metric),
                "train_loss": (training or {}).get("train_loss"),
                "train_top1": (training or {}).get("train_top1"),
                "has_training": training is not None,
                "version": record["visbench_version"],
            }
        )

    rows.sort(
        key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0.0),
        reverse=task not in LOWER_IS_BETTER,
    )

    print(f"\n=== {task}  ({metric}, {len(rows)} backbones)", file=out)

    if task in ZERO_SHOT:
        print("    zero-shot: fits nothing, so there is no fit to diagnose", file=out)
        return {"task": task, "verdict": "zero_shot"}

    # A record without the headline metric means the metric table names a key
    # this board does not emit -- which is how the copied HEADLINE_METRICS drifts
    # (`occlusion_edge` reports `occlusion_edge_correlation`, not
    # `edge_correlation`). Said plainly here rather than crashing three lines
    # down in a format string, which names neither the board nor the key.
    unscored = [r["backbone"] for r in rows if r["score"] is None]
    if unscored:
        print(
            f"    !!! {len(unscored)}/{len(rows)} records carry no {metric!r}: "
            f"{unscored[:3]}{' ...' if len(unscored) > 3 else ''}",
            file=out,
        )
        print(
            f"    !!! HEADLINE_METRICS names {metric!r} for this board; "
            f"one record emits {sorted(k for k in next(iter(board.values()))['metrics'])[:6]}",
            file=out,
        )
        if len(unscored) == len(rows):
            return {"task": task, "verdict": "unscored"}

    missing = [r["backbone"] for r in rows if not r["has_training"]]
    if missing:
        print(
            f"    training: null on {len(missing)}/{len(rows)} records "
            f"(pre-v8) -- this board CANNOT answer the underfitting question",
            file=out,
        )
        if len(missing) == len(rows):
            return {"task": task, "verdict": "unanswerable"}

    width = max(len(r["backbone"]) for r in rows)
    header = f"    {'backbone':{width}s} {metric:>12s} {'train_loss':>12s}"
    if any(r["train_top1"] is not None for r in rows):
        header += f" {'train_top1':>11s}"
    print(header, file=out)
    for row in rows:
        score = f"{row['score']:12.4f}" if row["score"] is not None else f"{'--':>12s}"
        line = f"    {row['backbone']:{width}s} {score}"
        line += f" {row['train_loss']:12.4f}" if row["train_loss"] is not None else f" {'--':>12s}"
        if any(r["train_top1"] is not None for r in rows):
            line += (
                f" {row['train_top1']:11.4f}" if row["train_top1"] is not None else f" {'--':>11s}"
            )
        print(line, file=out)

    fitted = [r for r in rows if r["train_loss"] is not None and r["score"] is not None]
    verdict: dict = {"task": task, "verdict": "ok", "n": len(fitted)}

    top1s = [r["train_top1"] for r in fitted if r["train_top1"] is not None]
    if top1s and min(top1s) >= 0.9999:
        print(
            "    SATURATED: every train_top1 is 1.0000, so nothing here underfits "
            "and the board's whole spread is generalisation",
            file=out,
        )
        verdict["verdict"] = "saturated"

    if len(fitted) >= 3:
        # Both sides are turned into "higher is better" before correlating --
        # a fit by negating the loss, a score by negating it too when the
        # headline metric is an error -- so a positive rho always reads the same
        # way whichever direction the board runs. The alternative was to
        # correlate the raw values and negate the coefficient at the end, which
        # is right twice and unreadable once.
        quality = -1.0 if task in LOWER_IS_BETTER else 1.0
        rho = spearman([-r["train_loss"] for r in fitted], [quality * r["score"] for r in fitted])
        if rho is not None:
            print(
                f"    rho(fit, score) = {rho:+.3f}  (+1 = the better fit scores better)",
                file=out,
            )
            verdict["rho"] = rho

        # An OUTLIER test, not a "worst fit" test. Every board has a worst fit,
        # and on a board where the ordering is real the worst fit is simply the
        # weakest representation -- `edge` spans 0.046 of loss across twelve
        # converged heads. What underfitting looks like instead is one head
        # sitting away from the distribution the others form, so the threshold
        # is stated in median absolute deviations of the board's own losses.
        losses = sorted(r["train_loss"] for r in fitted)
        median = losses[len(losses) // 2]
        mad = sorted(abs(loss - median) for loss in losses)[len(losses) // 2]

        # A SATURATED board cannot contain an underfitted head: `train_top1`
        # 1.0000 says that head fitted everything it was given, so flagging one
        # there contradicts the line printed just above it. The object
        # classification board is the case -- `mae_vitb16` sits at train_loss
        # 0.0061 against a median of 0.0000 and still fits every training image.
        #
        # A board whose losses are IDENTICAL has no spread to be an outlier
        # against, and `mad == 0` is how that arrives.
        #
        # What is deliberately NOT here is a rule about the median being small.
        # Two attempts at one -- half the loss spread, then a floor on MAD
        # relative to the median -- both fired on converged boards, and the
        # honest reading is that "3 MAD" is a flag for a run worth re-checking
        # rather than a verdict about it. So the message below leads with the
        # absolute excess, which is the part that means something, and reports
        # the multiple as a magnitude rather than a spurious 4905.6.
        if verdict["verdict"] == "saturated" or mad <= 0:
            outliers = []
        else:
            outliers = [r for r in fitted if r["train_loss"] > median + OUTLIER_MADS * mad]
        for row in outliers:
            excess = row["train_loss"] - median
            multiple = excess / mad
            magnitude = f"{multiple:.1f}" if multiple < 100 else f">{100 * (multiple // 100):.0f}"
            print(
                f"    SUSPECT: {row['backbone']}'s train_loss {row['train_loss']:.4f} is "
                f"{excess:.4f} above this board's median {median:.4f} ({magnitude} MAD) "
                "-- worth re-checking as a fit before reading it as a representation",
                file=out,
            )
        if outliers:
            verdict["verdict"] = "suspect"
            verdict["suspect"] = [r["backbone"] for r in outliers]

    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--board", action="append", help="only these boards (repeatable)")
    parser.add_argument(
        "--controls",
        action="store_true",
        help="also read results/controls/*.jsonl, each file as its own board set",
    )
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        print(f"No corpus at {args.corpus}", file=sys.stderr)
        return 1

    boards = latest_per_backbone(load_records(args.corpus))
    unknown = sorted(set(boards) - set(HEADLINE_METRICS))
    if unknown:
        print(f"No headline metric for {unknown}; update HEADLINE_METRICS", file=sys.stderr)
        return 1

    wanted = args.board or sorted(boards)
    missing = sorted(set(wanted) - set(boards))
    if missing:
        print(f"Not in the corpus: {missing}", file=sys.stderr)
        return 1

    verdicts = [report_board(task, boards[task], sys.stdout) for task in wanted]

    if args.controls:
        for path in sorted(CONTROLS.glob("*.jsonl")):
            print(f"\n--- control: {path.name}", file=sys.stdout)
            for task, board in latest_per_backbone(load_records(path)).items():
                if task in HEADLINE_METRICS:
                    report_board(task, board, sys.stdout)

    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict["verdict"]] = counts.get(verdict["verdict"], 0) + 1
    print("\n=== summary", file=sys.stdout)
    for name in sorted(counts):
        print(f"    {name:14s} {counts[name]}", file=sys.stdout)
    unanswerable = [v["task"] for v in verdicts if v["verdict"] == "unanswerable"]
    if unanswerable:
        print(f"    boards that cannot answer it: {unanswerable}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
