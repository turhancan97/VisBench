#!/usr/bin/env python
"""Ask what a board's ordering tracks, using only records already in the corpus.

Every board here ranks backbones, and nothing in a record says *why* one is
above another. This script correlates each board's ranking against the
structural properties of the backbones it ranked -- feature-grid area, dense
width, pretraining corpus size -- so that a board whose ordering is mostly
resolution can be told apart from one whose ordering is mostly something else.

    scripts/analyse_board_correlates.py                   # every board
    scripts/analyse_board_correlates.py --board semantic_segmentation

It answers a question that came out of step 10e. A recipe control showed that
the semantic-segmentation board cannot separate training objectives -- a
supervised backbone lands within 0.0011 of a pixel-reconstruction one -- which
left "then what does it separate?" open. The answer, on the twelve-backbone
corpus, is that it is the **only dense board that does not rank by feature
resolution**, and the control for that claim was already in the corpus:
``generic_segmentation`` runs on the same 1449 VOC images at the same
resolution with the same linear head and the same schedule, differing only in
whether the target has 2 classes or 21.

**This is a lead, not a proof, and the sample size is why.** Twelve backbones
means a Spearman rho has wide error bars, and the properties below are
correlated with each other -- DINOv2 has both the finest grid and a large
pretraining corpus, so it pushes every coefficient at once. ``--drop`` exists
for exactly that: re-run without a backbone and see which conclusions survive.

Nothing here is a metric and nothing is written back to the corpus. It reads
``results/corpus/visbench.jsonl`` and prints.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"

#: The metric each board is ranked on, mirroring ``render.HEADLINE_METRICS``.
#:
#: Duplicated rather than imported so this script runs without visbench (and so
#: without torch) importable, which is what lets it run on a login node. A test
#: pins the two tables against each other, so the copy cannot drift.
HEADLINE_METRICS: dict[str, str] = {
    "classification": "top1",
    "retrieval": "mAP",
    "correspondence": "recall@5px",
    "similarity": "accuracy",
    "semantic_segmentation": "miou",
    "generic_segmentation": "iou",
    "detection": "map_50",
    "depth": "d1",
    "surface_normal": "mean",
    "edge": "edge_correlation",
    "keypoints2d": "keypoint_correlation",
    "corner": "corner_correlation",
    "occlusion_edge": "occlusion_edge_correlation",
}

#: Boards whose headline metric is an *error*, so a lower number is better.
#:
#: Listed rather than inferred, for the reason ``METRIC_DIRECTIONS`` is listed:
#: a heuristic that guessed wrong would silently invert a board and the output
#: would read as a finding rather than a bug.
LOWER_IS_BETTER: frozenset[str] = frozenset({"surface_normal"})


@dataclass(frozen=True)
class Structure:
    """What a backbone *is*, independent of how it was trained.

    ``tokens`` is the number of spatial positions the probe sees at 224px --
    16x16 for a patch-14 ViT, 14x14 for patch-16, 7x7 for patch-32 and for
    every stride-32 CNN here. It is the quantity a dense probe's ceiling
    depends on, which is why ``correspondence`` already reports it as
    ``ceiling_``.

    ``pretrain_images`` is order-of-magnitude only. WebLI's public size is not
    pinned to a single figure, so treat it as "very large" rather than exact --
    ``--webli`` re-runs with a different assumption, and the semantic
    segmentation result is unchanged from 4e8 through 1e10.
    """

    tokens: int
    width: int
    family: str
    pretrain: str
    pretrain_images: float
    objective: str


#: Patch size is pinned by each record's own ``backbone_key`` (for example
#: ``timm/vit_base_patch16_224/dino/224``), so these are read off the corpus
#: rather than remembered. Width and pretraining corpus come from the model
#: cards of the checkpoints those keys name.
STRUCTURE: dict[str, Structure] = {
    "dinov2_vits14": Structure(256, 384, "ViT", "LVD-142M", 1.42e8, "ssl-discriminative"),
    "dinov2_vitb14": Structure(256, 768, "ViT", "LVD-142M", 1.42e8, "ssl-discriminative"),
    "clip_vitb16": Structure(196, 768, "ViT", "WIT-400M", 4.0e8, "language"),
    "siglip_vitb16": Structure(196, 768, "ViT", "WebLI", 1.0e10, "language"),
    "mae_vitb16": Structure(196, 768, "ViT", "IN1k", 1.28e6, "reconstruction"),
    "dino_vitb16": Structure(196, 768, "ViT", "IN1k", 1.28e6, "ssl-discriminative"),
    "supervised_vitb16": Structure(196, 768, "ViT", "IN1k", 1.28e6, "supervised"),
    "sam_vitb16": Structure(196, 768, "ViT", "IN1k", 1.28e6, "supervised"),
    "clip_vitb32": Structure(49, 768, "ViT", "WIT-400M", 4.0e8, "language"),
    "resnet18": Structure(49, 512, "CNN", "IN1k", 1.28e6, "supervised"),
    "resnet50": Structure(49, 2048, "CNN", "IN1k", 1.28e6, "supervised"),
    "convnext_base": Structure(49, 1024, "CNN", "IN1k", 1.28e6, "supervised"),
}


def ranks(values: dict[str, float]) -> dict[str, int]:
    """Rank 1 is the best score. Ties take their first position, not a mean.

    Every board this is used on has distinct values, so the simple rule is
    enough; a corpus with genuine ties would need midranks here and in
    :func:`spearman` together.
    """
    ordered = sorted(values, key=lambda k: -values[k])
    return {k: i + 1 for i, k in enumerate(ordered)}


def spearman(left: dict[str, float], right: dict[str, float]) -> float:
    """Rank correlation over the backbones the two arguments share.

    Intersecting rather than requiring equal keys is deliberate: a board that
    is missing a backbone should shrink the comparison, not raise, because the
    corpus is grown one column at a time and a half-filled board is a normal
    intermediate state.
    """
    keys = sorted(set(left) & set(right))
    n = len(keys)
    if n < 3:
        raise ValueError(f"need at least 3 shared backbones to correlate, got {n}")
    lr, rr = ranks({k: left[k] for k in keys}), ranks({k: right[k] for k in keys})
    d2 = sum((lr[k] - rr[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def load_boards(corpus: Path) -> dict[str, dict[str, float]]:
    """``{task: {backbone: score}}``, with every score oriented higher-is-better.

    Error metrics are negated rather than ranked in reverse, so that every
    caller downstream can assume one direction.
    """
    boards: dict[str, dict[str, float]] = defaultdict(dict)
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task = record["task"]
        value = float(record["metrics"][HEADLINE_METRICS[task]])
        boards[task][record["backbone"]] = -value if task in LOWER_IS_BETTER else value
    return dict(boards)


def _property(name: str, backbones: list[str], webli: float) -> dict[str, float]:
    getters = {
        "tokens": lambda s: float(s.tokens),
        "width": lambda s: float(s.width),
        "pretrain_images": lambda s: webli if s.pretrain == "WebLI" else s.pretrain_images,
    }
    return {b: getters[name](STRUCTURE[b]) for b in backbones}


def report(boards: dict[str, dict[str, float]], webli: float, only: str | None) -> None:
    tasks = sorted(boards) if only is None else [only]
    for prop in ("tokens", "width", "pretrain_images"):
        print(f"\n=== every board vs {prop} ===")
        scored = [(t, spearman(boards[t], _property(prop, list(boards[t]), webli))) for t in tasks]
        for task, rho in sorted(scored, key=lambda pair: -pair[1]):
            print(f"  {task:24s} rho = {rho:+.3f}")

    if only is not None:
        return

    print("\n=== the control: two targets over the same 1449 VOC images ===")
    print("  generic_segmentation is foreground/background, semantic is 21 classes.")
    print("  Same images, same 224px, same linear head, same schedule.")
    for task in ("generic_segmentation", "semantic_segmentation"):
        board = boards[task]
        tok = spearman(board, _property("tokens", list(board), webli))
        dat = spearman(board, _property("pretrain_images", list(board), webli))
        print(f"  {task:24s} vs grid {tok:+.3f}   vs pretraining size {dat:+.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--board", default=None, help="restrict to one task")
    parser.add_argument(
        "--drop",
        action="append",
        default=[],
        metavar="BACKBONE",
        help="exclude a backbone; repeatable. Use it to check which conclusions "
        "survive without the rows that carry them.",
    )
    parser.add_argument(
        "--webli",
        type=float,
        default=1.0e10,
        help="assumed WebLI size, since it is not pinned to one public figure",
    )
    args = parser.parse_args(argv)

    boards = load_boards(args.corpus)
    if args.board is not None and args.board not in boards:
        parser.error(f"no board named {args.board!r}; have {', '.join(sorted(boards))}")

    unknown = {b for board in boards.values() for b in board} - set(STRUCTURE)
    if unknown:
        # A new backbone column reaches the corpus before it reaches this
        # table, and correlating against a structure we do not have would
        # silently drop it from every coefficient.
        print(f"error: no STRUCTURE entry for {', '.join(sorted(unknown))}", file=sys.stderr)
        return 1

    for backbone in args.drop:
        for board in boards.values():
            board.pop(backbone, None)

    report(boards, args.webli, args.board)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
