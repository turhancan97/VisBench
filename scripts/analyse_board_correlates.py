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
    # A resolution control, not a corpus column: the same weights at 196px, so
    # its grid matches every ViT-B/16. Listed here so this script can be run
    # against results/controls/resolution.jsonl, which the corpus never holds.
    "dinov2_vitb14_196": Structure(196, 768, "ViT", "LVD-142M", 1.42e8, "ssl-discriminative"),
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


#: The image corpus each board reads, for telling a shared-data effect apart
#: from a shared-capability one.
#:
#: Hand-written because the records cannot supply it: three unrelated datasets
#: are all called ``val`` in the ``dataset`` field (Imagenette, NYUv2 and the
#: staged Taskonomy corner frames), so grouping on that would merge boards that
#: share nothing.
#:
#: Only ``semantic_segmentation`` and ``generic_segmentation`` read the *same
#: 1449 images*; the rest of a group shares a corpus, not a split. That
#: distinction is what makes the VOC trio worth reading pair by pair rather
#: than as a mean.
SOURCE_IMAGES: dict[str, str] = {
    "semantic_segmentation": "VOC",
    "generic_segmentation": "VOC",
    "detection": "VOC",
    "classification": "Imagenette",
    "retrieval": "Imagenette",
    "correspondence": "Imagenette",
    "depth": "NYUv2",
    "surface_normal": "NYUv2",
    "edge": "Taskonomy",
    "keypoints2d": "Taskonomy",
    "occlusion_edge": "Taskonomy",
    "corner": "Taskonomy",
    "similarity": "NIGHTS",
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


def load_levels(corpus: Path) -> dict[str, str]:
    """``{task: level}``, read from the records rather than from the task name.

    **Count tiers from ``record.level``, never from which boards feel
    semantic.** `similarity` is the trap: it is mid-level image similarity,
    kept deliberately distinct from high-level retrieval, and counting it as
    high-level is a mistake that shipped in this repository for a commit. The
    record already carries the answer, so there is no reason to infer one.
    """
    levels: dict[str, str] = {}
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task, level = record["task"], record["level"]
        if levels.setdefault(task, level) != level:
            raise ValueError(f"{task} appears at two levels: {levels[task]} and {level}")
    return levels


def agreement(boards: dict[str, dict[str, float]]) -> dict[tuple[str, str], float]:
    """Spearman between every pair of boards, keyed by a sorted task pair.

    This is the question the structural correlations cannot answer: two boards
    can both track resolution and still rank backbones differently, and two
    boards that track nothing structural can still agree with each other. What
    the taxonomy claims is that probes within a tier agree more than probes
    across tiers, and only a board-against-board matrix tests that.
    """
    tasks = sorted(boards)
    return {
        (a, b): spearman(boards[a], boards[b]) for i, a in enumerate(tasks) for b in tasks[i + 1 :]
    }


def tier_summary(
    pairs: dict[tuple[str, str], float], levels: dict[str, str]
) -> tuple[dict[str, float], float]:
    """Mean within-tier rho per level, and the single cross-tier mean.

    The claim under test is *relative*: every within-tier mean should exceed
    the cross-tier mean. An absolute value means little, because a corpus whose
    backbones all sit on one capacity axis makes every board agree with every
    other -- which is exactly what the n=6 version of this analysis found, and
    why it concluded the tiers do not separate.
    """
    within: dict[str, list[float]] = defaultdict(list)
    across: list[float] = []
    for (a, b), rho in pairs.items():
        if levels[a] == levels[b]:
            within[levels[a]].append(rho)
        else:
            across.append(rho)
    means = {level: sum(v) / len(v) for level, v in within.items()}
    return means, sum(across) / len(across)


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


def report_agreement(boards: dict[str, dict[str, float]], levels: dict[str, str]) -> None:
    """Boards against each other, and the tier claim that rests on it."""
    pairs = agreement(boards)
    n = len(next(iter(boards.values())))

    print(f"\n=== do probes within a tier agree more than probes across tiers? (n={n}) ===")
    means, across = tier_summary(pairs, levels)
    for level in sorted(means):
        count = sum(1 for a, b in pairs if levels[a] == levels[b] == level)
        verdict = "above" if means[level] > across else "BELOW"
        print(f"  within {level:11s} {means[level]:+.3f}  ({count:2d} pairs)  {verdict} cross-tier")
    print(
        f"  across tiers        {across:+.3f}  "
        f"({sum(1 for a, b in pairs if levels[a] != levels[b]):2d} pairs)"
    )
    failing = sorted(level for level, mean in means.items() if mean <= across)
    if not failing:
        print("  -> every within-tier mean exceeds the cross-tier mean: the taxonomy holds here")
    else:
        print(
            "  -> at least one tier is no tighter than chance pairing: the taxonomy does NOT hold"
        )

    # A tier mean averages over pairs, and an average cannot say whether a tier
    # is uniformly loose or is two tight clusters ignoring each other. Those
    # want different responses -- the first questions the probes, the second
    # questions the tier -- so a failing tier prints its pairs, not just a mean.
    for level in failing:
        print(f"\n  {level} pair by pair, since its mean is what failed:")
        members = sorted(t for t in boards if levels[t] == level)
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                print(f"    {pairs[a, b]:+.3f}  {a} / {b}")

    identical = [pair for pair, rho in pairs.items() if rho == 1.0]
    print(f"\n  boards ranking identically: {len(identical)} of {len(pairs)}")
    for a, b in identical:
        print(f"    {a} == {b}")
    if identical:
        print("    (two probes producing one ordering measure one thing, whatever they are named)")

    print("\n=== most and least agreement between boards ===")
    ordered = sorted(pairs.items(), key=lambda kv: -kv[1])
    for (a, b), rho in ordered[:5]:
        tag = "same tier" if levels[a] == levels[b] else "cross"
        print(f"  {rho:+.3f}  {a} / {b}  ({tag})")
    print("  ...")
    for (a, b), rho in ordered[-5:]:
        tag = "same tier" if levels[a] == levels[b] else "cross"
        print(f"  {rho:+.3f}  {a} / {b}  ({tag})")


def report_sources(boards: dict[str, dict[str, float]], levels: dict[str, str]) -> None:
    """Is a cluster of boards a shared capability, or just shared images?

    A fair question to ask of any agreement result here, because the probes do
    not each have their own dataset -- three read VOC and three read Imagenette.
    If sharing pixels were what made two boards agree, the tier findings would
    be an artefact of how the corpus was assembled.

    The decisive comparison is inside VOC and needs no new run: semantic and
    generic segmentation read the *same 1449 images* at the same resolution
    through the same head, while detection reads 600 different VOC frames.
    Shared pixels as the cause predicts the identical-image pair is the
    strongest of the three.
    """
    pairs = agreement(boards)
    within: dict[str, list[float]] = defaultdict(list)
    across: list[float] = []
    for (a, b), rho in pairs.items():
        if SOURCE_IMAGES[a] == SOURCE_IMAGES[b]:
            within[SOURCE_IMAGES[a]].append(rho)
        else:
            across.append(rho)

    print("\n=== do boards agree because they read the same images? ===")
    pooled = [rho for group in within.values() for rho in group]
    for source in sorted(within):
        group = within[source]
        print(f"  within {source:11s} {sum(group) / len(group):+.3f}  ({len(group):2d} pairs)")
    print(f"  within any source   {sum(pooled) / len(pooled):+.3f}  ({len(pooled):2d} pairs)")
    print(f"  across sources      {sum(across) / len(across):+.3f}  ({len(across):2d} pairs)")

    print("\n  the VOC trio, where two boards read the *same* 1449 images:")
    voc = sorted(t for t in boards if SOURCE_IMAGES[t] == "VOC")
    for i, a in enumerate(voc):
        for b in voc[i + 1 :]:
            same = (
                "<- same images"
                if {a, b} == {"semantic_segmentation", "generic_segmentation"}
                else ""
            )
            print(f"    {pairs[a, b]:+.3f}  {a} ({levels[a]}) / {b} ({levels[b]})  {same}")

    # The clincher, and the reason this section exists at all: rank the
    # identical-image board's neighbours. If shared pixels drove agreement its
    # VOC siblings would be at the top of this list.
    probe = "generic_segmentation"
    print(f"\n  what {probe} agrees with, best first (it reads VOC):")
    neighbours = sorted(
        ((pairs[tuple(sorted((probe, t)))], t) for t in boards if t != probe), reverse=True
    )
    for rho, task in neighbours[:6]:
        print(f"    {rho:+.3f}  {task:24s} {levels[task]:11s} {SOURCE_IMAGES[task]}")


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
    parser.add_argument(
        "--section",
        choices=("structure", "agreement", "sources", "all"),
        default="all",
        help="structure: boards against backbone properties. agreement: boards "
        "against each other, and the tier claim that rests on it. sources: "
        "whether agreement is really about sharing a dataset.",
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

    if args.section in ("structure", "all"):
        report(boards, args.webli, args.board)
    if args.section in ("agreement", "all") and args.board is None:
        report_agreement(boards, load_levels(args.corpus))
    if args.section in ("sources", "all") and args.board is None:
        report_sources(boards, load_levels(args.corpus))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
