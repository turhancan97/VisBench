#!/usr/bin/env python
"""Do the correlational findings survive a re-fit? (22b)

    python scripts/analyse_correlation_seeds.py
    python scripts/analyse_correlation_seeds.py --section claims
    python scripts/analyse_correlation_seeds.py --pairs        # every board pair

`scripts/analyse_board_correlates.py` reads the corpus, which holds **one seed
per cell**, so every coefficient it prints -- and every correlational claim
`CORPUS_FINDINGS.md` builds on them -- is one draw of a statistic whose
run-to-run scatter nobody had measured. 20a-22a showed what that costs for
*orderings*: a third of adjacent pairs are not separable and three are reversed.
This asks the same question of the *correlations*, using the sweeps those steps
already committed, so it costs no GPU.

**What a "draw" is here.** Every trained board has five seeds under
`results/controls/seeds/` (and `relative_pose` in `pose_seeds.jsonl`). A board
published at one seed is one of those five, so a draw assigns each board one of
its seeds, independently -- a different head was fitted on every board, so a
seed number couples nothing across boards. The three zero-shot boards fit no
head and are exact, the same in every draw. A pair of boards therefore has
exactly 25 equally valid values (5 if one side is zero-shot) and is enumerated;
a claim spanning many boards is sampled, ``--draws`` times from a fixed RNG.

Each claim is reduced to a **margin** that is positive exactly when the
published reading holds -- "A ranks with B more than with C" becomes
``rho(A,B) - rho(A,C)`` -- and is reported three ways: on the published corpus,
on the **seed-mean** board (each cell averaged over its five seeds, the least
noisy single estimate there is), and as the fraction of draws in which it holds.

**`scene_classification` is read from its held-out sweep and flagged**, as
`analyse_placements.py` reads it: its seed 0 does not reproduce its board (20c,
amplification per 20d), so "published" and "draws" describe two slightly
different configurations there.

**Nothing here is a score and nothing ranks a backbone.** It says how much of a
published *reading* is the reading and how much is the draw. It reads the
corpus and the sweeps and prints; it imports no visbench, like the script it
audits, so it runs on a login node.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
SWEEP_DIR = ROOT / "results" / "controls" / "seeds"
POSE_SWEEP = ROOT / "results" / "controls" / "pose_seeds.jsonl"
HELD_OUT = {
    "scene_classification": ROOT / "results" / "controls" / "scene_classification_seeds.jsonl"
}
ZERO_SHOT = ("retrieval", "correspondence", "similarity")
SEEDS = (0, 1, 2, 3, 4)

#: The pseudo-board the instance entry reads: the same runs ranked on their
#: *box* half. Never part of the board matrix -- it is not a board.
BOX = "instance_segmentation:box"


def _load_correlates():
    """The audited script, by path, so its tables and `spearman` are the ones used.

    A second copy of the rank arithmetic here would be free to disagree with
    the script the prose tells a reader to run -- the reason `analyse_placements`'
    test loads its subject the same way.
    """
    path = Path(__file__).resolve().parent / "analyse_board_correlates.py"
    spec = importlib.util.spec_from_file_location("analyse_board_correlates", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("analyse_board_correlates", module)
    spec.loader.exec_module(module)
    return module


correlates = _load_correlates()
spearman = correlates.spearman

Board = dict[str, float]


def _oriented(task: str, value: float) -> float:
    return -value if task in correlates.LOWER_IS_BETTER else value


def sweep_paths() -> dict[str, Path]:
    found = {path.stem: path for path in SWEEP_DIR.glob("*.jsonl")} if SWEEP_DIR.is_dir() else {}
    if POSE_SWEEP.is_file():
        found["relative_pose"] = POSE_SWEEP
    found.update({task: path for task, path in HELD_OUT.items() if path.is_file()})
    return found


@dataclass
class Data:
    """Published boards, their per-seed versions, and the levels and fits."""

    published: dict[str, Board]
    seeded: dict[str, dict[int, Board]]  # task -> seed -> board
    levels: dict[str, str]
    fits: dict[str, dict[int, Board]]  # task -> seed -> {backbone: train_loss}

    @property
    def tasks(self) -> list[str]:
        return sorted(t for t in self.published if t != BOX)

    def seed_mean(self) -> dict[str, Board]:
        return {
            task: {b: statistics.fmean(seeds[s][b] for s in SEEDS) for b in seeds[SEEDS[0]]}
            for task, seeds in self.seeded.items()
        }


def load(corpus: Path = CORPUS) -> Data:
    published = correlates.load_boards(corpus)
    levels = correlates.load_levels(corpus)
    # The box half of the instance board, published.
    box: Board = {}
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record["task"] == "instance_segmentation":
                box[record["backbone"]] = float(record["metrics"]["box_map_50"])
    published[BOX] = box

    paths = sweep_paths()
    seeded: dict[str, dict[int, Board]] = {}
    fits: dict[str, dict[int, Board]] = defaultdict(lambda: defaultdict(dict))
    for task in published:
        if task == BOX:
            continue
        if task in ZERO_SHOT:
            seeded[task] = {s: dict(published[task]) for s in SEEDS}
            continue
        if task not in paths:
            raise SystemExit(f"error: trained board {task!r} has no committed sweep")
        boards: dict[int, Board] = defaultdict(dict)
        metric = correlates.HEADLINE_METRICS[task]
        for line in paths[task].read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["task"] != task:
                continue
            seed, backbone = record["seed"], record["backbone"]
            boards[seed][backbone] = _oriented(task, float(record["metrics"][metric]))
            if task == "instance_segmentation":
                seeded.setdefault(BOX, defaultdict(dict))[seed][backbone] = float(
                    record["metrics"]["box_map_50"]
                )
            training = record.get("training") or {}
            if "train_loss" in training:
                fits[task][seed][backbone] = float(training["train_loss"])
        if set(boards) != set(SEEDS) or any(
            set(b) != set(published[task]) for b in boards.values()
        ):
            raise SystemExit(f"error: sweep for {task!r} does not cover 5 seeds x its board")
        seeded[task] = dict(boards)
    seeded[BOX] = dict(seeded[BOX])
    return Data(published, seeded, levels, {t: dict(v) for t, v in fits.items()})


class View:
    """One set of boards -- published, seed-mean, or a draw -- with cached rho."""

    def __init__(self, boards: dict[str, Board], levels: dict[str, str]):
        self.boards = boards
        self.levels = levels
        self._cache: dict[tuple[str, str], float] = {}

    def rho(self, a: str, b: str) -> float:
        key = (a, b) if a < b else (b, a)
        if key not in self._cache:
            self._cache[key] = spearman(self.boards[key[0]], self.boards[key[1]])
        return self._cache[key]

    def mean_rho(self, task: str, others: list[str]) -> float:
        return statistics.fmean(self.rho(task, o) for o in others if o != task)

    def tier(self, level: str, exclude: str | None = None) -> list[str]:
        return sorted(
            t for t, lv in self.levels.items() if lv == level and t != exclude and t in self.boards
        )


# --------------------------------------------------------------------- claims

IMAGE_LEVEL = ("classification", "retrieval")
LOCALISED = (
    "detection",
    "semantic_segmentation",
    "fine_grained_classification",
    "scene_classification",
)


def _two_clusters(v: View) -> float:
    """Every within-cluster pair above every between-cluster pair ("nothing between")."""
    within = [v.rho(*IMAGE_LEVEL)] + [
        v.rho(a, b) for i, a in enumerate(LOCALISED) for b in LOCALISED[i + 1 :]
    ]
    between = [v.rho(a, b) for a in IMAGE_LEVEL for b in LOCALISED]
    return min(within) - max(between)


def _tier_margin(level: str) -> Callable[[View], float]:
    def margin(v: View) -> float:
        tasks = sorted(v.levels)
        within = [
            v.rho(a, b)
            for i, a in enumerate(tasks)
            for b in tasks[i + 1 :]
            if v.levels[a] == v.levels[b] == level
        ]
        across = [
            v.rho(a, b)
            for i, a in enumerate(tasks)
            for b in tasks[i + 1 :]
            if v.levels[a] != v.levels[b]
        ]
        return statistics.fmean(within) - statistics.fmean(across)

    return margin


def _voc_weakest_is_same_image(v: View) -> float:
    voc = sorted(t for t in v.levels if correlates.SOURCE_IMAGES[t] == "VOC")
    pairs = [(a, b) for i, a in enumerate(voc) for b in voc[i + 1 :]]
    same = [v.rho(a, b) for a, b in pairs if {a, b} <= correlates.SAME_IMAGES]
    other = [v.rho(a, b) for a, b in pairs if not {a, b} <= correlates.SAME_IMAGES]
    return min(other) - min(same)


def _imagenette_lowest_source(v: View) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    tasks = sorted(v.levels)
    for i, a in enumerate(tasks):
        for b in tasks[i + 1 :]:
            if correlates.SOURCE_IMAGES[a] == correlates.SOURCE_IMAGES[b]:
                groups[correlates.SOURCE_IMAGES[a]].append(v.rho(a, b))
    means = {s: statistics.fmean(r) for s, r in groups.items()}
    return min(m for s, m in means.items() if s != "Imagenette") - means["Imagenette"]


def _instance_vs(task: str) -> Callable[[View], float]:
    def margin(v: View) -> float:
        mid = v.tier("mid_level")
        high = v.tier("high_level", exclude="instance_segmentation")
        return v.mean_rho(task, mid) - v.mean_rho(task, high)

    return margin


@dataclass(frozen=True)
class Claim:
    key: str
    text: str
    where: str
    margin: Callable[[View], float]


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "fine_grained_localised",
        "fine_grained ranks with detection, not with the object board it subclasses",
        "two-cluster entry",
        lambda v: (
            v.rho("fine_grained_classification", "detection")
            - v.rho("fine_grained_classification", "classification")
        ),
    ),
    Claim(
        "scene_localised",
        "scene_classification ranks with detection over classification and retrieval",
        "scene_classification entry",
        lambda v: (
            v.rho("scene_classification", "detection")
            - max(
                v.rho("scene_classification", "classification"),
                v.rho("scene_classification", "retrieval"),
            )
        ),
    ),
    Claim(
        "two_clusters",
        "high-level is two clusters with nothing between (min within > max between)",
        "two-cluster entry, 15-pair table",
        _two_clusters,
    ),
    Claim(
        "scene_parsing_question",
        "scene_parsing agrees with VOC semseg over its own NYUv2 frames (depth, normals)",
        "scene_parsing entry",
        lambda v: (
            v.rho("scene_parsing", "semantic_segmentation")
            - max(v.rho("scene_parsing", "depth"), v.rho("scene_parsing", "surface_normal"))
        ),
    ),
    Claim(
        "scene_parsing_localised",
        "scene_parsing sits with the localised cluster, not the image-level one",
        "scene_parsing entry",
        lambda v: (
            v.mean_rho(
                "scene_parsing",
                ["semantic_segmentation", "detection", "fine_grained_classification"],
            )
            - max(v.rho("scene_parsing", t) for t in IMAGE_LEVEL)
        ),
    ),
    Claim(
        "vehicle_closest_is_place",
        "vehicle's closest partner is scene_classification, not fine_grained",
        "vehicle entry",
        lambda v: (
            v.rho("vehicle_classification", "scene_classification")
            - v.rho("vehicle_classification", "fine_grained_classification")
        ),
    ),
    Claim(
        "vehicle_tier_order",
        "vehicle agrees high > mid > low (mean rho per tier)",
        "vehicle entry",
        lambda v: min(
            v.mean_rho("vehicle_classification", v.tier("high_level"))
            - v.mean_rho("vehicle_classification", v.tier("mid_level")),
            v.mean_rho("vehicle_classification", v.tier("mid_level"))
            - v.mean_rho("vehicle_classification", v.tier("low_level")),
        ),
    ),
    Claim(
        "pose_orthogonal",
        "relative_pose agrees with mid and low far more than with high (margin)",
        "relative_pose entry",
        lambda v: (
            min(
                v.mean_rho("relative_pose", v.tier("low_level")),
                v.mean_rho("relative_pose", v.tier("mid_level", exclude="relative_pose")),
            )
            - v.mean_rho("relative_pose", v.tier("high_level"))
        ),
    ),
    Claim(
        "instance_geometry",
        "instance_segmentation agrees with mid-level over its own tier",
        "instance entry",
        _instance_vs("instance_segmentation"),
    ),
    Claim(
        "instance_box_geometry",
        "the BOX half of instance alone already ranks with mid-level",
        "instance entry",
        _instance_vs(BOX),
    ),
    Claim(
        "voc_same_image_weakest",
        "the weakest of the six VOC pairs is a same-image pair (not sufficient)",
        "shared-datasets entry",
        _voc_weakest_is_same_image,
    ),
    Claim(
        "generic_reaches_off_voc",
        "generic_segmentation agrees more with a non-VOC board than with semseg (not necessary)",
        "shared-datasets entry",
        lambda v: (
            max(
                v.rho("generic_segmentation", t)
                for t in v.levels
                if correlates.SOURCE_IMAGES[t] != "VOC"
            )
            - v.rho("generic_segmentation", "semantic_segmentation")
        ),
    ),
    Claim(
        "imagenette_lowest",
        "Imagenette is the lowest within-source mean (sharing data is not sufficient)",
        "shared-datasets entry",
        _imagenette_lowest_source,
    ),
    Claim(
        "tier_low",
        "within low_level mean exceeds the cross-tier mean",
        "two-cluster entry, tier table",
        _tier_margin("low_level"),
    ),
    Claim(
        "tier_mid",
        "within mid_level mean exceeds the cross-tier mean",
        "two-cluster entry, tier table",
        _tier_margin("mid_level"),
    ),
    Claim(
        "tier_high",
        "within high_level mean exceeds the cross-tier mean (published: 'within noise')",
        "two-cluster entry, tier table",
        _tier_margin("high_level"),
    ),
)


def draws(data: Data, count: int, seed: int) -> list[View]:
    """``count`` views, each assigning every board one of its seeds independently."""
    rng = random.Random(seed)
    tasks = sorted(data.seeded)
    out = []
    for _ in range(count):
        pick = {t: rng.choice(SEEDS) for t in tasks}
        # The box half is the same runs as the mask half, so it takes the same seed.
        pick[BOX] = pick["instance_segmentation"]
        out.append(View({t: data.seeded[t][pick[t]] for t in tasks}, data.levels))
    return out


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def evaluate_claims(data: Data, views: list[View]) -> list[dict]:
    published = View(data.published, data.levels)
    mean = View(data.seed_mean(), data.levels)
    rows = []
    for claim in CLAIMS:
        values = [claim.margin(v) for v in views]
        rows.append(
            {
                "key": claim.key,
                "text": claim.text,
                "where": claim.where,
                "published": claim.margin(published),
                "seed_mean": claim.margin(mean),
                "low": _quantile(values, 0.05),
                "high": _quantile(values, 0.95),
                "min": min(values),
                "holds": sum(1 for x in values if x > 0) / len(values),
            }
        )
    return rows


def verdict(row: dict) -> str:
    if row["holds"] == 1.0 and row["seed_mean"] > 0:
        return "robust"
    if row["holds"] >= 0.95 and row["seed_mean"] > 0:
        return "holds"
    if row["seed_mean"] > 0 and row["holds"] >= 0.5:
        return "FRAGILE"
    return "FAILS"


# ---------------------------------------------------------------- the matrix


def pair_values(data: Data, a: str, b: str) -> list[float]:
    """Every value rho(a, b) can take under a re-fit: 25, 5 or 1 combinations."""
    sa = SEEDS if a not in ZERO_SHOT else (0,)
    sb = SEEDS if b not in ZERO_SHOT else (0,)
    return [spearman(data.seeded[a][i], data.seeded[b][j]) for i in sa for j in sb]


def matrix(data: Data) -> list[dict]:
    mean = View(data.seed_mean(), data.levels)
    tasks = data.tasks
    rows = []
    for i, a in enumerate(tasks):
        for b in tasks[i + 1 :]:
            values = pair_values(data, a, b)
            pub = spearman(data.published[a], data.published[b])
            rows.append(
                {
                    "pair": (a, b),
                    "published": pub,
                    "seed_mean": mean.rho(a, b),
                    "min": min(values),
                    "max": max(values),
                    "width": max(values) - min(values),
                    # Where the published draw sits in its own distribution,
                    # midranked: 0.5 is the middle, 1.0 the top.
                    "percentile": (
                        sum(1 for x in values if x < pub) + 0.5 * sum(1 for x in values if x == pub)
                    )
                    / len(values),
                    "n": len(values),
                    "sign_flips": min(values) < 0 < max(values),
                }
            )
    return rows


def report_matrix(rows: list[dict], *, verbose: bool) -> None:
    varying = [r for r in rows if r["n"] > 1]
    widths = [r["width"] for r in varying]
    print("\n=== every board pair under a re-fit ===")
    print(f"  {len(rows)} pairs; {len(varying)} can move (the rest are zero-shot on both sides)")
    print(
        f"  range of rho across re-fits: median width {statistics.median(widths):.3f}, "
        f"max {max(widths):.3f}"
    )
    shift = [r["published"] - r["seed_mean"] for r in varying]
    print(
        f"  published minus seed-mean: mean {statistics.fmean(shift):+.3f}, "
        f"median {statistics.median(shift):+.3f}, |max| {max(abs(x) for x in shift):.3f}"
    )
    pct = [r["percentile"] for r in varying]
    print(
        f"  where the published draw sits in its own range: mean percentile "
        f"{statistics.fmean(pct):.2f} (0.50 is unbiased); "
        f"top fifth on {sum(1 for p in pct if p >= 0.8)}, "
        f"bottom fifth on {sum(1 for p in pct if p <= 0.2)} of {len(pct)}"
    )
    flips = [r for r in varying if r["sign_flips"]]
    print(f"  pairs whose SIGN changes with the seed: {len(flips)} of {len(varying)}")
    strong = [r for r in varying if abs(r["published"]) >= 0.5]
    strong_flip = [r for r in strong if r["sign_flips"]]
    print(f"  ... of those published at |rho| >= 0.5: {len(strong_flip)} of {len(strong)}")

    print("\n  widest ranges:")
    for r in sorted(varying, key=lambda r: -r["width"])[:8]:
        a, b = r["pair"]
        print(
            f"    {r['min']:+.3f} .. {r['max']:+.3f}  (published {r['published']:+.3f})  {a} / {b}"
        )
    if verbose:
        print("\n  every pair:")
        for r in sorted(rows, key=lambda r: r["pair"]):
            a, b = r["pair"]
            print(
                f"    {r['published']:+.3f} pub  {r['seed_mean']:+.3f} mean  "
                f"{r['min']:+.3f} .. {r['max']:+.3f}  p{r['percentile']:.2f}  {a} / {b}"
            )


# ------------------------------------------------------------- structure, fit


def report_structure(data: Data) -> None:
    """Which structural property wins each grid-reading board, seed by seed."""
    print("\n=== which property a grid-reading board ranks by, per seed ===")
    print("  tokens wins = beats both width and pretraining size; 5 seeds per board\n")
    print(f"  {'board':24s} {'published':>9s}  {'tokens wins':>11s}  rho(tokens) range")
    exceptions_per_seed: dict[int, set[str]] = {s: set() for s in SEEDS}
    for task in sorted(correlates.GRID_READING_BOARDS):

        def winner(board: Board) -> tuple[str, float]:
            props = {
                p: spearman(board, correlates._property(p, list(board), 1.0e10))
                for p in ("tokens", "width", "pretrain_images")
            }
            return max(props, key=lambda p: props[p]), props["tokens"]

        pub_winner, _ = winner(data.published[task])
        per_seed = {s: winner(data.seeded[task][s]) for s in SEEDS}
        wins = sum(1 for w, _ in per_seed.values() if w == "tokens")
        toks = [t for _, t in per_seed.values()]
        for s, (w, _) in per_seed.items():
            if w != "tokens":
                exceptions_per_seed[s].add(task)
        print(
            f"  {task:24s} {pub_winner:>9.9s}  {wins:>6d} of 5  "
            f"{min(toks):+.3f} .. {max(toks):+.3f}"
        )
    print("\n  the exceptions, seed by seed (published: detection, semantic_segmentation):")
    for s, ex in exceptions_per_seed.items():
        print(f"    seed {s}: {', '.join(sorted(ex)) or '(none)'}")

    print("\n  the same-image control, seed by seed (margin > 0 is the published reading):")
    for task, sign in (("generic_segmentation", 1), ("semantic_segmentation", -1)):
        margins = []
        for s in SEEDS:
            board = data.seeded[task][s]
            tok = spearman(board, correlates._property("tokens", list(board), 1.0e10))
            dat = spearman(board, correlates._property("pretrain_images", list(board), 1.0e10))
            margins.append(sign * (tok - dat))
        label = "tokens - pretraining" if sign > 0 else "pretraining - tokens"
        print(f"    {task:24s} {label}: {min(margins):+.3f} .. {max(margins):+.3f}")


def report_fit(data: Data) -> None:
    """The `--section fit` claim of the audited script, seed by seed.

    Its two halves behave differently under a re-fit, which is the point of
    printing them per seed: the sign of rho is a property of the whole board,
    while "the single worst fit" is one row and moves with it.
    """
    print("\n=== does the fit track the grid, on every seed? ===")
    boards = sorted(correlates.GRID_READING_BOARDS)
    negative = total = 0
    rhos: list[float] = []
    for s in SEEDS:
        coarsest = 0
        for task in boards:
            losses = data.fits[task][s]
            tokens = {b: float(correlates.STRUCTURE[b].tokens) for b in losses}
            rho = spearman(tokens, losses)
            rhos.append(rho)
            negative += rho < 0
            total += 1
            worst = max(losses, key=lambda b: losses[b])
            coarsest += tokens[worst] == min(tokens.values())
        print(f"  seed {s}: worst fit on the coarsest grid on {coarsest} of {len(boards)} boards")
    print(
        f"  rho(tokens, train_loss) negative on {negative} of {total} board-seeds "
        f"(range {min(rhos):+.3f} .. {max(rhos):+.3f})"
    )


def report_claims(rows: list[dict], count: int) -> None:
    print(f"=== the published correlational claims, over {count} draws ===")
    print("  margin > 0 means the published reading holds; 'holds' is the share of draws\n")
    print(
        f"  {'claim':26s} {'published':>9s} {'seed-mean':>9s} {'p5..p95':>17s} "
        f"{'holds':>6s}  verdict"
    )
    for r in rows:
        mark = " *" if "scene_classification" in r["text"] or r["key"] == "two_clusters" else "  "
        print(
            f"{mark}{r['key']:26s} {r['published']:+9.3f} {r['seed_mean']:+9.3f} "
            f"{r['low']:+8.3f}..{r['high']:+.3f} {r['holds']:6.1%}  {verdict(r)}"
        )
    print("\n  * reads scene_classification, whose sweep is held out (see the docstring)")
    print("  robust: every draw and the seed-mean; holds: >= 95%; FRAGILE: a majority;")
    print("  FAILS: the seed-mean board does not show it\n")
    for r in rows:
        print(f"    {r['key']:26s} {r['text']}  [{r['where']}]")


def report_partners(data: Data, views: list[View], task: str) -> None:
    """Who a board's closest partner is across draws -- a superlative, not a margin."""
    counts: dict[str, int] = defaultdict(int)
    others = [t for t in data.tasks if t != task]
    for view in views:
        counts[max(others, key=lambda t: view.rho(task, t))] += 1
    mean = View(data.seed_mean(), data.levels)
    top = sorted(others, key=lambda t: -mean.rho(task, t))[:3]
    print(f"\n=== {task}: closest partner across {len(views)} draws ===")
    for partner, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n / len(views):6.1%}  {partner}")
    print("  seed-mean: " + ", ".join(f"{t} {mean.rho(task, t):+.3f}" for t in top))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=22)
    parser.add_argument(
        "--section",
        choices=("claims", "pairs", "structure", "fit", "all"),
        default="all",
    )
    parser.add_argument("--pairs", action="store_true", help="print every board pair")
    args = parser.parse_args(argv)

    data = load()
    if args.section in ("claims", "all"):
        views = draws(data, args.draws, args.rng_seed)
        report_claims(evaluate_claims(data, views), args.draws)
        report_partners(data, views, "vehicle_classification")
    if args.section in ("pairs", "all"):
        report_matrix(matrix(data), verbose=args.pairs)
    if args.section in ("structure", "all"):
        report_structure(data)
    if args.section in ("fit", "all"):
        report_fit(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
