"""Markdown tables built from result records.

``leaderboard.py`` decides what may be compared and in what order; this module
is the only place that turns an answer into text. The split is deliberate and
one-directional: a number that should not have gone in a table is wrong long
before anyone formats it, so nothing here may relax a rule from there.

Everything a table asserts therefore comes from the records themselves. There is
no backbone metadata table, no "known good" number, and no per-task special
case beyond two listed dicts — :data:`HEADLINE_METRICS`, which says what a board
is ordered by, and :data:`CAVEATS`, which says what a reader must know before
believing the ordering. Both are listed rather than inferred, for the reason
``METRIC_DIRECTIONS`` is: a heuristic that guessed wrong would produce a table
that reads as a finding rather than a bug.
"""

from collections.abc import Iterable, Sequence

from visbench.results.leaderboard import (
    CONTEXT_PREFIX,
    DIAGNOSTIC_METRICS,
    ComparabilityKey,
    UnknownMetric,
    group_comparable,
    is_context_metric,
    latest_per_backbone,
    metric_direction,
    rank,
    ranking_disagreements,
    shared_metrics,
)
from visbench.results.schema import ResultRecord

__all__ = [
    "CAVEATS",
    "COUNT_METRICS",
    "HEADLINE_METRICS",
    "board_columns",
    "render_board",
    "render_leaderboard",
]

#: What each board is ordered by. Listed, never guessed.
#:
#: A renderer that picked a metric silently would manufacture results: `edge`
#: orders its three metrics three different ways (``edge_correlation`` ranks
#: DINOv2-S first, ``mae`` ranks DINOv2-B first, ``rmse`` ranks DINOv2-S again),
#: so "the winner" is a property of the choice, not of the data. Naming the
#: choice here is what lets :func:`render_board` say so in the output.
#:
#: These are the metrics each probe's own documentation leads with, so a board
#: and the prose around it cannot disagree about what was being measured.
HEADLINE_METRICS: dict[str, str] = {
    "classification": "top1",
    "scene_classification": "top1",
    "fine_grained_classification": "top1",
    "retrieval": "mAP",
    # Pixels, not patch widths: a patch is a different physical distance on
    # every backbone, so `recall@1p` ranked this board upside down until v0.6.1.
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
    "orientation": "orientation_error",
    "occlusion_edge": "occlusion_edge_correlation",
}

#: What a reader must know before reading a board's ordering as a ranking.
#:
#: Keyed by task, and deliberately not by anything cleverer: a caveat is a claim
#: about a protocol, and inferring one would mean inventing the claim.
CAVEATS: dict[str, str] = {
    "correspondence": (
        "Thresholds are in **pixels**, which is the only unit two backbones can "
        "be compared in — a patch width is 14px on DINOv2/14 and 32px on a "
        "ResNet, so scoring in patch widths asks each backbone a different "
        "question. Read `ceiling_` beside every score: a 7x7 grid cannot place "
        "a match within 5px more than ~10% of the time whatever its features "
        "are, so part of this ordering is resolution rather than quality. "
        "`num_matches` is the denominator each backbone's own ratio test left, "
        "and it varies by more than 5x."
    ),
    "detection": (
        "Absolute mAP is low by design: the head is anchor-free and "
        "single-scale, so it has no feature pyramid and small objects fall "
        "between cells. The board ranks representations, which is what it is "
        "for — it is not a detector benchmark."
    ),
    "scene_classification": (
        "This is *scene* category, not object category — a distinct question "
        "from the `classification` board, and a backbone's rank can move "
        "between the two. Places365 scenes overlap what ImageNet-supervised "
        "backbones already saw, so for those the number is closer to "
        "in-distribution recall than transfer."
    ),
    "fine_grained_classification": (
        "This is *subordinate* category — 200 bird species that share a body "
        "plan — not the basic-level question the `classification` board asks, "
        "and a backbone's rank moves a long way between the two. **The "
        "in-distribution confound that shapes the object board does not carry "
        "over here**, which was measured rather than assumed: ImageNet-1k "
        "holds 59 bird classes, so the four ImageNet-1k-*supervised* backbones "
        "were expected to be flattered, and instead they take places 8, 9, 10 "
        "and 11 of twelve — `convnext_base`, `resnet50`, `supervised_vitb16`, "
        "`resnet18`, above only `mae_vitb16`. The controlled comparison says "
        "the same thing: among the four ViT-B/16 models, the supervised one is "
        "second-to-last, behind both `sam_vitb16` and `dino_vitb16`. "
        "Basic-level supervision appears to discard the within-class variation "
        "this board asks about. The probe also does **not** underfit despite "
        "200 classes over ~6k training images — `train_top1` is 1.0000 on all "
        "six backbones it was measured on directly, including the board's last "
        "place — so the spread is generalisation and a low score is a property "
        "of the representation."
    ),
}


#: Metrics that count things, and so have no fractional part worth showing.
#:
#: Listed, rather than "any diagnostic" or "any integral value", because both of
#: those get it wrong on a real board. `tie_rate` is a diagnostic and a *rate* —
#: it renders as ``0`` under the first rule when no triplet tied, which reads as
#: a count. A ``ceiling_recall@4p`` of exactly 1.0 is saturated, not a count, and
#: falls foul of the second. Only `num_matches` and `classes_scored` are
#: genuinely integers; `detections_per_image` is a mean and keeps its decimals.
COUNT_METRICS: frozenset[str] = frozenset({"num_matches", "classes_scored"})


def _format(name: str, value: float) -> str:
    """Four decimals, except counts, which have no fractional part to show."""
    if name in COUNT_METRICS and float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.4f}"


def board_columns(records: Sequence[ResultRecord]) -> tuple[list[str], list[str]]:
    """``(rankable, context_and_diagnostic)`` metric names for one group.

    The second list is what makes a board honest rather than merely ordered.
    Correspondence's ``num_matches`` is the denominator its own score is
    averaged over and detection's ``classes_scored`` is the denominator of its
    mAP; both are refused by :func:`rank` — correctly, since ranking on them
    would order runs by how much was measurable — but a table that omits them
    presents a comparison whose terms differ as though they did not.

    Present on *every* record or not shown at all: a column with a hole in it
    invites reading the gap as a zero.
    """
    ranked = shared_metrics(records)
    if not records:
        return [], []

    common: set[str] = set(records[0].metrics)
    for record in records[1:]:
        common &= set(record.metrics)

    extra = sorted(
        name
        for name in common
        if name not in ranked and (is_context_metric(name) or name in DIAGNOSTIC_METRICS)
    )
    return ranked, extra


def render_board(
    records: Iterable[ResultRecord],
    *,
    key: ComparabilityKey | None = None,
    heading_level: int = 3,
    metrics: Sequence[str] | None = None,
) -> str:
    """One comparability group as a markdown table, plus what qualifies it.

    Rows are ordered by the task's :data:`HEADLINE_METRICS` entry and the best
    cell in each *rankable* column is bolded — per column, so a board whose
    metrics disagree shows the disagreement instead of hiding it behind the row
    order. Context and diagnostic columns are never bolded, because "best" is
    not defined for them.

    ``metrics`` narrows the rankable columns, for a README where correspondence's
    sixteen threshold-by-metric combinations do not fit. It narrows **only**
    those: the context and diagnostic columns always survive, since they are
    what qualifies the numbers beside them, and a caller trimming for width must
    not be able to drop the denominator a score is an average over. The
    disagreement note is likewise computed over every shared metric, not the
    selection, so narrowing a board cannot quietly make it look consistent.

    Raises
    ------
    UnknownMetric
        If the task has no declared headline metric, or its records do not carry
        it. Both mean the board cannot be ordered, and ordering it by whatever
        happened to sort first is how a table starts asserting something nobody
        chose.
    """
    records = latest_per_backbone(records)
    if not records:
        return ""

    task = records[0].task
    try:
        headline = HEADLINE_METRICS[task]
    except KeyError:
        raise UnknownMetric(
            f"No headline metric declared for task {task!r}. Add it to "
            "HEADLINE_METRICS rather than letting the renderer pick: a board "
            "ordered by whichever metric sorted first asserts a ranking nobody "
            "chose, and reads as a finding rather than a bug."
        ) from None

    available, extra = board_columns(records)
    if headline not in available:
        raise UnknownMetric(
            f"Task {task!r} is ordered by {headline!r}, which is not present on "
            f"every record in this group (have: {available}). A board missing "
            "its own headline cannot be ordered."
        )

    if metrics is None:
        ranked = available
    else:
        unknown = [name for name in metrics if name not in available]
        if unknown:
            raise UnknownMetric(f"Not rankable on this board: {unknown}. Available: {available}.")
        # The headline is what the rows are ordered by, so a selection that
        # omitted it would leave the order unexplained by any visible column.
        ranked = [name for name in metrics if name != headline]
        ranked.insert(0, headline)
        # A ceiling qualifies the one score it is the ceiling *of*, so carrying
        # the ceilings of scores this board no longer shows is noise. Diagnostics
        # are not dropped: `num_matches` is the denominator of every recall here,
        # not of any single one.
        extra = [
            name
            for name in extra
            if name in DIAGNOSTIC_METRICS or name.removeprefix(CONTEXT_PREFIX) in ranked
        ]

    ordered = [record for record, _ in rank(records, headline)]

    # Winner per rankable column, by that column's own direction. Computed
    # against the metric, never against row position -- the two disagree
    # whenever a board's metrics do, which is the case worth showing.
    best: dict[str, float] = {}
    for name in ranked:
        values = [record.metrics[name] for record in ordered]
        best[name] = max(values) if metric_direction(name) == "higher" else min(values)

    columns = ["backbone", *ranked, *extra]
    lines = [
        f"{'#' * heading_level} {task}",
        "",
        "| " + " | ".join(f"`{c}`" if c != "backbone" else c for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in ordered:
        cells = [f"`{record.backbone}`"]
        for name in ranked:
            text = _format(name, record.metrics[name])
            cells.append(f"**{text}**" if record.metrics[name] == best[name] else text)
        cells.extend(_format(name, record.metrics[name]) for name in extra)
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", f"Ordered by `{headline}`."])

    # Over every shared metric, not the selection: narrowing a board for width
    # must not be able to make it look more consistent than it is.
    disagreements = ranking_disagreements(ordered)
    involving = sorted(
        {b for a, b in disagreements if a == headline}
        | {a for a, b in disagreements if b == headline}
    )
    if involving:
        names = ", ".join(f"`{name}`" for name in involving)
        lines[-1] = (
            f"Ordered by `{headline}`, which **disagrees with {names}** — this "
            "task does not rank its backbones the same way twice, so the row "
            "order is one of several defensible ones."
        )
    elif disagreements:
        lines[-1] += (
            " Some other metric pairs on this board disagree with each other, "
            "though not with the ordering."
        )

    if task in CAVEATS:
        lines.extend(["", f"> **Read this first.** {CAVEATS[task]}"])

    if key is not None:
        lines.extend(["", f"<sub>{key.describe()}</sub>"])
    return "\n".join(lines)


def render_leaderboard(
    records: Iterable[ResultRecord],
    *,
    heading_level: int = 3,
) -> str:
    """Every comparability group in a corpus, one board each.

    Groups are emitted in a fixed order — level, then task, then the key's
    digest — so regenerating an unchanged corpus produces an unchanged file and
    a diff means a number moved.

    A group holding one backbone is still rendered. It ranks nothing, but it
    says that a run exists, and silently dropping it would make a corpus with a
    missing backbone look like a corpus that never had one.
    """
    groups = group_comparable(records)
    levels = {"low_level": 0, "mid_level": 1, "high_level": 2}
    ordered_keys = sorted(
        groups,
        key=lambda k: (levels.get(k.level, 99), k.level, k.task, k.short_id()),
    )
    boards = [
        render_board(groups[key], key=key, heading_level=heading_level) for key in ordered_keys
    ]
    return "\n\n".join(board for board in boards if board)
