"""Deciding which result records may be compared, and ranking the ones that may.

The schema has carried the fields a leaderboard needs since v0.1 (see
``schema.py``). What it has never carried is the *rules* — which pairs of
records answer the same question and which merely look alike. Those rules have
lived as prose in ``CLAUDE.md``, which means every table in this repository was
assembled by hand against them, and one of those tables had already drifted by
the time it was noticed.

This module is those rules as code. It deliberately does no rendering and no
I/O beyond what ``writer.py`` already provides: a number that should not have
been put in a table is wrong long before anyone formats it.

The governing idea is **comparability**, not similarity. Two records are
comparable when every choice that decides *what the number means* agrees, so
that the only remaining difference is the thing being ranked. Anything else is
refused rather than silently ranked, because the failure this module exists to
prevent looks exactly like a result.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import astuple, dataclass
from typing import Literal

from visbench.results.schema import ResultRecord

__all__ = [
    "ComparabilityKey",
    "Direction",
    "IncomparableRecords",
    "UnknownMetric",
    "comparability_key",
    "group_comparable",
    "is_context_metric",
    "latest_per_backbone",
    "metric_direction",
    "rank",
    "ranking_disagreements",
    "shared_metrics",
]

Direction = Literal["higher", "lower"]

#: Prefix :meth:`BaseTask.context_metrics` uses for values that travel beside a
#: score without being one. Correspondence's ceiling is the only current case:
#: ``recall@1px`` has a ceiling of 0.015 on DINOv2 ViT-S/14 at 224px, so the
#: score alone says the wrong thing — but ranking backbones *by* the ceiling
#: would rank the geometry of the split, not the representation.
CONTEXT_PREFIX = "ceiling_"

#: Which direction is better, per metric name.
#:
#: Names are deliberately listed rather than inferred from a suffix. ``mean``
#: and ``median`` are surface-normal *angular error* in degrees, where lower is
#: better; nothing about the word says so, and a heuristic that guessed from
#: "mean" would get it backwards and produce a leaderboard ranked upside down.
#:
#: Several names are shared by tasks that measure different things in different
#: units — ``rmse`` is metres for depth, degrees for surface normals and target
#: units for a magnitude probe; ``d1`` is a depth ratio threshold and an angular
#: one. They agree on *direction*, which is all this table claims. Keeping them
#: apart is :func:`comparability_key`'s job, and it does it by requiring the
#: task to match.
METRIC_DIRECTIONS: dict[str, Direction] = {
    # classification, retrieval, similarity
    "top1": "higher",
    "top5": "higher",
    "accuracy": "higher",
    "precision": "higher",
    "recall": "higher",
    "f1": "higher",
    # segmentation
    "iou": "higher",
    "binary_iou": "higher",
    "miou": "higher",
    "miou_per_image": "higher",
    "pixel_acc": "higher",
    "mean_acc": "higher",
    # depth and surface normals
    "d1": "higher",
    "d2": "higher",
    "d3": "higher",
    "abs_rel": "lower",
    "mean": "lower",
    "median": "lower",
    # dense regression, shared by depth, normals and the magnitude probes
    "rmse": "lower",
    "mae": "lower",
    # magnitude probes
    "correlation": "higher",
    "edge_correlation": "higher",
    "keypoint_correlation": "higher",
    "occlusion_edge_correlation": "higher",
    # detection
    "map_50": "higher",
    "map_50_95": "higher",
}

#: Metrics that are diagnostics rather than scores, so ranking on one is
#: refused. These are not context metrics — they carry no ``ceiling_`` prefix
#: and a task returns them from ``evaluate`` like any other number.
#:
#: ``classes_scored`` is the sharp case. It is mAP's actual denominator, and it
#: is **not always** ``num_classes``: a class with no non-difficult objects
#: scores ``None`` and is excluded. So two detection runs whose
#: ``classes_scored`` differ have divided by different numbers, and comparing
#: their mAP compares two different averages. :func:`rank` checks it rather
#: than ranking it.
DIAGNOSTIC_METRICS: frozenset[str] = frozenset(
    {
        "classes_scored",
        "tie_rate",
        "num_corr",
    }
)

#: Diagnostics that must *agree* across a comparison, because they say what the
#: score averaged over. A mismatch is refused, not reported alongside.
GUARD_METRICS: tuple[str, ...] = ("classes_scored",)


class IncomparableRecords(ValueError):
    """Raised when records that do not answer the same question are compared."""


class UnknownMetric(KeyError):
    """Raised for a metric with no recorded direction.

    Deliberately fatal rather than defaulted to "higher is better". A new
    lower-is-better metric defaulting the wrong way produces a leaderboard
    ranked precisely backwards, which reads as a surprising finding rather than
    as a bug.
    """


@dataclass(frozen=True)
class ComparabilityKey:
    """Everything that must agree before two records may be ranked together.

    The backbone is deliberately **not** part of this key — it is the thing
    being compared. Everything else is.

    Attributes
    ----------
    task / level:
        Which probe. This is also what keeps ``rmse``-in-metres apart from
        ``rmse``-in-degrees, since both are recorded under the same name.
    protocol:
        From ``task_params["protocol"]``, or ``None`` for a task that declares
        none. Two probes can read the same dataset under the same task name and
        still not be comparable — v0.4.0 shipped a CLI that let ``edge`` record
        a keypoint number under ``visbench_edge_regression``.
    dataset / split / dataset_fingerprint:
        The data. The fingerprint is what distinguishes two runs over folders
        that share a name, and a limited split from a full one.
    finetuned / finetune_blocks / finetune_backbone_lr:
        A frozen score asks what a representation already carries; a fine-tuned
        one asks what it can be adapted into. Ranking across the two is
        meaningless, and this is the field that makes it impossible.
        ``trainable_params`` is deliberately excluded — it differs between
        ViT-S and ViT-B for the *same* setting, so including it would make two
        legitimately comparable runs look incomparable.
    pooling / feature_mode / layers:
        What representation the task asked the backbone for.
    task_params / dataset_params:
        Everything else that decides what the number means, as sorted tuples so
        the key stays hashable. Conservative on purpose: ``dataset_params``
        carries ``target_transform``, which is what separates the ``log1p``
        occlusion-edge number from a linear-space one, and ``task_params``
        carries the training budget, which is what separates a probe trained for
        forty epochs from one trained for ten. Neither can be ranked against
        the other, and enumerating "the settings that matter" would mean this
        module needed editing every time a task grew one.
    """

    task: str
    level: str
    protocol: str | None
    dataset: str
    split: str
    dataset_fingerprint: str | None
    finetuned: bool
    finetune_blocks: int | None
    finetune_backbone_lr: float | None
    pooling: str
    feature_mode: str
    layers: tuple[int, ...] | None
    task_params: tuple[tuple[str, str], ...]
    dataset_params: tuple[tuple[str, str], ...]

    def short_id(self) -> str:
        """Stable 8-hex digest of the whole key.

        Exists because :meth:`describe` is lossy and cannot stop being: it names
        the handful of fields a reader cares about, while the key separates on
        every setting that changes the number's meaning. The real corpus already
        contains a pair this matters for — two `edge` groups identical in task,
        dataset, split, protocol and frozen-ness, differing only in
        ``target_scale`` (65535 against 1000, from 6d-1's sweep, where the first
        scored 0.047 and the second 0.456). Described alone they read as one
        group listed twice.

        A digest rather than a diff because a key has no privileged summary —
        which field differs depends on which two keys you hold.
        """
        payload = repr(astuple(self)).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=4).hexdigest()

    def describe(self, *, with_id: bool = True) -> str:
        """One-line human summary, for error messages and group headings.

        Lossy by design — see :meth:`short_id`, which is appended so two groups
        are never described identically even when their readable parts agree.
        Pass ``with_id=False`` when the caller has already disambiguated.
        """
        parts = [f"{self.task} on {self.dataset}/{self.split}"]
        if self.protocol:
            parts.append(f"protocol={self.protocol}")
        if self.finetuned:
            parts.append(f"finetuned={self.finetune_blocks} blocks")
        else:
            parts.append("frozen")
        summary = ", ".join(parts)
        return f"{summary} [{self.short_id()}]" if with_id else summary


def _freeze(params: dict) -> tuple[tuple[str, str], ...]:
    """Sorted, hashable, comparable-by-value form of an open params dict.

    Values are stringified rather than kept as-is because these dicts hold
    lists (``iou_thresholds``, ``layers``) and nested dicts, neither of which
    is hashable, and because ``1`` and ``1.0`` arriving from JSON should not
    split a group. ``repr`` is stable within a run and that is all this needs —
    the value is never read back out, only compared.
    """
    return tuple(sorted((str(key), repr(value)) for key, value in params.items()))


def comparability_key(record: ResultRecord, *, ignore: Iterable[str] = ()) -> ComparabilityKey:
    """The key under which ``record`` may be ranked against others.

    Parameters
    ----------
    record:
        Any record, of any schema version. Fields a record predates come back
        as ``None`` and simply group with other records that also lack them,
        which is the additive-only schema working as intended.
    ignore:
        ``task_params`` / ``dataset_params`` keys to leave out of the key. The
        escape hatch for a setting that provably does not change what the number
        means — ``batch_size`` on a deterministic evaluation, say. Use it
        knowingly: every name passed here is a claim that two runs differing in
        it are still measuring the same thing.
    """
    ignored = set(ignore)
    task_params = {k: v for k, v in record.task_params.items() if k not in ignored}
    dataset_params = {k: v for k, v in record.dataset_params.items() if k not in ignored}

    finetune = record.finetune or {}
    return ComparabilityKey(
        task=record.task,
        level=record.level,
        protocol=record.task_params.get("protocol"),
        dataset=record.dataset,
        split=record.split,
        dataset_fingerprint=record.dataset_fingerprint,
        finetuned=record.finetune is not None,
        finetune_blocks=finetune.get("blocks"),
        finetune_backbone_lr=finetune.get("backbone_lr"),
        pooling=record.pooling,
        feature_mode=record.feature_mode,
        layers=tuple(record.layers) if record.layers is not None else None,
        task_params=_freeze(task_params),
        dataset_params=_freeze(dataset_params),
    )


def group_comparable(
    records: Iterable[ResultRecord], *, ignore: Iterable[str] = ()
) -> dict[ComparabilityKey, list[ResultRecord]]:
    """Partition records into groups that may be ranked internally.

    Insertion order is preserved within each group, so a caller that read a
    JSONL file gets its runs back in the order they were written.
    """
    grouped: dict[ComparabilityKey, list[ResultRecord]] = defaultdict(list)
    for record in records:
        grouped[comparability_key(record, ignore=ignore)].append(record)
    return dict(grouped)


def is_context_metric(name: str) -> bool:
    """Whether ``name`` is context travelling beside a score rather than a score.

    ``run()`` merges :meth:`BaseTask.context_metrics` into the same flat dict as
    the scores, under a prefix, and refuses a collision. A leaderboard must not
    rank on one: correspondence's ``ceiling_recall@1px`` measures how much of
    the split is *recoverable at all*, which is a property of the data.
    """
    return name.startswith(CONTEXT_PREFIX)


def metric_direction(name: str) -> Direction:
    """``"higher"`` or ``"lower"``, or raise :class:`UnknownMetric`.

    Context metrics raise too. They have a direction in the arithmetic sense
    and no meaning as a ranking, which is the more useful thing to refuse.
    """
    if is_context_metric(name):
        raise UnknownMetric(
            f"{name!r} is context, not a score: it describes the split rather than "
            "the representation, so ranking on it ranks the data."
        )
    if name in DIAGNOSTIC_METRICS:
        raise UnknownMetric(
            f"{name!r} is a diagnostic, not a score. Ranking on it would order runs "
            "by how much was measurable rather than by how well anything did."
        )
    try:
        return METRIC_DIRECTIONS[name]
    except KeyError:
        raise UnknownMetric(
            f"No recorded direction for metric {name!r}. Add it to "
            "METRIC_DIRECTIONS rather than defaulting: a lower-is-better metric "
            "assumed higher-is-better ranks the leaderboard backwards, and that "
            "reads as a finding rather than a bug."
        ) from None


def shared_metrics(records: Iterable[ResultRecord]) -> list[str]:
    """Rankable metric names present on *every* record, in sorted order.

    Context and diagnostic metrics are excluded, as are metrics with no
    recorded direction — this is the "what can I rank these on" question, so a
    name that cannot be ranked is not an answer to it.
    """
    records = list(records)
    if not records:
        return []
    common: set[str] = set(records[0].metrics)
    for record in records[1:]:
        common &= set(record.metrics)
    rankable = []
    for name in common:
        try:
            metric_direction(name)
        except UnknownMetric:
            continue
        rankable.append(name)
    return sorted(rankable)


def _check_comparable(records: list[ResultRecord], *, ignore: Iterable[str]) -> None:
    keys = {comparability_key(record, ignore=ignore) for record in records}
    if len(keys) > 1:
        described = sorted({comparability_key(r, ignore=ignore).describe() for r in records})
        raise IncomparableRecords(
            "These records do not answer the same question and must not be ranked "
            "together:\n  " + "\n  ".join(described)
        )


def _check_guards(records: list[ResultRecord], metric: str) -> None:
    for guard in GUARD_METRICS:
        values = {record.metrics[guard] for record in records if guard in record.metrics}
        if len(values) > 1:
            raise IncomparableRecords(
                f"{guard!r} differs across these records ({sorted(values)}), so their "
                f"{metric!r} values are averages over different denominators. "
                "Ranking them would compare two different quantities."
            )


def rank(
    records: Iterable[ResultRecord],
    metric: str,
    *,
    ignore: Iterable[str] = (),
) -> list[tuple[ResultRecord, float]]:
    """Records sorted best-first on ``metric``, or raise.

    Raises
    ------
    IncomparableRecords
        If the records do not share a :class:`ComparabilityKey`, or if a guard
        metric disagrees across them.
    UnknownMetric
        If ``metric`` has no recorded direction, or is context or a diagnostic.
    KeyError
        If any record lacks ``metric``. Skipping those would quietly rank a
        subset and present it as the whole comparison.

    Ties keep their input order, since :func:`sorted` is stable — two backbones
    that genuinely tie should not be given an ordering by accident.
    """
    records = list(records)
    if not records:
        return []

    direction = metric_direction(metric)
    _check_comparable(records, ignore=ignore)

    missing = [r.backbone for r in records if metric not in r.metrics]
    if missing:
        raise KeyError(
            f"{metric!r} is missing from records for {sorted(set(missing))}. Ranking "
            "the rest would present a partial comparison as a complete one."
        )
    _check_guards(records, metric)

    return sorted(
        ((record, float(record.metrics[metric])) for record in records),
        key=lambda pair: pair[1],
        reverse=direction == "higher",
    )


def ranking_disagreements(
    records: Iterable[ResultRecord],
    *,
    metrics: Iterable[str] | None = None,
    ignore: Iterable[str] = (),
) -> dict[tuple[str, str], tuple[list[str], list[str]]]:
    """Pairs of metrics that order the same records differently.

    Returns ``{(metric_a, metric_b): (order_a, order_b)}`` for every pair whose
    backbone ordering differs, where each order is a list of backbone names
    best-first.

    This exists because **a task can disagree with itself**. On Taskonomy
    normals, DINOv2-S wins on mean angular error while DINOv2-B wins on the
    11.25-degree threshold; quoting one and dropping the other manufactures a
    result. A leaderboard that picks a headline metric silently will do exactly
    that, so the disagreement has to be visible to whatever renders it.

    An empty dict means every metric agrees, which is a real answer and not the
    absence of one.
    """
    records = list(records)
    if len(records) < 2:
        return {}

    names = sorted(metrics) if metrics is not None else shared_metrics(records)
    orders: dict[str, list[str]] = {}
    for name in names:
        orders[name] = [record.backbone for record, _ in rank(records, name, ignore=ignore)]

    disagreements: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if orders[first] != orders[second]:
                disagreements[(first, second)] = (orders[first], orders[second])
    return disagreements


def latest_per_backbone(records: Iterable[ResultRecord]) -> list[ResultRecord]:
    """One record per backbone, keeping the newest by ``timestamp``.

    A results file accumulates re-runs — 6a alone wrote five records for one
    VOC configuration while chasing a timing question. Ranking them all would
    list one backbone several times and let a repeat outvote a single run.

    Timestamps are ISO 8601 UTC (``utc_timestamp``), so lexicographic ordering
    is chronological. Ties keep the last seen, matching "append-only file, later
    line wins".
    """
    newest: dict[str, ResultRecord] = {}
    for record in records:
        current = newest.get(record.backbone)
        if current is None or record.timestamp >= current.timestamp:
            newest[record.backbone] = record
    return list(newest.values())
