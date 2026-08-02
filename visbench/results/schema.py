"""The structured result record — one schema from the start.

Every task run logs a record with this shape so that leaderboard tooling never
needs a retrofit (CLAUDE.md, "Engineering conventions"). Fields are added only
additively; renaming or removing one is a breaking change to every historical
record.
"""

from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import datetime, timezone

__all__ = ["ResultRecord", "SCHEMA_VERSION", "utc_timestamp"]

#: Bumped whenever the record shape changes, so consumers can migrate.
#:
#: History
#: -------
#: 1. Initial schema.
#: 2. Added ``dataset_size`` and ``dataset_fingerprint``. Without them, two
#:    runs over different folders that happen to share a name produced records
#:    that were byte-identical and meant different things.
#: 3. Added ``task_params``. Trained probes have hyperparameters, and an
#:    accuracy reported without the optimiser settings that produced it is not
#:    reproducible. Deliberately one open dict rather than a column per
#:    setting, so a new task never forces another schema bump.
#: 4. Added ``layers``, for multi-layer extraction. A dense task reading four
#:    backbone depths is a different run from one reading the last, and the
#:    existing ``layer`` field cannot say so without changing its type — which
#:    would make every v1-v3 record on disk parse to something new.
#: 5. Added ``dataset_params``. The dataset half of a run had no equivalent of
#:    ``task_params``, so settings that decide what the number means were
#:    recorded nowhere: a correspondence run's ``max_warp``, or the
#:    ``image_size`` a dense split was cropped to. They changed the
#:    fingerprint, so two such runs were distinguishable — but only as "not the
#:    same data", with nothing saying how they differed.
#: 6. Added ``finetune``, for v0.3's opt-in unfreezing. A fine-tuned score and a
#:    frozen one answer different questions — "what does this representation
#:    already carry" against "what can it be adapted into" — and until now they
#:    would have sat in one JSONL under one task name with nothing to separate
#:    them. ``None`` means frozen, which is what every record written before
#:    this carries by absence, so no reader needs a version check to ask.
#: 7. Added ``pooling_requested``. ``pooling`` is recorded **resolved**, because
#:    the literal word "default" does not say what produced the number — and
#:    that is still right for interpreting a record. But it is wrong for
#:    *comparing* two: ``default`` resolves to ``cls`` on a ViT and ``mean`` on
#:    a CNN, so a leaderboard keyed on the resolved value put ResNets in a
#:    different group from ViTs and could never rank the two against each other.
#:    Found by widening the corpus to six backbones, where four of the twelve
#:    probes split in half. The request is the protocol; the resolution is a
#:    property of the backbone, so both are needed and neither replaces the
#:    other. ``None`` on every earlier record, where the resolved value is the
#:    only thing that was ever known.
SCHEMA_VERSION = 7


@dataclass
class ResultRecord:
    """One task run on one backbone over one dataset.

    Attributes
    ----------
    backbone / backbone_key:
        Registered name, and the weights-and-resolution identifier from
        :meth:`BaseBackbone.cache_key`. Both, because the name alone does not
        pin down what actually ran.
    task / level:
        Registered task name and its level (high/mid/low).
    dataset / split:
        Dataset identifier plus split.
    dataset_size / dataset_fingerprint:
        How many images, and a short hash of the file list from
        :meth:`BaseDataset.fingerprint`. The name alone does not identify data
        — two folders can share one — so without these, a run before and after
        the images changed produces indistinguishable records.
    pooling / feature_mode:
        Exactly what representation produced the number — without these two the
        metrics are not reproducible. ``pooling`` is **resolved**: the literal
        ``"default"`` means CLS on a ViT and mean over the grid on a CNN, so it
        would not say what ran.
    pooling_requested:
        What the task actually asked for, before resolution — usually
        ``"default"``. Kept beside the resolved value because the two answer
        different questions, and a leaderboard needs this one: two backbones
        asked for ``default`` ran the *same protocol* even though one resolved
        to ``cls`` and the other to ``mean``, whereas two runs that named ``cls``
        and ``mean`` explicitly did not. ``None`` on schema v6 and earlier,
        where only the resolved value was ever recorded.
    metrics:
        The flat dict returned by :meth:`BaseTask.evaluate`.
    seed:
        From :func:`visbench.utils.set_seed`, which returns the seed it used
        even when the caller passed ``None``. Irrelevant for zero-shot tasks;
        required to reproduce anything that trains.
    task_params:
        Task-specific settings from :meth:`BaseTask.describe` — for a trained
        probe, the optimiser, learning rate, epochs and so on. Empty for
        zero-shot tasks, which have none.
    dataset_params:
        Whatever :meth:`BaseDataset.describe` returned beyond the fields above
        — a correspondence split's ``max_warp``, a dense split's
        ``image_size``, a triplet split's ``num_triplets``. The dataset's
        counterpart to ``task_params``, and open for the same reason: a new
        dataset type must not force another schema bump.
    finetune:
        ``None`` for a frozen probe — every v0.1 and v0.2 run, and the default.
        Otherwise ``{"blocks", "backbone_lr", "trainable_params"}`` describing
        what was unfrozen. A frozen number measures what a representation
        already carries; a fine-tuned one measures what it can be adapted into,
        and averaging or ranking the two together is meaningless. This is the
        field that makes the difference legible to a leaderboard.
    layer / layers:
        Which backbone depth the features came from. ``layer`` is the
        single-layer form and ``layers`` the resolved list a multiscale head
        was fed; exactly one is set. Both hold **resolved** indices, never a
        relative ``-1``, so a record still names the same depth after a
        backbone of a different size is compared against it.
    """

    backbone: str
    backbone_key: str
    task: str
    level: str
    dataset: str
    split: str
    pooling: str
    feature_mode: str
    metrics: dict[str, float]
    timestamp: str
    visbench_version: str
    schema_version: int = SCHEMA_VERSION
    dataset_size: int | None = None
    dataset_fingerprint: str | None = None
    # Optional despite belonging beside `pooling`, because a dataclass cannot
    # put a defaulted field before undefaulted ones -- and it must default, so
    # that reading a v6 record does not raise.
    pooling_requested: str | None = None
    task_params: dict = field(default_factory=dict)
    dataset_params: dict = field(default_factory=dict)
    layer: int | None = None
    layers: list[int] | None = None
    finetune: dict | None = None
    seed: int | None = None
    duration_seconds: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        """JSON-serialisable dict, with ``None`` fields retained.

        Retained rather than dropped so every record has identical keys, which
        keeps downstream tabular loading trivial.
        """
        payload = asdict(self)
        # Metrics arrive from evaluate() and are the one field a task fills
        # freely; coerce here so a stray tensor or numpy float cannot make a
        # whole results file unreadable.
        payload["metrics"] = {key: float(value) for key, value in self.metrics.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ResultRecord":
        """Rebuild a record, rejecting a ``schema_version`` newer than this one.

        Older versions are read, not rejected: the schema is additive-only, so
        every field a v1 record has still exists, and the ones it predates come
        back as ``None``. Refusing them would throw away exactly the history a
        benchmark library exists to accumulate.
        """
        version = payload.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(f"schema_version must be an int, got {version!r}")
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version}; this VisBench reads up to "
                f"version {SCHEMA_VERSION}. Records are additive-only, so a "
                "newer file needs a newer VisBench."
            )

        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                f"Unknown fields for schema_version {SCHEMA_VERSION}: {sorted(unknown)}"
            )
        missing = {
            f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
        } - set(payload)
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        return cls(**payload)


def utc_timestamp() -> str:
    """ISO 8601 UTC timestamp, e.g. ``2026-07-24T09:15:04+00:00``.

    UTC always: a leaderboard aggregating runs from several machines cannot
    order local timestamps.
    """
    return datetime.now(timezone.utc).isoformat()
