"""One call from backbone + task + data to a scored, logged result.

The glue every caller was otherwise writing by hand: resolve pooling, extract
through the cache, fit if the task trains, evaluate, and assemble the
:class:`ResultRecord`. Deliberately written *after* all three v0.1 tasks
existed, because they need genuinely different shapes — zero-shot over one
split, trained over two, and zero-shot over pairs with geometry — and a helper
designed against one of them would have been wrong for the other two.

This is what the v0.2 CLI will be a thin wrapper over.

Named ``runner`` rather than ``run`` so the module does not shadow
:func:`visbench.run`. Python binds a submodule onto its parent package on
import, so ``visbench/run.py`` silently replaced the function with the module
the first time anything imported it — the same trap that kept the
result-logging package from being called ``logging``.
"""

import time
from pathlib import Path
from typing import Any

import visbench
from visbench.cache import FeatureCache
from visbench.data.pair_dataset import PairDataset, PairViewDataset
from visbench.results.schema import ResultRecord, utc_timestamp
from visbench.results.writer import ResultWriter
from visbench.types import Pooling
from visbench.utils.seed import set_seed

__all__ = ["run", "RunResult"]

#: Keys of :meth:`BaseDataset.describe` that have their own record field.
#: Anything else a dataset describes lands in ``dataset_params``.
_RECORD_FIELDS = frozenset({"dataset", "split", "dataset_size", "dataset_fingerprint"})


class RunResult:
    """Metrics plus the record that describes how they were produced."""

    def __init__(self, metrics: dict, record: ResultRecord, probe: Any):
        self.metrics = metrics
        self.record = record
        #: The fitted probe, for callers that want its internals — a trained
        #: classifier's ``train_top1``, say, which diagnoses underfitting and
        #: is therefore not a result.
        self.probe = probe

    def __repr__(self) -> str:
        scores = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"<RunResult {self.record.backbone} {self.record.task}: {scores}>"


def _resolve(backbone: Any, probe: Any) -> str:
    """Turn ``pooling="default"`` into what this backbone actually does.

    Recorded resolved, not deferred. ``"default"`` means CLS on a ViT and mean
    on a CNN, so a record carrying the literal word does not say what produced
    the number — and comparing two such records across architectures compares
    two different representations under one name.
    """
    return backbone.default_pooling() if probe.pooling == Pooling.DEFAULT else probe.pooling


def _extract(
    cache: FeatureCache, backbone: Any, dataset: Any, probe: Any, **kwargs: Any
) -> tuple[Any, Any]:
    """Extract features and their labels, in the form this task can afford.

    Returns ``(features, labels)``. A pooled task gets everything stacked and a
    label list; a dense task gets a streaming reader with its targets already
    paired by index, and ``None`` for labels — nothing about a dense split is
    small enough to stack, targets included. A pairwise task gets a sequence of
    ``(features_0, features_1)`` and the geometry to score them against.
    """
    shared = {
        "pooling": _resolve(backbone, probe),
        # The task declares which depths it needs — a multiscale head is
        # unusable without them — and asking here means one forward pass
        # covers all of them.
        "layers": probe.layers,
        "feature_mode": probe.feature_mode,
        **kwargs,
    }

    if probe.uses_pairs:
        # Correspondence's shape: two views per item plus geometry, which no
        # amount of squinting turns into images plus labels. Flattening the
        # pairs into 2N single images is what lets the cache stay unchanged —
        # see PairViewDataset, and TwoAFCDataset for the same move on triplets.
        if not isinstance(dataset, PairDataset):
            raise TypeError(
                f"{probe.name!r} consumes image pairs plus geometry, but got a "
                f"{type(dataset).__name__}. Pass a PairDataset — e.g. "
                "HomographyPairDataset."
            )
        views = cache.materialise(backbone, PairViewDataset(dataset), **shared)
        return PairViewDataset.regroup(views), dataset.labels()

    if not probe.uses_dense:
        # Pooled features are small enough to hold whole, and every zero-shot
        # task wants them all at once anyway (retrieval ranks N against N).
        return cache.extract_dataset(backbone, dataset, keep="pooled", **shared), dataset.labels()

    # Dense features are ~250x larger, so they stream from disk. So do their
    # targets, through the same index — `dataset.labels()` would stack every
    # depth map, which for NYUv2 is another ~4.8 GB on top of the features.
    targets = getattr(dataset, "target", None)
    features = cache.materialise(backbone, dataset, targets=targets, **shared)
    return features, (None if targets is not None else dataset.labels())


def run(
    backbone: str | Any,
    task: str | Any,
    dataset: Any,
    train_dataset: Any | None = None,
    cache: FeatureCache | None = None,
    results: str | Path | None = None,
    batch_size: int = 32,
    seed: int | None = 0,
    device: str | None = None,
    notes: str | None = None,
    **task_kwargs: Any,
) -> RunResult:
    """Run one task on one backbone over one dataset, and log the result.

    Parameters
    ----------
    backbone, task:
        Registered names, or already-constructed objects. Names are the common
        case; objects let a caller configure something first.
    dataset:
        The split that gets *scored*. For a trained probe, pass the training
        split as ``train_dataset``; the record then describes the split the
        number came from, which is the one a reader cares about.
    train_dataset:
        Required when the task is not zero-shot. Passing it for a zero-shot
        task raises rather than being ignored, since silently discarding it
        would mean the caller's intent and the result disagree.
    results:
        JSONL path to append to. ``None`` skips writing, for exploration.

    Returns
    -------
    RunResult
        ``.metrics``, ``.record`` and the fitted ``.probe``.

    Notes
    -----
    Correspondence is covered: a task declaring ``uses_pairs`` gets both views
    of every pair extracted and handed back paired, with the geometry as
    labels. Its ceiling is merged into the metrics under ``ceiling_*``, because
    a correspondence score read without it says the wrong thing (see
    :meth:`BaseTask.context_metrics`).
    """
    used_seed = set_seed(seed)
    started = time.perf_counter()

    if isinstance(backbone, str):
        backbone = visbench.get_backbone(backbone, device=device)
    if isinstance(task, str):
        task = visbench.get_probe(task, **task_kwargs)
    elif task_kwargs:
        raise TypeError("task_kwargs only apply when `task` is a name, not an instance")

    if task.zero_shot and train_dataset is not None:
        raise ValueError(
            f"{task.name!r} is zero-shot and has nothing to fit, but train_dataset was "
            "given. Drop it, or use a task that trains."
        )
    if not task.zero_shot and train_dataset is None:
        raise ValueError(f"{task.name!r} trains a probe and needs train_dataset")

    cache = cache if cache is not None else FeatureCache()

    if task.finetune_blocks:
        # The cache is bypassed entirely, not keyed differently. Its keys name
        # the weights through cache_key(), and fine-tuned weights differ at
        # every optimiser step: an entry written here would be stale on arrival
        # and — indistinguishable from a frozen one — served to every later
        # frozen run of this backbone. So the task gets the datasets and runs
        # the backbone itself, inside its own training loop.
        task.attach_backbone(backbone)
        features, labels = dataset, None
        task.fit(train_dataset)
    else:
        features, labels = _extract(cache, backbone, dataset, task, batch_size=batch_size)
        if train_dataset is None:
            task.fit(features)
        else:
            task.fit(*_extract(cache, backbone, train_dataset, task, batch_size=batch_size))

    metrics = task.evaluate(features, labels)
    # Merged after, and never allowed to overwrite: a task that returned a
    # context key colliding with one of its own scores would replace a result
    # with the thing that qualifies it.
    for name, value in task.context_metrics(features, labels).items():
        if name in metrics:
            raise ValueError(
                f"{task.name!r} returned context metric {name!r}, which is already one "
                "of its scores. Namespace it — these share one flat dict."
            )
        metrics[name] = value

    dataset_described = dataset.describe()
    described = {**dataset_described, **task.describe()}
    record = ResultRecord(
        backbone=backbone.name,
        backbone_key=backbone.cache_key(),
        task=described["task"],
        level=described["level"],
        dataset=described["dataset"],
        split=described["split"],
        dataset_size=described["dataset_size"],
        dataset_fingerprint=described["dataset_fingerprint"],
        pooling=_resolve(backbone, task),
        feature_mode=described["feature_mode"],
        # Resolved against this backbone's depth, for the same reason pooling
        # is: a record saying [-4, -1] does not name the layers that produced
        # the number, and means two different things on a 12- and a 24-block ViT.
        layers=(None if task.layers is None else backbone.resolve_layers(task.layers)),
        task_params=described["task_params"],
        # None for a frozen probe, which is every task that cannot fine-tune.
        # A fine-tuned score is not comparable to a frozen one and this is what
        # says so in the record itself.
        finetune=task.finetune(),
        # Whatever the dataset described beyond the fields above — max_warp,
        # image_size, num_triplets. Taken from the dataset's own describe()
        # rather than a fixed list, so a new dataset type carries its settings
        # into the record without a schema change.
        dataset_params={
            key: value for key, value in dataset_described.items() if key not in _RECORD_FIELDS
        },
        metrics=metrics,
        timestamp=utc_timestamp(),
        visbench_version=visbench.__version__,
        seed=used_seed,
        duration_seconds=time.perf_counter() - started,
        notes=notes,
    )

    if results is not None:
        with ResultWriter(Path(results)) as writer:
            writer.write(record)

    return RunResult(metrics, record, task)
