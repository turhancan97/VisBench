"""Shared machinery for dense probes trained on cached features — v0.2.

Depth, surface normals and (next) the two segmentation tasks are the same
experiment four times over: stream dense features off disk, train one pluggable
head on them under probe3d's ten-epoch schedule, score per pixel. They differ
in exactly three places — how the head's raw output becomes a prediction, what
loss supervises it, and what the metric is — and those three are the hooks
below.

Written by lifting the working, tested body of :class:`DepthTask` rather than
designed up front, because the second task is the first point at which the
shared part is actually knowable. The division of labour it settled on:

===========================  =================================================
this class                   feature sources, batching, head construction,
                             the optimiser and its schedule, the training
                             loop, batch-wise prediction and metric averaging
a subclass                   :attr:`out_channels`, :meth:`_activate`,
                             :meth:`_loss`, :meth:`_batch_metrics`
===========================  =================================================

**Memory.** Dense features are large: DINOv2-B at 224 is 768 x 16 x 16 x 4 B,
about 786 KB per image, so all 24k NYUv2 images come to roughly 19 GB. Pass
features from :meth:`~visbench.cache.FeatureCache.materialise` and they are
streamed from disk a batch at a time, with no such limit. Passing an
already-stacked feature dict still works and is convenient for small splits;
:meth:`fit` refuses past a ceiling and names the alternative rather than
letting a run discover the problem by being killed.
"""

import math
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from visbench.cache.streaming import CachedFeatures
from visbench.heads import build_head
from visbench.tasks.base import BaseTask
from visbench.types import FeatureMode, MetricsDict, Pooling
from visbench.utils.device import resolve_device

__all__ = ["DenseTrainingTask"]


class _MemoryFeatures(Dataset):
    """Already-stacked features, indexable like the streaming reader.

    Lets an in-memory feature dict go through exactly the same training loop as
    a :class:`~visbench.cache.CachedFeatures`, instead of the task carrying two.
    """

    def __init__(self, dense: torch.Tensor | list, targets: torch.Tensor | None = None) -> None:
        self.count = len(dense[0]) if isinstance(dense, list) else len(dense)
        if targets is not None and self.count != len(targets):
            raise ValueError(f"Got {self.count} feature maps for {len(targets)} targets")
        self.dense = dense
        self.targets = targets

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Any:
        if isinstance(self.dense, list):
            features: Any = [layer[index] for layer in self.dense]
        else:
            features = self.dense[index]
        if self.targets is None:
            return features
        return features, self.targets[index]


class _WithTargets(Dataset):
    """Pair a features-only source with targets, by index.

    By index and not by iteration order: that is what keeps each feature map
    with its own supervision once a loader starts shuffling.
    """

    def __init__(self, source: Dataset, targets: torch.Tensor) -> None:
        if len(source) != len(targets):  # type: ignore[arg-type]
            raise ValueError(
                f"Got {len(source)} feature maps for {len(targets)} targets"  # type: ignore[arg-type]
            )
        self.source = source
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        return self.source[index], self.targets[index]


class DenseTrainingTask(BaseTask):
    """A per-pixel probe: one pluggable head, trained on frozen dense features.

    ``head="linear"`` is the number to quote when comparing representations: it
    is the only head under which a difference between two backbones is a
    difference between two *feature maps*. ``head="dpt"`` is probe3d's own
    choice and scores higher for everyone, so report both, or say which.

    The DPT head is multiscale and needs several backbone depths: set
    :attr:`layers`, and extraction reads them all in one forward pass.
    """

    feature_mode = FeatureMode.DENSE_ONLY
    zero_shot = False
    uses_dense = True
    #: Dense tasks read the grid, never the pooled vector — but extraction
    #: still needs a pooling name for the cache key. Mean is the honest choice:
    #: it is what a CNN would use anyway, and it keeps these entries from
    #: colliding with a CLS-pooled run over the same images.
    pooling = Pooling.MEAN

    #: Roughly 8 GB of float32 features, past which :meth:`fit` refuses rather
    #: than letting the OOM killer end a run half an hour in.
    max_feature_elements = 2_000_000_000

    #: Channels in the ground truth. ``1`` is a scalar map stored as
    #: ``(N, H, W)`` or ``(N, 1, H, W)``; anything higher must arrive as
    #: ``(N, C, H, W)``. Targets are normalised to 4-D either way, so a
    #: subclass's loss and metric always see the same rank.
    target_channels: int = 1

    #: Human-readable names used in error messages, so a caller who forgot the
    #: ground truth is told what is missing rather than "labels".
    display_name: str = "This task"
    target_noun: str = "target maps"

    def __init__(
        self,
        head: str = "linear",
        layers: list[int] | None = None,
        hidden_dim: int = 512,
        epochs: int = 10,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        warmup_epochs: float = 1.5,
        head_kwargs: dict | None = None,
        device: str | None = None,
    ) -> None:
        """Configure the probe; the head is built lazily in :meth:`fit`.

        Parameters
        ----------
        head:
            Any registered head name — ``"linear"``, ``"dpt"``, or a
            contributor's — built once :meth:`fit` knows the feature width.
        layers:
            Backbone depths to extract. ``None`` means the last layer only,
            which a linear head is happy with and a DPT head is not: it refuses
            a single map rather than duplicating it. probe3d reads four evenly
            spaced blocks.
        epochs, lr, warmup_epochs:
            probe3d's ten-epoch schedule — AdamW at 5e-4, linear warmup over
            the first 1.5 epochs, then cosine decay to zero.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if warmup_epochs < 0 or warmup_epochs >= epochs:
            raise ValueError(
                f"warmup_epochs must be in [0, epochs), got {warmup_epochs} with "
                f"epochs={epochs} — the schedule would still be warming up when training "
                "ended. Pass warmup_epochs=0 for a short run; the default 1.5 assumes "
                "probe3d's 10 epochs. Clamping it silently would report a number produced "
                "by a schedule nobody chose."
            )
        if layers is not None and len(layers) == 0:
            raise ValueError("layers=[] requests nothing; pass None for the last layer")

        self.head_name = head
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.head_kwargs = dict(head_kwargs or {})
        self.device = resolve_device(device)

        self.head: nn.Module | None = None

        #: Set by :meth:`fit`. A diagnostic, not a result — a poor score with a
        #: high training loss means the probe underfitted, which is a different
        #: finding from a representation that does not carry this signal.
        self.train_loss: float | None = None

    # -- subclass hooks ------------------------------------------------------

    @property
    def out_channels(self) -> int:
        """How many channels the head must emit."""
        raise NotImplementedError

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        """Turn the head's raw ``(B, out_channels, H, W)`` output into a prediction.

        Applied everywhere — loss, metrics and :meth:`predict` — so those three
        can never disagree about what the model predicts.
        """
        raise NotImplementedError

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Training loss for one batch. Both arguments are ``(B, C, H, W)``."""
        raise NotImplementedError

    def _batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
        """Score one batch, on CPU. Both arguments are ``(B, C, H, W)``.

        Must return **per-image averages**, which is what lets :meth:`evaluate`
        weight each batch by its size and recover the whole-split number.
        """
        raise NotImplementedError

    def _on_epoch_start(self) -> None:
        """Called before each training epoch; a no-op by default.

        Exists so a subclass can reset a per-epoch diagnostic accumulator and
        end up describing the *last* epoch rather than an average over
        training, which for anything that drifts is the more useful of the two.
        """

    def _task_params(self) -> dict:
        """Subclass-specific entries for the result record's ``task_params``."""
        return {}

    # -- feature sources -----------------------------------------------------

    def _source(self, features: Any, labels: Any | None, targets_required: bool = True) -> Dataset:
        """Normalise whatever was passed into one indexable ``(features, target)``.

        Two front doors, one training loop. A
        :class:`~visbench.cache.CachedFeatures` streams from disk and is used as
        is; anything already in memory — a feature dict, a bare tensor, a list
        of per-layer tensors — is wrapped to look the same. The loop below never
        learns which it got.
        """
        if isinstance(features, CachedFeatures):
            if features.targets is not None:
                if labels is not None:
                    raise ValueError(
                        "These features already carry targets (materialise(targets=...)), "
                        "so passing labels as well gives two sources of truth for the "
                        "supervision. Pass one."
                    )
                return features
            if labels is None:
                if not targets_required:
                    return features
                raise ValueError(
                    f"{self.display_name} requires {self.target_noun}. Either pass them here, or "
                    "build the features with materialise(targets=dataset.target), which "
                    "keeps each target with its own image."
                )
            return _WithTargets(features, self._as_targets(labels))

        dense = self._dense(features)
        self._check_size(dense)
        if labels is None and not targets_required:
            return _MemoryFeatures(dense)
        return _MemoryFeatures(dense, self._as_targets(labels))

    def _loader(self, source: Dataset, shuffle: bool) -> DataLoader:
        """Batch a source, reshuffling each epoch when training.

        No explicit generator: without one, ``DataLoader`` seeds its shuffle
        from the global RNG, which :func:`visbench.utils.set_seed` owns — so the
        seed recorded next to the metrics still governs the run, exactly as the
        hand-rolled ``randperm`` it replaced did.
        """
        return DataLoader(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=CachedFeatures.collate,
        )

    def _dense(self, features: Any) -> torch.Tensor | list:
        """Pull what the head wants out of an in-memory feature dict."""
        if isinstance(features, dict):
            if self.layers is not None:
                if "dense_layers" not in features:
                    raise KeyError(
                        f"This task asked for layers={self.layers} but the features hold "
                        "only a single map. Extract with the same layers= the task "
                        "declares, or use visbench.run(), which does that for you."
                    )
                return [layer.float() for layer in features["dense_layers"]]
            if "dense" not in features:
                raise KeyError(
                    "Feature dict has no 'dense' entry. A dense task needs "
                    "extract_dataset(keep='dense') or keep='both'."
                )
            return features["dense"].float()
        if isinstance(features, (list, tuple)):
            return [layer.float() for layer in features]
        return features.float()

    def _check_size(self, dense: torch.Tensor | list) -> None:
        """Refuse an in-memory batch too large to hold, and name the way out."""
        maps = dense if isinstance(dense, list) else [dense]
        elements = sum(layer.numel() for layer in maps)
        if elements > self.max_feature_elements:
            raise ValueError(
                f"These features are about {elements * 4 / 1e9:.1f} GB in float32, over this "
                f"task's {self.max_feature_elements * 4 / 1e9:.0f} GB ceiling. Extract with "
                "FeatureCache.materialise(...) instead of extract_dataset(...) to stream "
                "them from disk, which has no such limit. Raise max_feature_elements to "
                "override."
            )

    def _shape_of(self, source: Dataset) -> tuple[Any, int]:
        """Read channel width(s) and target resolution off the first item.

        Sampled rather than declared: it is the same for a streamed and an
        in-memory source, so the head is built from what is actually there
        rather than from two code paths that could disagree.
        """
        sample_features, sample_target = source[0]  # type: ignore[index]
        if isinstance(sample_features, list):
            channels: Any = [layer.shape[0] for layer in sample_features]
        else:
            channels = int(sample_features.shape[0])

        target = self._as_4d(torch.as_tensor(sample_target).unsqueeze(0))
        height, width = target.shape[-2:]
        if height != width:
            raise ValueError(
                f"Targets are {height}x{width}; this task assumes square maps, which is "
                "what DenseFolderDataset produces."
            )
        return channels, int(width)

    def _build_head(self, channels: Any, output_size: int) -> nn.Module:
        """Instantiate the configured head, sized to these features."""
        if isinstance(channels, list):
            kwargs: dict = {"num_layers": len(channels), "hidden_dim": self.hidden_dim}
        else:
            kwargs = {}
        kwargs.update(self.head_kwargs)
        return build_head(
            self.head_name,
            in_channels=channels,
            out_channels=self.out_channels,
            output_size=output_size,
            **kwargs,
        ).to(self.device)

    # -- targets -------------------------------------------------------------

    def _as_targets(self, labels: Any) -> torch.Tensor:
        """Coerce a batch of targets to a stacked float tensor."""
        if labels is None:
            raise ValueError(f"{self.display_name} requires {self.target_noun}; got None")
        targets = labels if isinstance(labels, torch.Tensor) else torch.stack(list(labels))
        return self._as_4d(targets.float())

    def _as_4d(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalise stacked targets to ``(N, target_channels, H, W)``.

        A scalar map is accepted either as ``(N, H, W)`` or already channelled,
        so a dataset that emits one shape and a caller who stacks the other both
        work — and every loss and metric downstream sees one rank.
        """
        if self.target_channels == 1 and targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if targets.ndim != 4 or targets.shape[1] != self.target_channels:
            expected = (
                "(N, H, W) or (N, 1, H, W)"
                if self.target_channels == 1
                else f"(N, {self.target_channels}, H, W)"
            )
            raise ValueError(
                f"Expected {self.target_noun} of shape {expected}, got {tuple(targets.shape)}"
            )
        return targets

    # -- training ------------------------------------------------------------

    def fit(self, features: Any, labels: Any | None = None) -> "DenseTrainingTask":
        """Train the head on dense features and their per-pixel targets.

        ``features`` is either an in-memory feature dict or a streaming
        :class:`~visbench.cache.CachedFeatures`; the second is what makes a
        split larger than RAM trainable.

        Seeding is the caller's job (:func:`visbench.utils.set_seed`), so the
        seed recorded next to the metrics is the one that governed the run.
        """
        source = self._source(features, labels)
        count = len(source)  # type: ignore[arg-type]
        if count == 0:
            raise ValueError("Cannot fit on an empty feature set")

        channels, output_size = self._shape_of(source)
        self.head = self._build_head(channels, output_size)
        self.head.train()

        optimiser = torch.optim.AdamW(
            self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loader = self._loader(source, shuffle=True)
        steps_per_epoch = max(1, len(loader))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimiser, self._schedule(steps_per_epoch * self.epochs, steps_per_epoch)
        )

        running = 0.0
        for _ in range(self.epochs):
            running = 0.0
            self._on_epoch_start()
            for batch_features, batch_targets in loader:
                optimiser.zero_grad()
                predicted = self._forward(batch_features)
                targets = self._as_4d(torch.as_tensor(batch_targets).float()).to(self.device)
                loss = self._loss(predicted, targets)
                loss.backward()
                optimiser.step()
                scheduler.step()
                running += loss.item()
            running /= steps_per_epoch

        self.head.eval()
        self.train_loss = running
        return self

    def _schedule(self, total_steps: int, steps_per_epoch: int):
        """Linear warmup then cosine decay, as a step-indexed multiplier."""
        warmup_steps = int(self.warmup_epochs * steps_per_epoch)

        def multiplier(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            remaining = max(1, total_steps - warmup_steps)
            progress = min(1.0, (step - warmup_steps) / remaining)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return multiplier

    def _forward(self, dense: Any) -> torch.Tensor:
        """A batch of features to a batch of predictions."""
        head = self._require_head()
        if isinstance(dense, list):
            dense = [layer.to(self.device).float() for layer in dense]
        else:
            dense = dense.to(self.device).float()
        return self._activate(head(dense))

    def _require_head(self) -> nn.Module:
        if self.head is None:
            raise RuntimeError(
                "This probe has not been fitted. Call fit(train_features, train_targets) "
                "before predict() or evaluate()."
            )
        return self.head

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def predict(self, features: Any, labels: Any | None = None) -> torch.Tensor:
        """Predictions for every image, stacked ``(N, C, H, W)``.

        Holds every prediction in memory, because that is what was asked for —
        24k NYUv2 images at 224 square is about 4.8 GB per channel.
        :meth:`evaluate` does not use this; it scores batch by batch and keeps
        only the running metrics.
        """
        self._require_head()
        outputs = [
            self._forward(batch).cpu()
            for batch, _ in self._iter_batches(features, labels, targets_required=False)
        ]
        return torch.cat(outputs)

    def _iter_batches(
        self, features: Any, labels: Any | None, targets_required: bool = True
    ) -> Iterator[tuple[Any, torch.Tensor | None]]:
        """Yield ``(features, targets_or_None)`` in dataset order.

        Normalised here so callers need not care whether targets were available:
        ``collate`` returns a bare batch when there are none, and a multi-layer
        batch is itself a list, so only the tuple marks targets.
        """
        source = self._source(features, labels, targets_required=targets_required)
        for batch in self._loader(source, shuffle=False):
            if isinstance(batch, tuple):
                yield batch
            else:
                yield batch, None

    @torch.no_grad()
    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """Score the split, a batch at a time.

        Accumulates per-image totals rather than collecting every prediction
        first. Because :meth:`_batch_metrics` returns per-image averages,
        weighting each batch by its size and dividing by the total reproduces
        the whole-split number exactly, at bounded memory.
        """
        self._require_head()
        totals: dict[str, float] = {}
        count = 0

        for batch_features, batch_targets in self._iter_batches(features, labels):
            assert batch_targets is not None  # targets_required defaults to True
            predicted = self._forward(batch_features).cpu()
            targets = self._as_4d(torch.as_tensor(batch_targets).float())
            batch_metrics = self._batch_metrics(predicted, targets)
            size = len(targets)
            for name, value in batch_metrics.items():
                totals[name] = totals.get(name, 0.0) + value * size
            count += size

        if count == 0:
            raise ValueError("Cannot evaluate on an empty feature set")
        return {name: total / count for name, total in totals.items()}

    # -- provenance ----------------------------------------------------------

    def describe(self) -> dict:
        """Task metadata plus everything that shaped the number."""
        described = super().describe()
        described["task_params"] = {
            "head": self.head_name,
            "layers": self.layers,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "warmup_epochs": self.warmup_epochs,
            "optimizer": "adamw",
            "protocol": "probe3d",
            **self._task_params(),
        }
        return described
