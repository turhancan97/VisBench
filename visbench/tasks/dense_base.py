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

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from visbench.cache.keys import hash_image, make_prefix_key
from visbench.cache.streaming import CachedFeatures
from visbench.heads import build_head
from visbench.tasks.base import BaseTask
from visbench.tasks.schedule import check_schedule, warmup_cosine
from visbench.types import FeatureMode, MetricsDict, Pooling
from visbench.utils.device import resolve_device

__all__ = ["DenseTrainingTask", "pool_to_grid"]


def pool_to_grid(targets: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """Area-average a ``(B, C, H, W)`` target down to a ``(B, C, gh, gw)`` grid.

    The information a patch token can carry about its own patch, and nothing
    more. :meth:`DenseTrainingTask.evaluate_oracle` upsamples the result back
    and scores it, which is how the oracle asks whether a target survives the
    feature grid at all.

    **Invalid pixels are excluded rather than averaged in.** ``NaN`` is the
    magnitude probes' invalid marker (the fourth validity convention in this
    codebase), and a plain mean over a patch holding one hole would return
    ``NaN`` for the whole patch — which would quietly shrink the population the
    oracle is scored over relative to the population :meth:`evaluate` scores,
    and the two numbers would stop being comparable. The mean is taken over the
    finite pixels only; a cell with *no* finite pixel comes back ``NaN``, which
    is the same marker its pixels carried.

    Implemented as two adaptive pools rather than a reshape so that a grid which
    does not divide the image evenly — 224 over a 15x15 grid, say — still pools
    each cell over exactly the window the second pool counted.
    """
    if targets.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W) targets, got {tuple(targets.shape)}")
    grid_h, grid_w = grid_hw
    if grid_h < 1 or grid_w < 1:
        raise ValueError(f"grid_hw must be positive, got {grid_hw}")
    if grid_h > targets.shape[-2] or grid_w > targets.shape[-1]:
        raise ValueError(
            f"grid_hw {grid_hw} is finer than the target {tuple(targets.shape[-2:])}. "
            "The oracle asks what survives a *coarser* grid; a finer one is not a bound."
        )

    finite = torch.isfinite(targets)
    filled = targets.masked_fill(~finite, 0.0)
    # Both pools average over the same adaptive windows, so their ratio is the
    # mean over the finite pixels of each cell. An all-invalid cell is 0/0 = NaN.
    total = F.adaptive_avg_pool2d(filled, (grid_h, grid_w))
    weight = F.adaptive_avg_pool2d(finite.to(targets.dtype), (grid_h, grid_w))
    return total / weight


class _TargetsOf:
    """The targets of a :class:`~visbench.cache.CachedFeatures`, and nothing else.

    `CachedFeatures` keeps its targets as a callable by index and loads a
    feature file only when the item itself is asked for, so this reaches the
    supervision of a streamed run without touching the features — which is what
    lets the ceiling cost a pass over the targets rather than over the 19 GB
    beside them. Shaped as ``__len__`` + ``target(index)`` so that
    :meth:`DenseTrainingTask._iter_target_batches` sees the same thing a dataset
    presents.
    """

    def __init__(self, cached: CachedFeatures) -> None:
        if cached.targets is None:
            raise ValueError("These features carry no targets")
        self._cached = cached

    def __len__(self) -> int:
        return len(self._cached)

    def target(self, index: int) -> Any:
        assert self._cached.targets is not None
        return self._cached.targets(index)


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


def _collate_images(batch: list) -> tuple[list, torch.Tensor]:
    """Keep PIL images as a list, stack their targets.

    The fine-tuning counterpart to :meth:`CachedFeatures.collate`. Images reach
    the loop undecoded into tensors because each backbone owns its own
    normalisation and resolution, so ``backbone.preprocess`` is what turns them
    into a batch — the default collation cannot stack a PIL image at all.
    """
    return [item[0] for item in batch], torch.stack([torch.as_tensor(item[1]) for item in batch])


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

    #: dtype targets are coerced to before every loss and metric. Float suits a
    #: measurement — a depth, a normal component, a foreground probability.
    #: A *classification* target is different: cross-entropy needs class indices
    #: as ``long``, and silently handing it floats is a type error at best and a
    #: wrong answer at worst. Set by :class:`SemanticSegmentationTask`; the
    #: coercion happens in one place so training, evaluation and ``predict``
    #: cannot disagree about what a target is.
    target_dtype: torch.dtype = torch.float32

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
        finetune_blocks: int = 0,
        backbone_lr: float | None = None,
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
        finetune_blocks:
            Unfreeze this many trailing backbone blocks and train them with the
            head (v0.3). ``0``, the default, is the frozen probe every v0.1 and
            v0.2 number was measured with — **the two are not comparable**, and
            the result record says which was which.

            Fine-tuning reads images rather than cached features, because a
            cache entry is keyed on the weights and fine-tuned weights change at
            every step.

            **Slower, and increasingly so with backbone size.** On VOC with two
            unfrozen blocks: 200 s against a cached frozen run's 126-156 s on
            DINOv2-S, and 279 s against 126 s on DINOv2-B. The frozen path's
            cost barely moves between the two — it is dominated by per-file
            reads over the split, not by the bytes streamed — while the
            fine-tuned path tracks compute.
        backbone_lr:
            Learning rate for the unfrozen blocks. Defaults to ``lr / 100``:
            pretrained weights at the head's rate are destroyed within the first
            epoch, which shows up as a fine-tuned score *below* the frozen
            baseline rather than as an error.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        check_schedule(epochs, warmup_epochs)
        if layers is not None and len(layers) == 0:
            raise ValueError("layers=[] requests nothing; pass None for the last layer")
        if finetune_blocks < 0:
            raise ValueError(f"finetune_blocks must be >= 0, got {finetune_blocks}")
        if backbone_lr is not None and finetune_blocks == 0:
            raise ValueError(
                f"backbone_lr={backbone_lr} was given with finetune_blocks=0, so nothing "
                "in the backbone trains and the value would be silently ignored. Set "
                "finetune_blocks, or drop backbone_lr."
            )

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
        self.finetune_blocks = finetune_blocks
        self.backbone_lr = backbone_lr if backbone_lr is not None else lr / 100

        self.head: nn.Module | None = None

        #: Set by :meth:`attach_backbone` when fine-tuning. ``None`` for every
        #: frozen probe, which reads features and never sees a backbone at all.
        self._backbone: Any = None
        #: Parameter count the unfreeze actually made trainable, recorded so a
        #: reader can tell a two-block fine-tune of ViT-S from one of ViT-L.
        self.trainable_params: int = 0

        #: Set by :meth:`attach_backbone` (step 6b), and only when the backbone
        #: can serve this run's ``layers`` from a cut. ``None`` means the frozen
        #: blocks are recomputed every epoch, which is 6a's behaviour.
        self._prefix_cache: Any = None
        #: Whether the prefix cache was actually used, for the result record.
        #: Distinct from "one was offered": a DPT run reading layers below the
        #: cut declines it, and a record saying otherwise would misattribute
        #: the run's cost.
        self.uses_prefix_cache: bool = False

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

    def training_summary(self) -> dict | None:
        """The final epoch's mean training loss, for the record.

        Every dense probe inherits this, so all nine report it. It is the
        number that separates an underfitting probe from a weak representation:
        binary segmentation on 80 training images reads 0.16 IoU at the
        defaults and 0.87 at ``epochs=40`` on *identical* features, and only
        the loss says which of those a given low score is.

        ``None`` before :meth:`fit`, so a record can never claim a fit that did
        not happen.
        """
        if self.train_loss is None:
            return None
        return {"train_loss": self.train_loss}

    def _task_params(self) -> dict:
        """Subclass-specific entries for the result record's ``task_params``."""
        return {}

    # -- fine-tuning ---------------------------------------------------------

    def attach_backbone(self, backbone: Any, prefix_cache: Any = None) -> None:
        """Unfreeze the backbone and keep it, so :meth:`fit` trains through it.

        Called by :func:`visbench.run` when :attr:`finetune_blocks` is set. The
        task holds the backbone only in this mode; a frozen probe is handed
        features and never learns which model produced them, which is what lets
        it read them from a cache.

        The unfreeze happens here rather than in ``__init__`` because the task
        is constructed before it is paired with a backbone — and it is
        :meth:`BaseBackbone.unfreeze_last` that refuses a family it cannot
        support, or an unfreeze that would train nothing.

        ``prefix_cache`` is step 6b's saving: a
        :class:`~visbench.cache.PrefixCache` for the frozen blocks below the
        cut. It is used only where it is *provably* equivalent — see
        :meth:`BaseBackbone.can_use_prefix_cache` — and ignored otherwise, so
        passing one can change how long a run takes but never what it reports.
        """
        if self.finetune_blocks == 0:
            raise ValueError(
                "attach_backbone() on a probe with finetune_blocks=0. A frozen probe "
                "reads cached features and must not hold a backbone, or it would run one "
                "forward pass per epoch for no reason."
            )
        self.trainable_params = backbone.unfreeze_last(self.finetune_blocks)
        self._backbone = backbone
        # Resolved once, after the unfreeze: can_use_prefix_cache depends on
        # trainable_blocks, so asking before this line would always say no.
        #
        # A *disabled* cache is treated as no cache at all rather than as one
        # that always misses. The difference matters twice: the run takes 6a's
        # code path, which is what `--no-prefix-cache` exists to measure, and
        # the record says `prefix_cache: false` instead of claiming a saving it
        # did not get.
        usable = (
            prefix_cache is not None
            and getattr(prefix_cache, "enabled", True)
            and backbone.can_use_prefix_cache(self.layers)
        )
        self._prefix_cache = prefix_cache if usable else None
        self.uses_prefix_cache = usable

    def _backbone_dense(self, images: list) -> Any:
        """Preprocess a batch of PIL images and run the backbone over it.

        Picks the trainable entry point only while grads are enabled, so
        :meth:`evaluate` and :meth:`predict` — both under ``no_grad`` — take the
        ordinary frozen path and this method stays honest about which one ran.
        """
        backbone = self._backbone
        if self._prefix_cache is not None:
            return self._dense_from_prefix(images)

        batch = backbone.preprocess(images)
        extract = (
            backbone.extract_features_trainable
            if torch.is_grad_enabled()
            else backbone.extract_features
        )
        features = extract(
            batch,
            pooling=self.pooling,
            layers=self.layers,
            feature_mode=self.feature_mode,
        )
        return features["dense_layers"] if self.layers is not None else features["dense"]

    def _dense_from_prefix(self, images: list) -> Any:
        """Serve the frozen half from disk and recompute only the unfrozen blocks.

        The saving of step 6b, and it is a saving of *compute only*: the tokens
        below the cut are a pure function of the image and weights that do not
        move, so a hit here returns exactly what recomputing would have. A test
        asserts bit-equality against :meth:`_backbone_dense`'s other branch.

        Misses are computed for the whole batch in one forward pass and written
        back, so the first epoch pays roughly what the unbypassed path pays and
        every later one reads.
        """
        backbone = self._backbone
        cache = self._prefix_cache
        backbone_key = backbone.cache_key()
        cut = backbone.prefix_cut()

        keys = [make_prefix_key(hash_image(image), backbone_key, cut) for image in images]
        entries: list[Any] = [cache.get(key) for key in keys]

        missing = [index for index, entry in enumerate(entries) if entry is None]
        if missing:
            batch = backbone.preprocess([images[index] for index in missing])
            tokens, grid_hw = backbone.forward_prefix(batch)
            for position, index in enumerate(missing):
                cache.put(keys[index], tokens[position], grid_hw)
                entries[index] = (tokens[position], grid_hw)

        grids = {entry[1] for entry in entries}
        if len(grids) != 1:
            raise RuntimeError(
                f"Cached prefixes in one batch disagree on the patch grid: {sorted(grids)}. "
                "Stacking them would align every feature map against the wrong target."
            )
        grid_hw = entries[0][1]
        prefix = torch.stack([entry[0].to(backbone.device) for entry in entries])

        features = backbone.extract_features_from_prefix(
            prefix,
            grid_hw,
            pooling=self.pooling,
            layers=self.layers,
            feature_mode=self.feature_mode,
        )
        return features["dense_layers"] if self.layers is not None else features["dense"]

    # -- feature sources -----------------------------------------------------

    def _source(self, features: Any, labels: Any | None, targets_required: bool = True) -> Dataset:
        """Normalise whatever was passed into one indexable ``(features, target)``.

        Two front doors, one training loop. A
        :class:`~visbench.cache.CachedFeatures` streams from disk and is used as
        is; anything already in memory — a feature dict, a bare tensor, a list
        of per-layer tensors — is wrapped to look the same. The loop below never
        learns which it got.
        """
        if self._backbone is not None:
            # Fine-tuning is handed the dataset itself: it already yields
            # (image, target) at the working geometry, which is exactly the
            # pairing the loop needs, and nothing about it may be cached.
            if isinstance(features, (CachedFeatures, dict, torch.Tensor)):
                raise TypeError(
                    f"{self.display_name} is fine-tuning and needs the image dataset, not "
                    f"extracted features (got {type(features).__name__}). Features are "
                    "extracted with frozen weights and would not move as the backbone "
                    "trains. visbench.run() passes the dataset for you."
                )
            if labels is not None:
                raise ValueError(
                    f"{self.display_name} is fine-tuning and reads its {self.target_noun} "
                    "from the dataset, which already pairs each one with its own image. "
                    "Passing them separately gives two sources of truth for the "
                    "supervision, and these would be the ignored one."
                )
            return features

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
            collate_fn=_collate_images if self._backbone is not None else CachedFeatures.collate,
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
        if self._backbone is not None:
            # The sample is a PIL image, so the width has to come from a real
            # forward pass rather than from embed_dim: feature_mode changes it
            # (dense_cls_broadcast concatenates the CLS token onto every
            # location), and building the head against the wrong width is a
            # shape error a whole epoch later.
            with torch.no_grad():
                sample_features = self._backbone_dense([sample_features])
            sample_features = (
                [layer[0] for layer in sample_features]
                if isinstance(sample_features, list)
                else sample_features[0]
            )
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
        kwargs.update(
            in_channels=channels,
            out_channels=self.out_channels,
            output_size=output_size,
        )
        # Recorded at build time, not reconstructed at save time. `channels` and
        # `output_size` are measured from the first batch of features, so
        # nothing outside this call knows them -- and a saved head rebuilt at a
        # guessed width fails on a shape mismatch at best, and at worst loads
        # because the guess happened to be right for one backbone.
        self._head_spec: dict = {"kind": "registered", "name": self.head_name, "kwargs": kwargs}
        return build_head(self.head_name, **kwargs).to(self.device)

    def head_spec(self) -> dict | None:
        return getattr(self, "_head_spec", None)

    # -- targets -------------------------------------------------------------

    def _as_targets(self, labels: Any) -> torch.Tensor:
        """Coerce a batch of targets to a stacked tensor of :attr:`target_dtype`."""
        if labels is None:
            raise ValueError(f"{self.display_name} requires {self.target_noun}; got None")
        targets = labels if isinstance(labels, torch.Tensor) else torch.stack(list(labels))
        return self._as_4d(targets.to(self.target_dtype))

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

        # Two groups when fine-tuning, at deliberately different rates: the head
        # starts from noise and needs probe3d's 5e-4, while the backbone starts
        # from pretrained weights that the same rate destroys inside one epoch.
        groups: list[dict] = [{"params": list(self.head.parameters()), "lr": self.lr}]
        if self._backbone is not None:
            groups.append({"params": self._backbone.trainable_parameters(), "lr": self.backbone_lr})
        optimiser = torch.optim.AdamW(groups, lr=self.lr, weight_decay=self.weight_decay)
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
                targets = self._as_4d(torch.as_tensor(batch_targets).to(self.target_dtype)).to(
                    self.device
                )
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
        """Linear warmup then cosine decay, as a step-indexed multiplier.

        Shared with :class:`~visbench.tasks.high_level.detection.DetectionTask`,
        which cannot subclass this base but must train under the same schedule.
        """
        return warmup_cosine(total_steps, steps_per_epoch, self.warmup_epochs)

    def _forward(self, dense: Any) -> torch.Tensor:
        """A batch of features to a batch of predictions.

        When fine-tuning, the batch arrives as images and the backbone runs
        here, inside the graph — that is the whole difference between the two
        modes, and it is one branch rather than a second training loop.
        """
        head = self._require_head()
        if self._backbone is not None:
            dense = self._backbone_dense(dense)
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

        def scored() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for batch_features, batch_targets in self._iter_batches(features, labels):
                assert batch_targets is not None  # targets_required defaults to True
                predicted = self._forward(batch_features).cpu()
                yield predicted, self._as_4d(torch.as_tensor(batch_targets).to(self.target_dtype))

        return self._average_per_image(scored(), "feature set")

    def _average_per_image(
        self, batches: Iterator[tuple[torch.Tensor, torch.Tensor]], noun: str
    ) -> MetricsDict:
        """Weighted mean of :meth:`_batch_metrics` over ``(prediction, target)`` batches.

        Because :meth:`_batch_metrics` returns per-image averages, weighting
        each batch by its size and dividing by the total reproduces the
        whole-split number exactly, at bounded memory.

        Shared with :meth:`evaluate_oracle` deliberately. A ceiling computed by
        a second averaging rule would be an average over a different population
        from the score it qualifies — the mistake
        :meth:`~visbench.tasks.mid_level.correspondence.CorrespondenceTask.evaluate_ceiling`
        documents having made once, where restricting to a different set of
        matches let a real run report 127% of its own ceiling.
        """
        totals: dict[str, float] = {}
        count = 0
        for predicted, targets in batches:
            size = len(targets)
            for name, value in self._batch_metrics(predicted, targets).items():
                totals[name] = totals.get(name, 0.0) + value * size
            count += size

        if count == 0:
            raise ValueError(f"Cannot evaluate on an empty {noun}")
        return {name: total / count for name, total in totals.items()}

    # -- the oracle ----------------------------------------------------------

    def context_metrics(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """:meth:`evaluate_oracle` for this run's grid, prefixed ``ceiling_``.

        A dense score read without its ceiling says the wrong thing for the same
        reason a correspondence score does: the head sees one feature vector per
        patch, so part of every target is out of reach before the backbone is
        chosen, and how much varies by *backbone* — a ResNet's 7x7 grid leaves
        `corner` a ceiling of 0.67 where a ViT/14's 16x16 leaves 0.83. Ranking
        those two against each other without saying so invites a reader to
        attribute a grid difference to a representation.

        ``{}`` for a probe that declares no oracle, which is most of them —
        pooling is meaningful only for a target that averages, and the refusal
        in :meth:`oracle_prediction` is deliberate. The check is on the
        *subclass* rather than a flag, so a probe cannot claim one it has not
        implemented.

        **The ceiling is not a score and must never be ranked on.** It says what
        was available, not what the backbone recovered — and because it falls
        with the grid, ranking on it would rank feature resolution directly. It
        is also not a denominator: `corner` reaches 80% of its ceiling and
        `keypoints2d` 41%, and both rank backbones perfectly well.

        Costs a pass over the split's *targets* — not its features, which are
        the expensive half and are not read here.
        """
        if type(self).oracle_prediction is DenseTrainingTask.oracle_prediction:
            return {}

        targets = self._targets_only(features, labels)
        if targets is None:
            return {}

        grid_hw = self._grid_of(self._source(features, labels))
        return {
            f"ceiling_{name}": value
            for name, value in self.evaluate_oracle(targets, grid_hw).items()
        }

    def _targets_only(self, features: Any, labels: Any | None) -> Any:
        """The run's targets, reached without loading a single feature map.

        The three shapes :meth:`_source` accepts each keep the targets somewhere
        different, and only one of them is the ``labels`` argument. Going
        through :meth:`_source` instead would be simpler and would stream every
        dense feature in the split off disk to throw it away — which for a
        24k-image run is the 19 GB the streaming path exists to avoid.
        """
        if labels is not None:
            return labels
        if isinstance(features, CachedFeatures):
            return _TargetsOf(features) if features.targets is not None else None
        if hasattr(features, "target") and hasattr(features, "__len__"):
            # Fine-tuning: `features` is the image dataset, which pairs its own
            # targets by index. Reading them costs no forward pass.
            return features
        return None

    def _grid_of(self, source: Dataset) -> tuple[int, int]:
        """The feature grid the head reads, sampled off the first item.

        Sampled the same way :meth:`_shape_of` samples the channel width, and
        for the same reason: it is what is actually there rather than what a
        backbone's patch size implies.

        **The finest map, when several are requested.** A DPT head fuses
        multiple depths, and detail it can carry is bounded by its *finest*
        input, not its coarsest. Every ViT in this corpus returns one grid for
        all of them, so this only differs for a multi-stage CNN.
        """
        sample_features, _ = source[0]  # type: ignore[index]
        if self._backbone is not None:
            with torch.no_grad():
                sample_features = self._backbone_dense([sample_features])
            sample_features = (
                [layer[0] for layer in sample_features]
                if isinstance(sample_features, list)
                else sample_features[0]
            )
        maps = sample_features if isinstance(sample_features, list) else [sample_features]
        grids = [(int(one.shape[-2]), int(one.shape[-1])) for one in maps]
        return max(grids, key=lambda hw: hw[0] * hw[1])

    def oracle_prediction(self, targets: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """The best prediction a head reading a ``grid_hw`` feature map could emit.

        A dense probe sees one feature vector per patch, so whatever the
        backbone is, the head's output varies at grid resolution and is
        interpolated up from there. This returns the target itself passed
        through that bottleneck — pooled to the grid, upsampled back, and run
        through :meth:`_activate` — which is what a *perfect* backbone would
        make available.

        **Not implemented by default, and that is the same posture as**
        ``TARGET_STYLES`` **and** ``METRIC_DIRECTIONS``. Pooling is only the
        right bottleneck for a target that averages: a class-index map does not
        (the mean of classes 1 and 15 is class 8), and a bin-expectation depth
        target is a measurement whose pooled value is a different quantity from
        the one the head emits. A default that silently mean-pooled either would
        produce a confident number about nothing, so a probe opts in — see
        :meth:`_averaging_oracle`, which both opted-in probes call.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares no oracle. The oracle pools the target to the "
            "feature grid, which is only meaningful for a target that averages — see "
            "DenseMagnitudeTask.oracle_prediction and OrientationTask.oracle_prediction. "
            "If this target does average, implement oracle_prediction as a call to "
            "self._averaging_oracle(targets, grid_hw)."
        )

    def _averaging_oracle(self, targets: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """:meth:`oracle_prediction` for a target whose patch mean is meaningful.

        Bilinear on the way back up because that is what
        :meth:`~visbench.heads.base.BaseHead._resize` does — the oracle is
        meant to face the same interpolation a real head does, not a kinder one.

        An all-invalid cell is filled with the frame's mean before upsampling.
        Left as ``NaN`` it would spread into the neighbouring cells' pixels,
        which *are* scored, and turn the whole frame's metric into ``NaN``; the
        cell's own pixels are masked out by the metric either way, so the fill
        value only reaches pixels for which it is a neutral guess.
        """
        pooled = pool_to_grid(targets, grid_hw)
        holes = ~torch.isfinite(pooled)
        if bool(holes.any()):
            per_frame = torch.nan_to_num(pooled).flatten(1).sum(dim=1) / (~holes).flatten(1).sum(
                dim=1
            ).clamp(min=1)
            pooled = torch.where(holes, per_frame.view(-1, 1, 1, 1).expand_as(pooled), pooled)
        upsampled = F.interpolate(
            pooled, size=targets.shape[-2:], mode="bilinear", align_corners=False
        )
        return self._activate(upsampled)

    @torch.no_grad()
    def evaluate_oracle(
        self,
        labels: Any,
        grid_hw: tuple[int, int] | int,
        batch_size: int | None = None,
    ) -> MetricsDict:
        """What this probe could score if the features contained the answer.

        Needs **no backbone, no features and no fitted head** — only the targets
        and the grid a backbone would hand the head. That is the whole point: it
        costs one pass over a split rather than a board, and it answers the one
        question the rest of the derived-target gauntlet does not ask.

        The gauntlet's other checks are all about the *target* — its tail, its
        overlap with targets that already ship. None of them asks whether the
        target is **recoverable from patch features at all**. Photometric
        superpixels passed every one of them and then scored 0.02-0.04, because
        a 1px SLIC boundary in a flat region is placed by the seeding lattice
        and sub-threshold colour noise, which is not in a 14px patch token in
        principle. Its oracle would have said so before a backbone was loaded.

        Parameters
        ----------
        labels:
            Targets: a stacked tensor, a sequence of per-image tensors, or any
            dataset exposing ``__len__`` and ``target(index)`` — which
            :class:`~visbench.data.derived.DerivedTargetDataset` and every dense
            folder dataset do, so a candidate target can be measured straight
            off the dataset that generates it.
        grid_hw:
            Feature grid the head would read, as ``(h, w)`` or a single int for
            a square one. 16 is DINOv2 ViT-B/14 at 224px, 14 a ViT-B/16, 7 a
            ResNet's ``layer4``. Coarser grids give lower ceilings; running two
            says how much of a probe's spread is grid resolution rather than
            representation.

        Notes
        -----
        **This is an achievable score, not a proven upper bound**, and the
        distinction from
        :meth:`~visbench.tasks.mid_level.correspondence.CorrespondenceTask.evaluate_ceiling`
        is worth keeping. That one takes a minimum over the candidates a match
        could land on, so ``score <= ceiling`` holds by construction. This one
        reconstructs the target from its patch means, which is the natural
        near-optimal reconstruction and not provably the best one — a head could
        in principle emit grid values that upsample closer. Read it as "a
        competent head with perfect features scores about this", which is what
        the gate needs, and do not report a probe's score as a percentage of it.
        """
        if isinstance(grid_hw, int):
            grid_hw = (grid_hw, grid_hw)
        size = batch_size if batch_size is not None else self.batch_size
        if size < 1:
            raise ValueError(f"batch_size must be >= 1, got {size}")

        def scored() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for batch in self._iter_target_batches(labels, size):
                yield self.oracle_prediction(batch, grid_hw), batch

        return self._average_per_image(scored(), "target set")

    def _iter_target_batches(self, labels: Any, batch_size: int) -> Iterator[torch.Tensor]:
        """Yield ``(B, C, H, W)`` target batches from a tensor, a sequence or a dataset.

        A dataset is read through ``target(index)`` rather than ``__getitem__``
        so the images are never decoded: the oracle does not need them, and on a
        derived target decoding is most of the cost.
        """
        if isinstance(labels, torch.Tensor):
            stacked = self._as_4d(labels.to(self.target_dtype))
            for start in range(0, len(stacked), batch_size):
                yield stacked[start : start + batch_size]
            return

        getter: Any
        if hasattr(labels, "target") and hasattr(labels, "__len__"):
            count, getter = len(labels), labels.target
        elif isinstance(labels, (list, tuple)):
            count, getter = len(labels), labels.__getitem__
        else:
            raise TypeError(
                f"{self.display_name} needs {self.target_noun} as a tensor, a sequence, or a "
                f"dataset with target(index); got {type(labels).__name__}"
            )

        for start in range(0, count, batch_size):
            indices = range(start, min(start + batch_size, count))
            batch = torch.stack([torch.as_tensor(getter(index)) for index in indices])
            yield self._as_4d(batch.to(self.target_dtype))

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

    def finetune(self) -> dict | None:
        """What was unfrozen, for the result record's ``finetune`` field.

        ``None`` for a frozen probe — which is every v0.1 and v0.2 run, and the
        value old records carry by absence. A fine-tuned score and a frozen one
        are not the same measurement, and this is what keeps them apart when
        both sit in the same JSONL under the same task name.
        """
        if self.finetune_blocks == 0:
            return None
        return {
            "blocks": self.finetune_blocks,
            "backbone_lr": self.backbone_lr,
            "trainable_params": self.trainable_params,
            # Cost, not result: a prefix-cached run and a recomputed one must
            # report identical metrics, and recording which was which is what
            # lets a reader check that rather than take it on faith.
            "prefix_cache": self.uses_prefix_cache,
        }
