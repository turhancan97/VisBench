"""Object detection — v0.3, step 6c-3. An anchor-free single-scale probe.

The last piece of step 6c, and built last on purpose: the box dataset (6c-1)
and the VOC metric (6c-2) came first so that this head could be judged against
a scorer already cross-checked against ``VOCevaldet.m`` to zero difference. A
number from a head measured by an untrusted metric would have proved nothing.

**The protocol, and why each piece is what it is**

*Assignment* is FCOS's (Tian et al., 2019) reduced to one scale: a grid cell is
positive for a ground-truth box when the cell's **centre falls inside** it, and
an ambiguous cell takes the box of smallest area. FCOS resolves ambiguity by
assigning each scale a size range and letting the pyramid separate overlapping
objects; with one feature level there is no pyramid to do that, so the
smallest-area rule is what remains. It is also FCOS's own tie-break within a
level, so this is a subset of the reference rule rather than a substitute for
it.

*Losses* are sigmoid focal loss (Lin et al., 2017) on the classification branch
and GIoU loss (Rezatofighi et al., 2019) on the positives' box distances.
Focal because a dense anchor-free grid is overwhelmingly background — 256 cells
against a handful of objects — and plain BCE there converges to predicting
nothing while reporting a falling loss. GIoU because the plain IoU loss has
**zero gradient when the boxes do not overlap**, which is exactly the state
every box is in at initialisation.

*Decoding* is threshold, then per-class NMS, then a cap on detections per
image. All three are recorded in ``task_params``: they are not incidental, a
threshold change moves mAP, and two runs that disagree about them are not
comparable.

**The number this produces is low, and that is the design.** A single-scale
linear head over a 16x16 patch grid cannot localise the way an FPN detector
does; small objects fall between cells and are unrecoverable. What it can do is
rank representations, which is what VisBench is for. Do not read its mAP
against published VOC numbers — read it against another backbone's.

**Not a subclass of** :class:`~visbench.tasks.dense_base.DenseTrainingTask`,
and it was not a close call. That base assumes a stackable ``(B, C, H, W)``
target and recovers a split metric by weighting per-image metrics by batch
size. Detection has neither: its target is a variable-length list of boxes, and
average precision is a **dataset-level** quantity built by ranking every
detection in the split at once, which no weighted mean of per-batch numbers
reproduces (see :mod:`visbench.metrics.detection`). What it does share — the
optimiser and probe3d's warmup/cosine schedule — is shared explicitly, through
:mod:`visbench.tasks.schedule`, so a detection number and a segmentation number
differ in the head and the loss rather than in the optimisation.

**Fine-tuning (6a/6b) is not wired up here.** ``finetune_blocks`` stays 0; the
trainable-backbone path lives on ``DenseTrainingTask`` and detection does not
inherit it. A frozen probe is what every VisBench number is anyway.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from visbench.cache.streaming import CachedFeatures
from visbench.heads import build_head
from visbench.metrics.detection import COCO_IOU_THRESHOLDS, box_iou, detection_metrics
from visbench.registry import register_task
from visbench.tasks.base import BaseTask
from visbench.tasks.schedule import check_schedule, warmup_cosine
from visbench.types import FeatureMode, MetricsDict, Pooling
from visbench.utils.device import resolve_device

__all__ = ["DetectionTask"]

#: Ceiling on ``exp(raw)`` in the box branch. ``exp`` is unbounded and a single
#: early step can send it to ``inf``, which turns every later loss into ``nan``
#: and reports 0.0 mAP as though the representation were useless. Clamping the
#: *exponent* keeps the gradient intact everywhere below the ceiling; e^8 is
#: 2981 strides, far beyond any box in a 224px frame.
_MAX_LOG_DISTANCE = 8.0


class _PairedFeatures(Dataset):
    """Features plus their box annotations, paired **by index**.

    By index and never by iteration order — the loop below shuffles every
    epoch, and boxes that travel by position drift from their features the
    moment it does. This is the same rule dense targets follow; it is repeated
    here because the target is a dict rather than a tensor and so cannot reuse
    :class:`~visbench.tasks.dense_base._WithTargets`.
    """

    def __init__(self, features: Any, targets: Sequence[dict] | Callable[[int], dict]) -> None:
        self.features = features
        self.targets = targets
        count = len(features)
        if not callable(targets) and len(targets) != count:
            raise ValueError(f"Got {count} feature maps for {len(targets)} annotations")
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[Any, dict]:
        target = self.targets(index) if callable(self.targets) else self.targets[index]
        return self.features[index], target


def _collate(batch: list) -> tuple[torch.Tensor, list[dict]]:
    """Stack the feature maps; leave the annotations a list.

    The default collation would try to stack the target dicts and fail, or
    worse, stack ``boxes`` tensors of different lengths. A batch of detection
    targets is *irreducibly* a list — that is the shape difference that keeps
    this task off ``DenseTrainingTask``.
    """
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


@register_task("detection")
class DetectionTask(BaseTask):
    """Anchor-free single-scale detection on frozen dense features.

    ``num_classes`` and ``image_size`` are both required and neither has a safe
    default. ``num_classes`` sets the head's width and a wrong value trains a
    head that cannot express some categories while reporting a plausible mAP.
    ``image_size`` is worse: box targets are in **absolute post-transform
    pixels** (see :mod:`visbench.data.detection`), so it is what converts a grid
    cell into a pixel coordinate, and a value disagreeing with the dataset's
    puts every cell centre in the wrong place — which trains, and scores badly,
    and looks like a weak backbone. Pass the same number to both; the CLI wires
    one flag to both for exactly that reason, and :meth:`fit` refuses targets
    that fall outside the resolution it was given.
    """

    name = "detection"
    level = "high_level"
    feature_mode = FeatureMode.DENSE_ONLY
    zero_shot = False
    uses_dense = True
    #: Dense tasks read the grid, not the pooled vector, but extraction still
    #: needs a pooling name for the cache key. Mean, matching every other dense
    #: probe, so these entries do not collide with a CLS-pooled run.
    pooling = Pooling.MEAN

    def __init__(
        self,
        num_classes: int,
        image_size: int = 224,
        head: str = "detection",
        hidden_dim: int = 0,
        epochs: int = 10,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        warmup_epochs: float = 1.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_weight: float = 2.0,
        score_threshold: float = 0.05,
        nms_iou: float = 0.5,
        max_detections: int = 100,
        iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
        head_kwargs: dict | None = None,
        device: str | None = None,
    ) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        check_schedule(epochs, warmup_epochs)
        if not 0.0 <= score_threshold < 1.0:
            raise ValueError(f"score_threshold must be in [0, 1), got {score_threshold}")
        if not 0.0 < nms_iou <= 1.0:
            raise ValueError(f"nms_iou must be in (0, 1], got {nms_iou}")
        if max_detections < 1:
            raise ValueError(f"max_detections must be >= 1, got {max_detections}")
        if focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {focal_gamma}")
        if not 0.0 < focal_alpha < 1.0:
            raise ValueError(f"focal_alpha must be in (0, 1), got {focal_alpha}")

        self.num_classes = num_classes
        self.image_size = int(image_size)
        self.head_name = head
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.box_weight = box_weight
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.iou_thresholds = tuple(float(value) for value in iou_thresholds)
        self.head_kwargs = dict(head_kwargs or {})
        self.device = resolve_device(device)

        self.head: nn.Module | None = None
        #: Grid the head was built for, set by :meth:`fit`. Kept so decoding
        #: cannot silently run against a different geometry than training did.
        self.grid_hw: tuple[int, int] | None = None
        #: Diagnostic, not a result: a low mAP with a high training loss is an
        #: underfitted probe, which is a different finding from a
        #: representation that does not carry the signal.
        self.train_loss: float | None = None

    # -- geometry ------------------------------------------------------------

    def _strides(self, grid_hw: tuple[int, int]) -> tuple[float, float]:
        """Pixels per grid cell, per axis."""
        return (self.image_size / grid_hw[1], self.image_size / grid_hw[0])

    def _centres(self, grid_hw: tuple[int, int]) -> torch.Tensor:
        """``(H * W, 2)`` cell centres in post-transform pixels, row-major.

        Row-major, matching ``(C, H, W).flatten(1)``, because the classification
        map and these coordinates are indexed by the same flat position. A
        transposed grid would put every prediction in the wrong place and still
        train.
        """
        height, width = grid_hw
        stride_x, stride_y = self._strides(grid_hw)
        ys = (torch.arange(height, dtype=torch.float32) + 0.5) * stride_y
        xs = (torch.arange(width, dtype=torch.float32) + 0.5) * stride_x
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    def _distances(self, raw: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """Raw regression output to left/top/right/bottom **pixel** distances.

        ``exp`` scaled by the stride, as FCOS does: the exponential keeps every
        distance positive without a hard floor that would kill its gradient, and
        the stride makes the parameterisation resolution-independent, so a head
        trained at one grid size starts sensibly at another.
        """
        stride_x, stride_y = self._strides(grid_hw)
        scale = raw.new_tensor([stride_x, stride_y, stride_x, stride_y])
        return torch.exp(raw.clamp(max=_MAX_LOG_DISTANCE)) * scale

    # -- assignment ----------------------------------------------------------

    def _assign(
        self, target: dict, centres: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ground truth for one image, as ``(class targets, ltrb, positives)``.

        Returns a ``(HW, num_classes)`` multi-hot classification target (all
        zeros for a background cell, which is what focal loss expects), a
        ``(HW, 4)`` distance target that is meaningful only where ``positives``
        is set, and the ``(HW,)`` positive mask.

        ``difficult`` objects are **dropped here**, always. The dataset's
        ``include_difficult`` decides what reaches this method, and a scoring
        run must set it True — VOC's protocol needs those boxes present so a
        detection matching one can be *ignored* rather than counted (6c-2
        measured the difference at 4.3 mAP). But a difficult object is one the
        annotators judged unreasonable to require, and training against it is a
        separate question from scoring against it. So this drops them whatever
        the dataset kept, and the same dataset can serve both halves.
        """
        count = centres.shape[0]
        class_target = torch.zeros((count, self.num_classes), dtype=torch.float32)
        distance_target = torch.zeros((count, 4), dtype=torch.float32)
        positives = torch.zeros(count, dtype=torch.bool)

        boxes, labels = self._boxes_and_labels(target, drop_difficult=True)
        if boxes.shape[0] == 0:
            return class_target, distance_target, positives

        centre_x = centres[:, 0:1]
        centre_y = centres[:, 1:2]
        # (HW, M, 4): every cell's distance to every box's four edges.
        distances = torch.stack(
            [
                centre_x - boxes[None, :, 0],
                centre_y - boxes[None, :, 1],
                boxes[None, :, 2] - centre_x,
                boxes[None, :, 3] - centre_y,
            ],
            dim=-1,
        )

        # Strictly inside: a centre exactly on an edge gives a zero-width
        # target box, whose GIoU is degenerate and whose gradient says nothing.
        inside = distances.min(dim=-1).values > 0

        # Smallest area wins an ambiguous cell. With one feature level there is
        # no pyramid to separate overlapping objects by size, so this rule —
        # FCOS's own within-level tie-break — is what is left of its assignment.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ranked = torch.where(
            inside,
            areas[None, :].expand_as(inside),
            torch.full_like(inside, float("inf"), dtype=torch.float32),
        )
        best_area, best = ranked.min(dim=1)
        positives = torch.isfinite(best_area)

        if bool(positives.any()):
            chosen = best[positives]
            class_target[positives, labels[chosen]] = 1.0
            distance_target[positives] = distances[positives, chosen]
        return class_target, distance_target, positives

    def _boxes_and_labels(
        self, target: dict, drop_difficult: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(boxes, labels)`` from an annotation dict, validated against the frame."""
        boxes = torch.as_tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(target["labels"], dtype=torch.int64).reshape(-1)
        if boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                f"An annotation has {boxes.shape[0]} boxes and {labels.shape[0]} labels; "
                "the two are paired by row."
            )
        if drop_difficult:
            difficult = target.get("difficult")
            if difficult is not None:
                keep = ~torch.as_tensor(difficult, dtype=torch.bool).reshape(-1)
                if keep.shape[0] != boxes.shape[0]:
                    raise ValueError(
                        f"An annotation has {boxes.shape[0]} boxes and {keep.shape[0]} "
                        "difficult flags; the two are paired by row."
                    )
                boxes, labels = boxes[keep], labels[keep]
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
            raise ValueError(
                f"An annotation names class index {int(labels.max())} but this probe was "
                f"built for num_classes={self.num_classes}. The head has no channel for it, "
                "and the class would be invisible to both training and scoring."
            )
        if boxes.numel() and float(boxes.max()) > self.image_size + 1e-3:
            raise ValueError(
                f"A box reaches {float(boxes.max()):.1f}px but this probe was built for "
                f"image_size={self.image_size}. Box targets are absolute pixels in "
                "post-transform space, so the dataset's image_size and this probe's must "
                "be the same number — otherwise every grid cell maps to the wrong "
                "coordinate and the probe trains against shifted supervision."
            )
        return boxes, labels

    # -- losses --------------------------------------------------------------

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Sigmoid focal loss, summed. Lin et al., 2017, unchanged."""
        probability = torch.sigmoid(logits)
        cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
        modulation = (1.0 - p_t) ** self.focal_gamma
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        return (alpha_t * modulation * cross_entropy).sum()

    @staticmethod
    def _giou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``1 - GIoU`` between two sets of ltrb distances about a shared centre, summed.

        Computed from the distances rather than from decoded corners because
        both boxes share a centre, which makes the intersection and the
        enclosing box exact one-liners — and because the plain IoU loss it
        generalises has no gradient at all when the boxes are disjoint, the
        state every prediction starts in.
        """
        predicted_area = (predicted[:, 0] + predicted[:, 2]) * (predicted[:, 1] + predicted[:, 3])
        target_area = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])

        intersect_w = torch.min(predicted[:, 0], target[:, 0]) + torch.min(
            predicted[:, 2], target[:, 2]
        )
        intersect_h = torch.min(predicted[:, 1], target[:, 1]) + torch.min(
            predicted[:, 3], target[:, 3]
        )
        intersection = intersect_w.clamp(min=0) * intersect_h.clamp(min=0)
        union = predicted_area + target_area - intersection

        enclose_w = torch.max(predicted[:, 0], target[:, 0]) + torch.max(
            predicted[:, 2], target[:, 2]
        )
        enclose_h = torch.max(predicted[:, 1], target[:, 1]) + torch.max(
            predicted[:, 3], target[:, 3]
        )
        enclose = (enclose_w * enclose_h).clamp(min=1e-7)

        iou = intersection / union.clamp(min=1e-7)
        giou = iou - (enclose - union) / enclose
        return (1.0 - giou).sum()

    # -- decoding ------------------------------------------------------------

    def _decode(
        self, logits: torch.Tensor, distances: torch.Tensor, centres: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """One image's raw head output to ``{boxes, scores, labels}``.

        Applied in :meth:`predict` and :meth:`evaluate` alike, so the boxes a
        caller inspects are the boxes that were scored.
        """
        scores = torch.sigmoid(logits)
        cells, classes = torch.where(scores > self.score_threshold)
        if cells.numel() == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }

        selected = scores[cells, classes]
        # Bound the work NMS does on a pathologically low threshold, keeping the
        # highest-scoring candidates — which are the only ones NMS can keep.
        if selected.numel() > self.max_detections * 10:
            top = torch.topk(selected, self.max_detections * 10).indices
            cells, classes, selected = cells[top], classes[top], selected[top]

        centre = centres[cells]
        distance = distances[cells]
        boxes = torch.stack(
            [
                centre[:, 0] - distance[:, 0],
                centre[:, 1] - distance[:, 1],
                centre[:, 0] + distance[:, 2],
                centre[:, 1] + distance[:, 3],
            ],
            dim=1,
        ).clamp(min=0.0, max=float(self.image_size))

        keep = self._nms(boxes, selected, classes)
        keep = keep[: self.max_detections]
        return {"boxes": boxes[keep], "scores": selected[keep], "labels": classes[keep]}

    def _nms(
        self, boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor
    ) -> torch.Tensor:
        """Per-class non-maximum suppression, returned as indices in score order.

        Per class via the usual offset trick: each class's boxes are shifted
        into their own disjoint region of the plane, so a single global pass
        can never suppress across classes. The offset is larger than the frame,
        which is what guarantees the regions do not touch.
        """
        offset = classes.to(boxes.dtype)[:, None] * (self.image_size + 1.0)
        shifted = boxes + offset

        order = torch.argsort(scores, descending=True, stable=True)
        kept: list[int] = []
        while order.numel() > 0:
            current = int(order[0])
            kept.append(current)
            if order.numel() == 1:
                break
            overlaps = box_iou(shifted[current][None, :], shifted[order[1:]])[0]
            order = order[1:][overlaps <= self.nms_iou]
        return torch.tensor(kept, dtype=torch.int64)

    # -- feature sources -----------------------------------------------------

    def _source(self, features: Any, labels: Any | None) -> Dataset:
        """Normalise whatever was passed into one indexable ``(features, annotation)``.

        Two front doors, one loop — the same arrangement
        :class:`~visbench.tasks.dense_base.DenseTrainingTask` uses. A
        :class:`~visbench.cache.CachedFeatures` built by ``materialise`` already
        carries its annotations by index and is used as is; anything else needs
        ``labels``.
        """
        if isinstance(features, CachedFeatures):
            if features.targets is not None:
                if labels is not None:
                    raise ValueError(
                        "These features already carry annotations "
                        "(materialise(targets=...)), so passing labels as well gives two "
                        "sources of truth for the supervision. Pass one."
                    )
                return features
            if labels is None:
                raise ValueError(
                    "Detection requires box annotations. Either pass them here, or build "
                    "the features with materialise(targets=dataset.target), which keeps "
                    "each annotation with its own image."
                )
            return _PairedFeatures(features, labels)

        dense = features["dense"] if isinstance(features, dict) else features
        if not isinstance(dense, torch.Tensor):
            raise TypeError(
                "Detection needs dense features as a (N, C, H, W) tensor or a streaming "
                f"CachedFeatures, got {type(dense).__name__}. A list means layers=[...], "
                "which this single-scale head does not take."
            )
        if dense.ndim != 4:
            raise ValueError(
                f"Expected dense features of shape (N, C, H, W), got {tuple(dense.shape)}"
            )
        if labels is None:
            raise ValueError("Detection requires box annotations; got None")
        return _PairedFeatures(dense.float(), labels)

    def _loader(self, source: Dataset, shuffle: bool) -> DataLoader:
        """Batch a source, reshuffling each epoch when training.

        No explicit generator, so the shuffle is seeded from the global RNG that
        :func:`visbench.utils.set_seed` owns and the seed recorded next to the
        metrics still governs the run.
        """
        return DataLoader(source, batch_size=self.batch_size, shuffle=shuffle, collate_fn=_collate)

    # -- training ------------------------------------------------------------

    def fit(self, features: Any, labels: Any | None = None) -> "DetectionTask":
        """Train the head on dense features and their box annotations."""
        source = self._source(features, labels)
        count = len(source)  # type: ignore[arg-type]
        if count == 0:
            raise ValueError("Cannot fit on an empty feature set")

        sample_features, _ = source[0]  # type: ignore[index]
        channels = int(sample_features.shape[0])
        self.grid_hw = (int(sample_features.shape[1]), int(sample_features.shape[2]))
        self.head = self._build_head(channels)
        self.head.train()

        optimiser = torch.optim.AdamW(
            self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loader = self._loader(source, shuffle=True)
        steps_per_epoch = max(1, len(loader))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimiser,
            warmup_cosine(steps_per_epoch * self.epochs, steps_per_epoch, self.warmup_epochs),
        )
        # Kept on CPU: assignment is per-image indexing over a handful of boxes,
        # which is cheaper here than shuttling every annotation to the device.
        centres = self._centres(self.grid_hw)

        running = 0.0
        for _ in range(self.epochs):
            running = 0.0
            for batch_features, batch_targets in loader:
                optimiser.zero_grad()
                loss = self._batch_loss(batch_features, batch_targets, centres)
                loss.backward()
                optimiser.step()
                scheduler.step()
                running += loss.item()
            running /= steps_per_epoch

        self.head.eval()
        self.train_loss = running
        return self

    def _build_head(self, channels: int) -> nn.Module:
        """Instantiate the configured head, sized to these features."""
        kwargs: dict = {"num_classes": self.num_classes, "hidden_dim": self.hidden_dim}
        kwargs.update(self.head_kwargs)
        kwargs["in_channels"] = channels
        # See DenseTrainingTask._build_head: `channels` is measured from the
        # features, so the recipe has to be captured where it is known.
        self._head_spec: dict = {"kind": "registered", "name": self.head_name, "kwargs": kwargs}
        head = build_head(self.head_name, **kwargs).to(self.device)
        expected = self.num_classes + 4
        emitted = getattr(head, "out_channels", None)
        if emitted != expected:
            raise ValueError(
                f"Head {self.head_name!r} emits {emitted} channels; detection needs "
                f"{expected} (num_classes={self.num_classes} logits plus 4 box distances). "
                "Register a head that does, or use the default 'detection' head."
            )
        return head

    def training_summary(self) -> dict | None:
        """The final epoch's mean training loss, for the record.

        Detection has no training *accuracy* to report — its metric is a
        dataset-level ranking, not a per-example decision — so the loss is the
        whole summary. It is worth reading here in particular: this probe's mAP
        is low **by design**, since an anchor-free single-scale head has no
        feature pyramid, and the loss is what distinguishes that intended floor
        from a run that failed to converge.

        ``None`` before :meth:`fit`, so a record can never claim a fit that did
        not happen.
        """
        if self.train_loss is None:
            return None
        return {"train_loss": self.train_loss}

    def head_spec(self) -> dict | None:
        return getattr(self, "_head_spec", None)

    def probe_state(self) -> dict[str, torch.Tensor]:
        """``grid_hw``, which is fitted state living outside the head.

        Every cell centre — and therefore every decoded box — is computed from
        the grid the head was fitted on. Saved without it, the weights load
        cleanly and ``predict`` raises "this probe has not been fitted", which
        is at least loud; the quieter hazard is a future caller that defaults
        the grid instead and decodes every box against the wrong pixel
        coordinates. This is the case ``probe_state`` exists for, and it is the
        same one ``ClassificationTask``'s standardiser was.
        """
        if self.grid_hw is None:
            return {}
        return {"grid_hw": torch.tensor(self.grid_hw, dtype=torch.int64)}

    def load_probe_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore :meth:`probe_state`, refusing a key this probe cannot use."""
        unexpected = sorted(set(state) - {"grid_hw"})
        if unexpected:
            raise ValueError(
                f"{type(self).__name__} cannot restore {unexpected}. Refusing rather "
                "than dropping it: the weights were fitted alongside these tensors."
            )
        if "grid_hw" in state:
            height, width = (int(value) for value in state["grid_hw"])
            self.grid_hw = (height, width)

    def _batch_loss(
        self, batch_features: torch.Tensor, batch_targets: list[dict], centres: torch.Tensor
    ) -> torch.Tensor:
        """Focal classification loss plus GIoU box loss, normalised by positives.

        Normalised by the number of positive cells rather than by batch size,
        which is what makes the loss comparable between an image holding one
        object and one holding twelve. A batch with no positives at all is
        legitimate — VOC images whose objects were all cropped away — and is
        divided by 1 rather than by 0.
        """
        classification, distances = self._forward(batch_features)

        class_targets = []
        distance_targets = []
        positive_masks = []
        for target in batch_targets:
            class_target, distance_target, positives = self._assign(target, centres)
            class_targets.append(class_target)
            distance_targets.append(distance_target)
            positive_masks.append(positives)

        class_target_batch = torch.stack(class_targets).to(self.device)
        distance_target_batch = torch.stack(distance_targets).to(self.device)
        positive_batch = torch.stack(positive_masks).to(self.device)
        num_positive = max(1, int(positive_batch.sum()))

        loss = self._focal_loss(classification, class_target_batch) / num_positive
        if bool(positive_batch.any()):
            loss = loss + self.box_weight * (
                self._giou_loss(distances[positive_batch], distance_target_batch[positive_batch])
                / num_positive
            )
        else:
            # Keep the box branch in the graph so its parameters still receive a
            # (zero) gradient; dropping it entirely makes the optimiser step
            # shape-dependent on the batch, which is a silent source of
            # irreproducibility across shuffles.
            loss = loss + 0.0 * distances.sum()
        return loss

    def _forward(self, batch_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A batch of features to ``(class logits, ltrb distances)``, both flat per cell.

        Shapes are ``(B, HW, num_classes)`` and ``(B, HW, 4)``. Flattening here
        rather than in each caller is what keeps the cell ordering — row-major,
        matching :meth:`_centres` — defined in exactly one place.
        """
        head = self._require_head()
        grid_hw = self._require_grid()
        dense = batch_features.to(self.device).float()
        if tuple(dense.shape[-2:]) != grid_hw:
            raise ValueError(
                f"These features are on a {tuple(dense.shape[-2:])} grid but the head was "
                f"fitted on {grid_hw}. The cell centres would name different pixels, so "
                "every prediction and every target would be misplaced."
            )
        raw = head(dense)
        classification = raw[:, : self.num_classes].flatten(2).transpose(1, 2)
        distances = self._distances(raw[:, self.num_classes :].flatten(2).transpose(1, 2), grid_hw)
        return classification, distances

    def _require_head(self) -> nn.Module:
        if self.head is None:
            raise RuntimeError(
                "This probe has not been fitted. Call fit(train_features, train_targets) "
                "before predict() or evaluate()."
            )
        return self.head

    def _require_grid(self) -> tuple[int, int]:
        if self.grid_hw is None:
            raise RuntimeError("This probe has not been fitted; there is no feature grid yet.")
        return self.grid_hw

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def predict(self, features: Any, labels: Any | None = None) -> list[dict[str, torch.Tensor]]:
        """Detections per image, in dataset order.

        One dict per image with ``boxes`` ``(N, 4)`` ``xyxy`` in post-transform
        pixels, ``scores`` ``(N,)`` and ``labels`` ``(N,)`` — exactly the shape
        :func:`visbench.metrics.detection.average_precision` consumes, so what
        a caller inspects and what gets scored are the same objects.
        """
        return [prediction for prediction, _ in self._iter_predictions(features, labels)]

    def _iter_predictions(self, features: Any, labels: Any | None):
        """Yield ``(prediction, annotation)`` per image, in dataset order."""
        self._require_head()
        source = self._source(features, labels)
        centres = self._centres(self._require_grid())
        for batch_features, batch_targets in self._loader(source, shuffle=False):
            classification, distances = self._forward(batch_features)
            for index, target in enumerate(batch_targets):
                yield (
                    self._decode(classification[index].cpu(), distances[index].cpu(), centres),
                    target,
                )

    @torch.no_grad()
    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """mAP@50, mAP@50:95 and the number of classes that had a defined AP.

        Every prediction in the split is collected before scoring, because
        average precision **is** a split-level ranking and cannot be
        accumulated batch by batch (see :mod:`visbench.metrics.detection`).
        That is bounded work, not the memory hazard dense predictions are:
        ``max_detections`` boxes per image is at most a few hundred floats,
        against a dense map's quarter-million.

        ``difficult`` objects must still be **present** in the annotations here.
        The metric ignores a detection that matches one, which is VOC's rule;
        annotations with them already removed silently score 4.3 mAP lower on
        VOC val, and look like a weaker detector rather than a changed protocol.
        Build the scored split with ``include_difficult=True``.
        """
        predictions: list[dict] = []
        targets: list[dict] = []
        for prediction, target in self._iter_predictions(features, labels):
            predictions.append(prediction)
            targets.append(target)

        if not predictions:
            raise ValueError("Cannot evaluate on an empty feature set")

        metrics = detection_metrics(
            predictions,
            targets,
            num_classes=self.num_classes,
            iou_thresholds=self.iou_thresholds,
        )
        metrics["detections_per_image"] = float(
            sum(len(prediction["scores"]) for prediction in predictions) / len(predictions)
        )
        return metrics

    # -- provenance ----------------------------------------------------------

    def describe(self) -> dict:
        """Task metadata plus everything that shaped the number."""
        described = super().describe()
        described["task_params"] = {
            "head": self.head_name,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "warmup_epochs": self.warmup_epochs,
            "optimizer": "adamw",
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "box_weight": self.box_weight,
            # Decoding is part of the protocol, not a display setting: raising
            # the threshold or loosening NMS moves mAP, so two runs that
            # disagree about these are not comparable.
            "score_threshold": self.score_threshold,
            "nms_iou": self.nms_iou,
            "max_detections": self.max_detections,
            "iou_thresholds": list(self.iou_thresholds),
            # Neither probe3d (which has no detection task) nor VOC's own
            # detector. The *metric* is VOC's; the head and losses are this
            # codebase's, and a protocol field that overclaims is worse than
            # none — see SemanticSegmentationTask for the same call.
            "protocol": "visbench_anchor_free_det",
        }
        return described
