"""Detect-then-segment instance segmentation on frozen features — step 14a-3.

**This is :class:`~visbench.tasks.high_level.detection.DetectionTask` plus a
mask branch, by inheritance, because everything about the boxes is already
right.** The FCOS-style assignment, the focal and GIoU losses, the distance
decode, the per-class NMS and the ``grid_hw`` bookkeeping are used unchanged; a
second copy of any of them would be a second thing to keep in step with a
published board. What this class adds is: RoI-aligned features, one BCE term,
and a decode that pastes a mask back into its box.

**Why instance segmentation is expressible as a probe at all**, measured in
14a-1 over all 1,449 VOC val images at a 16x16 grid: the median instance covers
16.92 patches, and **zero of 3,207 instance pairs share a grid cell**. No patch
is contested, so per-patch features can carry instance information — even though
an instance *index* is only annotation order and could never be a stable output
channel. A perfect per-patch predictor scores **mask mAP@50 0.6666** at that
grid; the connected-components shortcut scores 0.1376 mean IoU. That band is
what this probe measures inside.

**The mask branch trains on ground-truth boxes and predicts on detected ones**,
which is a real asymmetry and is deliberate. Mask R-CNN trains on positive
*proposals*; here the boxes come from the same head being trained, so early
epochs would supply RoIs that contain no object and the mask branch would spend
them learning to segment background. Training on ground truth decouples "can
these features localise an object" from "can they outline one", which are the
two questions this probe reports separately — mask AP falls if either fails,
and ``box_map_50`` is reported beside it so a reader can tell which. Recorded as
``mask_train_boxes: "ground_truth"``; a variant training on proposals is a
different measurement, not a tweak.

**The mask target comes through the same RoIAlign as the prediction.** Cropping
the ground-truth mask by hand and resizing it separately would put target and
prediction on two sampling grids that agree almost everywhere — the failure
mode this codebase paid for once at ``recall@1px = 0.003``. Running both
through :func:`torchvision.ops.roi_align` with the same box makes the alignment
structural rather than tested.

**Void pixels are weighted out of the mask loss, not labelled.** VOC marks the
outline between touching objects 255 and it is 6.0% of pixels; scoring or
training a prediction there penalises it for a pixel the annotation declines to
label. The ``ignore`` map rides the same RoIAlign and becomes a per-pixel loss
weight, which is the label-map convention (CLAUDE.md, "Validity convention")
rather than the depth one.
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from visbench.heads.base import build_head, get_head
from visbench.heads.instance import InstanceHead
from visbench.metrics.detection import COCO_IOU_THRESHOLDS, detection_metrics
from visbench.metrics.instance import instance_metrics
from visbench.registry import register_task
from visbench.tasks.high_level.detection import DetectionTask
from visbench.types import MetricsDict

__all__ = ["InstanceSegmentationTask"]


@register_task("instance_segmentation")
class InstanceSegmentationTask(DetectionTask):
    """Anchor-free detection plus a linear mask readout, on frozen dense features.

    Registered at 14a-4, alongside the board that gives its dozen fixed tables
    something to say. The name is load-bearing in every one of them — the CLI's
    ``SPECS``, ``HEADLINE_METRICS``, ``METRIC_DIRECTIONS``, ``TARGET_STYLES``,
    both corpus arrays, a gallery figure and a docs page — which is why 14a-3
    shipped the implementation unregistered rather than claiming a name it had
    no board for.

    Parameters
    ----------
    num_classes, image_size:
        As :class:`~visbench.tasks.high_level.detection.DetectionTask`, and with
        the same warning: ``image_size`` converts a grid cell into a pixel
        coordinate and must equal the dataset's, or every cell centre and every
        pasted mask lands in the wrong place while still training.
    mask_size:
        Side of the square RoI the mask is predicted at, before being pasted
        back into its box. 14 rather than Mask R-CNN's 28: the features being
        sampled are on a 16x16 grid, so a 28x28 RoI asks RoIAlign to invent
        detail between patch centres that the backbone never produced. Recorded
        in ``task_params``, because two runs at different values are not
        comparable.
    mask_weight:
        Weight on the mask BCE term relative to the classification loss.
    mask_threshold:
        Probability above which a pasted pixel is foreground. 0.5, and it is
        part of the protocol rather than a display setting: raising it shrinks
        every mask and moves mask AP.
    mask_hidden_dim:
        Passed to :class:`~visbench.heads.instance.InstanceHead`. ``0`` keeps
        the mask branch a single 1x1 convolution, which is what makes a
        difference between two backbones a difference between representations.
    """

    name = "instance_segmentation"
    level = "high_level"

    def __init__(
        self,
        num_classes: int,
        image_size: int = 224,
        head: str = "instance",
        mask_size: int = 14,
        mask_weight: float = 1.0,
        mask_threshold: float = 0.5,
        mask_hidden_dim: int = 0,
        iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            image_size=image_size,
            head=head,
            iou_thresholds=iou_thresholds,
            **kwargs,
        )
        if mask_size < 1:
            raise ValueError(f"mask_size must be >= 1, got {mask_size}")
        if mask_weight < 0:
            raise ValueError(f"mask_weight must be >= 0, got {mask_weight}")
        if not 0.0 < mask_threshold < 1.0:
            raise ValueError(f"mask_threshold must be in (0, 1), got {mask_threshold}")
        if mask_hidden_dim < 0:
            raise ValueError(f"mask_hidden_dim must be >= 0, got {mask_hidden_dim}")

        self.mask_size = int(mask_size)
        self.mask_weight = float(mask_weight)
        self.mask_threshold = float(mask_threshold)
        self.mask_hidden_dim = int(mask_hidden_dim)
        #: Diagnostic, not a result: separates "the features cannot outline an
        #: object" from "the mask branch never converged".
        self.train_mask_loss: float | None = None

    # -- head ----------------------------------------------------------------

    def _build_head(self, channels: int) -> nn.Module:
        """Build the instance head and refuse one with no mask branch.

        Mirrors ``DetectionTask._build_head``'s posture: that method refuses a
        head emitting the wrong channel count, because a silently mis-sized head
        trains and reports a plausible mAP. The mask equivalent is a head with
        no :meth:`mask_logits`, which would fail partway through the first batch
        with an ``AttributeError`` pointing at this file rather than at the
        incompatible choice.
        """
        # Checked on the CLASS, before constructing. Building first and asking
        # afterwards means a head with no mask branch also has no
        # ``mask_hidden_dim`` parameter, so it dies on an unexpected keyword
        # argument -- a traceback pointing at ``build_head`` rather than at the
        # incompatible choice, which is the failure this check exists to avoid.
        head_class = get_head(self.head_name)
        if not callable(getattr(head_class, "mask_logits", None)):
            raise ValueError(
                f"Head {self.head_name!r} has no mask_logits(); instance segmentation needs "
                "a mask branch. Use the default 'instance' head, or register one that "
                "maps (N, C, M, M) RoI features to (N, 1, M, M) logits."
            )

        kwargs: dict = {
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "mask_hidden_dim": self.mask_hidden_dim,
        }
        kwargs.update(self.head_kwargs)
        kwargs["in_channels"] = channels
        self._head_spec: dict = {"kind": "registered", "name": self.head_name, "kwargs": kwargs}
        head = build_head(self.head_name, **kwargs).to(self.device)

        expected = self.num_classes + 4
        emitted = getattr(head, "out_channels", None)
        if emitted != expected:
            raise ValueError(
                f"Head {self.head_name!r} emits {emitted} channels; this probe needs "
                f"{expected} (num_classes={self.num_classes} logits plus 4 box distances)."
            )
        return head

    # -- geometry ------------------------------------------------------------

    def _spatial_scale(self, grid_hw: tuple[int, int]) -> float:
        """Feature cells per image pixel, for :func:`roi_align`.

        ``roi_align`` takes a single scalar, so a non-square grid would be
        sampled at the wrong scale on one axis — silently, since every shape
        still matches. Every backbone in this corpus produces a square grid from
        a square crop, so this refuses the case rather than guessing which axis
        to honour.
        """
        height, width = grid_hw
        if height != width:
            raise ValueError(
                f"This probe samples RoIs with a single spatial scale, so it needs a square "
                f"feature grid; got {grid_hw}. A rectangular grid would be sampled at the "
                "wrong scale on one axis without any shape disagreeing."
            )
        return width / float(self.image_size)

    def _roi_features(
        self, dense: torch.Tensor, boxes_per_image: list[torch.Tensor]
    ) -> torch.Tensor:
        """RoI-aligned features, ``(sum(N_i), C, mask_size, mask_size)``.

        ``aligned=True``, which is the ``roi_align`` fix for the half-pixel
        offset the original implementation carried. It matters more here than in
        detection: the same call produces the mask *target*, so an offset would
        shift target and prediction together and hide itself.
        """
        grid_hw = self._require_grid()
        return roi_align(
            dense,
            [boxes.to(dense.device, dense.dtype) for boxes in boxes_per_image],
            output_size=(self.mask_size, self.mask_size),
            spatial_scale=self._spatial_scale(grid_hw),
            aligned=True,
        )

    def _boxes_labels_masks(
        self, target: dict, drop_difficult: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(boxes, labels, masks, ignore)`` for one annotation, rows aligned.

        ``masks`` is filtered by the same mask that filters ``boxes``, which is
        the whole reason this exists rather than calling
        ``DetectionTask._boxes_and_labels`` and indexing masks separately: a
        dense target and its boxes are paired by row, and filtering one alone
        pairs an instance's box with another instance's mask while every length
        still matches.
        """
        if "masks" not in target:
            raise ValueError(
                "An annotation carries no 'masks'. This probe needs per-instance masks; "
                "VOCInstanceDataset supplies them, DetectionFolderDataset does not."
            )
        masks = torch.as_tensor(target["masks"])
        if masks.ndim != 3:
            raise ValueError(f"'masks' must be (N, H, W), got {tuple(masks.shape)}")

        boxes = torch.as_tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(target["labels"], dtype=torch.int64).reshape(-1)
        if not (boxes.shape[0] == labels.shape[0] == masks.shape[0]):
            raise ValueError(
                f"An annotation has {boxes.shape[0]} boxes, {labels.shape[0]} labels and "
                f"{masks.shape[0]} masks; the three are paired by row."
            )

        keep = torch.ones(boxes.shape[0], dtype=torch.bool)
        if drop_difficult:
            difficult = target.get("difficult")
            if difficult is not None:
                flags = torch.as_tensor(difficult, dtype=torch.bool).reshape(-1)
                if flags.shape[0] != boxes.shape[0]:
                    raise ValueError(
                        f"An annotation has {boxes.shape[0]} boxes and {flags.shape[0]} "
                        "difficult flags; the two are paired by row."
                    )
                keep = ~flags

        ignore = target.get("ignore")
        if ignore is None:
            ignore_map = torch.zeros(masks.shape[-2:], dtype=torch.bool)
        else:
            ignore_map = torch.as_tensor(ignore, dtype=torch.bool)
            if tuple(ignore_map.shape) != tuple(masks.shape[-2:]):
                raise ValueError(
                    f"'ignore' is {tuple(ignore_map.shape)} but the masks are "
                    f"{tuple(masks.shape[-2:])}; both describe the same frame."
                )
        return boxes[keep], labels[keep], masks[keep], ignore_map

    def _require_head(self) -> InstanceHead:
        """The fitted head, narrowed to the one that has a mask branch.

        `DetectionTask`'s version returns `nn.Module`, and `nn.Module`'s
        `__getattr__` is typed `Tensor | Module` — so `head.mask_logits(...)`
        type-checks as calling a tensor rather than as reaching the real
        method. Narrowing here is what `ClassificationTask` already does for
        its `nn.Linear`, and it is a real check rather than a cast: `fit`
        builds this head, but a `probe_state` round trip could hand back
        something else.
        """
        head = super()._require_head()
        if not isinstance(head, InstanceHead):
            raise TypeError(
                f"This probe needs an InstanceHead to predict masks, but its head is a "
                f"{type(head).__name__}. A head built for boxes alone has no mask branch."
            )
        return head

    def _mask_targets(
        self, masks: torch.Tensor, ignore: torch.Tensor, boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mask target, loss weight)``, both ``(N, 1, mask_size, mask_size)``.

        Both produced by the *same* :func:`roi_align` call shape the prediction
        goes through, at ``spatial_scale=1.0`` because a mask is already at image
        resolution. That is what makes target and prediction share one sampling
        grid by construction.

        The weight is ``1 - ignore``, sampled the same way, so a void pixel
        contributes nothing rather than being asserted as background.
        """
        if masks.shape[0] == 0:
            empty = torch.zeros((0, 1, self.mask_size, self.mask_size), dtype=torch.float32)
            return empty, empty.clone()

        # roi_align pairs its `boxes` list with the batch axis of its input, so
        # sampling instance i's mask with instance i's box means presenting the
        # masks as a batch of N and the boxes as N lists of one. Passing a
        # single (N, 4) tensor instead would sample every mask with the *first*
        # box and return correctly shaped nonsense.
        rois = [box[None] for box in boxes]
        target = roi_align(
            masks[:, None].to(torch.float32),
            rois,
            output_size=(self.mask_size, self.mask_size),
            spatial_scale=1.0,
            aligned=True,
        )
        weight = roi_align(
            (~ignore)[None, None]
            .to(torch.float32)
            .expand(masks.shape[0], 1, *ignore.shape)
            .contiguous(),
            rois,
            output_size=(self.mask_size, self.mask_size),
            spatial_scale=1.0,
            aligned=True,
        )
        return target, weight

    # -- training ------------------------------------------------------------

    def _batch_loss(
        self, batch_features: torch.Tensor, batch_targets: list[dict], centres: torch.Tensor
    ) -> torch.Tensor:
        """Detection's loss plus a mask BCE over ground-truth RoIs.

        The detection half is :meth:`DetectionTask._batch_loss` unchanged. The
        mask half is normalised by the number of RoIs, so an image holding one
        instance and one holding twelve contribute comparably — the same
        reasoning that normalises the box loss by positives.

        A batch with no instances at all is legitimate (VOC images whose objects
        the crop removed) and keeps the mask branch in the graph with a zero
        gradient, for the reason the box branch does: dropping a branch makes the
        optimiser step shape-dependent on the batch, which is a silent source of
        irreproducibility across shuffles.
        """
        loss = super()._batch_loss(batch_features, batch_targets, centres)

        head = self._require_head()
        dense = batch_features.to(self.device).float()
        boxes_per_image: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for target in batch_targets:
            boxes, _, masks, ignore = self._boxes_labels_masks(target, drop_difficult=True)
            boxes_per_image.append(boxes)
            mask_target, weight = self._mask_targets(masks, ignore, boxes)
            targets.append(mask_target)
            weights.append(weight)

        roi_features = self._roi_features(dense, boxes_per_image)
        logits = head.mask_logits(roi_features)
        if logits.shape[0] == 0:
            return loss + 0.0 * logits.sum()

        mask_target = torch.cat(targets).to(self.device)
        weight = torch.cat(weights).to(self.device)
        elementwise = F.binary_cross_entropy_with_logits(
            logits, mask_target, weight=weight, reduction="sum"
        )
        # Divided by the weight actually applied, not by the element count, so
        # masking void pixels does not quietly shrink the loss and with it the
        # effective learning rate on the mask branch.
        mask_loss = elementwise / weight.sum().clamp(min=1.0)
        if hasattr(self, "_mask_losses"):
            self._mask_losses.append(float(mask_loss.detach()))
        return loss + self.mask_weight * mask_loss

    def fit(self, features: Any, labels: Any | None = None) -> "InstanceSegmentationTask":
        """Train both branches, recording the mask loss separately.

        Delegates to :meth:`DetectionTask.fit` and reports the **final epoch's
        mean** mask loss, matching what ``train_loss`` reports for the total
        rather than being one batch of it. ``_batch_loss`` appends per batch and
        the tail is averaged here, since the epoch boundary lives in the
        inherited ``fit`` and is not otherwise observable.

        The two losses are worth separating: a low mask AP with a converged mask
        loss says the features do not carry the outline, while a high one says
        the branch never fitted, and those are opposite conclusions from one
        number.
        """
        self._mask_losses: list[float] = []
        super().fit(features, labels)
        if self._mask_losses:
            per_epoch = max(1, len(self._mask_losses) // max(1, self.epochs))
            tail = self._mask_losses[-per_epoch:]
            self.train_mask_loss = float(sum(tail) / len(tail))
        return self

    def training_summary(self) -> dict | None:
        """Total and mask training loss, or ``None`` before :meth:`fit`."""
        summary = super().training_summary()
        if summary is None:
            return None
        if self.train_mask_loss is not None:
            summary["train_mask_loss"] = self.train_mask_loss
        return summary

    # -- inference -----------------------------------------------------------

    def _paste_masks(self, probabilities: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """``(N, 1, M, M)`` RoI probabilities to ``(N, image_size, image_size)`` masks.

        Each RoI is resized to its box's integer pixel extent and written into a
        zero canvas, then thresholded — Mask R-CNN's paste. Bilinear, because the
        RoI probabilities are a continuous field rather than class indices: this
        is the one place in this codebase where interpolating a target-shaped
        thing is right, and it is right because the thresholding happens *after*.
        """
        count = probabilities.shape[0]
        pasted = torch.zeros((count, self.image_size, self.image_size), dtype=torch.bool)
        if count == 0:
            return pasted

        for index in range(count):
            x_min, y_min, x_max, y_max = (float(value) for value in boxes[index])
            left, top = int(x_min), int(y_min)
            right = max(left + 1, int(x_max + 0.5))
            bottom = max(top + 1, int(y_max + 0.5))
            right = min(right, self.image_size)
            bottom = min(bottom, self.image_size)
            if right <= left or bottom <= top:
                continue
            resized = F.interpolate(
                probabilities[index][None],
                size=(bottom - top, right - left),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            pasted[index, top:bottom, left:right] = resized > self.mask_threshold
        return pasted

    def _iter_predictions(self, features: Any, labels: Any | None):
        """Yield ``(prediction, annotation)`` per image, with masks attached.

        The decode is detection's, then each surviving box is RoI-aligned and
        segmented. Boxes come from the head being evaluated rather than from
        ground truth — the asymmetry with training, stated in the module
        docstring — so a mask can only be right where the box was.
        """
        head = self._require_head()
        source = self._source(features, labels)
        centres = self._centres(self._require_grid())
        for batch_features, batch_targets in self._loader(source, shuffle=False):
            classification, distances = self._forward(batch_features)
            dense = batch_features.to(self.device).float()
            for index, target in enumerate(batch_targets):
                prediction = self._decode(
                    classification[index].cpu(), distances[index].cpu(), centres
                )
                boxes = prediction["boxes"]
                roi_features = self._roi_features(dense[index : index + 1], [boxes])
                logits = head.mask_logits(roi_features)
                prediction["masks"] = self._paste_masks(torch.sigmoid(logits).cpu(), boxes)
                yield prediction, target

    @torch.no_grad()
    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """Mask mAP, with the box mAP beside it so a low score is attributable.

        ``mask_map_50`` is the headline. ``box_map_50`` is reported from the
        *same* predictions through detection's own metric, because mask AP falls
        when either half fails and the two numbers together say which: masks
        that are poor inside good boxes is a statement about the features'
        outlines, and it is a different finding from boxes that miss.

        Every prediction is collected before scoring, because average precision
        **is** a split-level ranking. **Masks make that genuinely expensive**,
        which is worth stating rather than tucking away: ``max_detections``
        masks at ``image_size**2`` is 5.0 MB per image at the defaults, and
        VisBench's own DINOv2-S run over VOC val decodes 74.4 detections an
        image, so the collected predictions come to **5.4 GB**. Storing them as
        ``bool`` rather than ``float32`` is what makes that feasible at all —
        the float version is 21.6 GB. Lowering ``max_detections`` or raising
        ``score_threshold`` is the lever if that does not fit, and both are
        recorded in ``task_params`` because both move mask AP.
        """
        predictions: list[dict] = []
        targets: list[dict] = []
        for prediction, target in self._iter_predictions(features, labels):
            predictions.append(prediction)
            targets.append(target)

        if not predictions:
            raise ValueError("Cannot evaluate on an empty feature set")

        metrics = dict(
            instance_metrics(
                predictions,
                targets,
                num_classes=self.num_classes,
                iou_thresholds=self.iou_thresholds,
            )
        )
        # Scored from the predictions already collected, not by calling
        # DetectionTask.evaluate -- that would run the whole split through the
        # backbone a second time, and the two passes could then disagree about
        # which boxes were scored.
        boxed = detection_metrics(
            predictions,
            targets,
            num_classes=self.num_classes,
            iou_thresholds=self.iou_thresholds,
        )
        metrics["box_map_50"] = boxed["map_50"]
        metrics["detections_per_image"] = float(
            sum(len(prediction["scores"]) for prediction in predictions) / len(predictions)
        )
        return metrics

    # -- provenance ----------------------------------------------------------

    def describe(self) -> dict:
        """Task metadata plus everything that shaped the number."""
        described = super().describe()
        described["task_params"].update(
            {
                "mask_size": self.mask_size,
                "mask_weight": self.mask_weight,
                # Part of the protocol, not a display setting: raising it
                # shrinks every mask and moves mask AP.
                "mask_threshold": self.mask_threshold,
                "mask_hidden_dim": self.mask_hidden_dim,
                # The training/inference asymmetry, recorded because a variant
                # training on proposals is a different measurement.
                "mask_train_boxes": "ground_truth",
                "mask_classes": "agnostic",
                # Neither Mask R-CNN's (four convolutions and a deconvolution,
                # on an FPN) nor VOC's own. The *metric* is VOC's matching with
                # a mask overlap; the head and losses are this codebase's, and a
                # protocol field that overclaims is worse than none.
                "protocol": "visbench_anchor_free_instance",
            }
        )
        return described
