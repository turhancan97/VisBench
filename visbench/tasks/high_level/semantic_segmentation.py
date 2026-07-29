"""Semantic (multi-class) segmentation — a trained dense probe, v0.2.

The high-level counterpart to mid-level *generic* (binary) object segmentation:
same base class, same schedule, same feature path, and deliberately so, because
the point of having both is that a difference between the numbers is a
difference between *what is being asked of the representation* — "is this pixel
part of an object" versus "which of 21 categories is it" — and not a difference
in how they were trained.

**Not probe3d's protocol.** That paper has no semantic segmentation task, so
there is nothing of its to borrow beyond the optimiser schedule its dense probes
share. The record says ``visbench_semantic_seg`` rather than ``probe3d``: a
protocol field that overclaims is worse than none, because the only thing it is
for is saying what a number may be compared against.

**Two mIoUs, both reported.** The dataset-level reduction — one confusion matrix
over the whole split, ratios taken once — is what VOC, ADE20K and Cityscapes
define, and the only one comparable to published numbers. This codebase's
standing rule is per-image-then-averaged, for the good reason that pooling every
pixel lets uneven valid-pixel coverage reweight a dataset silently. The two
genuinely disagree, so both are reported under distinct names (``miou`` and
``miou_per_image``) rather than one being chosen and the reader left to guess
which they are reading.
"""

from typing import Any

import torch
import torch.nn.functional as F

from visbench.metrics.dense import (
    confusion_matrix,
    metrics_from_confusion,
    semantic_metrics,
)
from visbench.registry import register_task
from visbench.tasks.dense_base import DenseTrainingTask
from visbench.types import MetricsDict

__all__ = ["SemanticSegmentationTask", "IGNORE_INDEX"]

#: Value marking an unlabelled pixel in a target map. Negative, not 0, because
#: **0 is a real class** — background — in every label map. Reusing the depth
#: convention, where 0 means invalid, would discard every background pixel and
#: train the probe to answer foreground everywhere.
IGNORE_INDEX = -1


@register_task("semantic_segmentation")
class SemanticSegmentationTask(DenseTrainingTask):
    """Per-pixel category prediction, linear or DPT head on frozen features.

    ``num_classes`` has no default on purpose. It is a property of the dataset,
    it sets the head's output width, and getting it wrong does not raise — it
    trains a head that cannot express some categories, or one carrying dead
    channels, and reports a plausible number either way.
    """

    level = "high_level"
    display_name = "Semantic segmentation"
    target_noun = "label maps"

    #: One channel of class *indices*, not one channel per class. The head emits
    #: ``num_classes`` scores; the ground truth stays a single index map.
    target_channels = 1

    #: Cross-entropy needs class indices, and the base coerces targets in one
    #: place so training, evaluation and predict cannot disagree.
    target_dtype = torch.long

    def __init__(
        self,
        num_classes: int,
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
        if num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2, got {num_classes}. One class is not a "
                "classification; use generic_segmentation for foreground/background."
            )
        super().__init__(
            head=head,
            layers=layers,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            head_kwargs=head_kwargs,
            device=device,
            finetune_blocks=finetune_blocks,
            backbone_lr=backbone_lr,
        )
        self.num_classes = num_classes
        self.name = "semantic_segmentation"

    @property
    def out_channels(self) -> int:
        return self.num_classes

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        """Identity: the head's raw scores are the prediction.

        Deliberately not a softmax. The base applies this in the loss, the
        metrics *and* ``predict``, so whatever it returns is what all three see —
        and cross-entropy needs logits, being defined as log-softmax plus NLL
        with the two fused for numerical stability. Applying softmax here and
        taking its log inside the loss would undo that fusion for no gain:
        ``argmax`` is indifferent to any monotone transform, so the metrics score
        identically either way.

        ``predict`` therefore returns ``(B, num_classes, H, W)`` scores. Take
        ``.argmax(dim=1)`` for a label map, or call :meth:`predict_labels`.
        """
        return raw

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Masked cross-entropy over class indices.

        ``ignore_index`` drops unlabelled pixels from the gradient entirely, and
        it is the same set the metric drops — the loss and the score must mask
        identically, or the probe is optimised against pixels it is not scored
        on.
        """
        if pred.shape[1] != self.num_classes:
            raise ValueError(
                f"Head emitted {pred.shape[1]} channels for {self.num_classes} classes"
            )
        indices = target.squeeze(1) if target.ndim == 4 else target
        return F.cross_entropy(pred, indices.long(), ignore_index=IGNORE_INDEX)

    def _batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
        """Per-image averages, which is what lets the base weight batches by size."""
        return semantic_metrics(pred, target, self.num_classes)

    @torch.no_grad()
    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """Score the split, reporting both reductions of mIoU.

        Overrides the base because a dataset-level mIoU cannot be recovered from
        per-batch averages: it needs one confusion matrix accumulated over every
        image, and no weighted mean of per-batch ratios equals the ratio of the
        sums. The per-image numbers are accumulated in the same pass, so this
        costs one traversal, not two.
        """
        self._require_head()
        matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        totals: dict[str, float] = {}
        count = 0

        for batch_features, batch_targets in self._iter_batches(features, labels):
            assert batch_targets is not None  # targets_required defaults to True
            predicted = self._forward(batch_features).cpu()
            targets = self._as_4d(torch.as_tensor(batch_targets).to(self.target_dtype))

            matrix += confusion_matrix(predicted, targets, self.num_classes)

            size = len(targets)
            for name, value in self._batch_metrics(predicted, targets).items():
                totals[name] = totals.get(name, 0.0) + value * size
            count += size

        if count == 0:
            raise ValueError(f"{self.display_name} got an empty split to evaluate")

        metrics = metrics_from_confusion(matrix)
        metrics.update({name: total / count for name, total in totals.items()})
        return metrics

    def predict_labels(self, features: Any) -> torch.Tensor:
        """``(N, H, W)`` predicted class indices, for saving or visualising."""
        return self.predict(features).argmax(dim=1)

    def _task_params(self) -> dict:
        return {
            "protocol": "visbench_semantic_seg",
            "loss": "cross_entropy",
            "num_classes": self.num_classes,
            "ignore_index": IGNORE_INDEX,
            "miou_reduction": "dataset_and_per_image",
        }
