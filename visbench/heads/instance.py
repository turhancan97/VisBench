"""The detect-then-segment instance head — step 14a-3.

A :class:`~visbench.heads.detection.DetectionHead` for the boxes, plus **one
1x1 convolution** for the masks. That second branch is the whole of the mask
model: RoI-aligned features in, one logit per RoI pixel out.

**Why one convolution and not Mask R-CNN's four.** Mask R-CNN's mask branch is
four 3x3 convolutions and a transposed convolution, and every point they
contribute is a point about the *branch* rather than about the frozen features
underneath it — the argument that keeps
:class:`~visbench.heads.linear.LinearHead` the head a dense VisBench number is
quoted with, and the argument behind ``DetectionHead``'s ``hidden_dim=0``
default. A single 1x1 convolution over RoI-aligned features is a *linear*
readout of those features, per RoI pixel. So the absolute mask AP here is well
below the segmentation literature's, deliberately, and is meant to be compared
across backbones rather than against published detectors.

**RoIAlign carries no parameters**, which is what makes that claim hold. The
only learned thing between a backbone's features and a predicted mask is the
1x1 convolution; the sampling is fixed geometry, so a difference between two
backbones on this probe is a difference between two representations.

**The mask branch is class-agnostic — one channel, not one per class.** Mask
R-CNN found class-specific masks slightly better and it costs ``num_classes``
times the parameters, which is the wrong trade for a probe: the class is
already decided by the detection branch, and asking the mask branch to relearn
it makes the readout less linear rather than more informative. Recorded in
``task_params`` so a future class-specific variant is a different measurement
rather than a silent change.

**One module, two branches, because fitted state outside ``self.head`` is a bug
this codebase has already shipped** (9a: ``DetectionTask.grid_hw`` was missing
from ``probe_state``, so a saved detection probe loaded and then refused to
predict). Holding the mask convolution in a *second* module beside the head
would put its weights outside ``head.state_dict()`` and reintroduce exactly
that, one artifact round-trip later. So this is a single
:class:`~visbench.heads.base.BaseHead` whose ``forward`` is the detection
tensor — satisfying the one-tensor contract every other head honours — with the
mask branch reachable as :meth:`mask_logits`.
"""

import torch
import torch.nn as nn

from visbench.heads.base import BaseHead, register_head
from visbench.heads.detection import DetectionHead
from visbench.types import FeatureMode

__all__ = ["InstanceHead"]


@register_head("instance")
class InstanceHead(BaseHead):
    """Per-cell class logits and box distances, plus a per-RoI mask logit.

    ``forward`` returns exactly what :class:`DetectionHead` returns — ``(B,
    num_classes + 4, H, W)`` — so a task that only wants boxes cannot tell the
    difference, and ``out_channels`` matches what
    :class:`~visbench.tasks.high_level.detection.DetectionTask` validates.
    :meth:`mask_logits` is the extra surface, taking RoI-aligned features
    rather than the dense grid.

    Parameters
    ----------
    in_channels:
        Feature channels. Under ``dense_cls_broadcast`` this is doubled.
    num_classes:
        Detection classes, sizing the classification branch.
    hidden_dim:
        Width of ``DetectionHead``'s optional shared stem. ``0``, the default,
        keeps the box and class branches linear; see that class.
    prior_probability:
        Passed to ``DetectionHead``'s focal prior.
    mask_hidden_dim:
        Width of an optional 3x3 stem *before* the mask convolution. ``0``, the
        default, is what makes the mask branch a linear readout — see the module
        docstring before raising it.
    """

    supported_feature_modes: tuple[str, ...] = (
        FeatureMode.DENSE_ONLY,
        FeatureMode.DENSE_CLS_BROADCAST,
    )
    multiscale = False

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 0,
        prior_probability: float = 0.01,
        mask_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        if mask_hidden_dim < 0:
            raise ValueError(f"mask_hidden_dim must be >= 0, got {mask_hidden_dim}")

        # Composition rather than inheritance: the detection branch is used
        # unchanged, including its two bias initialisations, and subclassing
        # would put this head's mask parameters inside something whose docstring
        # promises exactly two branches.
        self.detection = DetectionHead(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            prior_probability=prior_probability,
        )
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.out_channels = self.detection.out_channels
        self.mask_hidden_dim = mask_hidden_dim

        if mask_hidden_dim:
            self.mask_stem: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, mask_hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            mask_in = mask_hidden_dim
        else:
            self.mask_stem = nn.Identity()
            mask_in = in_channels

        #: One channel, class-agnostic. See the module docstring.
        self.mask_predictor = nn.Conv2d(mask_in, 1, kernel_size=1, bias=True)
        bias = self.mask_predictor.bias
        assert bias is not None  # bias=True above
        with torch.no_grad():
            # Zero, so training starts predicting 0.5 everywhere inside a RoI.
            # A RoI is a *detected object's* box, so roughly half its pixels are
            # foreground — unlike the dense classification branch, where nearly
            # every cell is background and the focal prior is what corrects for
            # it. Copying that prior here would start every mask empty.
            bias.zero_()

    def forward(self, features) -> torch.Tensor:
        """``(B, C, H, W)`` features to ``(B, num_classes + 4, H, W)``.

        Delegated to the detection branch unchanged, so every shape check and
        error message a caller sees is that class's.
        """
        return self.detection(features)

    def mask_logits(self, roi_features: torch.Tensor) -> torch.Tensor:
        """``(N, C, M, M)`` RoI-aligned features to ``(N, 1, M, M)`` mask logits.

        Raw logits, not probabilities: the task owns every activation, so that
        the loss, the metric and ``predict`` cannot disagree about what a mask
        means — the rule ``_activate`` follows on every dense probe.

        An empty RoI batch returns a correctly shaped empty tensor rather than
        raising. An image in which nothing was detected is ordinary, and it is
        the case a decode step reaches on almost every early epoch.
        """
        if roi_features.ndim != 4:
            raise ValueError(f"Expected (N, C, M, M) RoI features, got {tuple(roi_features.shape)}")
        if roi_features.shape[1] != self.in_channels:
            raise ValueError(
                f"RoI features have {roi_features.shape[1]} channels but this head was "
                f"built for {self.in_channels}."
            )
        if roi_features.shape[0] == 0:
            return roi_features.new_zeros((0, 1, *roi_features.shape[-2:]))
        return self.mask_predictor(self.mask_stem(roi_features))
