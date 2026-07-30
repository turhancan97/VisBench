"""The anchor-free single-scale detection head — v0.3, step 6c-3.

One 1x1 convolution for classification and one for box regression, both over
the backbone's patch grid at its native stride. No anchors, no feature pyramid,
no deformable anything.

**That is a deliberate ceiling, not an oversight.** VisBench probes what a
frozen representation already carries; a competitive detector would add an FPN,
multi-scale assignment and a heavier neck, and every point those contribute is
a point about the *neck* rather than about the features underneath it. The same
argument is why :class:`~visbench.heads.linear.LinearHead` is the head a dense
number is quoted with. So the absolute mAP here is low by the standards of the
detection literature and is not meant to be compared with it — it is meant to
be compared across backbones, which is the only comparison it supports.

The two branches share an input and are otherwise independent, which is what
lets the classification bias carry a *prior*: at initialisation nearly every
cell is background, and starting from a uniform 0.5 objectness makes the first
epochs' loss almost entirely the model discovering that. The standard fix
(Lin et al., focal loss) is to initialise the classification bias to
``-log((1 - pi) / pi)`` so training starts already predicting ``pi``, and
without it a dense anchor-free head can sit at near-zero mAP long enough to look
like a broken probe.
"""

import math

import torch
import torch.nn as nn

from visbench.heads.base import BaseHead, register_head
from visbench.types import FeatureMode

__all__ = ["DetectionHead"]


@register_head("detection")
class DetectionHead(BaseHead):
    """Per-cell class logits and box distances, emitted as one tensor.

    ``forward`` returns ``(B, num_classes + 4, H, W)``: the first
    ``num_classes`` channels are classification logits, the last four are the
    *raw* left/top/right/bottom distances.

    One tensor rather than a tuple because :meth:`BaseHead.forward` is declared
    to return a tensor and every other head honours that. The split index is
    :attr:`num_classes`, read off the head by the task rather than passed
    separately — one source of truth, so the two cannot drift.

    The distances are raw: turning them into pixels needs the grid stride,
    which depends on the input resolution the head never sees.
    :class:`~visbench.tasks.high_level.detection.DetectionTask` owns that step,
    for the same reason it owns every other activation — it applies it in the
    loss, the metric *and* ``predict``, so those three cannot disagree.
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
    ) -> None:
        """Configure the two branches.

        Parameters
        ----------
        hidden_dim:
            Width of an optional shared 3x3 stem. ``0``, the default, means no
            stem at all: the head is then exactly two linear maps of the
            features, which is what makes a difference between two backbones a
            difference between two *representations*. Anything above 0 makes
            this a small conv net and the number less attributable.
        prior_probability:
            Foreground probability the classification branch starts at. See the
            module docstring; 0.01 is the value focal loss was published with.
        """
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if hidden_dim < 0:
            raise ValueError(f"hidden_dim must be >= 0, got {hidden_dim}")
        if not 0.0 < prior_probability < 1.0:
            raise ValueError(f"prior_probability must be in (0, 1), got {prior_probability}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.out_channels = num_classes + 4
        self.hidden_dim = hidden_dim

        if hidden_dim:
            self.stem: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            branch_channels = hidden_dim
        else:
            self.stem = nn.Identity()
            branch_channels = in_channels

        # bias=True on both, which is not a default worth leaving implicit here:
        # the two initialisations below are the whole reason this class exists
        # rather than two bare 1x1 convolutions.
        self.classifier = nn.Conv2d(branch_channels, num_classes, kernel_size=1, bias=True)
        self.regressor = nn.Conv2d(branch_channels, 4, kernel_size=1, bias=True)

        class_bias, box_bias = self.classifier.bias, self.regressor.bias
        assert class_bias is not None and box_bias is not None  # bias=True above
        with torch.no_grad():
            # The prior. Without it the head spends its first epochs learning
            # that background is common, which on a ten-epoch schedule is most
            # of them.
            class_bias.fill_(-math.log((1.0 - prior_probability) / prior_probability))
            # Zero, so exp(raw) starts at 1 and every predicted box is one
            # stride wide before anything is learned — a scale-free start.
            box_bias.zero_()

    def forward(self, features) -> torch.Tensor:
        """``(B, C, H, W)`` features to ``(B, num_classes + 4, H, W)``."""
        if isinstance(features, (tuple, list)):
            raise TypeError(
                f"{type(self).__name__} takes a single dense tensor. A list means either "
                "layers=[...] or feature_mode='dense_plus_cls'; this head is single-scale "
                f"and accepts {self.supported_feature_modes}."
            )
        if features.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W) dense features, got {tuple(features.shape)}")
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"Features have {features.shape[1]} channels but this head was built for "
                f"{self.in_channels}. Under dense_cls_broadcast the channel dim is doubled."
            )
        shared = self.stem(features)
        return torch.cat([self.classifier(shared), self.regressor(shared)], dim=1)
