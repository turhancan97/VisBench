"""Linear heads — the simplest probe, and the baseline every dense task needs.

The linear probe head is what makes results comparable across backbones: any
gain from a fancier head is a property of the head, not the representation.

v0.2 for the dense variant; the classification task's linear layer in v0.1 is
deliberately self-contained and does not depend on this module.
"""

import torch
import torch.nn as nn

from visbench.heads.base import BaseHead, register_head
from visbench.types import FeatureMode

__all__ = ["LinearHead"]


@register_head("linear")
class LinearHead(BaseHead):
    """1x1 convolution over the dense grid, upsampled to the target resolution.

    Deliberately the least expressive head available. A dense task's headline
    number should be reported with this one, because it is the only head under
    which a difference between two backbones is a difference between two
    *representations* — anything deeper can compensate for a weak feature map
    and quietly narrow the gap it was meant to measure.

    Upsampling after the 1x1 rather than before is the cheaper order and the
    conventional one: the convolution runs on the coarse grid, and no
    interpolated feature ever enters the linear map.
    """

    supported_feature_modes: tuple[str, ...] = (
        FeatureMode.DENSE_ONLY,
        FeatureMode.DENSE_CLS_BROADCAST,
    )
    multiscale = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_size: int | tuple[int, int] | None = None,
        bias: bool = True,
    ) -> None:
        """Configure the projection.

        Parameters
        ----------
        in_channels:
            Channel dim of the dense features. Under
            ``dense_cls_broadcast`` this is the *widened* dim — the backbone's
            ``embed_dim`` doubled for a ViT — since the CLS token has already
            been concatenated on by the time the head sees it.
        output_size:
            Resolution to upsample the prediction to, typically the input image
            size so it can be compared against a ground-truth map. ``None``
            leaves it on the feature grid.
        """
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError(f"channels must be >= 1, got {in_channels} -> {out_channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_size = (
            (output_size, output_size) if isinstance(output_size, int) else output_size
        )
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, features) -> torch.Tensor:
        """``(B, C, H, W)`` dense features to ``(B, out_channels, *output_size)``."""
        if isinstance(features, (tuple, list)):
            raise TypeError(
                f"{type(self).__name__} takes a single dense tensor. A "
                "(dense, cls) pair means feature_mode='dense_plus_cls', which "
                f"this head does not support — it accepts {self.supported_feature_modes}."
            )
        if features.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W) dense features, got {tuple(features.shape)}")
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"Features have {features.shape[1]} channels but this head was built for "
                f"{self.in_channels}. Under dense_cls_broadcast the channel dim is doubled."
            )
        return self._resize(self.proj(features), self.output_size)
