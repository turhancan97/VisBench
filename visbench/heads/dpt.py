"""DPT-style multiscale head — v0.2, not implemented.

Follows probe3d (El Banani et al., CVPR 2024, arXiv:2404.08476), which is also
where the depth and surface-normal protocols come from — using the same head
as the reference protocol keeps VisBench numbers comparable to published ones.

Genuinely multiscale, so this head is the reason multi-layer extraction must be
wired up in v0.2: it consumes features from several backbone depths at once.
"""


import torch

from visbench.heads.base import BaseHead
from visbench.types import FeatureMode

__all__ = ["DPTHead"]


class DPTHead(BaseHead):
    """Fuses features from multiple backbone layers into a dense prediction."""

    supported_feature_modes: tuple[str, ...] = (
        FeatureMode.DENSE_ONLY,
        FeatureMode.DENSE_CLS_BROADCAST,
        FeatureMode.DENSE_PLUS_CLS,
    )

    def forward(self, features) -> torch.Tensor:
        raise NotImplementedError("DPT head lands in v0.2.")
