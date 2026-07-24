"""Linear heads — the simplest probe, and the baseline every dense task needs.

The linear probe head is what makes results comparable across backbones: any
gain from a fancier head is a property of the head, not the representation.

v0.2 for the dense variant; the classification task's linear layer in v0.1 is
deliberately self-contained and does not depend on this module.
"""

import torch

from visbench.heads.base import BaseHead
from visbench.types import FeatureMode

__all__ = ["LinearHead"]


class LinearHead(BaseHead):
    """1x1 convolution over the dense grid, upsampled to the target resolution.

    Deferred to v0.2.
    """

    supported_feature_modes: tuple[str, ...] = (
        FeatureMode.DENSE_ONLY,
        FeatureMode.DENSE_CLS_BROADCAST,
    )

    def forward(self, features) -> torch.Tensor:
        raise NotImplementedError("Dense linear head lands in v0.2.")
