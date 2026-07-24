"""DINOv2 backbone (ViT-S/14, ViT-B/14). First backbone implemented — v0.1.

Weights via torch.hub (facebookresearch/dinov2). Register tokens are stripped
before pooling or grid reshaping when the variant has them.
"""

from typing import Optional

import torch

from visbench.backbones.base import BaseBackbone
from visbench.types import LayerSpec

__all__ = ["DINOv2"]

#: Registered name -> (torch.hub entrypoint, embed dim, patch size, n registers).
_VARIANTS = {
    "dinov2_vits14": ("dinov2_vits14", 384, 14, 0),
    "dinov2_vitb14": ("dinov2_vitb14", 768, 14, 0),
}


# Activated at build step 2, once ``register_backbone`` has a body — decorators
# run at import, so registering now would make this module unimportable.
# @register_backbone("dinov2_vitb14")
class DINOv2(BaseBackbone):
    """DINOv2 ViT with a CLS token, so default pooling is CLS.

    Input resolution must be a multiple of the patch size (14); the grid is
    ``(H // 14, W // 14)``.
    """

    has_cls_token = True

    def __init__(self, variant: str = "dinov2_vitb14", device: Optional[str] = None) -> None:
        """Load the hub checkpoint for ``variant``, freeze it, set eval mode."""
        raise NotImplementedError

    def _forward_features(self, image: torch.Tensor, layers: LayerSpec):
        """Run the ViT and return ``(patch_tokens, cls_token, grid_hw)``.

        Uses ``get_intermediate_layers`` so the multi-layer path in v0.2 is a
        widening of this call rather than a rewrite. Register tokens, if the
        variant has them, are dropped here.
        """
        raise NotImplementedError

    def preprocess(self, images):
        """Resize to the configured resolution and apply ImageNet normalisation."""
        raise NotImplementedError

    def cache_key(self) -> str:
        """``"dinov2/<variant>/<resolution>"`` — changes if weights or input size change."""
        raise NotImplementedError
