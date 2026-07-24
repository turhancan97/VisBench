"""OpenCLIP ViT-B image-tower backbone — v0.1.

Only the visual tower is exposed; the text tower is out of scope for probing.
Note that CLIP uses its own normalisation constants, not ImageNet's.
"""

from typing import Optional

import torch

from visbench.backbones.base import BaseBackbone
from visbench.types import LayerSpec

__all__ = ["CLIP"]

#: Registered name -> (open_clip model name, pretrained tag, embed dim, patch size).
_VARIANTS = {
    "clip_vitb16": ("ViT-B-16", "openai", 768, 16),
    "clip_vitb32": ("ViT-B-32", "openai", 768, 32),
}


# Activated at build step 2 — see the note in dinov2.py.
# @register_backbone("clip_vitb16")
class CLIP(BaseBackbone):
    """CLIP visual tower.

    Has a CLS token, so default pooling is CLS — but note this is the
    pre-projection CLS, i.e. the representation used for probing, not the
    projected image embedding used for text alignment. Which of the two a task
    wants is a real decision; document it wherever it is made.
    """

    has_cls_token = True

    def __init__(self, variant: str = "clip_vitb16", device: Optional[str] = None) -> None:
        """Load the OpenCLIP visual tower for ``variant``, freeze it, set eval mode."""
        raise NotImplementedError

    def _forward_features(self, image: torch.Tensor, layers: LayerSpec):
        """Run the visual transformer, returning ``(patch_tokens, cls_token, grid_hw)``."""
        raise NotImplementedError

    def preprocess(self, images):
        """Resize to 224 and apply CLIP normalisation constants."""
        raise NotImplementedError

    def cache_key(self) -> str:
        """``"clip/<model>/<pretrained-tag>/<resolution>"``."""
        raise NotImplementedError
