"""Backbone implementations.

Importing this package runs the ``@register_backbone`` decorators, which is
what makes names visible to :func:`visbench.get_backbone`.

v0.1 ships DINOv2 and CLIP only. ResNet/timm and user-supplied custom
backbones (arbitrary ``nn.Module`` + preprocessing fn) arrive in v0.2.
"""

from visbench.backbones.base import BaseBackbone
from visbench.backbones.dinov2 import DINOv2

# CLIP is added here once it is implemented; importing an unimplemented module
# would register a name that cannot be constructed.
__all__ = ["BaseBackbone", "DINOv2"]
