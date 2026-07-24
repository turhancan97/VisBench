"""Backbone implementations.

Importing this package runs the ``@register_backbone`` decorators, which is
what makes names visible to :func:`visbench.get_backbone`.

v0.1 ships DINOv2 and CLIP only. ResNet/timm and user-supplied custom
backbones (arbitrary ``nn.Module`` + preprocessing fn) arrive in v0.2.
"""

from visbench.backbones.base import BaseBackbone
from visbench.backbones.dinov2 import DINOv2

__all__ = ["BaseBackbone", "DINOv2", "CLIP"]


def __getattr__(name: str):
    """Import CLIP lazily.

    open_clip is an optional extra, so a plain ``import visbench.backbones``
    must not require it. Attribute access does.
    """
    if name == "CLIP":
        from visbench.backbones.clip import CLIP

        return CLIP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
