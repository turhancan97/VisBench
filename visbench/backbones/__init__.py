"""Backbone implementations.

Importing this package runs the ``@register_backbone`` decorators, which is
what makes names visible to :func:`visbench.get_backbone`.

v0.1 ships DINOv2 and CLIP only. ResNet/timm and user-supplied custom
backbones (arbitrary ``nn.Module`` + preprocessing fn) arrive in v0.2.
"""

from visbench.backbones.base import BaseBackbone
from visbench.backbones.custom import CustomBackbone
from visbench.backbones.dinov2 import DINOv2

__all__ = ["BaseBackbone", "CustomBackbone", "DINOv2", "CLIP", "TimmBackbone"]


#: Backbones behind an optional extra, imported only on attribute access so a
#: plain ``import visbench.backbones`` does not require every extra.
_LAZY = {"CLIP": "visbench.backbones.clip", "TimmBackbone": "visbench.backbones.timm_backbone"}


def __getattr__(name: str):
    """Import optional-extra backbones lazily."""
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
