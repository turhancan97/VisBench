"""Task heads.

Interface lands in v0.1 so tasks can reference it; implementations arrive in
v0.2 with the dense tasks. At minimum a linear probe head and a DPT-style
multiscale head, selectable per task run; the interface stays open for more.
"""

from visbench.heads.base import BaseHead, build_head, get_head, list_heads, register_head
from visbench.heads.dpt import DPTHead
from visbench.heads.linear import LinearHead

__all__ = [
    "BaseHead",
    "LinearHead",
    "DPTHead",
    "register_head",
    "get_head",
    "build_head",
    "list_heads",
]
