"""Small shared helpers. Nothing task- or backbone-specific belongs here."""

from visbench.utils.device import resolve_device
from visbench.utils.seed import set_seed

__all__ = ["resolve_device", "set_seed"]
