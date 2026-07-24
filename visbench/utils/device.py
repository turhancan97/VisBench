"""Device selection and batching helpers."""

from typing import Optional

import torch

__all__ = ["resolve_device"]


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve ``None`` to the best available device (cuda > mps > cpu).

    Explicit values pass through unchanged, so a caller can always force cpu.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
