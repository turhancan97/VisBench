"""Device selection and batching helpers."""

from collections.abc import Iterator, Sequence
from typing import Optional

import torch

__all__ = ["resolve_device", "batched"]


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


def batched(items: Sequence, batch_size: int) -> Iterator[Sequence]:
    """Yield consecutive slices of ``items``, the last possibly short."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
