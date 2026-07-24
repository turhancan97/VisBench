"""Device selection and batching helpers."""

from collections.abc import Iterator, Sequence
from typing import Optional

__all__ = ["resolve_device", "batched"]


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve ``None`` to the best available device (cuda > mps > cpu).

    Explicit values pass through unchanged, so a caller can always force cpu.
    """
    raise NotImplementedError


def batched(items: Sequence, batch_size: int) -> Iterator[Sequence]:
    """Yield consecutive slices of ``items``, the last possibly short."""
    raise NotImplementedError
