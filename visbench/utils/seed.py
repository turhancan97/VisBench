"""Seeding, for reproducibility.

This is a benchmark library, so a reported number must be reproducible from
the record that logged it — the seed is part of :class:`ResultRecord`.
"""

from typing import Optional

__all__ = ["set_seed"]


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    """Seed python, numpy and torch RNGs; return the seed actually used.

    Returns the seed so a caller that passed ``None`` can still log which one
    was drawn. ``deterministic=True`` additionally enables torch's
    deterministic algorithms, which is slower and off by default.
    """
    raise NotImplementedError
