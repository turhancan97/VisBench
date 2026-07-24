"""Seeding, for reproducibility.

This is a benchmark library, so a reported number must be reproducible from
the record that logged it — the seed is part of :class:`ResultRecord`.
"""

import random
from typing import Optional

import numpy as np
import torch

__all__ = ["set_seed"]

#: Upper bound for a drawn seed. Kept inside 2**32 so the same value is
#: accepted by numpy, which rejects anything wider.
_MAX_SEED = 2**32 - 1


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    """Seed python, numpy and torch RNGs; return the seed actually used.

    Returns the seed so a caller that passed ``None`` can still log which one
    was drawn. ``deterministic=True`` additionally enables torch's
    deterministic algorithms, which is slower and off by default.
    """
    if seed is None:
        seed = random.randrange(_MAX_SEED)
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # warn_only: some ops have no deterministic kernel, and a hard failure
        # deep inside a backbone is worse than a warning on an op we do not
        # control.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

    return seed
