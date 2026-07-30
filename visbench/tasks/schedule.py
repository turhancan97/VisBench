"""The optimiser schedule every trained probe in this codebase shares.

Lifted out of :class:`~visbench.tasks.dense_base.DenseTrainingTask` when
detection arrived (v0.3, step 6c-3) — the same move that produced
``DenseTrainingTask`` itself, from a working ``DepthTask``. Detection cannot
subclass that base (its targets are variable-length box lists and its metric is
split-level, not per-image), but it must train under the *same* schedule, or a
difference between a detection number and a segmentation number would be partly
a difference in optimisation.

probe3d's schedule, unchanged: linear warmup then cosine decay to zero.
"""

import math
from collections.abc import Callable

__all__ = ["warmup_cosine", "check_schedule"]


def warmup_cosine(
    total_steps: int, steps_per_epoch: int, warmup_epochs: float
) -> Callable[[int], float]:
    """A step-indexed learning-rate multiplier: linear warmup, then cosine decay.

    Returned as a multiplier rather than a scheduler so it can be handed
    straight to ``LambdaLR`` and, more usefully, tested as a pure function.
    """
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / remaining)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return multiplier


def check_schedule(epochs: int, warmup_epochs: float) -> None:
    """Refuse a warmup that outlasts training, naming what to do instead.

    Clamping silently would report a number produced by a schedule nobody
    chose, which is the failure mode this whole codebase is arranged against.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError(
            f"warmup_epochs must be in [0, epochs), got {warmup_epochs} with "
            f"epochs={epochs} — the schedule would still be warming up when training "
            "ended. Pass warmup_epochs=0 for a short run; the default 1.5 assumes "
            "probe3d's 10 epochs. Clamping it silently would report a number produced "
            "by a schedule nobody chose."
        )
