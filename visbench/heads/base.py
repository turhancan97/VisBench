"""Pluggable task-head interface.

Dense task heads must be selectable per run, never hardcoded to one
architecture (CLAUDE.md, v0.2). This is a known extension point for
contributors, so the interface is intentionally small.

The interface is defined in v0.1 so tasks can reference it, but no head is
implemented before v0.2.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

__all__ = ["BaseHead"]


class BaseHead(nn.Module, ABC):
    """Maps backbone features to a task output.

    A head declares which feature modes it accepts; a task requesting an
    unsupported mode should fail at construction, not mid-training.
    """

    #: Feature modes this head can consume, from :class:`FeatureMode`.
    supported_feature_modes: tuple[str, ...] = ()

    @abstractmethod
    def forward(self, features) -> torch.Tensor:
        """Produce the task output from backbone features.

        ``features`` is a dense tensor for most modes, or a
        ``(dense, cls)`` tuple under ``dense_plus_cls``.
        """
        raise NotImplementedError
