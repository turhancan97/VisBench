"""Shared type definitions and enums.

Centralised here so that backbones, tasks and the feature cache all agree on
the vocabulary (pooling names, feature modes, the feature dict layout) without
importing each other.
"""

from typing import Optional, TypedDict

import torch

__all__ = [
    "FeatureDict",
    "Pooling",
    "FeatureMode",
    "POOLING_CHOICES",
    "FEATURE_MODE_CHOICES",
]


class FeatureDict(TypedDict, total=False):
    """Return value of :meth:`BaseBackbone.extract_features`.

    Every backbone returns both representations from a single forward pass;
    tasks pick whichever they need. See CLAUDE.md ("Feature extraction design").

    Keys
    ----
    dense:
        Spatial features, ``(B, C, H, W)``. For ViTs this is the patch-token
        grid reshaped from ``(B, num_patches, C)``; for CNNs it is the last
        conv feature map before global pooling.
    pooled:
        Single vector per image, ``(B, C)``. Which reduction produced it is
        determined by the ``pooling`` argument (CLS by default for ViTs with a
        CLS token, mean-pooling otherwise).
    grid_hw:
        ``(H, W)`` of the dense grid, so callers can reshape/unflatten without
        re-deriving patch size from the model.
    cls:
        Present only when ``feature_mode="dense_plus_cls"``: the global CLS
        vector, ``(B, C)``, kept apart from the grid so a head can fuse them
        where it chooses. Absent otherwise, so a task that never asks for the
        mode cannot accidentally read a stale one.
    """

    dense: torch.Tensor
    pooled: torch.Tensor
    grid_hw: tuple[int, int]
    cls: torch.Tensor


class Pooling:
    """How the ``pooled`` vector is reduced from the backbone output.

    Chosen by the *task*, not the backbone (CLAUDE.md, "BaseTask"). ``DEFAULT``
    defers to the backbone's architecture-appropriate default: CLS for ViTs
    that have a CLS token, mean-pooling for everything else.
    """

    DEFAULT = "default"
    CLS = "cls"
    MEAN = "mean"


POOLING_CHOICES: tuple[str, ...] = (Pooling.DEFAULT, Pooling.CLS, Pooling.MEAN)


class FeatureMode:
    """How dense features are presented to a dense-prediction task head.

    All three modes exist in the interface from v0.1 so that no refactor is
    needed later, but only :attr:`DENSE_ONLY` is enabled in v0.1; the others
    must be requested explicitly and are wired up in v0.2.
    """

    #: Spatial grid only, no CLS involved. The v0.1 default.
    DENSE_ONLY = "dense_only"

    #: CLS token broadcast across every spatial location and concatenated onto
    #: the channel dim, giving ``(B, C_dense + C_cls, H, W)``.
    DENSE_CLS_BROADCAST = "dense_cls_broadcast"

    #: Dense grid and global CLS vector kept separate and both handed to the
    #: head, which decides how to fuse them (e.g. only at a bottleneck).
    DENSE_PLUS_CLS = "dense_plus_cls"


FEATURE_MODE_CHOICES: tuple[str, ...] = (
    FeatureMode.DENSE_ONLY,
    FeatureMode.DENSE_CLS_BROADCAST,
    FeatureMode.DENSE_PLUS_CLS,
)

#: Layer selection for multi-layer extraction. Accepted by the interface from
#: v0.1; only single-layer extraction is wired up until v0.2.
LayerSpec = Optional[list[int]]

#: Flat metrics dict returned by :meth:`BaseTask.evaluate`.
MetricsDict = dict[str, float]
