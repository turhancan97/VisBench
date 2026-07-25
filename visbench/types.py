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
    "LayerSpec",
    "LayerOutput",
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
    dense_layers:
        Present only when ``layers=[...]`` was passed: one dense map per
        requested layer, shallowest first, each assembled in the same
        ``feature_mode`` as ``dense``. The last element **is** ``dense``.

        This is a separate key rather than making ``dense`` sometimes-a-list on
        purpose. A key whose type depends on how many layers were requested
        would break every single-layer consumer the moment a caller widened a
        layer list, and the breakage would surface deep inside a task rather
        than at the call.

        Per-layer grids are not returned separately: these are ``(B, C, H, W)``
        maps, so each carries its own. They can differ — a CNN's stages are at
        different strides — which is precisely what a multiscale head is for.
    layer_indices:
        Present alongside ``dense_layers``: the resolved, non-negative layer
        indices, in the same order. ``layers=[-1]`` records as ``[11]`` on a
        12-block ViT, so a result record says which depth was actually read
        rather than repeating a relative index.
    """

    dense: torch.Tensor
    pooled: torch.Tensor
    grid_hw: tuple[int, int]
    cls: torch.Tensor
    dense_layers: list[torch.Tensor]
    layer_indices: list[int]


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

#: Layer selection for multi-layer extraction, as passed by a caller: ``None``
#: for "the last layer", or indices which may be negative. Accepted by the
#: interface from v0.1 and wired up in v0.2.
LayerSpec = Optional[list[int]]

#: What one layer of a backbone contributes: ``(patch_tokens, cls, grid_hw)``
#: with ``patch_tokens`` ``(B, N, C)``, ``cls`` ``(B, C)`` or ``None``, and
#: ``N == grid_h * grid_w``. Every architecture family normalises to this in
#: :meth:`BaseBackbone._forward_features`, which returns one per requested
#: layer.
LayerOutput = tuple[torch.Tensor, Optional[torch.Tensor], tuple[int, int]]

#: Flat metrics dict returned by :meth:`BaseTask.evaluate`.
MetricsDict = dict[str, float]
