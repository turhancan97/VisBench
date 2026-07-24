"""The backbone abstraction.

One method, one return shape, for every architecture family. ViT and CNN
internals differ completely but callers cannot tell the difference — that is
the whole point of this class (CLAUDE.md, "Feature extraction design").
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from visbench.types import FeatureDict, LayerSpec, Pooling

__all__ = ["BaseBackbone"]


class BaseBackbone(nn.Module, ABC):
    """Frozen feature extractor with a uniform dual pooled+dense output.

    Subclasses implement :meth:`_forward_features` (architecture-specific) and
    declare :attr:`has_cls_token` / :attr:`embed_dim`; pooling, reshaping and
    validation are handled once, here.

    Backbones are deliberately "dumb": they execute whatever ``pooling`` the
    task asks for and hold no opinion about which representation a task needs.
    """

    #: Registered name, set by the ``@register_backbone`` decorator.
    name: str = ""

    #: Whether the architecture exposes a CLS token. Drives the default pooling
    #: rule: CLS for ViTs that have one, mean-pooling for everything else.
    has_cls_token: bool = False

    #: Channel dim of both ``dense`` and ``pooled`` outputs.
    embed_dim: int = 0

    #: ViT patch size; ``None`` for CNNs, where the stride is architectural.
    patch_size: Optional[int] = None

    def __init__(self, device: Optional[str] = None) -> None:
        """Load weights, freeze all parameters, and put the module in eval mode."""
        raise NotImplementedError

    @torch.no_grad()
    def extract_features(
        self,
        image: torch.Tensor,
        pooling: str = Pooling.DEFAULT,
        layers: LayerSpec = None,
    ) -> FeatureDict:
        """Extract dense and pooled features in a single forward pass.

        Parameters
        ----------
        image:
            Preprocessed batch, ``(B, 3, H, W)``. Use :meth:`preprocess` to
            build it from PIL images.
        pooling:
            One of :data:`visbench.types.POOLING_CHOICES`. ``"default"``
            resolves per :meth:`default_pooling`.
        layers:
            Indices for multi-layer extraction. Accepted from v0.1 so the
            signature never changes, but only single-layer extraction is wired
            up until v0.2 — passing more than one layer raises until then.

        Returns
        -------
        FeatureDict
            ``{"dense": (B, C, H, W), "pooled": (B, C), "grid_hw": (H, W)}``.
        """
        raise NotImplementedError

    @abstractmethod
    def _forward_features(self, image: torch.Tensor, layers: LayerSpec):
        """Architecture-specific forward returning raw tokens/maps plus CLS.

        Implementations return whatever their family produces natively; the
        base class normalises it into a :class:`FeatureDict`. ViTs return the
        patch-token sequence ``(B, N, C)`` and the CLS token ``(B, C)``; CNNs
        return the last conv map ``(B, C, H, W)`` and ``None``.
        """
        raise NotImplementedError

    def default_pooling(self) -> str:
        """Resolve ``pooling="default"`` for this architecture.

        CLS when :attr:`has_cls_token`, mean-pooling otherwise.
        """
        raise NotImplementedError

    def preprocess(self, images):
        """Convert PIL image(s) into a normalised, resized batch tensor.

        Each backbone owns its own normalisation constants and input
        resolution, so preprocessing lives with the backbone rather than in a
        shared transform.
        """
        raise NotImplementedError

    def cache_key(self) -> str:
        """Stable identifier for this backbone + weights, used in cache keys.

        Must change whenever the weights or extraction behaviour change, or
        stale cached features would be silently reused.
        """
        raise NotImplementedError

    def forward(self, image: torch.Tensor) -> FeatureDict:
        """Alias for :meth:`extract_features` with default pooling."""
        raise NotImplementedError
