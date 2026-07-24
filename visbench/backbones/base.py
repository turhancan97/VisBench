"""The backbone abstraction.

One method, one return shape, for every architecture family. ViT and CNN
internals differ completely but callers cannot tell the difference — that is
the whole point of this class (CLAUDE.md, "Feature extraction design").
"""

from abc import ABC, abstractmethod
from typing import Optional, cast

import torch
import torch.nn as nn

from visbench.backbones.pooling import apply_feature_mode, pool_tokens, tokens_to_grid
from visbench.types import (
    FEATURE_MODE_CHOICES,
    POOLING_CHOICES,
    FeatureDict,
    FeatureMode,
    LayerSpec,
    Pooling,
)
from visbench.utils.device import resolve_device

__all__ = ["BaseBackbone"]


class BaseBackbone(nn.Module, ABC):
    """Frozen feature extractor with a uniform dual pooled+dense output.

    Subclasses implement :meth:`_forward_features` (architecture-specific) and
    declare :attr:`has_cls_token` / :attr:`embed_dim`; pooling, reshaping and
    validation are handled once, here.

    Backbones are deliberately "dumb": they execute whatever ``pooling`` the
    task asks for and hold no opinion about which representation a task needs.
    """

    #: Registered name. Set per instance in ``__init__`` rather than by the
    #: ``@register_backbone`` decorator, because one class may serve several
    #: registered names (see :mod:`visbench.registry`).
    name: str = ""

    #: Whether the architecture exposes a CLS token. Drives the default pooling
    #: rule: CLS for ViTs that have one, mean-pooling for everything else.
    has_cls_token: bool = False

    #: Channel dim of both ``dense`` and ``pooled`` outputs.
    embed_dim: int = 0

    #: ViT patch size; ``None`` for CNNs, where the stride is architectural.
    patch_size: Optional[int] = None

    def __init__(self, device: Optional[str] = None) -> None:
        """Record the target device. Subclasses load weights, then call
        :meth:`_finalize` to freeze, ``eval()`` and move the module.

        Freezing cannot happen here: at this point the subclass has not built
        its weights yet, so there is nothing to freeze.
        """
        super().__init__()
        self.device = resolve_device(device)

    def _finalize(self) -> None:
        """Freeze every parameter, switch to eval mode, move to the device.

        Called by subclasses at the end of ``__init__``. Probing evaluates
        *frozen* representations, so a backbone that arrives in train mode
        (BatchNorm updating, dropout active) silently changes the numbers it
        reports — hence one shared implementation rather than per-backbone
        boilerplate.
        """
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        self.to(self.device)

    @torch.no_grad()
    def extract_features(
        self,
        image: torch.Tensor,
        pooling: str = Pooling.DEFAULT,
        layers: LayerSpec = None,
        feature_mode: str = FeatureMode.DENSE_ONLY,
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
        feature_mode:
            How ``dense`` is assembled, one of
            :data:`visbench.types.FEATURE_MODE_CHOICES`:

            ``dense_only``
                the patch grid alone, ``(B, C, H, W)``. The default.
            ``dense_cls_broadcast``
                CLS repeated at every location, ``(B, C + C_cls, H, W)``.
            ``dense_plus_cls``
                grid and CLS kept separate; ``dense`` is the grid and the CLS
                vector is returned under ``cls``, for a head that fuses them at
                a bottleneck rather than at every pixel.

            ``pooled`` is unaffected — it answers a different question, and a
            task wanting mean-pooled patches with a broadcast dense grid must
            be able to ask for both.

        Returns
        -------
        FeatureDict
            ``{"dense": ..., "pooled": (B, C), "grid_hw": (H, W)}``, plus
            ``cls`` when ``feature_mode="dense_plus_cls"``.
        """
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"Unknown pooling {pooling!r}; expected one of {POOLING_CHOICES}")
        if feature_mode not in FEATURE_MODE_CHOICES:
            raise ValueError(
                f"Unknown feature_mode {feature_mode!r}; expected one of {FEATURE_MODE_CHOICES}"
            )
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"extract_features expects a preprocessed tensor, got {type(image).__name__}. "
                "Call backbone.preprocess(images) first."
            )
        if image.ndim != 4:
            raise ValueError(
                f"Expected a batch of shape (B, 3, H, W), got {tuple(image.shape)}. "
                "For a single image use image.unsqueeze(0)."
            )
        if layers is not None and len(layers) > 1:
            raise NotImplementedError(
                f"Multi-layer extraction is wired up in v0.2; got layers={layers}. "
                "The single-layer path must be proven first (CLAUDE.md, "
                '"Multi-layer extraction").'
            )

        resolved = self.default_pooling() if pooling == Pooling.DEFAULT else pooling
        image = image.to(self.device)

        patch_tokens, cls_token, grid_hw = self._forward_features(image, layers)
        grid = tokens_to_grid(patch_tokens, grid_hw)

        features: FeatureDict = {
            "pooled": pool_tokens(patch_tokens, cls_token, resolved),
            "grid_hw": grid_hw,
        }
        assembled = apply_feature_mode(grid, cls_token, feature_mode)
        if feature_mode == FeatureMode.DENSE_PLUS_CLS:
            # The one mode that returns two things: keeping them separate is
            # the point, so `cls` is a distinct key rather than a tuple the
            # caller has to unpack differently from every other mode.
            # apply_feature_mode's return type depends on the *value* of
            # feature_mode, which the type system cannot express, and it has
            # already rejected a missing CLS token for this mode.
            dense, cls_vector = cast(tuple[torch.Tensor, torch.Tensor], assembled)
            features["dense"] = dense
            features["cls"] = cls_vector
        else:
            features["dense"] = cast(torch.Tensor, assembled)
        return features

    @abstractmethod
    def _forward_features(
        self,
        image: torch.Tensor,
        layers: LayerSpec,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], tuple[int, int]]:
        """Architecture-specific forward returning ``(patch_tokens, cls, grid_hw)``.

        Every family normalises to a **token sequence** here: ``patch_tokens``
        is ``(B, N, C)`` with ``N == grid_h * grid_w``, ``cls`` is ``(B, C)`` or
        ``None``. ViTs return this natively; CNN subclasses (v0.2) flatten their
        ``(B, C, H, W)`` conv map into it and return ``None`` for CLS.

        Making the *subclass* do that flattening — rather than having the base
        class branch on architecture family — is what keeps
        :meth:`extract_features` a single code path. The flatten/unflatten
        round-trip costs nothing next to a forward pass, and the alternative
        puts an ``if is_vit`` in the one method that exists to hide that
        distinction.

        Register tokens, if the variant has them, must be stripped here.
        """
        raise NotImplementedError

    def default_pooling(self) -> str:
        """Resolve ``pooling="default"`` for this architecture.

        CLS when :attr:`has_cls_token`, mean-pooling otherwise.
        """
        return Pooling.CLS if self.has_cls_token else Pooling.MEAN

    @abstractmethod
    def preprocess(self, images) -> torch.Tensor:
        """Convert PIL image(s) into a normalised, resized batch tensor.

        Each backbone owns its own normalisation constants and input
        resolution, so preprocessing lives with the backbone rather than in a
        shared transform.

        Accepts a single PIL image or a sequence of them; always returns
        ``(B, 3, H, W)``.
        """
        raise NotImplementedError

    @abstractmethod
    def cache_key(self) -> str:
        """Stable identifier for this backbone + weights, used in cache keys.

        Must change whenever the weights or extraction behaviour change, or
        stale cached features would be silently reused. Abstract rather than
        defaulted: a plausible-looking inherited key that does not actually
        track the weights is exactly how one model's features get served as
        another's.
        """
        raise NotImplementedError

    def forward(self, image: torch.Tensor) -> FeatureDict:
        """Alias for :meth:`extract_features` with default pooling."""
        return self.extract_features(image)
