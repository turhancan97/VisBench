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
    LayerOutput,
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
            Which backbone depths to read, shallowest first. ``None`` means the
            last layer, the single-layer path every v0.1 task uses. Indices may
            be negative (``-1`` is the last layer) and must be strictly
            increasing; see :meth:`resolve_layers`.

            Passing a list adds ``dense_layers`` and ``layer_indices`` to the
            result, which is what a multiscale head such as
            :class:`~visbench.heads.DPTHead` consumes. All requested layers
            come from **one** forward pass — that is the entire reason this is
            a list rather than a loop over single-layer calls.
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
            ``cls`` when ``feature_mode="dense_plus_cls"`` and
            ``dense_layers``/``layer_indices`` when ``layers`` is given.

            ``dense``, ``pooled``, ``grid_hw`` and ``cls`` always describe the
            **last** requested layer. A multi-layer call is therefore a superset
            of the single-layer one: a task that only reads ``dense`` behaves
            identically whether or not the layer list was widened underneath it.
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

        resolved = self.default_pooling() if pooling == Pooling.DEFAULT else pooling
        indices = self.resolve_layers(layers)
        image = image.to(self.device)

        outputs = self._forward_features(image, indices)
        if len(outputs) != len(indices):
            raise RuntimeError(
                f"{type(self).__name__}._forward_features returned {len(outputs)} layers "
                f"for {len(indices)} requested. A backbone must return one per index, in "
                "the order asked, or the caller cannot tell which depth it is holding."
            )

        assembled_layers = [self._assemble(output, feature_mode) for output in outputs]

        # The last requested layer is the headline one, so a task reading only
        # `dense` sees the deepest features whether or not shallower ones were
        # also requested.
        patch_tokens, cls_token, grid_hw = outputs[-1]
        features: FeatureDict = {
            "pooled": pool_tokens(patch_tokens, cls_token, resolved),
            "grid_hw": grid_hw,
            "dense": assembled_layers[-1][0],
        }
        if feature_mode == FeatureMode.DENSE_PLUS_CLS:
            # The one mode that returns two things: keeping them separate is
            # the point, so `cls` is a distinct key rather than a tuple the
            # caller has to unpack differently from every other mode.
            cls_vector = assembled_layers[-1][1]
            assert cls_vector is not None  # _assemble guarantees it for this mode
            features["cls"] = cls_vector
        if layers is not None:
            features["dense_layers"] = [dense for dense, _ in assembled_layers]
            features["layer_indices"] = indices
        return features

    def _assemble(
        self,
        output: LayerOutput,
        feature_mode: str,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One layer's tokens to ``(dense, cls_or_None)`` in ``feature_mode``."""
        patch_tokens, cls_token, grid_hw = output
        grid = tokens_to_grid(patch_tokens, grid_hw)
        assembled = apply_feature_mode(grid, cls_token, feature_mode)
        if feature_mode == FeatureMode.DENSE_PLUS_CLS:
            # apply_feature_mode's return type depends on the *value* of
            # feature_mode, which the type system cannot express, and it has
            # already rejected a missing CLS token for this mode.
            return cast(tuple[torch.Tensor, torch.Tensor], assembled)
        return cast(torch.Tensor, assembled), None

    def resolve_layers(self, layers: LayerSpec) -> list[int]:
        """Normalise a caller's ``layers`` into concrete, in-range indices.

        ``None`` becomes the last layer. Negative indices count from the end,
        so ``-1`` is the last layer and resolves to the same entry as its
        absolute index — which matters because the resolved value is what
        reaches the cache key, and ``[-1]`` and ``[11]`` on a 12-block model
        must not occupy two entries holding identical features.

        Indices must be **strictly increasing**. Order carries meaning
        downstream — a multiscale head treats the first layer it is handed as
        the coarsest — so a descending or repeated list would build a pyramid
        that silently disagrees with the caller's intent. Rejecting it is the
        only option that cannot be wrong; reordering would quietly overrule the
        caller, and accepting it would quietly mislabel the output.
        """
        depth = self.num_layers
        if layers is None:
            return [depth - 1]
        if not isinstance(layers, (list, tuple)):
            raise TypeError(f"layers must be a list of ints or None, got {type(layers).__name__}")
        if len(layers) == 0:
            raise ValueError("layers=[] requests nothing; pass None for the last layer.")

        resolved = []
        for index in layers:
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError(f"Layer indices must be ints, got {index!r}")
            absolute = index + depth if index < 0 else index
            if not 0 <= absolute < depth:
                raise ValueError(
                    f"Layer index {index} is out of range for {self.name or type(self).__name__}, "
                    f"which exposes {depth} layers (valid: 0..{depth - 1}, or -1..-{depth})."
                )
            resolved.append(absolute)

        if any(b <= a for a, b in zip(resolved, resolved[1:])):
            raise ValueError(
                f"layers must be strictly increasing, shallowest first; got {layers} "
                f"which resolves to {resolved}. A multiscale head reads the first layer "
                "as the coarsest, so the order is part of the request."
            )
        return resolved

    @property
    def num_layers(self) -> int:
        """How many depths this backbone can be asked for.

        Transformer blocks for a ViT, feature stages for a CNN. Defined by the
        subclass because only it knows what an index means; the base class uses
        it to resolve negative indices and to reject out-of-range ones once,
        rather than in four places with four different messages.

        A backbone that exposes only its final output returns 1, which makes
        every multi-layer request fail with a clear message instead of a
        confusing one from inside the model.
        """
        return 1

    @abstractmethod
    def _forward_features(
        self,
        image: torch.Tensor,
        layers: list[int],
    ) -> list[LayerOutput]:
        """Architecture-specific forward returning one output per requested layer.

        ``layers`` arrives already resolved by :meth:`resolve_layers`: concrete,
        in-range, strictly increasing, never empty and never negative. A
        subclass never has to interpret ``None`` or normalise an index.

        Every family normalises to a **token sequence** here: each element is
        ``(patch_tokens, cls, grid_hw)`` with ``patch_tokens`` ``(B, N, C)``,
        ``N == grid_h * grid_w``, and ``cls`` ``(B, C)`` or ``None``. ViTs
        return this natively; CNN subclasses flatten their ``(B, C, H, W)``
        conv map into it and return ``None`` for CLS.

        Making the *subclass* do that flattening — rather than having the base
        class branch on architecture family — is what keeps
        :meth:`extract_features` a single code path. The flatten/unflatten
        round-trip costs nothing next to a forward pass, and the alternative
        puts an ``if is_vit`` in the one method that exists to hide that
        distinction.

        All requested layers must come from **one** forward pass. Looping over
        single-layer calls would multiply the cost of the thing the feature
        cache exists to avoid.

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
