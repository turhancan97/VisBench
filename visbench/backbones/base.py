"""The backbone abstraction.

One method, one return shape, for every architecture family. ViT and CNN
internals differ completely but callers cannot tell the difference — that is
the whole point of this class (CLAUDE.md, "Feature extraction design").
"""

from abc import ABC, abstractmethod
from typing import Any, cast

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
    patch_size: int | None = None

    def __init__(self, device: str | None = None) -> None:
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

    # -- fine-tuning (v0.3) --------------------------------------------------

    #: How many trailing blocks :meth:`unfreeze_last` has unfrozen. ``0`` — the
    #: default and the only value any v0.1/v0.2 run ever had — means a frozen
    #: probe, and is what :meth:`extract_features_trainable` refuses on.
    trainable_blocks: int = 0

    def _blocks(self) -> Any:
        """The sequence of blocks/stages fine-tuning may unfreeze, shallowest first.

        Not implemented on the base class, and deliberately not defaulted to
        something plausible: a wrong answer here silently unfreezes the wrong
        parameters, which trains and reports a number rather than failing.
        Subclasses that can support fine-tuning override it; the rest inherit
        the refusal below, which names the backbone and the limitation.
        """
        raise NotImplementedError(
            f"Fine-tuning is not supported for {self.name or type(self).__name__} yet — "
            "step 6a covers DINOv2 only. Run it frozen, or open an issue for this family."
        )

    def unfreeze_last(self, n: int) -> int:
        """Make the last ``n`` blocks trainable. Returns the parameter count.

        Undoes exactly one half of :meth:`_finalize`: ``requires_grad`` on the
        chosen blocks. The module **stays in eval() mode**, which is not an
        oversight. Unfreezing a stage in train mode would start BatchNorm
        updating its running statistics and activate dropout, so a fine-tuned
        number would differ from its frozen baseline for two reasons at once
        and neither would be visible in the record. Keeping eval() is also
        standard practice for the small batches a probe trains with.

        Raises rather than returning zero when nothing was unfrozen. A run that
        unfreezes no parameters trains exactly like a frozen probe and reports
        the result as fine-tuned — the same shape of failure as a warning filter
        matching a phrase that is never emitted, and just as invisible.
        """
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError(f"n must be an int, got {n!r}")
        if n < 1:
            raise ValueError(
                f"unfreeze_last(n) needs n >= 1, got {n}. For a frozen probe do not call "
                "it at all — n=0 would leave a task claiming to fine-tune while training "
                "nothing."
            )

        blocks = self._blocks()
        depth = len(blocks)
        if n > depth:
            raise ValueError(
                f"Cannot unfreeze {n} blocks: {self.name or type(self).__name__} has {depth}. "
                "Fine-tuning the whole backbone is not probing, so this is not clamped."
            )

        trainable = 0
        for block in list(blocks)[depth - n :]:
            for param in block.parameters():
                param.requires_grad_(True)
                trainable += param.numel()

        if trainable == 0:
            raise RuntimeError(
                f"unfreeze_last({n}) on {self.name or type(self).__name__} made 0 parameters "
                "trainable, so this run would train exactly like a frozen probe while "
                "reporting itself as fine-tuned. _blocks() is returning something without "
                "parameters."
            )

        self.trainable_blocks = n
        return trainable

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Every parameter :meth:`unfreeze_last` made trainable, for the optimiser."""
        return [param for param in self.parameters() if param.requires_grad]

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
        return self._features(image, pooling, layers, feature_mode)

    def extract_features_trainable(
        self,
        image: torch.Tensor,
        pooling: str = Pooling.DEFAULT,
        layers: LayerSpec = None,
        feature_mode: str = FeatureMode.DENSE_ONLY,
    ) -> FeatureDict:
        """:meth:`extract_features`, but building a graph for backpropagation.

        The one entry point that does not suppress gradients, for fine-tuning
        (v0.3). Separate from :meth:`extract_features` rather than a flag on it,
        because every existing caller — the cache above all — depends on getting
        detached tensors, and a keyword whose default preserved that would put
        the expensive mistake one typo away.

        **Nothing here may be cached.** Cache keys name the weights through
        :meth:`cache_key`, and fine-tuned weights differ at every optimiser
        step, so an entry written from this path is stale on arrival and, worse,
        indistinguishable from a frozen one — it would be served to every later
        frozen run of the same backbone.

        Refuses outright until :meth:`unfreeze_last` has been called: a
        graph-building forward pass over a fully frozen model produces no
        gradients at all, so a caller who forgot would train nothing and report
        the result as fine-tuned.
        """
        if self.trainable_blocks == 0:
            raise RuntimeError(
                f"{self.name or type(self).__name__} is fully frozen, so a trainable "
                "forward pass would produce no gradients and train nothing. Call "
                "unfreeze_last(n) first, or use extract_features() for a frozen probe."
            )
        return self._features(image, pooling, layers, feature_mode)

    def _features(
        self,
        image: torch.Tensor,
        pooling: str,
        layers: LayerSpec,
        feature_mode: str,
    ) -> FeatureDict:
        """The extraction itself, shared by the frozen and trainable entry points.

        Split out so the two cannot drift on validation, layer resolution or
        assembly — they must differ in exactly one respect, whether a graph is
        built, and that difference lives in the decorator above.
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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

        # strict=False: the two arguments are deliberately ragged. Pairing a list
        # with its own tail is how consecutive pairs are formed, and the shorter
        # one is meant to end the walk.
        if any(b <= a for a, b in zip(resolved, resolved[1:], strict=False)):
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
