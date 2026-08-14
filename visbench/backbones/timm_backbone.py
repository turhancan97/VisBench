"""timm backbones — the first non-ViT family. v0.2.

CNNs have no CLS token and no patch size, so this is the first real test of the
claims :class:`BaseBackbone` has been making since v0.1: that one
``extract_features`` covers every architecture family, and that
``pooling="default"`` resolving to mean is the right answer when there is no
CLS token.

Both hold, and not just by shape. Mean-pooling the flattened conv map
reproduces a ResNet's own ``global_pool`` output exactly — so "the pooled
vector" means the same thing here as it does for a ViT's CLS token: the
representation the model itself hands to its classifier.

**ConvNeXt is the exception, and it is worth stating rather than smoothing
over.** Its head is a ``NormMlpClassifierHead`` — average pool, *then* a
``LayerNorm2d`` — so what the model hands its classifier is ``norm(mean(x))``
while this class returns ``mean(x)``. The two differ substantially (max
absolute difference 27.5 on one frame), and there is no way to satisfy both
invariants at once: LayerNorm across channels does not commute with a spatial
mean, so normalising the *dense* map first would not reproduce the head either.

The invariant kept instead is the structural one: **``pooled`` is always a
reduction of ``dense``**, for every backbone. A caller who mean-pools the dense
features this class returns gets exactly its ``pooled``. Breaking that for one
model — so its pooled vector came from a path its dense features did not — is
the worse trade, because the cache stores the dense features and every task
that pools does so from them.

So for ConvNeXt, ``pooled`` is the mean of the final stage *before* the head's
normalisation. That is a real representation and a defensible one; it is simply
not the same vector as ``model(x)``. ``tests/backbones/test_timm.py`` pins
which backbones match their own head and which does not, so this cannot drift
into an unexamined assumption.

The module is ``timm_backbone`` rather than ``timm`` so it cannot shadow the
package it imports.
"""

from collections.abc import Callable
from typing import cast

import torch
from PIL import Image

from visbench.backbones.base import BaseBackbone
from visbench.registry import register_backbone
from visbench.types import LayerOutput, LayerSpec, Pooling

__all__ = ["TimmBackbone", "describe_transformer"]

#: Registered name -> (timm model, pretrained tag).
#:
#: The tag is not decoration. timm ships several training recipes per
#: architecture — ``resnet50`` alone has a1/a2/a3/am/b1k and more — and they are
#: different weights producing different features under one architecture name.
#: It goes in the cache key for the same reason DINOv2 pins a commit.
_VARIANTS = {
    "resnet18": ("resnet18", "a1_in1k"),
    "resnet50": ("resnet50", "a1_in1k"),
    # `fb_in1k` rather than `fb_in22k_ft_in1k`: the ResNets above are in1k, so
    # this keeps the CNN rows a like-for-like comparison of architecture rather
    # than of pretraining data. The 22k recipe is the stronger model and would
    # be the more flattering number, which is exactly why it is not the default.
    "convnext_base": ("convnext_base", "fb_in1k"),
    # The two ViTs. Both are transformers, which this class refused outright
    # until it learned to read a model's own structure -- see `_describe_model`.
    "mae_vitb16": ("vit_base_patch16_224", "mae"),
    # The *GAP* SigLIP, not the canonical one. SigLIP's own head is an
    # `AttentionPoolLatent` (`global_pool='map'`), a learned pooling VisBench
    # has no mode for; this official sibling pools by global average, which is
    # exactly VisBench's `mean`. Same features, a head that already fits.
    "siglip_vitb16": ("vit_base_patch16_siglip_gap_224", "webli"),
}

#: ``global_pool`` values this class can express, mapped to VisBench's modes.
#:
#: The rule is the one the ResNets already follow: **the pooled vector is the
#: representation the model itself hands to its classifier.** timm records that
#: per model, so it is read rather than inferred -- MAE reports ``token`` and
#: SigLIP-GAP reports ``avg``, and both are then honest by construction.
#:
#: ``map`` is deliberately absent. It is a learned attention head, not a
#: reduction over tokens, so it cannot be expressed as a pooling *mode* over
#: features the cache already stores.
_POOL_TYPES = {"token": Pooling.CLS, "avg": Pooling.MEAN, "": Pooling.MEAN}


def describe_transformer(model: object, model_name: str) -> tuple[bool, int, str]:
    """``(has_cls_token, patch_size, pool_type)`` read off a timm transformer.

    A module-level function taking the model rather than a method taking
    ``self``, so it can be tested against a stub in the **fast** suite. Every
    timm backbone test needs real weights and is therefore marked ``slow``,
    which CI does not run — and the three decisions made here are exactly the
    kind that produce a silently wrong number rather than an error:

    - a wrong ``has_cls_token`` discards the CLS token while the record claims
      the model never had one;
    - a wrong ``patch_size`` maps every token to the wrong pixel, which
      correspondence would report as a weak backbone;
    - an unrecognised ``global_pool`` would otherwise fall back to some default
      that is not what the model pools with.

    Raises rather than guessing on all three.
    """
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))
    if prefix_tokens > 1:
        raise NotImplementedError(
            f"{model_name} has {prefix_tokens} prefix tokens; this class reads the first "
            "as the CLS token and has nowhere to put the rest (DINOv2's registers are "
            "handled by its own backbone)."
        )

    patch = getattr(getattr(model, "patch_embed", None), "patch_size", None)
    if patch is None:
        raise NotImplementedError(
            f"{model_name} produces token sequences but exposes no patch_embed, so there "
            "is no way to map tokens back to pixel positions."
        )

    pool = getattr(model, "global_pool", "")
    pool_type = pool if isinstance(pool, str) else ""
    if pool_type not in _POOL_TYPES:
        raise NotImplementedError(
            f"{model_name} pools with global_pool={pool_type!r}, which VisBench has no "
            f"mode for — it expresses {sorted(_POOL_TYPES)}. 'map' is SigLIP's learned "
            "AttentionPoolLatent head: a trained module, not a reduction over the tokens "
            "the cache stores, so it cannot be a pooling mode. Use the `_gap_` sibling, "
            "whose own pooling is a global average."
        )

    size = int(patch[0] if isinstance(patch, tuple | list) else patch)
    return prefix_tokens > 0, size, pool_type


@register_backbone("resnet18", variant="resnet18")
@register_backbone("resnet50", variant="resnet50")
@register_backbone("convnext_base", variant="convnext_base")
@register_backbone("mae_vitb16", variant="mae_vitb16")
@register_backbone("siglip_vitb16", variant="siglip_vitb16")
class TimmBackbone(BaseBackbone):
    """Any timm model — CNN or transformer — through the one interface.

    For a **CNN**, dense features are the last conv feature map before global
    pooling (a ResNet's ``layer4`` output), flattened to a token sequence so the
    base class needs no branch on architecture family. There is no CLS token and
    no patch grid, so ``pooling="cls"`` raises rather than falling back.

    For a **transformer**, dense features are the patch tokens and the CLS token
    is kept when the model has one. This class refused transformers outright
    until v0.9: ``has_cls_token`` and ``patch_size`` were *class* attributes
    declaring "CNN" for everything, and a false ``has_cls_token`` discards the
    CLS token while the record claims there was none to keep. They are read per
    instance now — see :meth:`_describe_model` — which is what makes a timm ViT
    honest rather than merely loadable.

    **What ``pooling="default"`` resolves to is the model's own choice**, read
    from timm's ``global_pool`` rather than inferred from whether a CLS token
    exists. Mean-pooling a ResNet's conv map reproduces its ``global_pool``
    exactly; MAE reports ``token`` and SigLIP-GAP reports ``avg``; in every case
    the pooled vector is the representation the model hands its own classifier.

    Registered names cover ResNet-18/50, ConvNeXt-B, MAE ViT-B/16 and
    SigLIP-GAP ViT-B/16; any other timm model works by passing ``model_name=``.
    """

    #: Defaults for the CNN case. Both are overwritten per instance for a
    #: transformer, which is the whole point of :meth:`_describe_model`.
    has_cls_token = False
    #: CNN stride is architectural, not a patch grid — see the base class.
    patch_size = None

    def __init__(
        self,
        variant: str | None = None,
        model_name: str | None = None,
        pretrained_tag: str | None = None,
        device: str | None = None,
        image_size: int | None = None,
    ) -> None:
        """Load a pretrained timm model, freeze it, set eval mode.

        Parameters
        ----------
        variant:
            A registered name from :data:`_VARIANTS`. Mutually exclusive with
            ``model_name``.
        model_name / pretrained_tag:
            Any timm model, for architectures without a registered name.
            Leaving the tag ``None`` takes timm's default for that model, which
            is recorded in the cache key so the run stays identifiable.
        image_size:
            Defaults to the model's own training resolution. CNNs accept other
            sizes — there is no patch-multiple constraint — but the dense grid
            scales with it, so correspondence needs it fixed across a run.
        """
        super().__init__(device)

        if variant is not None and model_name is not None:
            raise ValueError("Pass variant or model_name, not both")
        if variant is not None:
            if variant not in _VARIANTS:
                raise ValueError(
                    f"Unknown variant {variant!r}; expected one of {sorted(_VARIANTS)}. "
                    "For any other timm model, pass model_name= instead."
                )
            model_name, pretrained_tag = _VARIANTS[variant]
        elif model_name is None:
            raise ValueError("One of variant or model_name is required")

        try:
            import timm
            from timm.data import create_transform, resolve_data_config
        except ImportError as exc:  # pragma: no cover - needs the extra uninstalled
            raise ImportError(
                "timm backbones need timm. Install it with `pip install visbench[timm]`."
            ) from exc

        spec = f"{model_name}.{pretrained_tag}" if pretrained_tag else model_name
        model = timm.create_model(spec, pretrained=True, num_classes=0)
        # Read config off the local before assigning: once it is an attribute of
        # an nn.Module, __getattr__ widens every lookup to Tensor | Module.
        pretrained_cfg: dict = dict(getattr(model, "pretrained_cfg", {}))
        num_features = int(getattr(model, "num_features", 0))
        # Same reason: feature_info describes the stages and is read on every
        # layer_channels() call, so copy it out while it is still a plain list.
        self._feature_info: list[dict] = [dict(info) for info in cast(list, model.feature_info)]
        self.model = model

        # Which family this is decides everything below, and it is settled here
        # rather than at the first extraction. Once `forward_intermediates` is
        # asked for NCHW it reshapes a ViT's tokens into a grid, so by then the
        # output looks exactly like a conv map and nothing downstream would
        # notice that the CLS token had been dropped. One forward pass at
        # construction is cheap next to the weight download that preceded it.
        model.eval()
        forward_features = cast(Callable[[torch.Tensor], torch.Tensor], model.forward_features)
        with torch.no_grad():
            probe = forward_features(
                torch.zeros(1, *pretrained_cfg.get("input_size", (3, 224, 224)))
            )
        #: 4D is a conv map; 3D is a token sequence.
        self.is_transformer = probe.ndim == 3
        if probe.ndim not in (3, 4):
            raise NotImplementedError(
                f"{model_name} returns {probe.ndim}D features, which is neither a conv "
                "map (4D) nor a token sequence (3D)."
            )
        self._describe_model(model, model_name)

        config = resolve_data_config({}, model=model)
        if image_size is not None:
            config["input_size"] = (3, image_size, image_size)
        self.image_size = config["input_size"][-1]

        self.name = variant or model_name
        self.variant = variant
        self.model_name = model_name
        # Resolved rather than echoed: passing None means "timm's default", and
        # the record has to say which weights that actually was.
        self.pretrained_tag = pretrained_cfg.get("tag") or "default"
        self.embed_dim = num_features

        # timm knows each model's own normalisation, crop ratio and
        # interpolation; a shared transform would silently mis-preprocess
        # anything whose recipe differs from ImageNet's.
        self._transform = create_transform(**config)
        self._finalize()

    def _describe_model(self, model: object, model_name: str) -> None:
        """Read this model's own structure, rather than assuming a CNN's.

        ``has_cls_token`` and ``patch_size`` were class attributes declaring
        "CNN" for every model, which is why transformers used to be refused
        outright: for a ViT both were false, and a false ``has_cls_token``
        silently discards the CLS token while the record claims there was none.
        Read per instance, they are true statements about whatever was loaded,
        and any timm ViT becomes usable rather than only the CNNs.

        The pooling this resolves to is **the model's own** — ``global_pool``,
        which timm records per checkpoint. That is the rule the ResNets already
        follow (mean-pooling their conv map reproduces `global_pool` exactly),
        applied to a family where the answer is not always the same: MAE says
        ``token`` and SigLIP-GAP says ``avg``, so `default` means CLS for one
        and mean for the other, each matching what the model hands its own
        classifier.
        """
        if not self.is_transformer:
            self._pool_type = "avg"
            return
        self.has_cls_token, self.patch_size, self._pool_type = describe_transformer(
            model, model_name
        )

    def default_pooling(self) -> str:
        """The pooling the loaded model itself uses.

        Overrides the base's "CLS if there is one, mean otherwise", which is a
        good default but only a proxy. A ViT can carry a CLS token and still be
        trained to average — timm records which, so this reads it instead of
        guessing, and the two can never disagree.
        """
        return _POOL_TYPES[self._pool_type]

    @property
    def num_layers(self) -> int:
        """Feature stages timm exposes, stem included — 5 for a ResNet.

        These are not equivalent to a ViT's blocks. A ResNet's stages halve the
        resolution and double the width as they go, so a multi-layer request
        returns maps of **different shapes**, and a head consuming them needs
        per-layer ``in_channels``. A ViT's blocks all share one width and grid.
        That difference is architectural and is not something this class should
        paper over.
        """
        return len(self._feature_info)

    def layer_channels(self, layers: LayerSpec = None) -> list[int]:
        """Channel width of each stage, for building a head that fits.

        ``DPTHead(in_channels=backbone.layer_channels([1, 2, 3, 4]))`` is the
        intended use: a CNN's stages differ in width, so the head cannot assume
        a single number the way it can for a ViT.
        """
        indices = self.resolve_layers(layers)
        return [int(self._feature_info[index]["num_chs"]) for index in indices]

    def _forward_features(
        self,
        image: torch.Tensor,
        layers: list[int],
    ) -> list[LayerOutput]:
        """Return ``(patch_tokens, None, grid_hw)`` per requested stage.

        ``forward_intermediates`` runs the network once and taps each stage,
        which is how a multi-layer request costs one forward pass rather than
        one per layer. Maps are flattened here, in the subclass, rather than in
        the base class — the round-trip costs nothing next to a forward pass
        and keeps ``extract_features`` free of any ``if is_vit``.
        """
        forward_intermediates = cast(Callable[..., list], self.model.forward_intermediates)
        if self.is_transformer:
            return self._transformer_features(forward_intermediates, image, layers)

        feature_maps = forward_intermediates(
            image,
            indices=list(layers),
            output_fmt="NCHW",
            intermediates_only=True,
        )

        result: list[LayerOutput] = []
        for feature_map in feature_maps:
            _, _, grid_h, grid_w = feature_map.shape
            # (B, C, H, W) -> (B, H*W, C), matching the row-major order
            # patch_centers() assumes when it maps tokens back to pixels.
            tokens = feature_map.flatten(2).transpose(1, 2)
            result.append((tokens, None, (grid_h, grid_w)))
        return result

    def _transformer_features(
        self,
        forward_intermediates: Callable[..., list],
        image: torch.Tensor,
        layers: list[int],
    ) -> list[LayerOutput]:
        """Patch tokens and the CLS token, from one forward pass.

        ``return_prefix_tokens`` is timm's name for what open_clip calls
        ``output_extra_tokens``; the two APIs differ, which is why this cannot
        simply reuse :class:`~visbench.backbones.clip.CLIPBackbone`'s call.

        **``norm=True`` is load-bearing.** ``forward_intermediates`` returns raw
        block outputs, so the final LayerNorm has not been applied — patch-token
        standard deviation comes out at 1.43 rather than 1.0. With it, the last
        layer reproduces the model's own ``forward_features`` exactly, tokens
        and CLS alike, which is what makes "the features" mean the same thing
        here as for DINOv2 (which normalises by default) and CLIP (which
        applies ``ln_post`` for the same reason).
        """
        patch = self.patch_size
        assert patch is not None  # set by _describe_model for every transformer
        _, _, height, width = image.shape
        if height % patch or width % patch:
            raise ValueError(f"Input {height}x{width} is not a multiple of patch size {patch}")
        grid_hw = (height // patch, width // patch)

        outputs = forward_intermediates(
            image,
            indices=list(layers),
            output_fmt="NLC",
            return_prefix_tokens=True,
            norm=True,
            intermediates_only=True,
        )

        result: list[LayerOutput] = []
        for output in outputs:
            # timm pairs each layer with its prefix tokens *only when there are
            # any*: a model with `num_prefix_tokens = 0` (SigLIP-GAP) gets a
            # bare tensor back even though `return_prefix_tokens=True` was
            # asked for. Unpacking unconditionally then iterates the token
            # tensor itself and fails with an unrelated message.
            if isinstance(output, tuple | list):
                patch_tokens, prefix_tokens = output
            else:
                patch_tokens, prefix_tokens = output, None

            expected = grid_hw[0] * grid_hw[1]
            if patch_tokens.shape[1] != expected:
                raise ValueError(
                    f"Expected {expected} patch tokens for a {grid_hw} grid, got "
                    f"{patch_tokens.shape[1]}"
                )
            cls_token = None
            if self.has_cls_token:
                assert prefix_tokens is not None  # guaranteed by num_prefix_tokens > 0
                cls_token = prefix_tokens[:, 0]
            result.append((patch_tokens, cls_token, grid_hw))
        return result

    def preprocess(self, images: Image.Image | list) -> torch.Tensor:
        """Apply the model's own timm transform."""
        if isinstance(images, Image.Image):
            images = [images]
        return torch.stack([self._transform(img.convert("RGB")) for img in images])

    def cache_key(self) -> str:
        """``"timm/<model>/<tag>/<resolution>"``.

        The tag is included because ``resnet50.a1_in1k`` and ``resnet50.a3_in1k``
        are different weights behind one architecture name.
        """
        return f"timm/{self.model_name}/{self.pretrained_tag}/{self.image_size}"
