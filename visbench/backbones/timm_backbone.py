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

The module is ``timm_backbone`` rather than ``timm`` so it cannot shadow the
package it imports.
"""

from typing import Optional, Union

import torch
from PIL import Image

from visbench.backbones.base import BaseBackbone
from visbench.registry import register_backbone
from visbench.types import LayerSpec

__all__ = ["TimmBackbone"]

#: Registered name -> (timm model, pretrained tag).
#:
#: The tag is not decoration. timm ships several training recipes per
#: architecture — ``resnet50`` alone has a1/a2/a3/am/b1k and more — and they are
#: different weights producing different features under one architecture name.
#: It goes in the cache key for the same reason DINOv2 pins a commit.
_VARIANTS = {
    "resnet18": ("resnet18", "a1_in1k"),
    "resnet50": ("resnet50", "a1_in1k"),
}


@register_backbone("resnet18", variant="resnet18")
@register_backbone("resnet50", variant="resnet50")
class TimmBackbone(BaseBackbone):
    """Any timm CNN, exposed through the same interface as the ViTs.

    Dense features are the last conv feature map before global pooling — a
    ResNet's ``layer4`` output — flattened to a token sequence so the base
    class needs no branch on architecture family (CLAUDE.md, "CNN vs ViT
    handling").

    No CLS token, so :meth:`default_pooling` returns mean, and asking for
    ``pooling="cls"`` raises rather than silently falling back.

    Registered names cover ResNet-18 and ResNet-50; any other timm CNN works by
    constructing this class directly with ``model_name=``. Arbitrary
    user-supplied ``nn.Module`` backbones are a separate v0.2 step.
    """

    has_cls_token = False
    #: CNN stride is architectural, not a patch grid — see the base class.
    patch_size = None

    def __init__(
        self,
        variant: Optional[str] = None,
        model_name: Optional[str] = None,
        pretrained_tag: Optional[str] = None,
        device: Optional[str] = None,
        image_size: Optional[int] = None,
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
        self.model = timm.create_model(spec, pretrained=True, num_classes=0)

        config = resolve_data_config({}, model=self.model)
        if image_size is not None:
            config["input_size"] = (3, image_size, image_size)
        self.image_size = config["input_size"][-1]

        self.name = variant or model_name
        self.variant = variant
        self.model_name = model_name
        # Resolved rather than echoed: passing None means "timm's default", and
        # the record has to say which weights that actually was.
        self.pretrained_tag = self.model.pretrained_cfg.get("tag") or "default"
        self.embed_dim = self.model.num_features

        # timm knows each model's own normalisation, crop ratio and
        # interpolation; a shared transform would silently mis-preprocess
        # anything whose recipe differs from ImageNet's.
        self._transform = create_transform(**config)
        self._finalize()

    def _forward_features(
        self,
        image: torch.Tensor,
        layers: LayerSpec,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], tuple[int, int]]:
        """Return ``(patch_tokens, None, grid_hw)`` from the last conv map.

        ``forward_features`` gives the feature map before global pooling. It is
        flattened here, in the subclass, rather than in the base class — the
        round-trip costs nothing next to a forward pass and keeps
        ``extract_features`` free of any ``if is_vit``.
        """
        feature_map = self.model.forward_features(image)

        if feature_map.ndim != 4:
            raise NotImplementedError(
                f"{self.model_name} returns {feature_map.ndim}D features, so it is a "
                "transformer rather than a CNN. Use the dinov2_* or clip_* backbones; "
                "timm ViTs are not wired up."
            )

        _, _, grid_h, grid_w = feature_map.shape
        # (B, C, H, W) -> (B, H*W, C), matching the row-major order
        # patch_centers() assumes when it maps tokens back to pixels.
        tokens = feature_map.flatten(2).transpose(1, 2)
        return tokens, None, (grid_h, grid_w)

    def preprocess(self, images: Union[Image.Image, list]) -> torch.Tensor:
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
