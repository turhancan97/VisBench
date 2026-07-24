"""DINOv2 backbone (ViT-S/14, ViT-B/14). First backbone implemented — v0.1.

Weights via torch.hub (facebookresearch/dinov2). Register tokens are stripped
before pooling or grid reshaping when the variant has them.
"""

from typing import Optional, Union

import torch
from PIL import Image
from torchvision import transforms

from visbench.backbones.base import BaseBackbone
from visbench.registry import register_backbone
from visbench.types import LayerSpec
from visbench.utils.image import IMAGENET_MEAN, IMAGENET_STD

__all__ = ["DINOv2"]

#: Registered name -> (torch.hub entrypoint, embed dim, patch size, n registers).
_VARIANTS = {
    "dinov2_vits14": ("dinov2_vits14", 384, 14, 0),
    "dinov2_vitb14": ("dinov2_vitb14", 768, 14, 0),
}

_HUB_REPO = "facebookresearch/dinov2"


@register_backbone("dinov2_vits14", variant="dinov2_vits14")
@register_backbone("dinov2_vitb14", variant="dinov2_vitb14")
class DINOv2(BaseBackbone):
    """DINOv2 ViT with a CLS token, so default pooling is CLS.

    Input resolution must be a multiple of the patch size (14); the grid is
    ``(H // 14, W // 14)``.
    """

    has_cls_token = True

    def __init__(
        self,
        variant: str = "dinov2_vitb14",
        device: Optional[str] = None,
        image_size: int = 224,
    ) -> None:
        """Load the hub checkpoint for ``variant``, freeze it, set eval mode."""
        super().__init__(device)

        if variant not in _VARIANTS:
            raise ValueError(
                f"Unknown DINOv2 variant {variant!r}; expected one of {sorted(_VARIANTS)}"
            )
        entrypoint, embed_dim, patch_size, num_registers = _VARIANTS[variant]

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} is not a multiple of the patch size {patch_size}. "
                "A ragged final patch would silently change the grid shape."
            )

        self.name = variant
        self.variant = variant
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_registers = num_registers
        self.image_size = image_size

        self.model = torch.hub.load(_HUB_REPO, entrypoint)
        self._finalize()

    def _forward_features(
        self,
        image: torch.Tensor,
        layers: LayerSpec,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], tuple[int, int]]:
        """Run the ViT and return ``(patch_tokens, cls_token, grid_hw)``.

        Uses ``get_intermediate_layers`` so the multi-layer path in v0.2 is a
        widening of this call rather than a rewrite. Register tokens, if the
        variant has them, are dropped here.
        """
        _, _, height, width = image.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input {height}x{width} is not a multiple of patch size {self.patch_size}"
            )
        grid_hw = (height // self.patch_size, width // self.patch_size)

        # `layers=None` means "the last block". Negative indices are resolved
        # here because get_intermediate_layers compares against range(depth).
        depth = len(self.model.blocks)
        index = depth - 1 if layers is None else layers[0]
        if index < 0:
            index += depth
        if not 0 <= index < depth:
            raise ValueError(
                f"Layer index {layers[0] if layers else -1} out of range for depth {depth}"
            )

        outputs = self.model.get_intermediate_layers(
            image,
            n=[index],
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        patch_tokens, cls_token = outputs[0]

        # get_intermediate_layers already strips CLS and registers; assert the
        # count anyway, since a silently misaligned grid is the worst failure
        # mode in this file.
        expected = grid_hw[0] * grid_hw[1]
        if patch_tokens.shape[1] != expected:
            raise RuntimeError(
                f"DINOv2 returned {patch_tokens.shape[1]} patch tokens, expected {expected} "
                f"for a {grid_hw[0]}x{grid_hw[1]} grid"
            )

        return patch_tokens, cls_token, grid_hw

    def preprocess(self, images: Union[Image.Image, list]) -> torch.Tensor:
        """Resize to the configured resolution and apply ImageNet normalisation.

        Resize-short-side + centre-crop, the standard ImageNet eval protocol,
        rather than a square resize: distorting aspect ratio changes the
        geometry that mid-level tasks are meant to measure.
        """
        if isinstance(images, Image.Image):
            images = [images]
        return torch.stack([self._transform(img.convert("RGB")) for img in images])

    @property
    def _transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(
                    self.image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def cache_key(self) -> str:
        """``"dinov2/<variant>/<resolution>"`` — changes if weights or input size change.

        Built from ``self.variant``, not ``self.name``: both are equal today,
        but only ``variant`` is guaranteed to select the weights.
        """
        return f"dinov2/{self.variant}/{self.image_size}"
