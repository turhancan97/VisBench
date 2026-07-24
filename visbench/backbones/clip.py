"""OpenCLIP ViT-B image-tower backbone — v0.1.

Only the visual tower is exposed; the text tower is out of scope for probing.
Note that CLIP uses its own normalisation constants, not ImageNet's.
"""

import warnings
from typing import Optional, Union

import torch
from PIL import Image
from torchvision import transforms

from visbench.backbones.base import BaseBackbone
from visbench.registry import register_backbone
from visbench.types import LayerSpec
from visbench.utils.image import CLIP_MEAN, CLIP_STD

__all__ = ["CLIP"]

#: Registered name -> (open_clip model name, pretrained tag, embed dim, patch size).
#:
#: The ``-quickgelu`` suffix is not cosmetic. OpenAI's original CLIP weights
#: were trained with QuickGELU, and open_clip pairs them with a plain-GELU
#: architecture unless asked otherwise — it warns and continues, producing a
#: model that loads cleanly and computes subtly wrong activations. The pairing
#: is validated in ``__init__`` so that combination raises instead.
_VARIANTS = {
    "clip_vitb16": ("ViT-B-16-quickgelu", "openai", 768, 16),
    "clip_vitb32": ("ViT-B-32-quickgelu", "openai", 768, 32),
}

#: Dim of the shared image-text embedding for ViT-B, after ``visual.proj``.
_PROJECTED_DIM = 512


@register_backbone("clip_vitb16", variant="clip_vitb16")
@register_backbone("clip_vitb32", variant="clip_vitb32")
class CLIP(BaseBackbone):
    """CLIP visual tower.

    Has a CLS token, so default pooling is CLS — but note this is the
    pre-projection CLS, i.e. the representation used for probing, not the
    projected image embedding used for text alignment. Which of the two a task
    wants is a real decision; document it wherever it is made.

    **Pre-projection is the default here.** CLIP's visual tower ends with a
    linear projection into the space shared with the text encoder, and
    ``encode_image`` returns that 512-d vector. VisBench returns the 768-d CLS
    token from *before* it, because the projection is trained to discard
    whatever does not help match a caption — which is exactly the visual detail
    a mid-level probe exists to measure. It is also the wrong comparison to
    draw against DINOv2, which has no such head. Pass ``use_projection=True``
    for the 512-d embedding, which is what a published zero-shot number used.

    Patch size is 16 or 32 against DINOv2's 14, so the two produce different
    grids at the same input resolution. That is why correspondence measures
    error in patch widths rather than pixels.
    """

    has_cls_token = True

    def __init__(
        self,
        variant: str = "clip_vitb16",
        device: Optional[str] = None,
        image_size: int = 224,
        use_projection: bool = False,
    ) -> None:
        """Load the OpenCLIP visual tower for ``variant``, freeze it, set eval mode."""
        super().__init__(device)

        if variant not in _VARIANTS:
            raise ValueError(
                f"Unknown CLIP variant {variant!r}; expected one of {sorted(_VARIANTS)}"
            )
        model_name, pretrained, width, patch_size = _VARIANTS[variant]

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} is not a multiple of the patch size {patch_size}. "
                "A ragged final patch would silently change the grid shape."
            )

        self.name = variant
        self.variant = variant
        self.model_name = model_name
        self.pretrained = pretrained
        self.patch_size = patch_size
        self.image_size = image_size
        self.use_projection = use_projection
        self.embed_dim = _PROJECTED_DIM if use_projection else width

        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - needs the extra uninstalled
            raise ImportError(
                "The CLIP backbone needs open_clip_torch. Install it with "
                "`pip install visbench[clip]`."
            ) from exc

        # Promote open_clip's QuickGELU warning to an error. It is the one
        # failure mode here that yields a working model with wrong numbers, and
        # a warning in the middle of a benchmark run is a warning nobody reads.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*QuickGELU mismatch.*")
            try:
                model = open_clip.create_model(model_name, pretrained=pretrained)
            except UserWarning as exc:
                raise RuntimeError(
                    f"{model_name} + {pretrained!r} is a QuickGELU mismatch: {exc}. "
                    "That combination loads cleanly and computes wrong activations."
                ) from exc

        self.model = model.visual
        self._transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                # CLIP's own constants, not ImageNet's — see visbench.utils.image.
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )
        self._finalize()

    def _forward_features(
        self,
        image: torch.Tensor,
        layers: LayerSpec,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], tuple[int, int]]:
        """Run the visual transformer, returning ``(patch_tokens, cls_token, grid_hw)``.

        Uses ``forward_intermediates``, open_clip's counterpart to DINOv2's
        ``get_intermediate_layers``, so the v0.2 multi-layer path widens this
        call rather than rewriting it.
        """
        _, _, height, width = image.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input {height}x{width} is not a multiple of patch size {self.patch_size}"
            )
        grid_hw = (height // self.patch_size, width // self.patch_size)

        depth = len(self.model.transformer.resblocks)
        index = depth - 1 if layers is None else layers[0]
        if index < 0:
            index += depth
        if not 0 <= index < depth:
            raise ValueError(f"Layer index {layers[0] if layers else -1} out of range for {depth}")

        outputs = self.model.forward_intermediates(
            image,
            indices=[index],
            output_fmt="NLC",
            output_extra_tokens=True,
            intermediates_only=True,
        )
        patch_tokens = outputs["image_intermediates"][0]
        cls_token = outputs["image_intermediates_prefix"][0][:, 0]

        # forward_intermediates returns raw block outputs, so the final
        # LayerNorm has not been applied. DINOv2's equivalent normalises by
        # default; applying it here keeps "the features" meaning the same thing
        # across backbones, and it is what `proj` expects downstream —
        # ln_post(cls) @ proj reproduces encode_image exactly.
        patch_tokens = self.model.ln_post(patch_tokens)
        cls_token = self.model.ln_post(cls_token)

        if self.use_projection:
            if self.model.proj is None:
                raise RuntimeError(f"{self.model_name} has no visual projection")
            patch_tokens = patch_tokens @ self.model.proj
            cls_token = cls_token @ self.model.proj

        expected = grid_hw[0] * grid_hw[1]
        if patch_tokens.shape[1] != expected:
            raise RuntimeError(
                f"CLIP returned {patch_tokens.shape[1]} patch tokens, expected {expected} "
                f"for a {grid_hw[0]}x{grid_hw[1]} grid"
            )

        return patch_tokens, cls_token, grid_hw

    def preprocess(self, images: Union[Image.Image, list]) -> torch.Tensor:
        """Resize to 224 and apply CLIP normalisation constants."""
        if isinstance(images, Image.Image):
            images = [images]
        return torch.stack([self._transform(img.convert("RGB")) for img in images])

    def cache_key(self) -> str:
        """``"clip/<model>/<pretrained-tag>/<resolution>/<head>"``.

        The pretrained tag is in the key because ``openai`` and ``laion2b``
        weights are different models behind one name, and ``head`` because the
        projected and pre-projection vectors are different representations of
        the same forward pass.
        """
        head = "proj" if self.use_projection else "preproj"
        return f"clip/{self.model_name}/{self.pretrained}/{self.image_size}/{head}"
