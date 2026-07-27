"""OpenCLIP ViT-B image-tower backbone — v0.1.

Only the visual tower is exposed; the text tower is out of scope for probing.
Note that CLIP uses its own normalisation constants, not ImageNet's.
"""

import warnings

import torch
from PIL import Image
from torchvision import transforms

from visbench.backbones.base import BaseBackbone
from visbench.registry import register_backbone
from visbench.types import LayerOutput
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

#: Substring identifying open_clip's QuickGELU warnings, matched case-insensitively.
#:
#: Deliberately one token rather than a sentence. This guard was originally a
#: ``warnings.filterwarnings("error", message=".*QuickGELU mismatch.*")``, and
#: open_clip has never emitted the phrase "QuickGELU mismatch" — so the filter
#: never matched, the warning was never promoted, and the guard was dead code
#: from the day it was written. Its own test only caught it under ``-m slow``,
#: which CI does not run.
#:
#: open_clip warns in *both* directions — weights trained with QuickGELU loaded
#: under a plain-GELU config, and the reverse — with different wording each way
#: (see ``open_clip/factory.py``). Both are genuine mismatches. This is the one
#: token common to the two, and the only part of the wording worth depending on.
_QUICKGELU_MARKER = "quickgelu"


def _promote_quickgelu_warning(
    caught: "list[warnings.WarningMessage]", model_name: str, pretrained: str
) -> None:
    """Turn open_clip's QuickGELU warning into an error; re-emit anything else.

    open_clip only warns on an activation mismatch and hands back a model that
    loads cleanly and computes subtly wrong features. A warning in the middle of
    a benchmark run is a warning nobody reads, and this is the one failure mode
    here that silently changes a number rather than raising.

    Every other warning is re-emitted rather than swallowed: recording warnings
    suppresses them, and a guard against one specific problem has no business
    hiding unrelated deprecation notices from the caller.
    """
    mismatch: str | None = None
    for entry in caught:
        text = str(entry.message)
        if mismatch is None and _QUICKGELU_MARKER in text.lower():
            mismatch = text
            continue
        # stacklevel=2 points at CLIP.__init__ rather than this helper. The true
        # origin inside open_clip is not recoverable once a warning is recorded.
        warnings.warn(entry.message, stacklevel=2)

    if mismatch is not None:
        raise RuntimeError(
            f"{model_name} + {pretrained!r} is a QuickGELU mismatch: {mismatch} "
            "That combination loads cleanly and computes wrong activations."
        )


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
        device: str | None = None,
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

        # Record rather than filter-to-error: open_clip's wording is the only
        # signal available, and matching it as a regex is what broke this guard
        # once. See _promote_quickgelu_warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = open_clip.create_model(model_name, pretrained=pretrained)
        _promote_quickgelu_warning(caught, model_name, pretrained)

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

    @property
    def num_layers(self) -> int:
        """Residual attention blocks in the visual transformer."""
        return len(self.model.transformer.resblocks)

    def _forward_features(
        self,
        image: torch.Tensor,
        layers: list[int],
    ) -> list[LayerOutput]:
        """Run the visual transformer once, one output per requested layer.

        Uses ``forward_intermediates``, open_clip's counterpart to DINOv2's
        ``get_intermediate_layers``, which takes the whole index list.
        """
        _, _, height, width = image.shape
        patch = self.patch_size
        assert patch is not None  # set in __init__; Optional only on the base class
        if height % patch or width % patch:
            raise ValueError(f"Input {height}x{width} is not a multiple of patch size {patch}")
        grid_hw = (height // patch, width // patch)

        outputs = self.model.forward_intermediates(
            image,
            indices=list(layers),
            output_fmt="NLC",
            output_extra_tokens=True,
            intermediates_only=True,
        )

        expected = grid_hw[0] * grid_hw[1]
        result: list[LayerOutput] = []
        for position, index in enumerate(layers):
            patch_tokens = outputs["image_intermediates"][position]
            cls_token = outputs["image_intermediates_prefix"][position][:, 0]

            # forward_intermediates returns raw block outputs, so the final
            # LayerNorm has not been applied. DINOv2's equivalent normalises by
            # default; applying it here keeps "the features" meaning the same
            # thing across backbones, and it is what `proj` expects downstream —
            # ln_post(cls) @ proj reproduces encode_image exactly.
            #
            # Applied to every layer, not just the last. ln_post is trained for
            # the final block's scale, so on an intermediate layer it is a
            # convention rather than a reconstruction — but an unnormalised
            # layer sitting next to normalised ones in the same pyramid would
            # hand a multiscale head stages whose magnitudes differ by a factor
            # the head would have to unlearn.
            patch_tokens = self.model.ln_post(patch_tokens)
            cls_token = self.model.ln_post(cls_token)

            if self.use_projection:
                if self.model.proj is None:
                    raise RuntimeError(f"{self.model_name} has no visual projection")
                patch_tokens = patch_tokens @ self.model.proj
                cls_token = cls_token @ self.model.proj

            if patch_tokens.shape[1] != expected:
                raise RuntimeError(
                    f"CLIP layer {index} returned {patch_tokens.shape[1]} patch tokens, "
                    f"expected {expected} for a {grid_hw[0]}x{grid_hw[1]} grid"
                )
            result.append((patch_tokens, cls_token, grid_hw))
        return result

    def preprocess(self, images: Image.Image | list) -> torch.Tensor:
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
