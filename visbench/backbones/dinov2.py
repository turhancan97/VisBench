"""DINOv2 backbone (ViT-S/14, ViT-B/14). First backbone implemented — v0.1.

Weights via torch.hub (facebookresearch/dinov2). Register tokens are stripped
before pooling or grid reshaping when the variant has them.
"""

import hashlib
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from visbench.backbones.base import BaseBackbone
from visbench.registry import register_backbone
from visbench.types import LayerOutput
from visbench.utils.image import IMAGENET_MEAN, IMAGENET_STD

__all__ = ["DINOv2", "HUB_REF"]

#: Registered name -> (torch.hub entrypoint, embed dim, patch size, n registers).
_VARIANTS = {
    "dinov2_vits14": ("dinov2_vits14", 384, 14, 0),
    "dinov2_vitb14": ("dinov2_vitb14", 768, 14, 0),
}

_HUB_REPO = "facebookresearch/dinov2"

#: Pinned upstream commit. torch.hub defaults to the repository's default
#: branch, which means the weights behind ``dinov2_vitb14`` can change under a
#: fixed VisBench version — and because the ref is part of :meth:`cache_key`,
#: an unpinned load would let already-cached features from the old weights be
#: served for the new ones with nothing to indicate it. Pinning is what makes
#: a reported number reproducible from the record that logged it.
#:
#: Upstream publishes no tags, so this is a commit SHA. Bump it deliberately;
#: every cache entry and result record carries the short ref with it.
HUB_REF = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def _file_digest(path: Path) -> str:
    """Short content hash of a checkpoint file, for the cache key."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


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
        device: str | None = None,
        image_size: int = 224,
        hub_ref: str = HUB_REF,
        checkpoint: str | Path | None = None,
    ) -> None:
        """Load the hub checkpoint for ``variant``, freeze it, set eval mode.

        Parameters
        ----------
        hub_ref:
            Upstream git ref to load from. Defaults to the pinned
            :data:`HUB_REF`; override only to test against a newer upstream,
            and expect a different :meth:`cache_key`.
        checkpoint:
            Path to a local ``state_dict``, loaded instead of reaching the
            network. The architecture still comes from torch.hub (cached after
            one download), so this covers a pinned local copy of the weights
            rather than a fully offline install.
        """
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
        self.hub_ref = hub_ref
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None

        # trust_repo=True is defensible precisely because the ref is pinned:
        # the code being executed is a fixed commit, not whatever landed on the
        # default branch today.
        self.model = torch.hub.load(
            f"{_HUB_REPO}:{hub_ref}",
            entrypoint,
            pretrained=self.checkpoint is None,
            trust_repo=True,
        )
        if self.checkpoint is not None:
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            # Accept both a bare state_dict and a wrapped training checkpoint.
            state = state.get("model", state) if isinstance(state, dict) else state
            self.model.load_state_dict(state)

        # Built once: preprocess() runs it per image, and rebuilding a Compose
        # inside that loop is pure waste.
        self._transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self._finalize()

    @property
    def num_layers(self) -> int:
        """Transformer blocks — 12 for ViT-S/B, 24 for ViT-L."""
        return len(self.model.blocks)

    def _blocks(self) -> Any:
        """The transformer blocks, which are also what :attr:`num_layers` counts.

        The same sequence layer indices address, so ``unfreeze_last(2)`` on a
        12-block model unfreezes exactly the blocks a ``layers=[10, 11]``
        request would read.
        """
        return self.model.blocks

    def _forward_features(
        self,
        image: torch.Tensor,
        layers: list[int],
    ) -> list[LayerOutput]:
        """Run the ViT once and return one ``(patch_tokens, cls, grid_hw)`` per layer.

        ``get_intermediate_layers`` takes the whole index list, so every
        requested depth comes from a single forward pass. Register tokens, if
        the variant has them, are dropped here.
        """
        _, _, height, width = image.shape
        patch = self.patch_size
        assert patch is not None  # set in __init__; Optional only on the base class
        if height % patch or width % patch:
            raise ValueError(f"Input {height}x{width} is not a multiple of patch size {patch}")
        grid_hw = (height // patch, width // patch)

        outputs = self.model.get_intermediate_layers(
            image,
            n=layers,
            reshape=False,
            return_class_token=True,
            norm=True,
        )

        # get_intermediate_layers already strips CLS and registers; assert the
        # count anyway, since a silently misaligned grid is the worst failure
        # mode in this file.
        expected = grid_hw[0] * grid_hw[1]
        result: list[LayerOutput] = []
        for index, (patch_tokens, cls_token) in zip(layers, outputs, strict=True):
            if patch_tokens.shape[1] != expected:
                raise RuntimeError(
                    f"DINOv2 layer {index} returned {patch_tokens.shape[1]} patch tokens, "
                    f"expected {expected} for a {grid_hw[0]}x{grid_hw[1]} grid"
                )
            result.append((patch_tokens, cls_token, grid_hw))
        return result

    def preprocess(self, images: Image.Image | list) -> torch.Tensor:
        """Resize to the configured resolution and apply ImageNet normalisation.

        Resize-short-side + centre-crop, the standard ImageNet eval protocol,
        rather than a square resize: distorting aspect ratio changes the
        geometry that mid-level tasks are meant to measure.
        """
        if isinstance(images, Image.Image):
            images = [images]
        return torch.stack([self._transform(img.convert("RGB")) for img in images])

    def cache_key(self) -> str:
        """``"dinov2/<variant>/<resolution>/<ref>"`` — every input to the features.

        Built from ``self.variant``, not ``self.name``: both are equal today,
        but only ``variant`` is guaranteed to select the weights.

        The weights ref is in the key because it has to be. Without it, bumping
        :data:`HUB_REF` leaves every existing cache entry looking valid while
        describing the old model — a silently wrong number, not a crash. A local
        ``checkpoint`` replaces the ref with a hash of the file for the same
        reason.
        """
        if self.checkpoint is not None:
            weights = "local-" + _file_digest(self.checkpoint)
        else:
            weights = self.hub_ref[:12]
        return f"dinov2/{self.variant}/{self.image_size}/{weights}"
