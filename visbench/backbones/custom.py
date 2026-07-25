"""Wrap an arbitrary ``nn.Module`` as a VisBench backbone. v0.2.

The escape hatch: any model a user already has — a fine-tuned checkpoint, an
architecture VisBench has never heard of, something from a paper's repo — probed
by the same tasks as DINOv2 and CLIP, without adding a module to this package.

Constructed directly rather than looked up by name, since a registry name
cannot carry an ``nn.Module``::

    backbone = CustomBackbone(my_model, preprocess=my_transform, name="mine")
    visbench.run(backbone, "retrieval", dataset)

To give a custom backbone a registry name of its own, subclass
:class:`~visbench.backbones.base.BaseBackbone` and apply
:func:`visbench.register_backbone` — that path is unchanged and is how the
built-in backbones work.
"""

import hashlib
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from PIL import Image

from visbench.backbones.base import BaseBackbone
from visbench.types import LayerOutput

__all__ = ["CustomBackbone"]

#: Return signature of a user-supplied feature function.
FeatureFn = Callable[[nn.Module, torch.Tensor], tuple]

#: Multi-layer counterpart: ``(module, image_batch, layers) -> list`` of
#: ``(patch_tokens, cls_token, grid_hw)``, one per requested index. Kept
#: separate from :data:`FeatureFn` rather than overloading its signature, so a
#: function written for one convention can never be called under the other.
LayerFeatureFn = Callable[[nn.Module, torch.Tensor, list], list]


def hash_weights(module: nn.Module, sample_bytes: int = 1 << 20) -> str:
    """Short hash of a module's parameters, for the cache key.

    A custom backbone has no upstream commit or pretrained tag to point at, so
    the weights themselves are the only honest identifier. Hashing them means a
    fine-tuned checkpoint automatically gets a different cache key from the one
    it was fine-tuned from — the alternative is a user-supplied string that is
    correct only if they remember to change it.

    Large tensors are sampled rather than read whole: a full pass over a
    multi-GB state dict on every construction would cost more than the forward
    passes it protects. Shapes and dtypes are always folded in, so a change in
    architecture is caught even when the sampled bytes collide.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        flat = tensor.detach().cpu().flatten()
        digest.update(f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode())
        if flat.numel() == 0:
            continue
        values = flat.to(torch.float64)
        # A cheap moment-based summary plus a byte sample: catches retraining
        # and fine-tuning, which is what this needs to detect.
        digest.update(f"{values.sum().item():.6e}|{values.abs().max().item():.6e}".encode())
        digest.update(flat[: sample_bytes // flat.element_size()].numpy().tobytes())
    return digest.hexdigest()[:16]


class CustomBackbone(BaseBackbone):
    """Adapt a user-supplied ``nn.Module`` to the VisBench backbone contract.

    Parameters
    ----------
    module:
        Any ``nn.Module``. It is frozen and set to eval mode — probing measures
        fixed representations, and unfreezing is v0.3 scope.
    preprocess:
        Callable turning one PIL image into a ``(3, H, W)`` tensor, typically a
        torchvision ``Compose``. Required, and deliberately so: normalisation
        constants cannot be guessed, and getting them wrong is a silent
        accuracy loss rather than an error.
    feature_fn:
        Optional ``(module, image_batch) -> (patch_tokens, cls_token, grid_hw)``
        for full control. Without it, the module's own ``forward`` is called and
        its output interpreted — see :meth:`_forward_features`.
    has_cls_token:
        Whether the token sequence starts with a CLS token. Only consulted when
        the module returns tokens; ignored for a conv map.
    layer_feature_fn:
        Optional ``(module, image_batch, layers) -> [(tokens, cls, grid_hw), ...]``
        enabling multi-layer extraction. Requires ``num_layers``. VisBench
        cannot tap the intermediate activations of an arbitrary module — there
        is no equivalent of ``get_intermediate_layers`` to call — so this is
        the seam where a user says how their own model exposes depth.
    num_layers:
        How many depths ``layer_feature_fn`` can serve. Left at 1, the module
        exposes only its final output and any multi-layer request is rejected,
        which is the right default: silently returning the same map several
        times would let a multiscale head report a single-layer result.
    weights_id:
        Identifier for these weights in the cache key. Defaults to a hash of
        the module's parameters, which is usually what you want.
    """

    def __init__(
        self,
        module: nn.Module,
        preprocess: Callable[[Image.Image], torch.Tensor],
        name: str = "custom",
        embed_dim: Optional[int] = None,
        has_cls_token: bool = False,
        patch_size: Optional[int] = None,
        feature_fn: Optional[FeatureFn] = None,
        weights_id: Optional[str] = None,
        image_size: int = 224,
        device: Optional[str] = None,
        layer_feature_fn: Optional[LayerFeatureFn] = None,
        num_layers: int = 1,
    ) -> None:
        super().__init__(device)

        if not isinstance(module, nn.Module):
            raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
        if not callable(preprocess):
            raise TypeError(
                "preprocess must be callable (PIL image -> (3, H, W) tensor). "
                "VisBench cannot guess a model's normalisation constants."
            )

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_layers > 1 and layer_feature_fn is None:
            raise ValueError(
                f"num_layers={num_layers} promises depths VisBench cannot reach on its own. "
                "Pass layer_feature_fn=(module, images, layers) -> [(tokens, cls, grid_hw), ...]; "
                "an arbitrary nn.Module has no interface for tapping intermediate activations."
            )
        if layer_feature_fn is not None and num_layers == 1:
            raise ValueError(
                "layer_feature_fn was given but num_layers=1, so no multi-layer request "
                "can ever reach it. Pass num_layers= to say how many depths it serves."
            )

        self.name = name
        self.model = module
        self._num_layers = num_layers
        self._layer_feature_fn = layer_feature_fn
        self.has_cls_token = has_cls_token
        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim or 0
        self._preprocess = preprocess
        self._feature_fn = feature_fn
        self.weights_id = weights_id if weights_id is not None else hash_weights(module)

        self._finalize()

    @property
    def num_layers(self) -> int:
        """Depths this module exposes; 1 unless ``layer_feature_fn`` was given."""
        return self._num_layers

    def _forward_features(
        self,
        image: torch.Tensor,
        layers: list[int],
    ) -> list[LayerOutput]:
        """Interpret the module's output as ``(patch_tokens, cls, grid_hw)``.

        With ``layer_feature_fn`` supplied, that handles the whole request.
        Otherwise exactly one layer can be asked for — the base class has
        already rejected anything else against :attr:`num_layers`.

        With ``feature_fn`` supplied, that does the work and this only
        validates. Otherwise the module's ``forward`` output is read:

        ``(B, C, H, W)``
            A conv map. Unambiguous — the grid is the spatial shape.
        ``(B, N, C)``
            A token sequence. The grid is **not** recoverable from ``N`` alone,
            so ``patch_size`` must be set, or a square grid is assumed and a
            non-square input rejected rather than silently misaligned.

        Anything else raises, because the alternative is a feature map whose
        spatial layout is wrong in a way no shape check downstream would catch.
        """
        if self._layer_feature_fn is not None:
            outputs = list(self._layer_feature_fn(self.model, image, list(layers)))
            if len(outputs) != len(layers):
                raise ValueError(
                    f"layer_feature_fn returned {len(outputs)} outputs for layers={layers}. "
                    "It must return one (tokens, cls, grid_hw) per requested index, in order."
                )
            result: list[LayerOutput] = []
            for index, output in zip(layers, outputs):
                try:
                    tokens, cls_token, grid_hw = output
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"layer_feature_fn's output for layer {index} is not a "
                        f"(tokens, cls, grid_hw) triple: {output!r}"
                    ) from error
                self._note_embed_dim(tokens)
                result.append((tokens, cls_token, tuple(grid_hw)))
            return result

        if self._feature_fn is not None:
            tokens, cls_token, grid_hw = self._feature_fn(self.model, image)
            self._note_embed_dim(tokens)
            return [(tokens, cls_token, tuple(grid_hw))]

        output = self.model(image)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"The module returned {type(output).__name__}, not a tensor. "
                "Pass feature_fn= to extract features yourself."
            )

        if output.ndim == 4:
            _, _, grid_h, grid_w = output.shape
            tokens = output.flatten(2).transpose(1, 2)
            self._note_embed_dim(tokens)
            return [(tokens, None, (grid_h, grid_w))]

        if output.ndim == 3:
            tokens, cls_token, grid_hw = self._tokens_to_features(output, image)
            self._note_embed_dim(tokens)
            return [(tokens, cls_token, grid_hw)]

        raise ValueError(
            f"The module returned a {output.ndim}D tensor {tuple(output.shape)}. "
            "VisBench understands (B, C, H, W) conv maps and (B, N, C) token "
            "sequences; for anything else pass feature_fn=."
        )

    def _tokens_to_features(
        self,
        output: torch.Tensor,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], tuple[int, int]]:
        """Split a token sequence into patches, CLS, and a spatial grid."""
        cls_token = None
        tokens = output
        if self.has_cls_token:
            cls_token = output[:, 0]
            tokens = output[:, 1:]

        num_tokens = tokens.shape[1]
        _, _, height, width = image.shape

        if self.patch_size is not None:
            grid_hw = (height // self.patch_size, width // self.patch_size)
            if grid_hw[0] * grid_hw[1] != num_tokens:
                raise ValueError(
                    f"patch_size={self.patch_size} implies a {grid_hw[0]}x{grid_hw[1]} grid "
                    f"({grid_hw[0] * grid_hw[1]} tokens) but the module returned {num_tokens}. "
                    "Check has_cls_token, or pass feature_fn=."
                )
            return tokens, cls_token, grid_hw

        side = int(round(num_tokens**0.5))
        if side * side != num_tokens:
            raise ValueError(
                f"{num_tokens} patch tokens is not a square grid, and patch_size is unset, "
                "so the spatial layout is unknown. Set patch_size= or pass feature_fn=."
            )
        if height != width:
            raise ValueError(
                f"A square token grid was assumed, but the input is {height}x{width}. "
                "Set patch_size= so the grid can be derived rather than guessed."
            )
        return tokens, cls_token, (side, side)

    def _note_embed_dim(self, tokens: torch.Tensor) -> None:
        """Fill in ``embed_dim`` from the first forward pass if it was not given.

        Metadata rather than machinery — nothing depends on it — but a result
        record saying the feature width is 0 is worse than one saying 2048.
        """
        if not self.embed_dim:
            self.embed_dim = int(tokens.shape[-1])

    def preprocess(self, images: Union[Image.Image, list]) -> torch.Tensor:
        """Apply the user's transform to one image or a sequence of them."""
        if isinstance(images, Image.Image):
            images = [images]
        return torch.stack([self._preprocess(img) for img in images])

    def cache_key(self) -> str:
        """``"custom/<name>/<weights_id>/<resolution>"``.

        The weights id is what keeps two different checkpoints of the same
        architecture from sharing cached features — the failure a custom
        backbone is most exposed to, since there is no upstream ref to lean on.
        """
        return f"custom/{self.name}/{self.weights_id}/{self.image_size}"

    def extra_repr(self) -> str:
        return f"name={self.name!r}, weights_id={self.weights_id!r}"

    def describe(self) -> dict[str, Any]:
        """Metadata for a result record."""
        return {
            "backbone": self.name,
            "backbone_key": self.cache_key(),
            "has_cls_token": self.has_cls_token,
            "embed_dim": self.embed_dim,
        }
