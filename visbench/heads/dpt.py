"""DPT-style multiscale head.

Follows probe3d (El Banani et al., CVPR 2024, arXiv:2404.08476), which is also
where the depth and surface-normal protocols come from — using the same head
as the reference protocol keeps VisBench numbers comparable to published ones.
The architecture itself is Ranftl et al., "Vision Transformers for Dense
Prediction" (arXiv:2103.13413).

Written from those descriptions rather than adapted. probe3d's ``probes.py``
carries no separate licence header, so unlike its
``evals/utils/correspondence.py`` it falls under that repository's MIT licence
and would have been safe to adapt — implementing it keeps the package uniform
on that question. See NOTICE.

Genuinely multiscale, so this head is the reason multi-layer extraction must be
wired up in v0.2: it consumes features from several backbone depths at once.
"""

from collections.abc import Sequence
from typing import Optional, Union

import torch
import torch.nn as nn

from visbench.heads.base import BaseHead, register_head
from visbench.types import FeatureMode

__all__ = ["DPTHead"]


class ResidualConvUnit(nn.Module):
    """Two 3x3 convolutions with a skip connection.

    The fusion block's workhorse. No normalisation layer: batch statistics over
    a frozen backbone's features vary with batch composition, which would make
    a probe's score depend on how the loader happened to group images.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FusionBlock(nn.Module):
    """Merge a coarser stage into a finer one, then upsample.

    RefineNet-style: add the incoming stage if there is one, refine, project,
    and double the resolution. Running top-down like this is what lets the
    finest output carry information from every depth fed in.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = ResidualConvUnit(channels)
        self.project = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, previous: Optional[torch.Tensor] = None) -> torch.Tensor:
        if previous is not None:
            if previous.shape[-2:] != x.shape[-2:]:
                previous = nn.functional.interpolate(
                    previous, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            x = x + previous
        x = self.refine(x)
        x = self.project(x)
        return nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


@register_head("dpt")
class DPTHead(BaseHead):
    """Fuses features from multiple backbone layers into a dense prediction.

    Reassembles each layer to a common width, resamples them to a pyramid of
    resolutions — coarsest for the earliest layer fed in — then fuses top-down
    and projects to the output.

    **Requires several layers.** ``forward`` takes a list of dense maps, one per
    backbone depth, and rejects a single tensor rather than duplicating it: a
    DPT fed one layer is not multiscale, and reporting its score as a DPT number
    would misdescribe the architecture that produced it. Multi-layer extraction
    is the next v0.2 step; until then this head is complete but has nothing real
    to consume.

    **Output resolution differs from** :class:`LinearHead`. The last fusion
    block upsamples, so with ``output_size=None`` this head returns *twice* the
    feature-grid resolution where a linear head returns the grid itself — the
    same relationship DPT has to its backbone, where the pyramid ends one step
    above the input. Pass ``output_size`` explicitly on any task that swaps
    heads per run, or the two will produce predictions at different scales from
    identical features.

    Report a headline dense-task number with :class:`LinearHead` as well. A
    deeper head can compensate for a weak feature map and narrow the very gap
    between backbones the probe exists to measure.
    """

    supported_feature_modes: tuple[str, ...] = (
        FeatureMode.DENSE_ONLY,
        FeatureMode.DENSE_CLS_BROADCAST,
        FeatureMode.DENSE_PLUS_CLS,
    )
    multiscale = True

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        num_layers: int = 4,
        hidden_dim: int = 256,
        output_size: Optional[Union[int, tuple[int, int]]] = None,
        use_cls: bool = False,
        cls_dim: Optional[int] = None,
    ) -> None:
        """Configure the pyramid.

        Parameters
        ----------
        in_channels:
            One width for every layer, or a per-layer sequence when the depths
            fed in differ.
        num_layers:
            How many backbone depths this head expects. ``forward`` checks the
            list length against it, so a mismatch fails immediately rather than
            as a shape error inside a fusion block.
        output_size:
            Resolution to upsample the prediction to. ``None`` leaves it at
            twice the feature grid — see the class docstring, and prefer to be
            explicit when comparing against :class:`LinearHead`.
        use_cls:
            Under ``dense_plus_cls``, add a projection of the global vector to
            the coarsest stage. This is the fusion that mode exists to enable —
            at a bottleneck rather than into every pixel of every layer, which
            is what ``dense_cls_broadcast`` already does more cheaply.
        cls_dim:
            Width of that global vector, when ``use_cls``. Defaults to the first
            layer's width, which is right whenever the CLS token and the patch
            tokens come from the same backbone. Set it when they do not — the
            CLS width is a property of the backbone, not of whichever layer the
            vector happens to be injected alongside.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}")

        widths = [in_channels] * num_layers if isinstance(in_channels, int) else list(in_channels)
        if len(widths) != num_layers:
            raise ValueError(f"Got {len(widths)} in_channels for num_layers={num_layers}")

        self.in_channels = widths
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_cls = use_cls
        self.output_size = (
            (output_size, output_size) if isinstance(output_size, int) else output_size
        )

        # Reassemble: every layer to a common width before anything is fused.
        self.reassemble = nn.ModuleList(
            nn.Conv2d(width, hidden_dim, kernel_size=1) for width in widths
        )
        #: Resolution multiplier per stage, coarsest first. The earliest layer
        #: fed in is treated as the coarsest, matching DPT's use of shallow
        #: features for large-scale structure.
        self.scales = [2**-i for i in range(num_layers - 1, -1, -1)]
        self.fusion = nn.ModuleList(FusionBlock(hidden_dim) for _ in range(num_layers))
        # Sized from widths[0]: the vector is injected at stage 0, and for a
        # single backbone every layer shares the CLS width anyway.
        self.cls_dim = (cls_dim if cls_dim is not None else widths[0]) if use_cls else None
        self.cls_project = nn.Linear(self.cls_dim, hidden_dim) if self.cls_dim else None

        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, max(hidden_dim // 2, 1), kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(max(hidden_dim // 2, 1), out_channels, kernel_size=1),
        )

    def forward(self, features) -> torch.Tensor:
        """A list of ``(B, C, H, W)`` maps — one per layer — to a dense prediction."""
        stages, cls_vector = self._unpack(features)

        fused: Optional[torch.Tensor] = None
        for index, (stage, block) in enumerate(zip(stages, self.fusion)):
            projected = self.reassemble[index](stage)
            projected = self._to_scale(projected, stage.shape[-2:], self.scales[index])
            if index == 0 and cls_vector is not None and self.cls_project is not None:
                # Injected once, at the coarsest stage: a global vector is
                # global, and adding it at every level would only re-weight it.
                if cls_vector.ndim != 2 or cls_vector.shape[1] != self.cls_dim:
                    raise ValueError(
                        f"Expected a (B, {self.cls_dim}) CLS vector, got "
                        f"{tuple(cls_vector.shape)}. Pass cls_dim= if the backbone's "
                        "CLS width differs from its first layer's channel count."
                    )
                bias = self.cls_project(cls_vector)[:, :, None, None]
                projected = projected + bias
            fused = block(projected, fused)

        assert fused is not None  # num_layers >= 1 guarantees at least one pass
        return self._resize(self.output(fused), self.output_size)

    def _unpack(self, features):
        """Split the input into a per-layer list and an optional CLS vector."""
        cls_vector = None
        # A ``(stages, cls)`` pair is identified by its *first* element being a
        # sequence of maps. Keying off the second element instead would read a
        # plain two-layer tuple — ``head((stage0, stage1))``, which is how a
        # caller who did not notice the list/tuple distinction will write it —
        # as one stage plus a CLS vector, and report the resulting failure as
        # "got a single tensor" when the caller passed two.
        if isinstance(features, tuple) and len(features) == 2:
            if isinstance(features[0], (list, tuple)):
                features, cls_vector = features

        if isinstance(features, torch.Tensor):
            raise TypeError(
                f"{type(self).__name__} is multiscale and takes a list of dense maps, one "
                f"per backbone layer, but got a single {tuple(features.shape)} tensor. "
                "Duplicating it would report a single-layer result as a DPT number. "
                "Use LinearHead, or extract with layers=[...] once multi-layer "
                "extraction lands."
            )

        stages = list(features)
        if len(stages) != self.num_layers:
            raise ValueError(
                f"This head expects {self.num_layers} layers, got {len(stages)}. "
                "Build it with num_layers matching the layers you extract."
            )
        for index, stage in enumerate(stages):
            if stage.ndim != 4:
                raise ValueError(f"Layer {index} is {tuple(stage.shape)}; expected (B, C, H, W)")
            if stage.shape[1] != self.in_channels[index]:
                raise ValueError(
                    f"Layer {index} has {stage.shape[1]} channels but this head was built "
                    f"for {self.in_channels[index]}"
                )
        return stages, cls_vector

    @staticmethod
    def _to_scale(tensor: torch.Tensor, size, scale: float) -> torch.Tensor:
        """Resample a stage to ``scale`` times the feature grid."""
        target = (max(1, int(size[0] * scale)), max(1, int(size[1] * scale)))
        if tuple(tensor.shape[-2:]) == target:
            return tensor
        return nn.functional.interpolate(tensor, size=target, mode="bilinear", align_corners=False)
