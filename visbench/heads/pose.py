"""The pairwise pose head — probe3d's MLP, and the first board here to need hidden layers (16a-2).

Every other VisBench board is reported with the least expressive head that can
express the task, because a difference between two backbones under a linear
probe is a difference between two *representations* and anything deeper can
compensate for a weak feature map. In practice that has always meant a linear
map: an affine layer, a `LinearHead`, or the 1x1 convolutions `DetectionHead`
and `InstanceHead` are built from. `DPTHead` is the exception that proves it —
nonlinear, and used as a *control* rather than for a board. This head breaks
the rule **deliberately**,
and the reason is measured rather than argued: on the same 50,519 / 1,740 pairs
a single affine map underfits, reaching ``train_loss`` 0.0725-0.0732 — flat
across four backbones and 40x this head's — clearing the no-feature floor by
2.7-3.5 degrees where this one clears it by 24.5-42.6, and producing a residual
ordering that does not reproduce this one's and nearly inverts its top two.

So a pose board is a nonlinear-head board or it is nothing, and the linear run
is kept as a committed control rather than left in a docstring. probe3d's
published pose protocol *is* this MLP, which is what keeps
``protocol: "probe3d_pose"`` honest.

**It reads pooled vectors, two of them.** Every other registered head maps a
``(B, C, H, W)`` feature map; this one maps the concatenation of two images'
pooled features, which is what makes a pose probe cheap — no grid, no
streaming, no dense cache.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from visbench.heads.base import BaseHead, register_head
from visbench.types import FeatureMode

__all__ = ["PoseHead", "POSE_HIDDEN_DIMS", "POSE_OUTPUTS"]

#: probe3d's widths, as the reference implementation writes them. Changing them
#: changes what a board means rather than how fast it trains, so they are named
#: here and recorded in ``task_params`` rather than left as defaults nobody
#: reads.
POSE_HIDDEN_DIMS: tuple[int, ...] = (512, 256, 128)

#: ``[qw, qx, qy, qz, tx, ty, tz]`` — see
#: :data:`visbench.metrics.pose.POSE_COLUMNS`.
POSE_OUTPUTS = 7


@register_head("pose")
class PoseHead(BaseHead):
    """``[B, 2D] -> [B, 7]``: BatchNorm, then 512/256/128, then the 7-vector.

    The normalisation is the first layer and not an afterthought: the two
    halves of the input are pooled features from a frozen backbone, whose scale
    varies by an order of magnitude across the corpus, and the target is a unit
    quaternion beside a translation in metres.

    **The BatchNorm carries running statistics, which are fitted state.** They
    ride in ``state_dict()`` and therefore in a saved artifact, which is what
    keeps this head reloadable — but it is the same class of trap as
    ``DetectionTask.grid_hw``, so check any state a probe learns outside its
    head's parameters. It also means training needs batches of more than one
    row; :class:`~visbench.tasks.mid_level.pose.RelativePoseTask` drops a
    trailing singleton batch rather than crashing on it.

    Parameters
    ----------
    embed_dim : int
        The width of **one** view's pooled features. The head reads twice this,
        because a relative pose is a statement about a pair — passing the
        concatenated width by mistake builds a head twice as wide as intended
        and every shape still checks out at construction, which is why
        :meth:`forward` says which of the two it expected.
    hidden_dims : sequence of int
        Layer widths between the input and the 7-vector. Defaults to
        :data:`POSE_HIDDEN_DIMS`. An empty sequence makes this one affine map —
        the control, reachable by name rather than by a second class.
    """

    #: Named for completeness: this head consumes the *pooled* block, which
    #: every feature mode carries. The mode says what the extraction kept for
    #: the dense half, and a pose run keeps the default.
    supported_feature_modes: tuple[str, ...] = (
        FeatureMode.DENSE_ONLY,
        FeatureMode.DENSE_CLS_BROADCAST,
        FeatureMode.DENSE_PLUS_CLS,
    )
    multiscale = False

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: Sequence[int] = POSE_HIDDEN_DIMS,
        out_features: int = POSE_OUTPUTS,
    ) -> None:
        super().__init__()
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        widths = [int(width) for width in hidden_dims]
        if any(width < 1 for width in widths):
            raise ValueError(f"hidden widths must be >= 1, got {tuple(hidden_dims)}")

        self.embed_dim = int(embed_dim)
        self.hidden_dims = tuple(widths)
        self.out_features = int(out_features)

        layers: list[nn.Module] = [nn.BatchNorm1d(self.in_features)]
        width = self.in_features
        for hidden in widths:
            layers += [nn.Linear(width, hidden), nn.ReLU(inplace=True)]
            width = hidden
        layers.append(nn.Linear(width, self.out_features))
        self.mlp = nn.Sequential(*layers)

    @property
    def in_features(self) -> int:
        """``2 * embed_dim`` — the pair, not one view."""
        return 2 * self.embed_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``(B, 2 * embed_dim)`` concatenated pooled pairs to ``(B, 7)``."""
        if isinstance(features, (tuple, list)):
            raise TypeError(
                f"{type(self).__name__} takes one tensor holding both views of each pair, "
                "concatenated along the channel dim — not a (dense, cls) pair."
            )
        if features.ndim != 2:
            raise ValueError(
                f"Expected (B, 2 * embed_dim) pooled pairs, got {tuple(features.shape)}. "
                "This head reads pooled vectors, not a dense feature map."
            )
        if features.shape[1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} channels (2 x embed_dim {self.embed_dim}), "
                f"got {features.shape[1]}. Both views must come from the same backbone, "
                "and embed_dim is one view's width rather than the concatenated one."
            )
        return self.mlp(features)
