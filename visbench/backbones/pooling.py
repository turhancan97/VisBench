"""Pooling and dense feature-mode assembly.

The single place where "tokens in, requested representation out" is decided,
so no backbone reimplements it. Prior art: ``tokens_to_output`` in
probing-mid-level-vision (Chen, Marks & Cheng, arXiv:2411.17474), whose
``dense`` / ``dense-cls`` modes correspond to :attr:`FeatureMode.DENSE_ONLY`
and :attr:`FeatureMode.DENSE_CLS_BROADCAST` here.
"""

import torch

from visbench.types import FEATURE_MODE_CHOICES, FeatureMode, Pooling

__all__ = ["pool_tokens", "tokens_to_grid", "apply_feature_mode"]


def pool_tokens(
    patch_tokens: torch.Tensor,
    cls_token: torch.Tensor | None,
    pooling: str,
) -> torch.Tensor:
    """Reduce tokens to one vector per image, ``(B, C)``.

    Parameters
    ----------
    patch_tokens:
        ``(B, N, C)`` patch tokens, registers already stripped.
    cls_token:
        ``(B, C)`` or ``None`` for architectures without one.
    pooling:
        :attr:`Pooling.CLS` or :attr:`Pooling.MEAN`. ``"default"`` must be
        resolved by the caller before reaching here.

    Raises
    ------
    ValueError
        If ``pooling="cls"`` is requested from a backbone with no CLS token —
        an explicit failure is better than silently falling back to mean.
    """
    if pooling == Pooling.CLS:
        if cls_token is None:
            raise ValueError(
                "pooling='cls' requested from a backbone with no CLS token. "
                "Use pooling='mean', or pooling='default' to get this "
                "architecture's appropriate default."
            )
        return cls_token
    if pooling == Pooling.MEAN:
        return patch_tokens.mean(dim=1)
    if pooling == Pooling.DEFAULT:
        raise ValueError(
            "pooling='default' must be resolved by the caller "
            "(BaseBackbone.default_pooling) before reaching pool_tokens()"
        )
    raise ValueError(
        f"Unknown pooling {pooling!r}; expected one of {Pooling.CLS!r}, {Pooling.MEAN!r}"
    )


def tokens_to_grid(
    patch_tokens: torch.Tensor,
    grid_hw: tuple[int, int],
) -> torch.Tensor:
    """Reshape ``(B, N, C)`` patch tokens into a ``(B, C, H, W)`` spatial grid.

    Asserts ``N == H * W`` so that an unstripped CLS or register token surfaces
    as a loud error rather than a silently misaligned feature map.
    """
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"Expected patch tokens of shape (B, N, C), got {tuple(patch_tokens.shape)}"
        )
    b, n, c = patch_tokens.shape
    h, w = grid_hw
    if n != h * w:
        raise ValueError(
            f"Token count {n} does not match grid {h}x{w} = {h * w}. "
            "A CLS or register token was probably left in the sequence."
        )
    # transpose gives a non-contiguous view, so reshape (not view) is required.
    return patch_tokens.transpose(1, 2).reshape(b, c, h, w)


def apply_feature_mode(
    dense: torch.Tensor,
    cls_token: torch.Tensor | None,
    mode: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Assemble the representation a dense task head receives.

    Returns
    -------
    ``dense_only``
        ``(B, C, H, W)``.
    ``dense_cls_broadcast``
        ``(B, C + C_cls, H, W)`` — CLS repeated at every location.
    ``dense_plus_cls``
        ``((B, C, H, W), (B, C_cls))`` — kept separate; the head decides how to
        fuse them. No prior-art implementation to follow; design deliberately.

    All three modes work from v0.1 so the interface never has to change, but
    ``dense_only`` is the only one any v0.1 task requests; the other two exist
    for the dense-prediction heads landing in v0.2 (CLAUDE.md, "Dense-task
    feature modes").
    """
    if mode == FeatureMode.DENSE_ONLY:
        return dense

    if mode == FeatureMode.DENSE_CLS_BROADCAST:
        if cls_token is None:
            raise ValueError(
                f"feature_mode={mode!r} needs a CLS token, but this backbone has none. "
                f"Use {FeatureMode.DENSE_ONLY!r}."
            )
        b, _, h, w = dense.shape
        broadcast = cls_token[:, :, None, None].expand(b, cls_token.shape[1], h, w)
        return torch.cat([dense, broadcast], dim=1)

    if mode == FeatureMode.DENSE_PLUS_CLS:
        if cls_token is None:
            raise ValueError(
                f"feature_mode={mode!r} needs a CLS token, but this backbone has none. "
                f"Use {FeatureMode.DENSE_ONLY!r}."
            )
        return dense, cls_token

    raise ValueError(f"Unknown feature mode {mode!r}; expected one of {FEATURE_MODE_CHOICES}")
