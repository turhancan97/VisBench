"""Pooling and dense feature-mode assembly.

The single place where "tokens in, requested representation out" is decided,
so no backbone reimplements it. Prior art: ``tokens_to_output`` in
probing-mid-level-vision (Chen, Marks & Cheng, arXiv:2411.17474), whose
``dense`` / ``dense-cls`` modes correspond to :attr:`FeatureMode.DENSE_ONLY`
and :attr:`FeatureMode.DENSE_CLS_BROADCAST` here.
"""

from typing import Optional

import torch

__all__ = ["pool_tokens", "tokens_to_grid", "apply_feature_mode"]


def pool_tokens(
    patch_tokens: torch.Tensor,
    cls_token: Optional[torch.Tensor],
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
    raise NotImplementedError


def tokens_to_grid(
    patch_tokens: torch.Tensor,
    grid_hw: tuple[int, int],
) -> torch.Tensor:
    """Reshape ``(B, N, C)`` patch tokens into a ``(B, C, H, W)`` spatial grid.

    Asserts ``N == H * W`` so that an unstripped CLS or register token surfaces
    as a loud error rather than a silently misaligned feature map.
    """
    raise NotImplementedError


def apply_feature_mode(
    dense: torch.Tensor,
    cls_token: Optional[torch.Tensor],
    mode: str,
):
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

    Only ``dense_only`` is enabled in v0.1; the others raise until v0.2.
    """
    raise NotImplementedError
