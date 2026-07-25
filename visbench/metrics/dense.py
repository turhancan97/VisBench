"""Dense-prediction metrics (depth, surface normals, segmentation) — v0.2.

Definitions come from probe3d (El Banani et al., CVPR 2024, arXiv:2404.08476)
rather than being re-derived, so VisBench numbers stay comparable to published
ones. Its ``evals/utils/metrics.py`` carries an explicit MIT header — unlike
``evals/utils/correspondence.py``, which is CC BY-NC and could not be used this
way. See NOTICE.

Small differences in masking and averaging convention move these numbers
noticeably, so the conventions are spelled out rather than left implicit:

* **Valid pixels only.** A pixel is valid where ``target > 0``. Sensor depth
  maps are full of holes, and scoring a prediction against a hole measures the
  hole.
* **Per image, then averaged.** Each image contributes one number and images
  are weighted equally. Pooling every pixel of the split instead would weight
  images by how much valid depth they happen to contain, letting a dataset with
  uneven hole coverage silently reweight itself.
"""

import math
from collections.abc import Sequence
from typing import Optional

import torch

from visbench.types import MetricsDict

__all__ = [
    "depth_metrics",
    "match_scale_and_shift",
    "surface_normal_metrics",
    "binary_iou",
    "NYU_CROP",
]

#: NYUv2 evaluation crop, ``[45:471, 41:601]`` of a 480x640 map. Conventional
#: in the depth literature rather than principled — probe3d's own comment reads
#: "commonly used in many repos for some reason" — but a number computed
#: without it is not comparable to one computed with it, so it is offered and
#: left off by default.
NYU_CROP = (45, 471, 41, 601)


def _as_maps(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalise ``(B, 1, H, W)`` or ``(B, H, W)`` to ``(B, H, W)``."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction {tuple(pred.shape)} and target {tuple(target.shape)} must match. "
            "Resize the prediction to the ground-truth resolution before scoring."
        )
    if pred.ndim == 4:
        if pred.shape[1] != 1:
            raise ValueError(f"Expected one depth channel, got {pred.shape[1]}")
        return pred.squeeze(1), target.squeeze(1)
    if pred.ndim != 3:
        raise ValueError(f"Expected (B, H, W) or (B, 1, H, W) depth maps, got {tuple(pred.shape)}")
    return pred, target


def match_scale_and_shift(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Least-squares fit of a per-image scale and shift onto ``prediction``.

    From probe3d, which took it from MiDaS (Ranftl et al.). Solves the 2x2
    normal equations for ``a * pred + b ~= target`` over valid pixels only.

    This is what makes a *relative* depth prediction scorable against a metric
    ground truth. A backbone probed this way is being asked whether it encodes
    the shape of the scene, not whether a linear head also learned the dataset's
    absolute scale — a different question, and usually the one worth asking of a
    frozen representation.

    Degenerate images — a constant prediction, or one with no valid target —
    give a singular system, and are left unchanged rather than divided by a
    near-zero determinant.

    The fit is unconstrained, so the scale may come out **negative**: a depth
    map with near and far swapped is an exact affine solution and scores
    perfectly. probe3d and MiDaS both allow this, and VisBench keeps parity
    rather than adding a constraint that would make its numbers incomparable
    with theirs — but it is a reason to prefer plain metric scoring for
    anything load-bearing.
    """
    pred, gt = _as_maps(prediction, target)
    mask = (gt > 0).float()

    a_00 = (mask * pred * pred).sum(dim=(1, 2))
    a_01 = (mask * pred).sum(dim=(1, 2))
    a_11 = mask.sum(dim=(1, 2))
    b_0 = (mask * pred * gt).sum(dim=(1, 2))
    b_1 = (mask * gt).sum(dim=(1, 2))

    determinant = a_00 * a_11 - a_01 * a_01
    scale = torch.ones_like(b_0)
    shift = torch.zeros_like(b_1)
    solvable = determinant != 0
    scale[solvable] = (
        a_11[solvable] * b_0[solvable] - a_01[solvable] * b_1[solvable]
    ) / determinant[solvable]
    shift[solvable] = (
        -a_01[solvable] * b_0[solvable] + a_00[solvable] * b_1[solvable]
    ) / determinant[solvable]

    aligned = pred * scale.view(-1, 1, 1).detach() + shift.view(-1, 1, 1).detach()
    return aligned.unsqueeze(1) if prediction.ndim == 4 else aligned


def depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: Sequence[float] = (1.25, 1.25**2, 1.25**3),
    scale_invariant: bool = False,
    nyu_crop: bool = False,
) -> MetricsDict:
    """``{"d1", "d2", "d3", "rmse", "abs_rel"}`` over valid pixels only.

    Follows probe3d's ``evaluate_depth``. The threshold metrics are the Eigen
    et al. (NeurIPS 2014) delta accuracies: the fraction of valid pixels whose
    prediction and target are within a factor of ``1.25**k`` of each other, in
    whichever direction is worse.

    Parameters
    ----------
    scale_invariant:
        Fit a per-image scale and shift before scoring — see
        :func:`match_scale_and_shift`. Off by default because it changes what
        the number means, and a scale-invariant score quoted as a metric one
        flatters every backbone at once, which is the hardest kind of error to
        notice.
    nyu_crop:
        Apply :data:`NYU_CROP`. Requires 480x640 maps.

    Notes
    -----
    ``abs_rel`` is **not** part of probe3d's reported set. It is included
    because it is standard elsewhere in the depth literature and costs nothing
    here, but the other four are the ones to quote against that paper.
    """
    pred, gt = _as_maps(pred, target)
    pred = pred.float()
    gt = gt.float()

    if nyu_crop:
        top, bottom, left, right = NYU_CROP
        if tuple(gt.shape[-2:]) != (480, 640):
            raise ValueError(
                f"nyu_crop expects 480x640 maps, got {tuple(gt.shape[-2:])}. The crop is "
                "defined in raw NYUv2 pixels, so applying it to a resized map would cut out "
                "a different region than every published number using it."
            )
        pred = pred[..., top:bottom, left:right]
        gt = gt[..., top:bottom, left:right]

    if scale_invariant:
        pred = match_scale_and_shift(pred, gt)

    valid = (gt > 0).float()
    # Zeroed rather than indexed out, so each image keeps its own pixel count
    # and the per-image averages below stay independent of one another.
    pred = pred * valid
    num_valid = valid.sum(dim=(1, 2)).clamp(min=1)

    ratio = torch.maximum(gt / pred.clamp(min=1e-9), pred / gt.clamp(min=1e-9))

    metrics: MetricsDict = {}
    for index, threshold in enumerate(thresholds, start=1):
        within = ((ratio < threshold).float() * valid).sum(dim=(1, 2)) / num_valid
        metrics[f"d{index}"] = within.mean().item()

    squared_error = (gt - pred).pow(2)
    rmse = ((squared_error * valid).sum(dim=(1, 2)) / num_valid).sqrt()
    metrics["rmse"] = rmse.mean().item()

    relative = ((gt - pred).abs() / gt.clamp(min=1e-9)) * valid
    metrics["abs_rel"] = (relative.sum(dim=(1, 2)) / num_valid).mean().item()

    return metrics


def _normal_validity(target: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
    """Resolve the ``(B, H, W)`` float mask of scorable pixels.

    probe3d derives this from the *depth* map (``batch["depth"] > 0``), because
    its NYU loader carries both. VisBench scores normals against a normal map
    alone, so the default is the equivalent property of that map: a pixel is
    valid where its normal has non-zero length. Every normal-map format in
    circulation writes ``(0, 0, 0)`` for "unknown", so the two masks agree in
    practice — but pass ``valid`` explicitly if you have the depth map, since
    only that reproduces probe3d's masking exactly.
    """
    if valid is None:
        return (target.norm(dim=1) > 0).float()

    mask = valid.float()
    if mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError(f"Expected a single validity channel, got {mask.shape[1]}")
        mask = mask.squeeze(1)
    if mask.ndim != 3:
        raise ValueError(f"Expected a (B, H, W) or (B, 1, H, W) mask, got {tuple(valid.shape)}")
    if mask.shape != target.shape[:1] + target.shape[2:]:
        raise ValueError(
            f"Mask {tuple(mask.shape)} does not match target {tuple(target.shape)} "
            "over batch and spatial dimensions"
        )
    return mask


def surface_normal_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> MetricsDict:
    """Angular-error RMSE plus within-11.25/22.5/30-degree fractions.

    Follows probe3d's ``evaluate_surface_norm`` (itself from iDISC, and before
    that Fouhey et al. 2016): angular error via cosine similarity, clamped to
    [-1, 1] before ``acos`` so a floating-point overshoot past 1 cannot turn
    into NaN and poison the whole split's mean.

    Because the error comes from a *cosine*, neither input has to be a unit
    vector — a head that has not learned to normalise is scored on direction
    alone, exactly as probe3d scores it.

    Parameters
    ----------
    pred:
        ``(B, C, H, W)`` with ``C >= 3``. Only the first three channels are
        read, so an uncertainty-aware head's fourth (kappa) channel can be
        passed through untouched — probe3d slices the same way.
    target:
        ``(B, 3, H, W)`` ground-truth normals.
    valid:
        ``(B, H, W)`` or ``(B, 1, H, W)`` mask. Defaults to where ``target``
        has non-zero length; see :func:`_normal_validity`.

    Notes
    -----
    ``d1``/``d2``/``d3`` and ``rmse`` are probe3d's reported set and are the
    ones to quote against that paper. ``mean`` and ``median`` are included
    because the surface-normal literature reports them almost universally;
    both are per-image statistics averaged over images, matching the
    convention in this module's docstring — a median over the pooled pixels of
    the whole split would be a different and less comparable number.

    ``rmse`` and the two extras are in **degrees**; the ``d`` metrics are
    fractions in [0, 1].
    """
    if target.ndim != 4 or target.shape[1] != 3:
        raise ValueError(f"Expected (B, 3, H, W) target normals, got {tuple(target.shape)}")
    if pred.ndim != 4 or pred.shape[1] < 3:
        raise ValueError(
            f"Expected (B, C, H, W) predicted normals with C >= 3, got {tuple(pred.shape)}"
        )
    if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[2:]:
        raise ValueError(
            f"Prediction {tuple(pred.shape)} and target {tuple(target.shape)} must agree on "
            "batch and spatial dimensions. Resize the prediction to the ground-truth "
            "resolution before scoring."
        )

    pred = pred[:, :3].float()
    target = target.float()

    cosine = torch.cosine_similarity(pred, target, dim=1).clamp(min=-1.0, max=1.0)
    error = torch.acos(cosine) * 180.0 / math.pi

    mask = _normal_validity(target, valid)
    # Zeroed rather than indexed out, so each image keeps its own pixel count
    # and the per-image averages stay independent of one another.
    error = error * mask
    count = mask.sum(dim=(1, 2))
    num_valid = count.clamp(min=1)

    metrics: MetricsDict = {}
    for index, threshold in enumerate((11.25, 22.5, 30.0), start=1):
        within = ((error < threshold).float() * mask).sum(dim=(1, 2)) / num_valid
        metrics[f"d{index}"] = within.mean().item()

    metrics["rmse"] = (error.pow(2).sum(dim=(1, 2)) / num_valid).sqrt().mean().item()
    metrics["mean"] = (error.sum(dim=(1, 2)) / num_valid).mean().item()
    metrics["median"] = _masked_median(error, mask, count).mean().item()

    return metrics


def _masked_median(error: torch.Tensor, mask: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Per-image median of ``error`` over ``mask``, as a ``(B,)`` tensor.

    Invalid pixels are pushed to ``+inf`` so they sort above every real error;
    the median is then read at each image's own valid count. Exact, and cheaper
    than gathering a different number of pixels per image.
    """
    ranked = torch.where(mask.bool(), error, torch.full_like(error, float("inf")))
    ranked = ranked.flatten(1).sort(dim=1).values

    counts = count.long()
    lower_index = ((counts - 1).clamp(min=0) // 2).unsqueeze(1)
    # For an even count the median straddles two samples; for an odd one this
    # offset is zero and both reads land on the same pixel.
    upper_index = lower_index + ((counts + 1) % 2).unsqueeze(1)

    lower = ranked.gather(1, lower_index).squeeze(1)
    upper = ranked.gather(1, upper_index.clamp(max=ranked.shape[1] - 1)).squeeze(1)
    median = (lower + upper) / 2
    # An image with no valid pixels would read +inf and take the whole split's
    # average with it; it contributes zero to every other metric, so it does here.
    return torch.where(count > 0, median, torch.zeros_like(median))


def binary_iou(pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
    """Foreground IoU and pixel accuracy for generic object segmentation."""
    raise NotImplementedError("Generic object segmentation, and its metrics, land with that task.")
