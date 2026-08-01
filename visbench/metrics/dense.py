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
  hole. Segmentation is the one exception and reads ``target >= 0``, because
  there 0 is a real label — background — and an unlabelled pixel is marked
  negative instead; see :func:`binary_iou`.
* **Per image, then averaged.** Each image contributes one number and images
  are weighted equally. Pooling every pixel of the split instead would weight
  images by how much valid depth they happen to contain, letting a dataset with
  uneven hole coverage silently reweight itself.
"""

import math
from collections.abc import Sequence

import torch

from visbench.types import MetricsDict

__all__ = [
    "depth_metrics",
    "match_scale_and_shift",
    "surface_normal_metrics",
    "binary_iou",
    "edge_metrics",
    "magnitude_metrics",
    "SEGMENTATION_THRESHOLD",
    "confusion_matrix",
    "metrics_from_confusion",
    "semantic_metrics",
    "NYU_CROP",
]

#: NYUv2 evaluation crop, ``[45:471, 41:601]`` of a 480x640 map. Conventional
#: in the depth literature rather than principled — probe3d's own comment reads
#: "commonly used in many repos for some reason" — but a number computed
#: without it is not comparable to one computed with it, so it is offered and
#: left off by default.
NYU_CROP = (45, 471, 41, 601)


def _as_maps(
    pred: torch.Tensor, target: torch.Tensor, noun: str = "depth"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalise ``(B, 1, H, W)`` or ``(B, H, W)`` to ``(B, H, W)``.

    ``noun`` only names the quantity in the error messages — every scalar dense
    target arrives in one of these two shapes, so they all normalise here rather
    than each metric growing its own copy of this.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction {tuple(pred.shape)} and target {tuple(target.shape)} must match. "
            "Resize the prediction to the ground-truth resolution before scoring."
        )
    if pred.ndim == 4:
        if pred.shape[1] != 1:
            raise ValueError(f"Expected one {noun} channel, got {pred.shape[1]}")
        return pred.squeeze(1), target.squeeze(1)
    if pred.ndim != 3:
        raise ValueError(f"Expected (B, H, W) or (B, 1, H, W) {noun} maps, got {tuple(pred.shape)}")
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


def _normal_validity(target: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
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
    valid: torch.Tensor | None = None,
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


#: A predicted probability at or above this counts as foreground. Half is the
#: only threshold that needs no justification, and a probe's score is meant to
#: measure the representation rather than a tuned operating point — sweeping it
#: would report the best threshold for each backbone, which is a different and
#: much more flattering number.
SEGMENTATION_THRESHOLD = 0.5


def binary_iou(pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
    """Foreground IoU and pixel accuracy for generic object segmentation.

    ``{"iou", "f1", "pixel_acc"}``, each a per-image value averaged over the
    batch — the same convention as :func:`depth_metrics` and
    :func:`surface_normal_metrics`, and for the same reason: pooling every pixel
    of the split instead would weight images by how much of the frame their
    object happens to fill.

    Unlike depth and normals this protocol is **not** probe3d's — that paper has
    no binary segmentation task. Foreground IoU is the near-universal choice in
    the figure-ground literature, and it is reported alongside ``f1`` (Dice, the
    other convention) and ``pixel_acc`` because the three disagree in a useful
    way: accuracy alone looks excellent for a probe that predicts background
    everywhere, which on a dataset where objects cover a fifth of the frame is
    already 80%. IoU is the one to quote.

    Parameters
    ----------
    pred:
        ``(B, 1, H, W)`` or ``(B, H, W)`` foreground **probabilities**, not
        logits — thresholded at :data:`SEGMENTATION_THRESHOLD`.
    target:
        Same shape. ``1`` is foreground and ``0`` background; anything
        **negative** marks a pixel as unlabelled and excludes it from all three
        metrics, which is how a dataset with an explicit ignore region travels
        through. This mirrors depth's "0 means no ground truth", differing only
        because 0 is a real label here.

    Notes
    -----
    An image with neither predicted nor ground-truth foreground scores 1.0
    rather than 0/0. That is the honest reading — nothing was there and nothing
    was claimed — but it does mean a split full of empty targets flatters every
    probe equally, so it is worth knowing whether yours contains any.
    """
    pred, gt = _as_maps(pred, target, noun="mask")
    valid = (gt >= 0).float()

    predicted = ((pred >= SEGMENTATION_THRESHOLD).float()) * valid
    actual = ((gt > SEGMENTATION_THRESHOLD).float()) * valid

    intersection = (predicted * actual).sum(dim=(1, 2))
    union = ((predicted + actual) > 0).float().mul(valid).sum(dim=(1, 2))
    correct = ((predicted == actual).float() * valid).sum(dim=(1, 2))
    num_valid = valid.sum(dim=(1, 2))

    # An image with no foreground in either map has an empty union: the probe
    # got it exactly right, so it scores 1 instead of the 0/0 the ratio gives.
    # An image with no *valid* pixels at all is a different case and contributes
    # zero to every metric, exactly as it does for depth and normals.
    empty = (union == 0) & (num_valid > 0)
    iou = torch.where(empty, torch.ones_like(union), intersection / union.clamp(min=1))
    denominator = intersection + union
    f1 = torch.where(empty, torch.ones_like(union), 2 * intersection / denominator.clamp(min=1))

    return {
        "iou": iou.mean().item(),
        "f1": f1.mean().item(),
        "pixel_acc": (correct / num_valid.clamp(min=1)).mean().item(),
    }


# -- dense magnitude (edges, keypoint heatmaps) --------------------------------


def magnitude_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    correlation_key: str = "correlation",
) -> MetricsDict:
    """Agreement between a predicted and a ground-truth dense magnitude map.

    ``{correlation_key, "rmse", "mae"}``, each a per-image value averaged over
    the batch. **Quote the correlation.** :func:`edge_metrics` is this under the
    key ``edge_correlation``.

    Parameters
    ----------
    pred, target:
        ``(B, 1, H, W)`` or ``(B, H, W)`` magnitudes.
    correlation_key:
        Name the correlation is returned under. Each probe passes its own, so
        two magnitude numbers from different targets cannot be mistaken for the
        same measurement in a record — an occlusion-edge correlation and a
        texture-edge correlation are not comparable and must not share a key.

    Notes
    -----
    **Invalid pixels are marked ``NaN``, and this is the fourth validity
    convention in this codebase.** Depth uses 0, normals a zero-length vector,
    label maps a negative index — each an in-band sentinel, available because
    the value in question cannot occur legitimately. A magnitude map has no
    such value: 0 means "no edge here", which is a real reading covering most of
    most frames (this is exactly why :func:`edge_metrics` on an image-derived
    target masks nothing). So a domain that *does* have holes —
    Taskonomy's ``edge_occlusion`` and ``keypoints3d``, whose targets come from
    the 3D reconstruction — must carry validity out of band, and ``NaN`` is the
    only float that is not a possible magnitude.

    It is also the loud choice: a ``NaN`` that reaches an unmasked loss makes
    the loss ``NaN`` on the first step, where a fabricated 0 would train
    quietly and merely score badly. Both this function and
    :meth:`~visbench.tasks.low_level.magnitude.DenseMagnitudeTask._loss` mask
    on ``isfinite``, so the two agree about which pixels exist — which they
    must, or the probe is optimised against pixels it is not scored on.

    On an all-finite target every pixel is valid and this reduces exactly to the
    unmasked computation, which a test pins; the mask costs the image-derived
    domains nothing.
    """
    return _magnitude_metrics(pred, target, correlation_key)


def edge_metrics(pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
    """Agreement between a predicted and a ground-truth edge-magnitude map.

    ``{"edge_correlation", "rmse", "mae"}``, each a per-image value averaged
    over the batch. **Quote ``edge_correlation``.**

    :func:`magnitude_metrics` under the key this codebase's texture-edge numbers
    are published with; see there for the ``NaN`` validity convention, which
    Taskonomy's ``edge_texture`` never exercises.

    Parameters
    ----------
    pred, target:
        ``(B, 1, H, W)`` or ``(B, H, W)`` edge magnitudes. For an image-derived
        edge map there is **no validity mask**: 0 means "no edge", a real
        reading, so every pixel is scored. See
        :func:`~visbench.data.dense.load_edge_map`.

    Notes
    -----
    **Why correlation leads, and it is not a stylistic choice.** Edge magnitude
    is heavily concentrated near zero — on Taskonomy the mean over a frame is
    about 0.011 of the container range while the peak is 0.13. A probe that
    ignores its input and predicts that constant everywhere therefore achieves a
    *small* RMSE, and a reader comparing two backbones on RMSE alone would be
    comparing how well each matched the mean intensity of the split. Pearson
    correlation is invariant to scale and offset, so it asks only the question
    the probe exists to answer — does the representation know **where** the
    edges are — and it scores a constant prediction at 0 by construction.

    ``rmse`` and ``mae`` are reported alongside because correlation is blind to
    the complementary failure: a prediction perfectly shaped but at the wrong
    magnitude scores 1.0. The pair together pin both.

    A map with no variance contributes ``0.0`` to ``edge_correlation``. For a
    constant *prediction* that is the honest score. For a constant *target* —
    a blank frame, which does not occur in Taskonomy and can in a fixture —
    there is no spatial structure to recover, and scoring it 0 rather than
    dropping the image keeps every image weighted equally, which is what lets
    :meth:`~visbench.tasks.dense_base.DenseTrainingTask.evaluate` recover the
    split number from batch means.
    """
    return _magnitude_metrics(pred, target, "edge_correlation")


def _magnitude_metrics(
    pred: torch.Tensor, target: torch.Tensor, correlation_key: str
) -> MetricsDict:
    """Shared body of :func:`magnitude_metrics` and :func:`edge_metrics`."""
    pred, gt = _as_maps(pred, target, noun="magnitude map")

    flat_pred = pred.flatten(1)
    flat_gt = gt.flatten(1)

    # Out of band, not in band: see magnitude_metrics' notes. All-finite input
    # makes this an all-ones mask and every line below reduces to the plain
    # computation.
    valid = torch.isfinite(flat_gt).float()
    count = valid.sum(dim=1)
    # Zeroed rather than left as NaN: a single NaN survives every sum and would
    # poison an image's correlation even after masking, since the mask
    # multiplies rather than selects (selection cannot be batched over images
    # with differing valid counts).
    flat_gt = torch.where(valid.bool(), flat_gt, torch.zeros_like(flat_gt))

    mean_pred = (flat_pred * valid).sum(dim=1, keepdim=True) / count.clamp(min=1).unsqueeze(1)
    mean_gt = (flat_gt * valid).sum(dim=1, keepdim=True) / count.clamp(min=1).unsqueeze(1)
    centred_pred = (flat_pred - mean_pred) * valid
    centred_gt = (flat_gt - mean_gt) * valid
    norm_pred = centred_pred.norm(dim=1)
    norm_gt = centred_gt.norm(dim=1)

    # Guarded rather than clamped: a zero norm means one side is constant, and
    # dividing by a floor would return an arbitrary large-magnitude ratio from
    # what is numerically noise. 1e-8 is comfortably below any real frame's
    # variation at float32. An image with no valid pixels at all lands here too,
    # and scores 0 for the same reason a constant target does.
    degenerate = (norm_pred < 1e-8) | (norm_gt < 1e-8) | (count < 1)
    correlation = (centred_pred * centred_gt).sum(dim=1) / (norm_pred * norm_gt).clamp(min=1e-8)
    correlation = torch.where(degenerate, torch.zeros_like(correlation), correlation)

    error = (flat_pred - flat_gt) * valid
    denominator = count.clamp(min=1)
    return {
        correlation_key: correlation.mean().item(),
        "rmse": (error.pow(2).sum(dim=1) / denominator).sqrt().mean().item(),
        "mae": (error.abs().sum(dim=1) / denominator).mean().item(),
    }


# -- semantic (multi-class) segmentation --------------------------------------


def _as_label_maps(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalise scores or labels plus a label map to two ``(B, H, W)`` long maps.

    ``pred`` may be ``(B, C, H, W)`` scores — logits or probabilities, since
    ``argmax`` is indifferent to any monotone transform of them — or an already
    reduced ``(B, H, W)``/``(B, 1, H, W)`` label map. Accepting both means the
    metric can be handed a head's raw output *or* a stored prediction and cannot
    disagree with itself about which one it scored.

    Unlike :func:`_as_maps` the two shapes deliberately do not have to match:
    ``C`` classes of score reduce to one label per pixel.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2 for a multi-class metric, got {num_classes}")

    if pred.ndim == 4 and pred.shape[1] == num_classes:
        labels = pred.argmax(dim=1)
    elif pred.ndim == 4 and pred.shape[1] == 1:
        labels = pred.squeeze(1)
    elif pred.ndim == 3:
        labels = pred
    else:
        raise ValueError(
            f"Expected (B, {num_classes}, H, W) scores or (B, H, W) labels, got {tuple(pred.shape)}"
        )

    gt = target.squeeze(1) if target.ndim == 4 and target.shape[1] == 1 else target
    if gt.ndim != 3:
        raise ValueError(f"Expected (B, H, W) or (B, 1, H, W) label map, got {tuple(target.shape)}")
    if labels.shape != gt.shape:
        raise ValueError(
            f"Prediction reduces to {tuple(labels.shape)} but the target is {tuple(gt.shape)}. "
            "Resize the prediction to the ground-truth resolution before scoring."
        )
    return labels.long(), gt.long()


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Accumulate a ``(num_classes, num_classes)`` count matrix, rows ground truth.

    Only valid pixels are counted: a label map marks unlabelled pixels negative
    (VOC's 255 object outlines become -1), and 0 is the real background class.

    Out-of-range labels are dropped rather than clamped or wrapped. A prediction
    cannot produce one, so a stray value means the target was built with the
    wrong class count, and folding it into a neighbouring class would quietly
    corrupt that class's score instead of leaving the discrepancy visible in the
    pixel counts.
    """
    labels, gt = _as_label_maps(pred, target, num_classes)
    valid = (gt >= 0) & (gt < num_classes) & (labels >= 0) & (labels < num_classes)
    if not valid.any():
        return torch.zeros(num_classes, num_classes, dtype=torch.long)

    indices = gt[valid] * num_classes + labels[valid]
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def metrics_from_confusion(matrix: torch.Tensor) -> MetricsDict:
    """Reduce a confusion matrix to mIoU, pixel accuracy and mean class accuracy.

    This is the **dataset-level** reduction: one matrix accumulated over every
    image, then the ratios taken once. It is how VOC, ADE20K and Cityscapes
    define mIoU, and the only version comparable to published numbers.

    A class absent from both the ground truth and the prediction is excluded
    rather than scored 0. Counting it would drag the mean down in proportion to
    how many of the dataset's categories a split happens not to contain, which
    says nothing about the representation.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square confusion matrix, got {tuple(matrix.shape)}")

    matrix = matrix.double()
    true_positive = matrix.diag()
    actual = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    union = actual + predicted - true_positive

    total = matrix.sum()
    if total == 0:
        return {"miou": 0.0, "pixel_acc": 0.0, "mean_acc": 0.0}

    present = union > 0
    iou = true_positive[present] / union[present]

    labelled = actual > 0
    accuracy = true_positive[labelled] / actual[labelled]

    return {
        "miou": iou.mean().item() if present.any() else 0.0,
        "pixel_acc": (true_positive.sum() / total).item(),
        "mean_acc": accuracy.mean().item() if labelled.any() else 0.0,
    }


def semantic_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> MetricsDict:
    """Per-image mIoU and pixel accuracy, averaged over the batch.

    The per-image reduction this codebase uses everywhere else (see the module
    docstring). It is **not** the number a VOC leaderboard reports: averaging
    per-image IoUs weights a class by how many images contain it, while the
    dataset-level reduction in :func:`metrics_from_confusion` weights it by
    pixels. The two disagree, often by several points, so
    :class:`~visbench.tasks.high_level.semantic_segmentation.SemanticSegmentationTask`
    reports both under distinct names rather than picking one and leaving the
    reader to guess which they are looking at.

    An image with no valid pixels contributes 0 to every metric, matching how
    :func:`binary_iou` treats a fully ignored frame.
    """
    labels, gt = _as_label_maps(pred, target, num_classes)

    ious = []
    accuracies = []
    for index in range(labels.shape[0]):
        matrix = confusion_matrix(labels[index : index + 1], gt[index : index + 1], num_classes)
        single = metrics_from_confusion(matrix)
        ious.append(single["miou"])
        accuracies.append(single["pixel_acc"])

    count = max(len(ious), 1)
    return {
        "miou_per_image": sum(ious) / count,
        "pixel_acc": sum(accuracies) / count,
    }
