"""Correspondence matching and metrics.

Protocol follows probe3d (El Banani et al., CVPR 2024, arXiv:2404.08476).
Nearest-neighbour matching with a ratio test is standard practice inherited
from classical local-feature matching, and is also what vismatch
(https://github.com/gmberton/vismatch) does with dedicated matchers.

Implemented from the published descriptions, not adapted from probe3d's code:
its ``evals/utils/correspondence.py`` derives from Meta code under CC BY-NC
4.0, which is incompatible with an MIT package. See NOTICE. The methods here —
ratio test, thresholded recall, area under the cumulative error curve — are
standard and predate both papers.

Matching uses plain torch rather than faiss. Patch grids are small (256 tokens
for a 224px ViT/14, ~1k at 512px), so an exact ``cdist`` is both fast enough
and exact, and it keeps faiss out of the dependency list.
"""

from collections.abc import Sequence

import torch

from visbench.types import MetricsDict

__all__ = ["nn_match", "ratio_test", "correspondence_recall", "error_auc"]


def nn_match(
    feats_0: torch.Tensor,
    feats_1: torch.Tensor,
    k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k-nearest-neighbour match every feature in ``feats_0`` against ``feats_1``.

    ``k=2`` by default because the ratio test needs the second neighbour.
    Returns ``(distances, indices)``, both ``(N, k)``.

    Features are L2-normalised first, so the ranking is by cosine similarity.
    On normalised vectors euclidean and cosine ranking agree exactly, since
    ``|x - y|^2 = 2 - 2 x.y``; normalising is what makes the ratio threshold
    mean the same thing for backbones whose features differ in magnitude.
    """
    if feats_0.ndim != 2 or feats_1.ndim != 2:
        raise ValueError(
            f"Expected (N, C) features, got {tuple(feats_0.shape)} and {tuple(feats_1.shape)}"
        )
    if feats_0.shape[1] != feats_1.shape[1]:
        raise ValueError(
            f"Feature dims differ ({feats_0.shape[1]} vs {feats_1.shape[1]}); "
            "both sides must come from the same backbone."
        )
    if k > len(feats_1):
        raise ValueError(f"Cannot take {k} neighbours from {len(feats_1)} candidates")

    normed_0 = torch.nn.functional.normalize(feats_0.float(), dim=1)
    normed_1 = torch.nn.functional.normalize(feats_1.float(), dim=1)
    distances = torch.cdist(normed_0, normed_1)
    return distances.topk(k, dim=1, largest=False)


def ratio_test(distances: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    """Lowe's ratio test: keep matches whose first/second neighbour ratio is low.

    Rejects matches in repetitive regions, where the nearest neighbour is no
    more convincing than the runner-up.

    Takes the ``(N, k>=2)`` distances from :func:`nn_match` and returns a
    boolean ``(N,)`` mask.
    """
    if distances.ndim != 2 or distances.shape[1] < 2:
        raise ValueError(
            f"The ratio test needs at least 2 neighbours, got {tuple(distances.shape)}"
        )
    nearest, runner_up = distances[:, 0], distances[:, 1]
    # A zero runner-up means both neighbours are identical to the query, so the
    # match carries no information; clamp keeps the ratio at 1 and rejects it.
    return nearest < threshold * runner_up.clamp(min=1e-12)


def correspondence_recall(
    errors: torch.Tensor,
    thresholds: Sequence[float] = (1, 2, 5, 10),
) -> MetricsDict:
    """Fraction of correspondences within each pixel-error threshold."""
    if errors.numel() == 0:
        # Every match was rejected. 0.0 is the honest score: the backbone
        # produced no usable correspondence, which is a result, not an error.
        return {f"recall@{threshold}px": 0.0 for threshold in thresholds}
    return {
        f"recall@{threshold}px": (errors <= threshold).float().mean().item()
        for threshold in thresholds
    }


def error_auc(errors: torch.Tensor, thresholds: Sequence[float]) -> MetricsDict:
    """Area under the cumulative error curve up to each threshold.

    Summarises the whole error distribution rather than a single cut-off, which
    is why probe3d reports it alongside thresholded recall: two backbones can
    share a recall@5px while one concentrates its errors at 1px and the other
    at 4.9px.

    The curve plots "fraction of matches with error <= x" against x, and the
    area is normalised by the threshold so a perfect result scores 1.0.

    The curve is **linearly interpolated** between observed errors, not treated
    as a step function. That is the convention in probe3d and the wider pose-AUC
    literature, and it is deliberately matched here: the two give different
    numbers for the same data (a single 2px error under a 4px threshold scores
    0.75 interpolated, 0.5 stepped), so a benchmark that quietly chose the other
    one would not be comparable with any published result.
    """
    if errors.numel() == 0:
        return {f"auc@{threshold}px": 0.0 for threshold in thresholds}

    ordered = torch.sort(errors.flatten().float()).values
    # Recall after each match, i.e. the y value the curve steps up to.
    recall = torch.arange(1, len(ordered) + 1, dtype=torch.float64) / len(ordered)

    results: MetricsDict = {}
    for threshold in thresholds:
        below = ordered <= threshold
        if not below.any():
            results[f"auc@{threshold}px"] = 0.0
            continue
        # (0,0) -> each error at its recall -> (threshold, final recall). The
        # final point carries the last recall flat out to the threshold, so
        # matches beyond it contribute nothing.
        xs = torch.cat(
            [
                torch.zeros(1, dtype=torch.float64),
                ordered[below].double(),
                torch.tensor([float(threshold)], dtype=torch.float64),
            ]
        )
        ys = torch.cat(
            [torch.zeros(1, dtype=torch.float64), recall[below], recall[below][-1:].clone()]
        )
        results[f"auc@{threshold}px"] = (torch.trapz(ys, xs) / threshold).item()
    return results
