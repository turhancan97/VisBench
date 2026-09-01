"""BSDS500's boundary protocol — ODS, OIS and AP.

The measurement VisBench's `edge` probe deliberately does **not** make. That one
regresses a magnitude and scores per-image Pearson correlation; this one asks
how many predicted boundary pixels a *person* also marked, which is a
correspondence question and needs a matching rather than a per-pixel comparison.

**Implemented from the published description, never from the benchmark code.**
The BSR package's `bench/` tree — `boundaryBench.m`, `correspondPixels` and the
CSA solver under it — carries no licence, so it may not be adapted here; see
`NOTICE`, which takes the same position on probe3d's CC BY-NC correspondence
helpers and for the same reason. What is reproduced is the protocol, which is
public: Martin, Fowlkes & Malik, *Learning to Detect Natural Image Boundaries*
(TPAMI 2004), and Arbelaez, Maire, Fowlkes & Malik, *Contour Detection and
Hierarchical Image Segmentation* (TPAMI 2011).

The protocol, in the order it runs:

1. Sweep a threshold over the soft boundary map (99 levels by default).
2. At each level, binarise and **thin to single-pixel width** — without this a
   thick prediction is paid for once in recall and many times in precision.
3. Match the thinned prediction against each annotator separately, by
   minimum-cost bipartite correspondence with a distance tolerance.
4. A predicted pixel is correct if it matched **any** annotator (union);
   recall is counted against **each** annotator and summed. That asymmetry is
   the whole point of keeping every annotation — see
   :class:`~visbench.data.bsds.BSDS500Dataset`, where the annotators disagree by
   a median factor of 1.92 on how much of an image is boundary.
5. Aggregate: **ODS** is the best F at one threshold shared by the whole
   dataset, **OIS** the best F when each image may choose its own, and **AP**
   the area under the dataset's precision-recall curve.

Only the *cardinality* of the matching reaches the score — precision is matched
predictions over predictions, recall is matched annotator pixels over annotator
pixels — so the tolerance decides the numbers and the tie-breaking between
equal-cardinality matchings barely does. :func:`correspond_pixels` still
minimises total distance among maximum-cardinality matchings, because that is
what the protocol specifies.
"""

import numpy as np
import torch

__all__ = [
    "DEFAULT_MAX_DIST",
    "DEFAULT_THRESHOLDS",
    "boundary_metrics",
    "correspond_pixels",
    "image_counts",
    "thin_boundaries",
]

#: Matching tolerance as a fraction of the image diagonal. 0.0075 is the value
#: every published BSDS500 boundary number uses; on a 481x321 image it is 4.3
#: pixels. Changing it changes what the score means, so it travels in the
#: record rather than being assumed.
DEFAULT_MAX_DIST = 0.0075

#: Threshold levels swept over the soft map. The published sweep is 99.
DEFAULT_THRESHOLDS = 99


def _as_numpy(array) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


# -- thinning -----------------------------------------------------------------


def thin_boundaries(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: reduce a binary map to single-pixel-wide curves.

    The benchmark thins before matching, and skipping it inflates nothing and
    deflates precision badly: a boundary predicted three pixels thick can match
    an annotator's one-pixel curve only once, so the other two count as false
    positives at every threshold.

    Zhang & Suen, *A fast parallel algorithm for thinning digital patterns*,
    CACM 27(3), 1984 — a published algorithm implemented here rather than taken
    from the unlicensed `bwmorph`, and chosen over a morphological skeleton
    because it preserves connectivity and endpoint position, which is what a
    boundary map is.
    """
    image = (_as_numpy(binary) != 0).astype(np.uint8)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D map, got {image.shape}")

    # Pad so the 8-neighbourhood is defined everywhere; strip it at the end.
    padded = np.pad(image, 1)
    while True:
        removed = False
        for step in (0, 1):
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            # Number of 0->1 transitions around the ring p2..p9,p2.
            ring = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            transitions = sum(
                ((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8)
            )

            if step == 0:
                first, second = p2 * p4 * p6, p4 * p6 * p8
            else:
                first, second = p2 * p4 * p8, p2 * p6 * p8

            condemned = (
                (padded[1:-1, 1:-1] == 1)
                & (neighbours >= 2)
                & (neighbours <= 6)
                & (transitions == 1)
                & (first == 0)
                & (second == 0)
            )
            if condemned.any():
                padded[1:-1, 1:-1][condemned] = 0
                removed = True
        if not removed:
            break
    return padded[1:-1, 1:-1].astype(bool)


# -- correspondence -----------------------------------------------------------


def correspond_pixels(
    prediction: np.ndarray, truth: np.ndarray, max_dist_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost correspondence between two binary boundary maps.

    Returns ``(matched_prediction, matched_truth)``, boolean maps of which
    pixels found a partner. Two pixels may correspond only within
    ``max_dist_px``; the cost of a pair is their Euclidean distance, and no
    pixel is used twice.

    **Cardinality first, distance second**, which is what the protocol asks for:
    leaving a pixel unmatched is penalised more than any admissible set of pairs
    can cost, so the result is a maximum-cardinality matching and total distance
    only breaks ties between them.

    **The tie-break is not cosmetic, which is why this is exact.** Recall
    depends only on cardinality and every maximum-cardinality matching gives the
    same one — but precision counts predictions matched by *any* annotator, and
    different maximum-cardinality matchings put different predictions in that
    union. Measured on four images at two thresholds, taking an arbitrary
    maximum-cardinality matching (Hopcroft-Karp) instead moved precision by up
    to **0.013**, in both directions. A benchmark quoted to three decimals
    cannot absorb that, so the cheap substitute is refused here.

    Greedy matching is refused for the same reason and a worse one: it is not
    even maximum-cardinality, since a prediction taking its nearest free partner
    can strand a neighbour that had only that one.

    Implementation: the admissible pairs form a sparse bipartite graph, padded
    with one prohibitively-priced private partner per prediction so that a full
    matching always exists and scipy's sparse solver applies. Without the
    padding the solver refuses every image where some prediction cannot be
    matched at all, which is most of them.
    """
    from scipy.sparse import csr_matrix, hstack
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    from scipy.spatial import cKDTree

    prediction = _as_numpy(prediction) != 0
    truth = _as_numpy(truth) != 0
    if prediction.shape != truth.shape:
        raise ValueError(f"Maps must match: {prediction.shape} vs {truth.shape}")
    if max_dist_px < 0:
        raise ValueError(f"max_dist_px must be >= 0, got {max_dist_px}")

    matched_pred = np.zeros(prediction.shape, dtype=bool)
    matched_truth = np.zeros(truth.shape, dtype=bool)

    pred_yx = np.argwhere(prediction)
    truth_yx = np.argwhere(truth)
    if len(pred_yx) == 0 or len(truth_yx) == 0:
        return matched_pred, matched_truth

    # query_ball_tree rather than sparse_distance_matrix: a distance of exactly
    # 0 -- a prediction sitting on an annotator's own pixel, the most common
    # case there is -- is a stored zero that a sparse matrix drops, which would
    # silently discard the best matches available.
    neighbours = cKDTree(pred_yx).query_ball_tree(cKDTree(truth_yx), max_dist_px)
    rows = np.fromiter(
        (i for i, partners in enumerate(neighbours) for _ in partners), dtype=np.int64
    )
    if len(rows) == 0:
        return matched_pred, matched_truth
    cols = np.fromiter((j for partners in neighbours for j in partners), dtype=np.int64)
    distances = np.linalg.norm(pred_yx[rows] - truth_yx[cols], axis=1)

    # Drop pixels with no candidate at all: they cannot be matched, and keeping
    # them only widens the matrix.
    active_pred, pred_slot = np.unique(rows, return_inverse=True)
    active_truth, truth_slot = np.unique(cols, return_inverse=True)

    # **Pad the smaller side.** The matching is symmetric, so either side may be
    # the rows -- but the solver must find a full matching over every row, and
    # one dummy column is added per row. Putting the larger side in the rows on
    # a real image means 14832 rows where the other orientation needs 3904, and
    # it measured 6.5x slower for the identical answer. Boundary predictions
    # outnumber annotator pixels at almost every threshold, so this is the
    # common case rather than a corner.
    n_pred, n_truth = len(active_pred), len(active_truth)
    flip = n_truth > n_pred
    if flip:
        row_slot, col_slot, n_rows, n_cols = pred_slot, truth_slot, n_pred, n_truth
    else:
        row_slot, col_slot, n_rows, n_cols = truth_slot, pred_slot, n_truth, n_pred

    # +1 keeps every real cost strictly positive, so a zero-distance pair stays
    # an explicitly stored entry. A constant added to every edge cannot change
    # which full matching is cheapest, because all of them use the same number
    # of edges.
    real = csr_matrix((distances + 1.0, (row_slot, col_slot)), shape=(n_rows, n_cols))

    # One private, prohibitively expensive partner per row. Its price exceeds
    # the total any full set of real edges could cost, so the solver spends
    # every real edge it can before taking one -- maximum cardinality -- and
    # only then minimises distance.
    outlier = (n_rows + 1.0) * (max_dist_px + 1.0) + 1.0
    padded = hstack(
        [
            real,
            csr_matrix(
                (np.full(n_rows, outlier), (np.arange(n_rows), np.arange(n_rows))),
                shape=(n_rows,) * 2,
            ),
        ],
        format="csr",
    )

    left, right = min_weight_full_bipartite_matching(padded)
    keep = right < n_cols
    matched_rows, matched_cols = left[keep], right[keep]
    if not flip:
        matched_rows, matched_cols = matched_cols, matched_rows
    matched_pred[tuple(pred_yx[active_pred[matched_rows]].T)] = True
    matched_truth[tuple(truth_yx[active_truth[matched_cols]].T)] = True
    return matched_pred, matched_truth


# -- accumulation and aggregation ---------------------------------------------


def _levels(thresholds) -> np.ndarray:
    """The threshold sweep, as an array. An int means that many even levels."""
    if isinstance(thresholds, int):
        if thresholds < 1:
            raise ValueError(f"thresholds must be >= 1, got {thresholds}")
        # Open interval: a threshold of exactly 0 keeps every pixel including
        # the zeros, and one of exactly 1 usually keeps nothing.
        return (np.arange(thresholds) + 1.0) / (thresholds + 1.0)
    levels = np.asarray(thresholds, dtype=float).ravel()
    if levels.size == 0:
        raise ValueError("thresholds must contain at least one level")
    return levels


def image_counts(
    prediction: np.ndarray,
    annotations: np.ndarray,
    thresholds=DEFAULT_THRESHOLDS,
    max_dist: float = DEFAULT_MAX_DIST,
    thin: bool = True,
) -> np.ndarray:
    """Match counts for one image at every threshold: ``(T, 4)``.

    The columns are ``(matched_predictions, predictions, matched_truth,
    truth)`` — the four running totals precision and recall are built from, kept
    separate because ODS sums them over the dataset *before* dividing. Summing
    per-image ratios instead would weight a sparse image the same as a dense
    one, which is not what the benchmark defines.

    **Precision unions over annotators, recall sums over them**, and the
    asymmetry is the protocol rather than an oversight. A predicted pixel is
    correct if *anyone* drew a boundary there, so it is counted once however
    many annotators it matched; but each annotator's own boundary is something
    the prediction should have found, so every annotator contributes their full
    pixel count to the recall denominator. With five annotators the recall
    denominator is therefore roughly five times the size of one map.

    Parameters
    ----------
    prediction:
        Soft boundary map ``(H, W)``, larger meaning more boundary. Not required
        to be in ``[0, 1]``; the sweep is over the values as given.
    annotations:
        ``(A, H, W)`` binary maps, one per annotator — what
        :meth:`~visbench.data.bsds.BSDS500Dataset.annotations` returns.
    max_dist:
        Tolerance as a fraction of the image diagonal.
    thin:
        Thin each binarised map before matching. The benchmark does; turning it
        off is for testing a prediction that is already single-pixel-wide.
    """
    prediction = _as_numpy(prediction).astype(float)
    annotations = _as_numpy(annotations)
    if annotations.ndim == 2:
        annotations = annotations[None]
    if annotations.ndim != 3:
        raise ValueError(f"Expected annotations (A, H, W), got {annotations.shape}")
    if prediction.shape != annotations.shape[1:]:
        raise ValueError(
            f"Prediction {prediction.shape} does not match annotations {annotations.shape[1:]}"
        )

    annotations = annotations != 0
    height, width = prediction.shape
    tolerance = max_dist * float(np.hypot(height, width))
    truth_total = int(annotations.sum())

    counts = np.zeros((len(_levels(thresholds)), 4), dtype=np.int64)
    for index, level in enumerate(_levels(thresholds)):
        binary = prediction >= level
        if thin:
            binary = thin_boundaries(binary)

        union = np.zeros_like(binary)
        matched_truth = 0
        for annotator in annotations:
            hit_pred, hit_truth = correspond_pixels(binary, annotator, tolerance)
            union |= hit_pred
            matched_truth += int(hit_truth.sum())
        counts[index] = (int(union.sum()), int(binary.sum()), matched_truth, truth_total)
    return counts


def _f_measure(matched_p, total_p, matched_r, total_r):
    """F = 2PR/(P+R), with the zero cases defined rather than left to divide."""
    precision = np.divide(
        matched_p, total_p, out=np.zeros_like(matched_p, float), where=total_p > 0
    )
    recall = np.divide(matched_r, total_r, out=np.zeros_like(matched_r, float), where=total_r > 0)
    denominator = precision + recall
    f = np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(precision), where=denominator > 0
    )
    return precision, recall, f


def boundary_metrics(
    counts, thresholds=DEFAULT_THRESHOLDS, max_dist: float = DEFAULT_MAX_DIST
) -> dict:
    """ODS, OIS and AP from per-image :func:`image_counts`.

    ``counts`` is ``(N, T, 4)`` — one image per row, the same threshold sweep
    for each. Returns ``ods``, ``ois``, ``ap``, and the threshold ODS chose.

    **ODS** ("optimal dataset scale") sums the four counts over the dataset at
    each threshold, forms one precision-recall pair per threshold, and takes the
    best F. One threshold serves every image, which is what a detector shipped
    with a fixed operating point would have to do.

    **OIS** ("optimal image scale") lets each image pick the threshold that
    maximises its own F, then sums *those* counts over the dataset and takes F
    once. Note it is not the mean of per-image bests: the benchmark aggregates
    counts and divides once, so a sparse image cannot outweigh a dense one.

    **AP** is the area under the dataset-level precision-recall curve, taken by
    the trapezoid rule over recall. It is a summary of the whole sweep rather
    than of one operating point, so a detector that is excellent at one
    threshold and useless elsewhere scores well on ODS and badly here.
    """
    counts = np.asarray(counts)
    if counts.ndim == 2:
        counts = counts[None]
    if counts.ndim != 3 or counts.shape[-1] != 4:
        raise ValueError(f"Expected counts (N, T, 4), got {counts.shape}")
    if len(counts) == 0:
        raise ValueError("Cannot summarise an empty set of images")
    levels = _levels(thresholds)
    if counts.shape[1] != len(levels):
        raise ValueError(
            f"counts has {counts.shape[1]} thresholds but {len(levels)} were given. "
            "The sweep that produced the counts is the one that must summarise them."
        )

    dataset = counts.sum(axis=0).astype(float)
    precision, recall, f = _f_measure(dataset[:, 0], dataset[:, 1], dataset[:, 2], dataset[:, 3])
    best = int(np.argmax(f))

    per_image = counts.astype(float)
    _, _, image_f = _f_measure(
        per_image[:, :, 0], per_image[:, :, 1], per_image[:, :, 2], per_image[:, :, 3]
    )
    chosen = np.argmax(image_f, axis=1)
    picked = counts[np.arange(len(counts)), chosen].sum(axis=0).astype(float)
    _, _, ois = _f_measure(*picked)

    # Sorted by recall so the trapezoid rule integrates a curve rather than a
    # zigzag: the sweep runs from high threshold to low, and recall is not
    # guaranteed monotone in it once thinning is in the loop.
    order = np.argsort(recall)
    ap = float(np.trapezoid(precision[order], recall[order]))

    return {
        "ods": float(f[best]),
        "ois": float(ois),
        "ap": ap,
        "ods_threshold": float(levels[best]),
        "ods_precision": float(precision[best]),
        "ods_recall": float(recall[best]),
        "max_dist": float(max_dist),
    }
