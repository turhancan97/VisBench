"""The external check: does this implementation reproduce a published number?

Everything in `test_boundary_metrics.py` is self-consistent — hand-built cases
and a brute-force reference. Self-consistency cannot catch a protocol
misunderstanding: a matcher that is exact, thinning that is correct, and an
aggregation that is wrong about *what BSDS measures* would pass every one of
those and still report a number nobody else's code would produce.

The one measurement available that a whole literature agrees on is **human
agreement**: score one annotator's boundary map against the others'. BSDS500
publishes this as **ODS 0.80** on the test split, and it is the ceiling every
detector is quoted against.

This implementation gives **0.8030** on the test split and **0.7870** on val,
leave-one-out over all annotators. That is the protocol reproduced, not merely
implemented — see the module docstring of `visbench/metrics/boundary.py` for
why it had to be written from the paper rather than adapted from `bench/`.

Marked `slow` because it reads the real dataset, and skipped outright when that
has not been fetched. `scripts/fetch_bsds500.py` puts it in place.
"""

from pathlib import Path

import numpy as np
import pytest

from visbench.data import BSDS500Dataset
from visbench.metrics.boundary import boundary_metrics, image_counts

REAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "bsds500"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (REAL_ROOT / "groundTruth" / "val").is_dir(),
        reason="run scripts/fetch_bsds500.py to populate data/bsds500",
    ),
]


def leave_one_out(dataset):
    """Every annotator in turn scored against the rest — the published form.

    Scoring only annotator 0 is the tempting shortcut and it is biased: the
    annotators differ by a median factor of 1.92 in how much they mark, so
    whichever one is fixed decides the precision/recall split. Averaging over
    all of them is what makes the number comparable.
    """
    rows = []
    for index in range(len(dataset)):
        annotations = dataset.annotations(index).numpy()
        for held in range(annotations.shape[0]):
            rows.append(
                image_counts(
                    annotations[held].astype(float),
                    np.delete(annotations, held, axis=0),
                    thresholds=[0.5],
                    thin=False,  # an annotator's map is already single-pixel
                )
            )
    return np.stack(rows)


def test_reproduces_the_published_human_agreement():
    """BSDS500's published human ODS is 0.80. Thirty val images land at 0.7932.

    The band is deliberately wider than the measurement's own reproducibility —
    this asserts "the protocol was understood", not a digit. A greedy matcher, a
    missing thinning step, or precision and recall aggregated the wrong way
    round all leave this band comfortably.
    """
    dataset = BSDS500Dataset(REAL_ROOT, split="val", max_images=30)
    result = boundary_metrics(leave_one_out(dataset), thresholds=[0.5])

    assert 0.75 < result["ods"] < 0.83, result
    # Annotators agree on where boundaries are more readily than on how many to
    # draw, so precision runs well above recall. A run where they crossed would
    # mean the two were swapped somewhere.
    assert result["ods_precision"] > result["ods_recall"]


def test_a_prediction_scored_against_itself_is_perfect():
    """The upper anchor, on real annotations rather than a fixture.

    Cheap, and it catches the class of bug where the tolerance or the matching
    silently drops pixels: an annotator scored against a copy of itself has
    nowhere to lose one.
    """
    dataset = BSDS500Dataset(REAL_ROOT, split="val", max_images=5)
    rows = []
    for index in range(len(dataset)):
        first = dataset.annotations(index).numpy()[:1]
        rows.append(image_counts(first[0].astype(float), first, thresholds=[0.5], thin=False))

    result = boundary_metrics(np.stack(rows), thresholds=[0.5])
    assert result["ods"] == pytest.approx(1.0)
