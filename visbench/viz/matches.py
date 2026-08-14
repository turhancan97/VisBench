"""Drawing correspondence: two views side by side, with the matches between.

This is the panel the rest of ``visbench.viz`` could not express. Every other
probe has one image and one target of the same shape, so a row is a strip of
aligned panels. Correspondence has *two* images and a geometric relation
between them, and what a reader needs to see is not either frame but the
**error vectors** — where a matched point landed against where the geometry
says it should have.

**It is the shape of the errors that diagnoses the failure, not their size.**
The bug that motivates this package — correspondence scoring
``recall@1px = 0.003`` because the homography was expressed in original pixels
while the features came from a 224 centre crop — does not look like noise. It
looks like every match being wrong *in the same direction*, which a person
recognises instantly and no scalar recall can report. Scattered short errors
are a weak backbone; a coherent field of long ones is a broken pipeline. Those
are the two readings that must not be confused, and a number cannot separate
them.

Matches are drawn from :meth:`CorrespondenceTask.match_details`, the same call
:meth:`~visbench.tasks.mid_level.correspondence.CorrespondenceTask.evaluate`
pools, so a panel cannot vouch for a number it disagrees with.
"""

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image, ImageDraw

from visbench.viz.panels import render_panels

__all__ = ["draw_matches", "error_coherence", "render_match_panels"]

#: A match inside the threshold, and one outside it. Not the target/prediction
#: colours from ``panels.py``: those distinguish *what* is drawn, these
#: distinguish whether it was right, and one legend should not mean two things.
_WITHIN = (60, 220, 90)
_BEYOND = (240, 70, 70)
#: The straight line from where a point should have landed to where it did.
_ERROR = (255, 200, 0)
_SEAM = 12


def error_coherence(details: dict) -> float:
    """How aligned the error vectors are: 1.0 all parallel, ~0 scattered.

    The mean resultant length of the error *directions*, ignoring their length.
    This is the visual diagnosis of :mod:`this module <visbench.viz.matches>`
    turned into a number, so a reader does not have to trust their eye — and so
    the claim can be tested rather than asserted.

    Measured on 224px homography pairs with ResNet-18 features: **0.29 and 0.40**
    for two correctly-scored pairs, against **0.98 and 1.00** for the same pairs
    with the homography deliberately expressed in the wrong pixel frame — the
    shape of the bug that scored ``recall@1px = 0.003``. Median error moved from
    10-23px to 227-294px at the same time, but the median alone cannot separate
    "broken" from "hopeless"; the coherence can.

    High coherence therefore means the geometry, the crop or the units are
    wrong, not that the representation is weak. It is a **diagnostic, never a
    score**: it says nothing about a backbone, and it is not recorded.
    """
    vectors = (details["target"] - details["expected"]).to(torch.float64)
    lengths = vectors.norm(dim=1)
    keep = lengths > 1e-6
    if int(keep.sum()) == 0:
        # Every match landed exactly right, which has no direction to average.
        return 0.0
    directions = vectors[keep] / lengths[keep].unsqueeze(1)
    return float(directions.mean(dim=0).norm())


def _sample(count: int, limit: int) -> torch.Tensor:
    """At most ``limit`` indices, spread evenly across ``count``.

    Evenly rather than the best-scoring few: matches are already sorted by
    descending similarity, so a prefix would draw the most confident ones and
    show a systematically better picture than the score describes. Evenly
    rather than randomly so the same pair draws the same way twice.
    """
    if count <= limit:
        return torch.arange(count)
    return torch.linspace(0, count - 1, limit).round().long()


def draw_matches(
    view_0: Image.Image,
    view_1: Image.Image,
    details: dict,
    *,
    threshold: float = 5.0,
    max_matches: int = 40,
    show_error_vectors: bool = True,
) -> Image.Image:
    """The two views side by side, with sampled matches drawn between them.

    A line runs from each matched point in view 0 to where it landed in view 1,
    green if it landed within ``threshold`` pixels of where the geometry says it
    should have and red otherwise. With ``show_error_vectors`` a short amber
    segment also runs from the expected position to the actual one, which is
    what makes a *systematic* offset legible: a hundred parallel amber stubs is
    a different diagnosis from a hundred scattered ones.

    Neither view is resized. The canvas is built around them, exactly as
    :func:`~visbench.viz.panels.render_panels` builds a page around its panels.
    """
    left = view_0.convert("RGB")
    right = view_1.convert("RGB")
    canvas = Image.new(
        "RGB",
        (left.width + _SEAM + right.width, max(left.height, right.height)),
        (24, 24, 24),
    )
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + _SEAM, 0))

    draw = ImageDraw.Draw(canvas)
    offset = left.width + _SEAM
    errors = details["errors_px"]
    chosen = _sample(len(errors), max_matches)

    for index in chosen.tolist():
        x0, y0 = (float(v) for v in details["source"][index])
        x1, y1 = (float(v) for v in details["target"][index])
        colour = _WITHIN if float(errors[index]) <= threshold else _BEYOND
        draw.line((x0, y0, offset + x1, y1), fill=colour, width=1)
        draw.ellipse((x0 - 1, y0 - 1, x0 + 1, y0 + 1), fill=colour)

        if show_error_vectors:
            ex, ey = (float(v) for v in details["expected"][index])
            draw.line((offset + ex, ey, offset + x1, y1), fill=_ERROR, width=1)
    return canvas


def render_match_panels(
    dataset: Any,
    task: Any,
    features: Sequence,
    indices: Sequence[int],
    *,
    threshold: float = 5.0,
    max_matches: int = 40,
) -> Image.Image:
    """One row per pair: both views, the matches, and how many landed.

    ``features`` is the regrouped ``(features_0, features_1)`` sequence
    :func:`visbench.run` builds for a pair task, ordered to match ``indices``.
    """
    geometries = dataset.labels()
    rows: list[tuple[str, list]] = []

    for position, index in enumerate(indices):
        view_0, view_1, _ = dataset[index]
        details = task.match_details(
            features[position][0], features[position][1], geometries[index]
        )

        errors = details["errors_px"]
        within = int((errors <= threshold).sum()) if len(errors) else 0
        median = float(errors.median()) if len(errors) else float("nan")
        # The count is part of the reading, not decoration: a pair keeping four
        # matches and a pair keeping four hundred are different evidence, and
        # the ratio test makes that vary a lot between backbones.
        label = (
            f"pair {index}\n{within}/{len(errors)} within {threshold:g}px"
            f"\nmedian {median:.1f}px"
            f"\ncoherence {error_coherence(details):.2f}"
        )
        rows.append(
            (
                label,
                [
                    draw_matches(
                        view_0,
                        view_1,
                        details,
                        threshold=threshold,
                        max_matches=max_matches,
                    )
                ],
            )
        )

    legend = (
        f"green = within {threshold:g}px, red = beyond; amber = expected -> actual. "
        "Coherence near 1 means every error points the same way -- a broken geometry, "
        "crop or unit, not a weak backbone."
    )
    return render_panels(rows, ["view 0  |  view 1"], footer=legend)
