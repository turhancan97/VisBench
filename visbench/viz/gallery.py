"""The three probes whose answer is a choice among images, not a map over one.

``classification``, ``retrieval`` and ``similarity`` have no spatial target, so
none of them fits the ``image | target | prediction`` grid. What they have
instead is a *decision*: which class, which neighbours, which of two candidates.
This module draws the decision.

Each of the three has a silent failure in this codebase's own history that the
corresponding picture catches instantly:

- **A labelled folder is grouped by class**, so ``subset(n)`` takes a prefix that
  is entirely class 0 and a run then scores 1.0 while measuring nothing.
  ``balanced_subset`` exists because of this. A sheet of frames with their labels
  shows it at a glance; a top-1 of 1.0 does not.
- **The NIGHTS CSV is read by column name** because the reference reads the vote
  from column 2 and the paths from 4/5/6. Reordering the file scores against the
  wrong column and *looks like a mediocre number rather than an error*. Draw the
  triplet with the human vote marked and a transposed vote is obvious: the
  "preferred" candidate is visibly the more distorted one.

Both readings are also printed as a footer figure, following
:func:`~visbench.viz.matches.error_coherence`: the diagnosis should not depend on
the reader's eye. Neither figure is a score, and neither is recorded.
"""

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image, ImageDraw

from visbench.viz.panels import CAPTION_INK, PAGE_INK, font_for_captions, render_panels

__all__ = [
    "RIGHT",
    "WRONG",
    "annotate",
    "class_balance",
    "render_retrieval_panels",
    "render_sheet",
    "render_triplet_panels",
    "vote_balance",
]

#: A decision that agreed with the ground truth, and one that did not. The same
#: meaning ``matches.py`` gives its two colours, which is why they are shared
#: rather than redefined: one legend across the package, or a reader has to
#: remember which page they are on.
RIGHT = (60, 220, 90)
WRONG = (240, 70, 70)
#: Drawn on a frame that is neither — the query in a retrieval row, or any frame
#: when no backbone was given and there is no decision to judge.
NEUTRAL = (130, 130, 130)

_CAPTION_HEIGHT = 13
_BORDER = 2


def annotate(
    image: Image.Image,
    caption: str = "",
    border: tuple[int, int, int] | None = None,
) -> Image.Image:
    """A copy of ``image`` with a caption bar beneath it and an optional border.

    **The caption goes below the frame, never over it.** Everything else in
    ``visbench.viz`` exists to show pixels as the dataset yielded them, and text
    drawn across those pixels would obscure the very thing being inspected —
    for a small thumbnail, a class name covers a good fraction of the subject.
    The canvas grows instead; the frame is pasted unchanged and a test pins it
    byte-for-byte.

    The border is drawn *inside* the frame's own area, since an outline around
    it would need the same room again on every side and thumbnails are small.
    Two pixels, which is enough to read at a glance and little enough to leave
    the subject legible.
    """
    canvas = Image.new("RGB", (image.width, image.height + _CAPTION_HEIGHT), PAGE_INK)
    canvas.paste(image.convert("RGB"), (0, 0))

    draw = ImageDraw.Draw(canvas)
    if border is not None:
        draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=border, width=_BORDER)
    if caption:
        font = font_for_captions()
        # Shortened to fit rather than clipped at the edge: a hard cut leaves
        # "triangl" looking like a value rather than a truncation, and on a
        # contact sheet of class names that is exactly the sort of thing a
        # reader would take at face value.
        #
        # ASCII only, here and in every caption this module writes. PIL's
        # built-in bitmap font has no glyph for an em dash or an ellipsis and
        # draws an empty box instead -- which looks like a corrupted label, and
        # was found by rendering a page rather than by any test.
        while caption and draw.textlength(caption, font=font) > image.width - 4:
            caption = caption[:-1]
            if len(caption) > 1:
                caption = caption[:-1] + "~"
        draw.text((2, image.height + 1), caption, fill=CAPTION_INK, font=font)
    return canvas


def class_balance(labels: Sequence[int], classes: Sequence[str] | None = None) -> str:
    """One line describing how a labelled split is distributed.

    The prefix bug stated as a figure. A split collapsed to a single class reads
    ``1 class`` here however flattering its top-1 is, and whichever frames the
    viewer happened to draw — so the diagnosis does not depend on the sample.

    A **diagnostic, never a score**: it describes the data, not a backbone, and
    nothing records it.
    """
    if not len(labels):
        return "empty split"

    counts: dict[int, int] = {}
    for label in labels:
        counts[int(label)] = counts.get(int(label), 0) + 1

    total = sum(counts.values())
    low, high = min(counts.values()), max(counts.values())
    plural = "class" if len(counts) == 1 else "classes"
    line = f"{len(counts)} {plural}, {total} items, {low}-{high} per class"
    if len(counts) == 1 and classes is not None:
        only = classes[next(iter(counts))]
        line += f" - every item is {only!r}, so any score here is an artefact"
    return line


def vote_balance(triplets: torch.Tensor) -> str:
    """One line describing which way the human votes fell.

    The NIGHTS vote is a binary preference over two candidates presented in an
    arbitrary order, so it should sit near 50%. A figure far from that means the
    vote column was read from the wrong CSV field — the failure that otherwise
    surfaces only as a mediocre accuracy.
    """
    if not len(triplets):
        return "no triplets"
    right = float((triplets[:, 3] == 1).float().mean())
    return (
        f"{len(triplets)} triplets, humans chose right in {right:.0%} "
        "(far from 50% means the vote column is wrong)"
    )


def render_sheet(
    dataset: Any,
    indices: Sequence[int],
    predictions: Sequence[int] | None = None,
    columns: int = 6,
) -> Image.Image:
    """A contact sheet of frames with their labels — ``classification``.

    Packed several to a row rather than one per row because the failures worth
    catching here are *class-level patterns*: a split collapsed to one class, or
    a whole category being confused for another. Neither is visible in four
    frames stacked vertically.
    """
    names = getattr(dataset, "classes", None)
    truth = dataset.labels()

    def label_of(value: Any) -> str:
        if value is None:
            return "?"
        return names[int(value)] if names is not None else str(int(value))

    tiles: list[Image.Image] = []
    for position, index in enumerate(indices):
        image, _ = dataset[index]
        actual = label_of(truth[index])
        if predictions is None:
            tiles.append(annotate(image, actual, NEUTRAL))
            continue
        guess = label_of(predictions[position])
        correct = guess == actual
        caption = actual if correct else f"{guess} != {actual}"
        tiles.append(annotate(image, caption, RIGHT if correct else WRONG))

    rows: list[tuple[str, list]] = []
    for start in range(0, len(tiles), columns):
        block = tiles[start : start + columns]
        rows.append((f"{start}-{start + len(block) - 1}", list(block)))

    footer = class_balance(truth, names)
    if predictions is not None:
        hits = sum(
            1
            for position, index in enumerate(indices)
            if label_of(predictions[position]) == label_of(truth[index])
        )
        footer += f" | {hits}/{len(indices)} of the drawn frames correct"
    return render_panels(rows, [""] * min(columns, max(1, len(tiles))), footer=footer)


def render_retrieval_panels(
    dataset: Any,
    ranking: torch.Tensor,
    queries: Sequence[int],
    topk: int = 5,
) -> Image.Image:
    """One row per query: the query, then its nearest neighbours — ``retrieval``.

    ``ranking`` is what :meth:`RetrievalTask.predict` returned for the *whole*
    split, so row ``i`` of it ranks every other image against image ``i``. The
    neighbours must be drawn from the full gallery: leave-one-out retrieval over
    four images ranks each against three alternatives and shows nothing.
    """
    names = getattr(dataset, "classes", None)
    labels = dataset.labels()

    def label_of(index: int) -> str:
        value = labels[index]
        if value is None:
            return "?"
        return names[int(value)] if names is not None else str(int(value))

    rows: list[tuple[str, list]] = []
    for query in queries:
        image, _ = dataset[query]
        tiles = [annotate(image, f"query: {label_of(query)}", NEUTRAL)]

        neighbours = [int(value) for value in ranking[query][:topk]]
        hits = 0
        for rank, neighbour in enumerate(neighbours, start=1):
            same = labels[neighbour] == labels[query]
            hits += int(bool(same))
            tiles.append(
                annotate(
                    dataset[neighbour][0],
                    f"{rank}. {label_of(neighbour)}",
                    RIGHT if same else WRONG,
                )
            )
        rows.append((f"item {query}\n{hits}/{len(neighbours)} same class", tiles))

    columns = ["query"] + [f"nn {rank}" for rank in range(1, topk + 1)]
    return render_panels(rows, columns, footer=class_balance(labels, names))


def render_triplet_panels(
    dataset: Any,
    triplet_indices: Sequence[int],
    predictions: Sequence[int] | None = None,
) -> Image.Image:
    """One row per triplet: reference, left, right — ``similarity``.

    The human vote is marked on the candidate it chose. With ``predictions`` the
    model's choice is marked too, so agreement and disagreement are visible
    directly rather than pooled into an accuracy.
    """
    triplets = dataset.labels()
    rows: list[tuple[str, list]] = []

    for position, index in enumerate(triplet_indices):
        reference, left, right, vote = (int(value) for value in triplets[index])
        chosen = "right" if vote == 1 else "left"

        tiles = [annotate(dataset[reference][0], "reference", NEUTRAL)]
        for side, image_index in (("left", left), ("right", right)):
            caption = side
            border = NEUTRAL
            if side == chosen:
                caption = f"{side} - humans"
                border = RIGHT
            if predictions is not None:
                model = "right" if int(predictions[position]) == 1 else "left"
                if side == model:
                    caption += " + model" if side == chosen else " - model"
                    # Agreement keeps the human colour; a split decision is the
                    # thing worth seeing, so it is drawn as a disagreement.
                    border = RIGHT if side == chosen else WRONG
            tiles.append(annotate(dataset[image_index][0], caption, border))

        label = f"triplet {index}\nhumans: {chosen}"
        if predictions is not None:
            model = "right" if int(predictions[position]) == 1 else "left"
            label += f"\nmodel: {model}" + ("" if model == chosen else "  (differs)")
        rows.append((label, tiles))

    footer = vote_balance(triplets)
    if predictions is not None:
        agree = sum(
            1
            for position, index in enumerate(triplet_indices)
            if int(predictions[position]) == int(triplets[index][3])
        )
        footer += f" | model agreed on {agree}/{len(triplet_indices)} drawn"
    return render_panels(rows, ["reference", "left", "right"], footer=footer)
