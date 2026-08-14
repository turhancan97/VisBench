"""Laying panels out on a page, and drawing boxes onto a frame.

The layout is deliberately dull. What matters here is one rule, and every
choice below follows from it:

**The viewer displays what the dataset yielded, and applies no geometry of its
own.** A panel is pasted at the dataset's own resolution — never resized to fit,
never re-read from the source file, never re-cropped. A viewer that resized for
layout could make a misaligned pipeline look fine and a correct one look
broken, which would make it worse than no viewer at all: the whole reason to
draw a target beside its image is that the pair is either aligned or it is not,
and a second geometry destroys exactly that evidence.

That rule is cheap to keep because ``DenseFolderDataset`` and its subclasses
already hand over a PIL image at the working resolution rather than a
normalised tensor, so there is nothing to invert and nothing to resample.
``tests/viz/test_panels.py`` pins the image panel byte-for-byte against
``np.asarray(dataset[i][0])``.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from visbench.viz.colour import display_range, target_to_rgb
from visbench.viz.styles import TargetStyle, style_for

__all__ = [
    "CAPTION_INK",
    "PAGE_INK",
    "draw_boxes",
    "font_for_captions",
    "render_panels",
    "render_probe_panels",
]

#: Ground truth, predictions, and the annotations VOC flags as difficult. Kept
#: apart by hue rather than by line style: a dashed 2px rectangle is unreadable
#: at the scale a 224px crop is actually looked at.
_BOX_TRUTH = (0, 255, 0)
_BOX_DIFFICULT = (140, 140, 0)
_BOX_PREDICTION = (255, 140, 0)

#: Kinds drawn against a stated numeric range. The others -- normals, labels,
#: binary masks -- carry their meaning in the colour itself, so a range would be
#: meaningless for them and, for the channelled ones, ill-shaped.
_SCALAR_KINDS = frozenset({"magnitude", "depth"})

_GAP = 8
_GUTTER = 200
_HEADER = 18

#: Page background and text colour. Public because ``gallery.py`` builds its own
#: tiles and a caption bar in a different shade would read as a different kind of
#: thing rather than as the same page.
PAGE_INK = (24, 24, 24)
CAPTION_INK = (235, 235, 235)
_PAGE = PAGE_INK
_INK = CAPTION_INK


def font_for_captions() -> Any:
    """PIL's built-in bitmap font.

    Deliberately not a TrueType lookup: a font path that exists on the machine
    the panel was drawn on and not on the next one turns a working viewer into
    an ``OSError``, and the labels here are short enough that legibility is not
    the binding constraint. This also keeps the package free of a font asset.
    """
    return ImageFont.load_default()


def draw_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    *,
    labels: Sequence[Any] | None = None,
    difficult: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
    colour: tuple[int, int, int] = _BOX_TRUTH,
) -> Image.Image:
    """Draw ``boxes`` onto a copy of ``image``, in its own pixel coordinates.

    ``boxes`` are ``xyxy`` in **post-transform** pixels, which is what
    :class:`~visbench.data.detection.DetectionFolderDataset` returns — the same
    frame as the image, so nothing is rescaled here. That is the point of the
    panel: a box drawn straight onto the crop either lands on the object or it
    does not, and the rescale-by-achieved-ratio arithmetic 6c-1 spent its care
    on is visible in one glance.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = font_for_captions()

    for index in range(len(boxes)):
        x1, y1, x2, y2 = (float(v) for v in boxes[index])
        is_difficult = difficult is not None and bool(difficult[index])
        outline = _BOX_DIFFICULT if is_difficult else colour
        draw.rectangle((x1, y1, x2, y2), outline=outline, width=2)

        caption = ""
        if labels is not None:
            caption = str(labels[index])
        if scores is not None:
            caption = f"{caption} {float(scores[index]):.2f}".strip()
        if is_difficult:
            caption = f"{caption} (difficult)".strip()
        if caption:
            draw.text((x1 + 2, max(0.0, y1 - 10)), caption, fill=outline, font=font)
    return canvas


def render_panels(
    rows: Sequence[tuple[str, Sequence[np.ndarray | Image.Image]]],
    columns: Sequence[str],
    footer: str = "",
) -> Image.Image:
    """Assemble labelled panels into one page.

    ``rows`` is ``(row_label, panels)``. Every panel keeps its own size — the
    grid is laid out around them rather than them being fitted to the grid.
    """
    if not rows:
        raise ValueError("Nothing to draw: no frames were selected.")

    images = [[_as_image(panel) for panel in panels] for _, panels in rows]
    # Rows may be ragged: a contact sheet's last row holds whatever is left over,
    # and padding it with blanks would put empty frames on the page as though
    # they were data. So a column's width is the widest panel among the rows that
    # *have* one, and a short row simply stops.
    widths = [
        max((row[index].width for row in images if index < len(row)), default=0)
        for index in range(len(columns))
    ]
    heights = [max((panel.height for panel in row), default=0) for row in images]

    page_width = _GUTTER + sum(widths) + _GAP * len(widths)
    font = font_for_captions()
    # Wrapped rather than left to run off the edge. The footer is the *legend* --
    # which convention marks an invalid pixel, what the colours mean -- so
    # truncating it loses the one line that says how to read the page. Wrapped
    # rather than widening the page, because the width belongs to the panels:
    # one long sentence should not stretch a figure past its content.
    lines = _wrap(footer, font, page_width - 12) if footer else []
    page_height = (
        _HEADER + sum(heights) + _GAP * len(heights) + _HEADER + 11 * max(0, len(lines) - 1)
    )

    page = Image.new("RGB", (page_width, page_height), _PAGE)
    draw = ImageDraw.Draw(page)

    offset = _GUTTER
    for title, width in zip(columns, widths, strict=True):
        draw.text((offset, 4), title, fill=_INK, font=font)
        offset += width + _GAP

    top = _HEADER
    for (label, _), row, height in zip(rows, images, heights, strict=True):
        for line_number, line in enumerate(label.split("\n")):
            draw.text((6, top + 2 + line_number * 11), line, fill=_INK, font=font)
        offset = _GUTTER
        # strict=False, deliberately: `row` is allowed to be shorter than
        # `widths`, which is what a ragged final row means. Everywhere else in
        # this codebase a zip over two index-paired sequences is strict; here
        # the raggedness is the point, as it is for `resolved`/`resolved[1:]`
        # in backbones/base.py.
        for panel, width in zip(row, widths, strict=False):
            page.paste(panel, (offset, top))
            offset += width + _GAP
        top += height + _GAP

    for number, line in enumerate(lines):
        draw.text((6, top + 2 + number * 11), line, fill=_INK, font=font)
    return page


def _wrap(text: str, font: Any, width: int) -> list[str]:
    """``text`` split on spaces into lines that fit ``width`` pixels."""
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and measure.textlength(candidate, font=font) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _as_image(panel: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(panel, Image.Image):
        return panel.convert("RGB")
    return Image.fromarray(np.ascontiguousarray(panel))


def _as_target_form(prediction: torch.Tensor, style: TargetStyle) -> torch.Tensor:
    """One probe's ``predict()`` output reduced to the shape its target has.

    Each branch mirrors what that task's ``_activate`` emits, so the prediction
    panel is drawn by exactly the rule the metric scores by. Getting this wrong
    is not subtle in the output — a logit map drawn as a class index is visibly
    nonsense — which is why it is a small table rather than a guarded one.
    """
    if style.kind == "labels":
        # Logits, passed through unchanged by SemanticSegmentationTask so that
        # cross_entropy sees them. The class is the argmax.
        return prediction.argmax(dim=0)
    if style.kind == "normals":
        # Channel 3, when present, is the uncertainty-aware loss's kappa.
        return prediction[:3]
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        return prediction[0]
    return prediction


def render_probe_panels(
    dataset: Any,
    probe: str,
    indices: Sequence[int],
    predictions: Any | None = None,
    class_names: Sequence[str] | None = None,
) -> Image.Image:
    """The page ``visbench show`` writes: one row per frame.

    ``predictions`` is whatever the probe's ``predict()`` returned for exactly
    these ``indices`` — a stacked ``(N, C, H, W)`` tensor for a dense probe, a
    list of per-image dicts for detection — or ``None`` for the target-only
    view, which needs no backbone at all.
    """
    style = style_for(probe)
    columns = ["image", "target"] + (["prediction"] if predictions is not None else [])
    rows: list[tuple[str, list[np.ndarray | Image.Image]]] = []

    for position, index in enumerate(indices):
        image, target = dataset[index]
        prediction = None if predictions is None else predictions[position]
        label, panels = _row(style, dataset, index, image, target, prediction, class_names)
        rows.append((label, panels))

    legend = f"magenta = no ground truth; {style.note}" if style.invalid else style.note
    return render_panels(rows, columns, footer=legend)


def _row(
    style: TargetStyle,
    dataset: Any,
    index: int,
    image: Image.Image,
    target: Any,
    prediction: Any,
    class_names: Sequence[str] | None,
) -> tuple[str, list[np.ndarray | Image.Image]]:
    """One frame: its label, and the two or three panels beside it."""
    stem = str(getattr(dataset, "stems", [])[index]) if hasattr(dataset, "stems") else str(index)
    stem = stem.rsplit("/", 1)[-1]

    if style.kind == "boxes":
        return _box_row(stem, image, target, prediction, class_names)

    # Only the scalar kinds have a range, and only they are asked for one. A
    # normal map's validity mask is (H, W) while the target is (3, H, W), so
    # computing a span for it is not merely wasted -- it is a shape error, which
    # is how this was found: the first three-channel figure ever rendered
    # through a full page raised instead of drawing.
    span = None
    if style.kind in _SCALAR_KINDS:
        valid = None if style.invalid is None else ~style.invalid(target)
        span = display_range(target, valid)

    panels: list[np.ndarray | Image.Image] = [
        image,
        target_to_rgb(target, style, span),
    ]
    if prediction is not None:
        # The target's range, not the prediction's own. A head whose output is
        # uniformly half the target draws identically to a correct one if each
        # panel is scaled to its own extremes.
        panels.append(target_to_rgb(_as_target_form(prediction, style), style, span))

    label = stem if span is None else f"{stem}\n{span.caption(style.unit)}"
    return label, panels


def _box_row(
    stem: str,
    image: Image.Image,
    target: dict,
    prediction: Any,
    class_names: Sequence[str] | None,
) -> tuple[str, list[np.ndarray | Image.Image]]:
    def named(labels: torch.Tensor) -> list[str]:
        if class_names is None:
            return [str(int(value)) for value in labels]
        return [class_names[int(value)] for value in labels]

    panels: list[np.ndarray | Image.Image] = [
        image,
        draw_boxes(
            image,
            target["boxes"],
            labels=named(target["labels"]),
            difficult=target.get("difficult"),
        ),
    ]
    if prediction is not None:
        panels.append(
            draw_boxes(
                image,
                prediction["boxes"],
                labels=named(prediction["labels"]),
                scores=prediction.get("scores"),
                colour=_BOX_PREDICTION,
            )
        )
    kept = len(target["boxes"])
    original = target.get("num_original", kept)
    # "0 of 3" is a legitimate frame -- a centre crop genuinely removes objects
    # -- and is not distinguishable from a parsing failure without this.
    return f"{stem}\n{kept} of {original} boxes", panels
