"""Target tensors to RGB arrays. Pure functions, no I/O, no file formats.

Two rules govern everything here, and both exist because a picture is believed
more readily than a number.

**An invalid pixel is drawn as invalid.** :data:`INVALID_RGB` is magenta, which
appears in no colouriser's own output — greyscale has no hue, the normal-map
convention cannot reach full magenta for a unit vector, and VOC's palette does
not contain it. So a hole reads as a hole at a glance instead of as a plausible
dark pixel, which is the failure that makes a misread validity convention
survive inspection.

**A target and a prediction share one display range.** Scaling each panel to its
own extremes is the obvious implementation and it hides the most common way a
regression head is wrong: a prediction uniformly half the target's magnitude
renders *identically* to a correct one. :func:`display_range` is computed once,
from the target, over valid pixels only, and applied to both.

Greyscale rather than a perceptual colour map is a deliberate second-order
choice **for a magnitude map**. It needs no lookup table and therefore no new
dependency, and it cannot manufacture structure: a colour ramp puts visible
boundaries at its own transitions, and on a noisy magnitude map those read as
edges in the data.

**Depth is the exception, and the exception is measured rather than asserted.**
A magnitude answers "how much is here", where mid-grey is a reading like any
other; a depth map answers "how far", and the eye reads no ordinal meaning into
mid-grey at all, so a depth panel comes out as texture rather than as near and
far. :data:`_DEPTH_ANCHORS` is therefore a ramp — generated inline by
interpolating five anchors, the way :func:`voc_palette` and
:func:`_orientation` stay dependency-free. What makes it safe against the
objection above is that its luminance **never reverses**: the greyscale panel is
recoverable as this one's luminance channel, so the ramp cannot introduce a
boundary greyscale does not already have. A test asserts that, and asserts the
ramp stays far from magenta, which is the other property the palette owes
:data:`INVALID_RGB`.

**Non-decreasing, not strictly increasing** — the distinction is worth stating
because the stronger word shipped here through 0.16.1 and is wrong. Measured
over 256 samples, the luminance rises 24.3 to 229.2 with **5 ties in 255
steps**: the ramp spans about 205 of the 255 available levels, so at this
sample count uint8 quantisation *forces* repeats and strict monotonicity is
unreachable rather than merely absent. Non-decreasing is also all the argument
needs — a ramp that never reverses adds hue without adding structure — so the
test asserts ``>= 0`` deliberately. Do not tidy it to ``> 0``; it would fail,
and it would be asserting a property nothing here requires.
"""

from dataclasses import dataclass

import numpy as np
import torch

from visbench.viz.styles import COMPOSITE_KINDS, TargetStyle

__all__ = [
    "INVALID_RGB",
    "DisplayRange",
    "display_range",
    "target_to_rgb",
    "voc_palette",
]

#: Colour for a pixel with no ground truth. Chosen because no colouriser here
#: can produce it, so it is never ambiguous with real data.
INVALID_RGB: tuple[int, int, int] = (255, 0, 255)

#: Percentiles the display range spans. Not the full min/max: a single outlying
#: pixel — a specular highlight in a corner response, one far-wall depth reading
#: — otherwise compresses the whole frame into the bottom of the ramp and the
#: panel goes flat black while the data is fine.
_DISPLAY_PERCENTILES: tuple[float, float] = (2.0, 98.0)


@dataclass(frozen=True)
class DisplayRange:
    """The low and high values mapped to black and white.

    Carried explicitly, and reported in the caption, because it is the one
    piece of scaling a viewer cannot avoid applying. Stating it is what keeps
    the panel a measurement rather than an impression.
    """

    low: float
    high: float

    def normalise(self, values: np.ndarray) -> np.ndarray:
        """``values`` mapped to ``[0, 1]``, clipped at both ends."""
        span = self.high - self.low
        if not np.isfinite(span) or span <= 0:
            # A constant frame. Everything is "at the value", so mid-grey says
            # that honestly; scaling by a zero span would divide by zero and
            # scaling by an epsilon would turn rounding noise into structure.
            return np.full(values.shape, 0.5, dtype=np.float64)
        return np.clip((values - self.low) / span, 0.0, 1.0)

    def caption(self, unit: str = "") -> str:
        """``"0.41 to 6.24 m"``, for the panel label.

        Spelled ``to`` rather than a hyphen because a negative low makes a
        hyphen ambiguous: ``keypoints2d`` renders ranges like
        ``-0.3318 to 1.956``, which as ``-0.3318--1.956`` reads as a subtraction
        or a typo. Negative lows are ordinary here — a magnitude probe's
        ``_activate`` is the identity, so a head is free to predict below zero
        and the 2nd percentile of one often is.

        ASCII, like every caption this package writes: PIL's built-in bitmap
        font has no glyph for an en dash and draws an empty box.
        """
        suffix = f" {unit}" if unit else ""
        return f"{self.low:.4g} to {self.high:.4g}{suffix}"


def display_range(target: torch.Tensor, valid: torch.Tensor | None = None) -> DisplayRange:
    """The range to draw ``target`` against, over its valid pixels only.

    ``valid`` is a ``(H, W)`` bool mask, or ``None`` when every pixel counts.
    Excluding the invalid ones is not cosmetic: depth's holes are stored as 0,
    so including them pins the low end at 0 for every frame and flattens the
    real range into the top of the ramp. ``NaN`` holes are worse — a percentile
    over them is ``NaN``, and the whole panel goes black.
    """
    values = target.detach().to(torch.float64).cpu().numpy().ravel()
    if valid is not None:
        values = values[valid.detach().cpu().numpy().ravel()]
    values = values[np.isfinite(values)]

    if values.size == 0:
        # Nothing to measure. The panel will be entirely invalid anyway, so any
        # range renders the same thing; a defined one keeps normalise() total.
        return DisplayRange(0.0, 1.0)

    low, high = np.percentile(values, _DISPLAY_PERCENTILES)
    return DisplayRange(float(low), float(high))


def voc_palette(count: int = 256) -> np.ndarray:
    """Pascal VOC's class colours as ``(count, 3)`` uint8.

    Generated by VOC's own bit-shuffle rather than hard-coded, so class 15 is
    ``(192, 128, 128)`` — the colour someone who has looked at VOC before will
    recognise as *person*. A palette invented here would be equally legible and
    would silently disagree with every VOC figure ever published, which is the
    kind of difference that costs an hour to notice.
    """
    palette = np.zeros((count, 3), dtype=np.uint8)
    for index in range(count):
        remaining, red, green, blue = index, 0, 0, 0
        for shift in range(8):
            red |= ((remaining >> 0) & 1) << (7 - shift)
            green |= ((remaining >> 1) & 1) << (7 - shift)
            blue |= ((remaining >> 2) & 1) << (7 - shift)
            remaining >>= 3
        palette[index] = (red, green, blue)
    return palette


#: The depth ramp, as five anchors interpolated at draw time. Ordered dark blue
#: -> blue -> teal -> green -> pale yellow, which is the viridis family with its
#: purple end dropped: that end sits at hue ~296 degrees, four degrees from
#: magenta, and a marker that means "no ground truth" must not share a
#: neighbourhood with real data however dark that data is drawn. Luminance rises
#: 24 -> 80 -> 122 -> 175 -> 229, and interpolation is linear, so luminance is
#: linear within every segment and increasing across all of them.
_DEPTH_ANCHORS: tuple[tuple[int, int, int], ...] = (
    (13, 24, 61),
    (26, 90, 140),
    (32, 148, 133),
    (120, 200, 90),
    (250, 235, 110),
)


def _greyscale(target: torch.Tensor, span: DisplayRange) -> np.ndarray:
    values = target.detach().to(torch.float64).cpu().numpy()
    # NaN survives normalise() as NaN and would cast to 0 silently. It is always
    # masked out by the caller for the one probe that produces it, but a target
    # with an unexpected NaN must not be drawn as black — nan_to_num puts it at
    # the top of the ramp, where it is at least visible.
    grey = np.nan_to_num(span.normalise(values), nan=1.0)
    scaled = np.round(grey * 255.0).astype(np.uint8)
    return np.repeat(scaled[..., None], 3, axis=-1)


def _depth(target: torch.Tensor, span: DisplayRange) -> np.ndarray:
    """A distance as a ramp, not as grey. See the module docstring.

    Shares every scaling decision with :func:`_greyscale` — the same
    :class:`DisplayRange`, so a prediction is still drawn against the target's
    range, and the same ``nan_to_num`` to the top of the ramp — and differs only
    in what the normalised value is looked up in.
    """
    values = target.detach().to(torch.float64).cpu().numpy()
    position = np.nan_to_num(span.normalise(values), nan=1.0)

    anchors = np.asarray(_DEPTH_ANCHORS, dtype=np.float64)
    stops = np.linspace(0.0, 1.0, len(anchors))
    channels = [np.interp(position, stops, anchors[:, channel]) for channel in range(3)]
    return np.round(np.stack(channels, axis=-1)).astype(np.uint8)


def _normals(target: torch.Tensor) -> np.ndarray:
    """``(3, H, W)`` unit vectors as RGB, by the universal ``(n + 1) / 2``.

    Takes the first three channels only: ``SurfaceNormalTask`` emits a fourth
    when its uncertainty-aware loss is on, and that channel is a concentration
    parameter rather than a direction — folded into the colour it would tint
    the whole panel by how confident the head was.
    """
    directions = target.detach().to(torch.float64).cpu().numpy()[:3]
    rgb = np.clip((directions + 1.0) / 2.0, 0.0, 1.0)
    return np.round(np.moveaxis(rgb, 0, -1) * 255.0).astype(np.uint8)


def _orientation(target: torch.Tensor) -> np.ndarray:
    """``(2, H, W)`` double-angle field as RGB: hue is orientation, value is coherence.

    The two channels are ``coherence * cos 2theta`` and ``coherence * sin 2theta``
    (see :class:`~visbench.data.derived.OrientationResponse`), so ``atan2`` of
    them recovers ``2theta`` and the vector length is the coherence. Hue carries
    the orientation — cyclic, like the quantity — and brightness carries how
    well-defined it is, so a flat patch reads as black rather than as a
    confident wrong colour. Saturation is fixed, which keeps the one free
    channel (value) unambiguous.

    HSV is converted inline rather than pulled from a colour library, matching
    the "no lookup table, no new dependency" rule the greyscale path follows.
    """
    field = target.detach().to(torch.float64).cpu().numpy()[:2]
    two_theta = np.arctan2(field[1], field[0])
    hue = (two_theta % (2.0 * np.pi)) / (2.0 * np.pi)
    coherence = np.clip(np.hypot(field[0], field[1]), 0.0, 1.0)

    sextant = np.floor(hue * 6.0).astype(np.int64) % 6
    frac = hue * 6.0 - np.floor(hue * 6.0)
    p = np.zeros_like(coherence)
    q = coherence * (1.0 - frac)
    t = coherence * frac
    v = coherence
    stack = np.stack(
        [
            np.choose(sextant, [v, q, p, p, t, v]),
            np.choose(sextant, [t, v, v, q, p, p]),
            np.choose(sextant, [p, p, t, v, v, q]),
        ],
        axis=-1,
    )
    return np.round(stack * 255.0).astype(np.uint8)


def _binary(target: torch.Tensor) -> np.ndarray:
    """Non-zero is foreground, matching ``load_mask``'s own rule.

    Thresholded at 0.5 rather than at "non-zero" so a prediction — which
    arrives as a sigmoid probability, not a label — is drawn by the same rule
    ``binary_iou`` scores it by.
    """
    values = target.detach().to(torch.float64).cpu().numpy()
    mask = np.where(values > 0.5, 255, 0).astype(np.uint8)
    return np.repeat(mask[..., None], 3, axis=-1)


def _labels(target: torch.Tensor) -> np.ndarray:
    """Class indices through VOC's palette, with no intermediate conversion.

    The index is used as an index. That sounds too obvious to write down, and
    it is exactly what ``convert("L")`` broke when VOC's palette PNGs were first
    read: classes ``[0, 1, 15, 255]`` became ``[0, 38, 147, 220]``, which loads,
    trains and scores against labels that mean nothing.
    """
    indices = target.detach().cpu().numpy()
    palette = voc_palette()
    # Negative entries are the ignore label; they are overwritten with
    # INVALID_RGB by the caller, and are clipped here only so the lookup is in
    # range rather than wrapping round to a real class's colour.
    safe = np.clip(np.round(indices).astype(np.int64), 0, len(palette) - 1)
    return palette[safe]


def target_to_rgb(
    target: torch.Tensor,
    style: TargetStyle,
    span: DisplayRange | None = None,
) -> np.ndarray:
    """Draw ``target`` as an ``(H, W, 3)`` uint8 array, per ``style``.

    ``span`` is required for a scalar map and ignored otherwise. It is passed in
    rather than derived here so a prediction can be drawn against the *target's*
    range — deriving it per call is what makes a systematically scaled
    prediction look correct.

    Invalid pixels are painted :data:`INVALID_RGB` last, after the colouriser
    has run, so no colouriser needs to know the convention and none of them can
    disagree about it.
    """
    if style.kind in COMPOSITE_KINDS:
        drawn_by = {
            "boxes": "panels.draw_boxes",
            "instances": "panels.draw_instances",
            "matches": "matches.draw_matches",
            "sheet": "gallery.render_sheet",
            "ranking": "gallery.render_retrieval_panels",
            "triplet": "gallery.render_triplet_panels",
        }
        raise ValueError(
            f"A {style.kind!r} target is drawn onto the frames it belongs to, not as "
            f"a panel of its own. Use visbench.viz.{drawn_by[style.kind]}."
        )

    if style.kind in ("magnitude", "depth"):
        if span is None:
            raise ValueError(
                f"A {style.kind} target needs a display range. Compute it once from "
                "the target with display_range() and pass the same one for the "
                "prediction, or a prediction at half the target's scale draws "
                "identically to a correct one."
            )
        rgb = _greyscale(target, span) if style.kind == "magnitude" else _depth(target, span)
    elif style.kind == "normals":
        rgb = _normals(target)
    elif style.kind == "orientation":
        rgb = _orientation(target)
    elif style.kind == "binary":
        rgb = _binary(target)
    elif style.kind == "labels":
        rgb = _labels(target)
    else:  # pragma: no cover - Kind is exhaustive above
        raise ValueError(f"No colouriser for kind {style.kind!r}")

    if style.invalid is not None:
        mask = style.invalid(target).detach().cpu().numpy()
        rgb = rgb.copy()
        rgb[mask] = INVALID_RGB
    return rgb
