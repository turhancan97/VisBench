"""How each probe's target is drawn, listed per probe rather than inferred.

Nine probes ship a spatial target and they use **four different conventions for
an invalid pixel**, none of which is visible in the tensor's shape or dtype:

===================  ===============================================
convention           probes
===================  ===============================================
``0`` is invalid     ``depth`` (and the zero *vector* for normals)
negative is invalid  both segmentations -- ``0`` is a real class
nothing is invalid   ``edge``, ``keypoints2d``, ``corner``
``NaN`` is invalid   ``occlusion_edge``
===================  ===============================================

A viewer that guessed -- "one channel, mask the zeros, apply a colour map" --
would draw a depth hole and a real 0-metre reading identically for four of the
nine, and would erase the background class from both segmentation probes. That
is the same failure the target *loaders* already guard against, arriving one
layer later: it renders, it looks plausible, and it says the wrong thing.

So :data:`TARGET_STYLES` is a listed table and :func:`style_for` raises on a
name it does not hold, exactly as ``METRIC_DIRECTIONS`` does in
``visbench.results.leaderboard``. Adding a probe and forgetting its style fails
immediately rather than at the first frame someone squints at.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

__all__ = [
    "COMPOSITE_KINDS",
    "TARGET_STYLES",
    "TargetStyle",
    "UnknownTargetStyle",
    "show_probes",
    "style_for",
]

#: What kind of quantity a target holds, which decides how it becomes pixels.
#:
#: ``magnitude`` and ``depth`` differ only in what the caption says -- both are
#: scalar maps drawn in greyscale against a stated range -- but they are kept
#: apart because a depth range is in metres and a magnitude range is in
#: whatever arbitrary unit ``target_scale`` produced, and a caption that
#: confused the two would invite reading one as the other.
Kind = Literal[
    "magnitude",
    "depth",
    "normals",
    "orientation",
    "binary",
    "labels",
    "boxes",
    "matches",
    "sheet",
    "ranking",
    "triplet",
]


def _invalid_zero(target: torch.Tensor) -> torch.Tensor:
    """Depth's convention: a pixel with no ground truth is stored as 0."""
    return target == 0


def _invalid_zero_vector(target: torch.Tensor) -> torch.Tensor:
    """Normals' convention: an unmeasured direction is the zero vector.

    Reduced over the channel axis, so the mask is ``(H, W)`` like every other
    one here — a normal is invalid as a *pixel*, not per component.
    """
    return target.norm(dim=0) == 0


def _invalid_negative(target: torch.Tensor) -> torch.Tensor:
    """Both segmentations: unlabelled is negative, because 0 is a real class.

    Reusing depth's rule here would erase every background pixel, which is the
    mistake ``SemanticSegmentationTask``'s ``IGNORE_INDEX = -1`` exists to make
    impossible in the loss. This is the same rule for the eye.
    """
    return target < 0


def _invalid_nan(target: torch.Tensor) -> torch.Tensor:
    """``occlusion_edge``: validity travels out of band, because 0 is a reading.

    A magnitude map has no spare in-band value — 0 means "no edge here" — so
    the reconstruction-derived domains mark their holes ``NaN``. Drawn, an
    unmasked ``NaN`` is not merely wrong but undrawable, which is the same
    loudness the loss relies on.
    """
    return torch.isnan(target)


@dataclass(frozen=True)
class TargetStyle:
    """How one probe's target becomes pixels, and what the caption must say."""

    #: Which colouriser draws it.
    kind: Kind
    #: ``(H, W)`` bool mask of pixels with no ground truth, or ``None`` when the
    #: target has no invalid value at all. ``None`` is a claim, not an omission:
    #: it says every pixel of this target is a real measurement.
    invalid: Callable[[torch.Tensor], torch.Tensor] | None
    #: Unit for the display range in the caption. Empty when the scale is
    #: arbitrary, which is most of them.
    unit: str = ""
    #: What a reader has to know to read the panel, appended to the caption.
    note: str = ""


#: The table. One row per probe ``visbench show`` can draw; an absent probe is
#: refused by name rather than drawn on a guess.
TARGET_STYLES: dict[str, TargetStyle] = {
    "depth": TargetStyle(
        kind="depth",
        invalid=_invalid_zero,
        unit="m",
        note="grey spans each row's stated range, taken over valid pixels only",
    ),
    "surface_normal": TargetStyle(
        kind="normals",
        invalid=_invalid_zero_vector,
        note="RGB is (n+1)/2 per axis, the usual normal-map convention",
    ),
    "generic_segmentation": TargetStyle(
        kind="binary",
        invalid=_invalid_negative,
        note="white is foreground; unlabelled pixels are negative, not 0",
    ),
    "semantic_segmentation": TargetStyle(
        kind="labels",
        invalid=_invalid_negative,
        note="VOC's palette, so class 15 is the colour a VOC user expects",
    ),
    "edge": TargetStyle(
        kind="magnitude",
        invalid=None,
        note="0 is a real reading (no edge here), so nothing is masked",
    ),
    "keypoints2d": TargetStyle(
        kind="magnitude",
        invalid=None,
        note="0 is a real reading, so nothing is masked",
    ),
    "occlusion_edge": TargetStyle(
        kind="magnitude",
        invalid=_invalid_nan,
        note="log1p space, so the range is compressed; NaN marks a hole",
    ),
    "corner": TargetStyle(
        kind="magnitude",
        invalid=None,
        note="log1p(1e4*lambda_min), computed from this exact crop",
    ),
    "orientation": TargetStyle(
        kind="orientation",
        invalid=None,
        note="hue is the local orientation, brightness its coherence; computed from this crop",
    ),
    "detection": TargetStyle(
        kind="boxes",
        invalid=None,
        note="boxes are post-transform pixels, drawn on the crop the probe saw",
    ),
    "correspondence": TargetStyle(
        kind="matches",
        invalid=None,
        note="two views and the matches between them; needs a backbone, trains nothing",
    ),
    "classification": TargetStyle(
        kind="sheet",
        invalid=None,
        note="a contact sheet of frames and their labels; the footer states the balance",
    ),
    "scene_classification": TargetStyle(
        kind="sheet",
        invalid=None,
        note="a contact sheet of frames and their scene labels; the footer states the balance",
    ),
    "fine_grained_classification": TargetStyle(
        kind="sheet",
        invalid=None,
        note="a contact sheet of frames and their species labels; the footer states the balance",
    ),
    "retrieval": TargetStyle(
        kind="ranking",
        invalid=None,
        note="each query and its nearest neighbours; needs a backbone and a real gallery",
    ),
    "similarity": TargetStyle(
        kind="triplet",
        invalid=None,
        note="reference and two candidates, with the human vote marked",
    ),
}

#: Kinds that are not a panel beside their image, and are drawn by their own
#: renderer instead. Listed so a caller can ask before reaching for
#: :func:`~visbench.viz.colour.target_to_rgb`, which refuses both by name.
COMPOSITE_KINDS: frozenset[str] = frozenset({"boxes", "matches", "sheet", "ranking", "triplet"})


class UnknownTargetStyle(KeyError):
    """Raised for a probe with no listed way to draw its target.

    Deliberately fatal rather than falling back to "scalar map, mask the
    zeros". That default is correct for exactly one of the nine probes here and
    silently wrong for four of them — it would draw a real 0-magnitude reading
    as a hole, and erase the background class from both segmentations.
    """


def style_for(probe: str) -> TargetStyle:
    """The drawing style for ``probe``, or a message naming what can be drawn.

    Distinguishes "registered but not drawable" from "not a probe at all", the
    way :func:`visbench.cli.datasets.spec_for` does: a probe VisBench knows but
    this module cannot draw is a real gap, not a typo.
    """
    if probe in TARGET_STYLES:
        return TARGET_STYLES[probe]

    import visbench

    drawable = ", ".join(show_probes())
    if probe in visbench.list_probes():
        raise UnknownTargetStyle(
            f"{probe!r} is a registered probe but has no listed way to draw its "
            f"target, so `visbench show` cannot render it. Probes it can draw: "
            f"{drawable}."
        )
    raise UnknownTargetStyle(f"Unknown probe {probe!r}. Probes that can be drawn: {drawable}.")


def show_probes() -> list[str]:
    """Probe names with a listed target style, sorted."""
    return sorted(TARGET_STYLES)
