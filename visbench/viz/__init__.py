"""Looking at what a probe saw, and at what it predicted.

Everything else in VisBench turns pixels into numbers. This turns them back,
and it exists because of two failures in this project's own history that were
found by reading code and would have been obvious in one frame:

- geometric correspondence scoring ``recall@1px = 0.003`` — a misalignment
  between a homography in original pixels and features from a 224 centre crop,
  not a weak backbone
- VOC's palette PNGs read through ``convert("L")``, turning classes
  ``[0, 1, 15, 255]`` into ``[0, 38, 147, 220]`` — which loads, trains and
  scores against labels that mean nothing

Both share a shape: a dense target drifting from the image it belongs to fails
*silently*. Nothing raises, the probe trains, and the number comes out merely
mediocre — which reads as a hard task or a weak representation, the two
explanations this library exists to distinguish.

The rules this package keeps, each with its own guard in ``tests/viz/``:

- **Display what the dataset yielded.** No resize, no re-read, no second crop.
- **Draw an invalid pixel as invalid**, per that probe's own convention — there
  are four of them across nine probes and none is visible in a tensor's shape.
- **Scale a prediction by the target's range**, so a systematically wrong
  magnitude cannot draw as a correct one.

Pillow and numpy only; nothing here adds a dependency.
"""

from visbench.viz.colour import (
    INVALID_RGB,
    DisplayRange,
    display_range,
    target_to_rgb,
    voc_palette,
)
from visbench.viz.gallery import (
    RIGHT,
    WRONG,
    annotate,
    class_balance,
    render_retrieval_panels,
    render_sheet,
    render_triplet_panels,
    vote_balance,
)
from visbench.viz.matches import draw_matches, error_coherence, render_match_panels
from visbench.viz.panels import draw_boxes, render_panels, render_probe_panels
from visbench.viz.styles import (
    COMPOSITE_KINDS,
    TARGET_STYLES,
    TargetStyle,
    UnknownTargetStyle,
    show_probes,
    style_for,
)

__all__ = [
    "COMPOSITE_KINDS",
    "RIGHT",
    "WRONG",
    "INVALID_RGB",
    "TARGET_STYLES",
    "DisplayRange",
    "TargetStyle",
    "UnknownTargetStyle",
    "annotate",
    "class_balance",
    "display_range",
    "draw_boxes",
    "draw_matches",
    "error_coherence",
    "render_match_panels",
    "render_panels",
    "render_retrieval_panels",
    "render_sheet",
    "render_triplet_panels",
    "render_probe_panels",
    "show_probes",
    "style_for",
    "target_to_rgb",
    "voc_palette",
    "vote_balance",
]
