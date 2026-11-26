"""The video's palette: the eight colours of the brief, and the role each one plays.

A colour means the same thing in every scene, so a viewer learns it once:
see the role table in `script.md`.
"""

from __future__ import annotations


def _rgb(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


APPLE_BLUE = _rgb(80, 122, 255)  # backbones, features
APPLE_PINK = _rgb(217, 127, 174)  # the probe head
APPLE_TEAL = _rgb(85, 163, 152)  # mid level
APPLE_PURPLE = _rgb(138, 107, 221)  # high level
APPLE_MUSTARD = _rgb(234, 179, 77)  # low level
APPLE_ORANGE = _rgb(255, 127, 80)  # warning, a reversal
APPLE_GREEN = _rgb(52, 199, 89)  # ordered, holds
APPLE_YELLOW = _rgb(255, 204, 0)  # highlight

# Neutrals for the stage, chosen to sit under the palette rather than compete.
BACKGROUND = "#16181D"
INK = "#E8E8EC"  # primary text
MUTED = "#8A8D96"  # secondary text, axes
DIM = "#3A3E48"  # de-emphasised marks

LEVEL_COLOUR = {
    "high_level": APPLE_PURPLE,
    "mid_level": APPLE_TEAL,
    "low_level": APPLE_MUSTARD,
}
