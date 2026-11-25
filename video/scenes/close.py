"""Scene 6: a number, and what it is comparable to written beside it.

manim -qh video/scenes/close.py Close
"""

from __future__ import annotations

from common import DATA, NarratedScene, label
from manim import (
    DOWN,
    UP,
    FadeIn,
    LaggedStart,
    VGroup,
    Write,
)
from palette import (
    APPLE_BLUE,
    APPLE_GREEN,
    APPLE_MUSTARD,
    APPLE_ORANGE,
    APPLE_PINK,
    APPLE_PURPLE,
    APPLE_TEAL,
    INK,
    MUTED,
)


class Close(NarratedScene):
    narration = "close"

    def construct(self) -> None:
        # s0  "That is what VisBench is for."
        value = DATA["boards"]["generic_segmentation"]["dino_vitb8"]
        number = label(f"{value:.3f}", 96, INK).move_to([0, 0.4, 0])
        what = label("DINO-B/8, figure-ground segmentation, IoU", 24, MUTED).next_to(
            number, DOWN, buff=0.3
        )
        self.beat(Write(number), FadeIn(what))

        # s1  "Not a winner, but a number you can compare ... the controls ..."
        context = [
            ("protocol: visbench_binary_seg", APPLE_PURPLE, [-4.4, 2.6, 0]),
            ("pooling: resolved, recorded", APPLE_TEAL, [4.3, 2.6, 0]),
            ("layers: resolved, recorded", APPLE_MUSTARD, [-4.6, -1.6, 0]),
            ("seed sweep: ordered vs. tied", APPLE_GREEN, [4.4, -1.6, 0]),
            ("controls: e.g. feature scale", APPLE_ORANGE, [0, 2.0, 0]),
        ]
        tags = VGroup(*[label(text, 24, colour).move_to(at) for text, colour, at in context])
        install = label("pip install visbench", 34, APPLE_PINK).move_to([0, -2.75, 0])
        repo = label("github.com/turhancan97/VisBench", 22, APPLE_BLUE).next_to(
            install, DOWN, buff=0.2
        )
        self.beat(
            LaggedStart(*[FadeIn(tag, shift=UP * 0.15) for tag in tags], lag_ratio=0.25),
            FadeIn(install, repo),
            share=0.75,
        )
        self.finish()
