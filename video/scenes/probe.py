"""Scene 2: what a probe is, and the three levels the twenty probes span.

manim -qh video/scenes/probe.py Probe
"""

from __future__ import annotations

from common import DATA, LEVEL_ORDER, PROBE_NAMES, NarratedScene, label, stack
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Create,
    FadeIn,
    GrowArrow,
    LaggedStart,
    Rectangle,
    RoundedRectangle,
    Square,
    VGroup,
    Write,
)
from palette import (
    APPLE_BLUE,
    APPLE_GREEN,
    APPLE_MUSTARD,
    APPLE_PINK,
    APPLE_PURPLE,
    APPLE_TEAL,
    DIM,
    INK,
    LEVEL_COLOUR,
    MUTED,
)

LEVEL_TITLE = {
    "high_level": "High level: semantics",
    "mid_level": "Mid level: geometry",
    "low_level": "Low level: signal",
}


def image_icon(size: float = 1.3) -> VGroup:
    """A stand-in photograph: a 4x4 mosaic in the palette's quieter tones."""
    colours = [APPLE_TEAL, APPLE_PURPLE, APPLE_MUSTARD, APPLE_BLUE, DIM]
    tile = size / 4
    return VGroup(
        *[
            Square(
                tile, fill_color=colours[(i * 3 + j * 2) % 5], fill_opacity=0.85, stroke_width=0
            ).move_to([j * tile, -i * tile, 0])
            for i in range(4)
            for j in range(4)
        ]
    )


class Probe(NarratedScene):
    narration = "probe"

    def construct(self) -> None:
        # s0  "Each question is a probe."
        title = label("A probe", 40).to_corner(UP + LEFT)
        image = image_icon().move_to([-5.6, 0, 0])
        backbone = RoundedRectangle(
            corner_radius=0.15, width=2.2, height=1.8, color=APPLE_BLUE, fill_opacity=0.15
        ).move_to([-2.7, 0, 0])
        frozen = label("frozen\nbackbone", 24, APPLE_BLUE).move_to(backbone)
        a1 = Arrow(image.get_right(), backbone.get_left(), buff=0.15, color=MUTED)
        self.beat(FadeIn(title), FadeIn(image), GrowArrow(a1), FadeIn(backbone, frozen))

        # s1  "Features come out of the frozen network once, and are cached."
        grid = VGroup(
            *[
                Square(
                    0.22,
                    fill_color=APPLE_BLUE,
                    fill_opacity=0.35 + 0.5 * ((i + j) % 3) / 2,
                    stroke_width=0,
                ).move_to([j * 0.26, -i * 0.26, 0])
                for i in range(6)
                for j in range(6)
            ]
        ).move_to([0.4, 0, 0])
        features = label("features", 20, MUTED).next_to(grid, DOWN, buff=0.2)
        a2 = Arrow(backbone.get_right(), grid.get_left(), buff=0.15, color=MUTED)
        cache = Rectangle(width=1.7, height=0.7, color=MUTED).move_to([0.4, -2.4, 0])
        cache_text = label("cache", 22, MUTED).move_to(cache)
        once = label("computed once", 20, MUTED).next_to(cache, RIGHT, buff=0.25)
        a3 = Arrow(features.get_bottom(), cache.get_top(), buff=0.1, color=DIM)
        self.beat(
            GrowArrow(a2),
            LaggedStart(*[FadeIn(s, scale=0.5) for s in grid], lag_ratio=0.01),
            FadeIn(features),
            GrowArrow(a3),
            Create(cache),
            FadeIn(cache_text, once),
        )

        # s2  "...the smallest head that can answer, usually a single linear map."
        head = RoundedRectangle(
            corner_radius=0.12, width=1.9, height=1.0, color=APPLE_PINK, fill_opacity=0.2
        ).move_to([3.3, 0, 0])
        head_text = label("linear\nhead", 24, APPLE_PINK).move_to(head)
        a4 = Arrow(grid.get_right(), head.get_left(), buff=0.15, color=MUTED)
        answer = label("answer", 24, INK).move_to([5.75, 0, 0])
        a5 = Arrow(head.get_right(), answer.get_left(), buff=0.12, color=MUTED)
        self.beat(GrowArrow(a4), FadeIn(head, head_text), GrowArrow(a5), FadeIn(answer))

        # s3  "If a linear map can read the answer off the features, the representation carries it."
        claim = label(
            "linearly readable  ⇒  carried by the representation", 26, APPLE_GREEN
        ).move_to([0, 2.3, 0])
        self.beat(
            head.animate.set_fill(APPLE_PINK, opacity=0.45),
            grid.animate.set_opacity(1.0),
            Write(claim),
        )

        # s4  "The twenty probes span three levels, following Chen, Marks and Cheng..."
        diagram = VGroup(
            title,
            image,
            a1,
            backbone,
            frozen,
            grid,
            features,
            a2,
            cache,
            cache_text,
            once,
            a3,
            head,
            head_text,
            a4,
            answer,
            a5,
            claim,
        )
        columns = VGroup()
        for k, level in enumerate(LEVEL_ORDER):
            colour = LEVEL_COLOUR[level]
            members = [p for p in PROBE_NAMES if DATA["levels"][p] == level]
            heading = label(LEVEL_TITLE[level], 26, colour)
            items = stack([PROBE_NAMES[p] for p in members])
            column = VGroup(heading, items).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            column.move_to([-6.4 + k * 4.6, 0.6, 0], aligned_edge=UP + LEFT)
            column.set_y(2.6, direction=UP)
            columns.add(column)
        title2 = label("Twenty probes, three levels", 40).to_corner(UP + LEFT)
        credit = label("levels after Chen, Marks & Cheng (2024)", 20, MUTED)
        credit.move_to([6.9 - credit.width / 2, -2.6, 0])  # above the subtitle band
        self.beat(
            self.swap(
                diagram,
                FadeIn(title2),
                LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in columns], lag_ratio=0.25),
                FadeIn(credit),
            ),
            share=0.7,
        )
        self.finish()
