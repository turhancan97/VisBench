"""Scene 6: the timm double crop -- another bias every seed shared (22c/23a).

    manim -qh video/scenes/crop.py Crop

The leader changes and the size of the re-run come from `data.json`
(`extract_data.crop_fix`), which asserts the counts the narration speaks.
"""

from __future__ import annotations

from common import DATA, DISPLAY, PROBE_NAMES, NarratedScene, label
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    AnimationGroup,
    Create,
    DashedVMobject,
    FadeIn,
    LaggedStart,
    Rectangle,
    Square,
    Succession,
    VGroup,
    Write,
)
from palette import (
    APPLE_BLUE,
    APPLE_GREEN,
    APPLE_MUSTARD,
    APPLE_ORANGE,
    APPLE_PURPLE,
    APPLE_TEAL,
    BACKGROUND,
    DIM,
    INK,
    MUTED,
)

#: The fraction of each side the second crop kept, at its tightest (87.5%).
KEPT = 0.875


def mosaic(size: float, n: int = 8) -> VGroup:
    """A stand-in photograph whose content runs to every edge."""
    colours = [APPLE_TEAL, APPLE_PURPLE, APPLE_MUSTARD, APPLE_BLUE, DIM]
    tile = size / n
    return VGroup(
        *[
            Square(
                tile,
                fill_color=colours[(i * 3 + j * 2 + i * j) % 5],
                fill_opacity=0.8,
                stroke_width=0,
            ).move_to([j * tile, -i * tile, 0])
            for i in range(n)
            for j in range(n)
        ]
    )


class Crop(NarratedScene):
    narration = "crop"

    def construct(self) -> None:
        fix = DATA["crop_fix"]

        # s0  "It is not the first bias every seed shared."
        title = label("A bias every seed shared", 40).to_corner(UP + LEFT)
        self.beat(FadeIn(title))

        # s1  "Until version 0.27, nine of the corpus's columns saw a frame cropped twice..."
        size = 3.4
        image = mosaic(size).move_to([-3.9, 0.35, 0])
        frame = Square(size, color=INK, stroke_width=3).move_to(image)
        first = label("dataset: resize + crop to 224 px", 22, INK).next_to(frame, UP, buff=0.2)
        inner = DashedVMobject(
            Square(size * KEPT, color=APPLE_ORANGE, stroke_width=4).move_to(image), num_dashes=40
        )
        # Dim the ring the second crop throws away: four strips around `inner`.
        m = size * (1 - KEPT) / 2
        ring = VGroup(
            Rectangle(width=size, height=m).next_to(inner, UP, buff=0),
            Rectangle(width=size, height=m).next_to(inner, DOWN, buff=0),
            Rectangle(width=m, height=size * KEPT).next_to(inner, LEFT, buff=0),
            Rectangle(width=m, height=size * KEPT).next_to(inner, RIGHT, buff=0),
        ).set_style(fill_color=BACKGROUND, fill_opacity=0.75, stroke_width=0)
        second = VGroup(
            label("timm eval transform: crops again", 22, APPLE_ORANGE),
            label("features see the central 87.5-95%", 22, APPLE_ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        second.next_to(frame, DOWN, buff=0.25)
        target = label("the dense target spans all of it", 20, MUTED).next_to(
            second, DOWN, buff=0.12, aligned_edge=LEFT
        )
        self.beat(
            Succession(
                AnimationGroup(FadeIn(image), Create(frame), FadeIn(first)),
                AnimationGroup(Create(inner), FadeIn(ring), FadeIn(second), FadeIn(target)),
            ),
            share=0.7,
        )

        # s2  "Fixing it re-ran 180 cells, and moved four leaders, all to DINO-B/8."
        head = label("Leader, before → after the fix", 26, INK)
        rows = VGroup()
        for change in fix["changes"]:
            row = VGroup(
                label(PROBE_NAMES[change["task"]], 22, MUTED),
                label(DISPLAY[change["before"]], 22, INK),
                label("→", 22, MUTED),
                label(DISPLAY[change["after"]], 22, APPLE_GREEN),
            )
            rows.add(row)
        # A fixed column grid, so names line up whatever their widths.
        xs = (0.4, 2.95, 5.05, 5.5)
        for i, row in enumerate(rows):
            for part, x in zip(row, xs, strict=True):
                part.move_to([x, 1.2 - i * 0.55, 0], aligned_edge=LEFT)
        head.move_to([0.4, 2.0, 0], aligned_edge=LEFT)
        size_note = label(
            f"{fix['rerun_cells']} cells re-run, {fix['columns']} columns", 22, MUTED
        ).move_to([0.4, 1.2 - len(rows) * 0.55 - 0.15, 0], aligned_edge=LEFT)
        self.beat(
            FadeIn(head),
            LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows], lag_ratio=0.2),
            FadeIn(size_note),
        )

        # s3  "A comparable number needs every step of its pipeline written down, and checked."
        moral = VGroup(
            label("every step of the pipeline:", 26, APPLE_GREEN),
            label("written down, and checked", 26, APPLE_GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        moral.move_to([0.4, -1.75, 0], aligned_edge=LEFT)
        self.beat(Write(moral))
        self.finish()
