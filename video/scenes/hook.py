"""Scene 1: one number per model -- and why a backbone is not one number.

    manim -qh video/scenes/hook.py Hook

The opening leaderboard is the backbones' *average rank over the twenty
boards*: a real number from the corpus, and exactly the kind of summary the
rest of the video argues against. It unfolds into the 14 x 20 rank grid.
"""

from __future__ import annotations

from common import (
    DATA,
    DISPLAY,
    LEVEL_ORDER,
    NarratedScene,
    by_mean_rank,
    label,
    probes_by_level,
    rank_grid,
)
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    AnimationGroup,
    Circle,
    FadeIn,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    VGroup,
    Write,
)
from palette import (
    APPLE_BLUE,
    APPLE_ORANGE,
    INK,
    LEVEL_COLOUR,
    MUTED,
)


class Hook(NarratedScene):
    narration = "hook"

    def construct(self) -> None:
        names = by_mean_rank()
        mean = DATA["mean_rank"]

        # s0  "...a leaderboard: one number per model, sorted."
        title = label("Leaderboard", 40).to_corner(UP + LEFT)
        sub = label("average rank over 20 tasks", 22, MUTED).next_to(
            title, RIGHT, buff=0.35, aligned_edge=DOWN
        )
        top, step = 2.55, 0.4
        # The numbers are their own group, not children of the rows, so that
        # fading them out really removes them (a FadeOut of mobjects that are
        # also inside a group still on screen leaves them drawn).
        rows, column = VGroup(), VGroup()
        for i, name in enumerate(names):
            y = top - i * step
            row = VGroup(
                label(f"{i + 1:>2}", 20, MUTED).move_to([-6.3, y, 0]),
                label(DISPLAY[name], 22, INK).move_to([-5.85, y, 0], aligned_edge=LEFT),
            )
            row[1].set_x(-5.85 + row[1].width / 2)
            rows.add(row)
            column.add(label(f"{mean[name]:.1f}", 22, APPLE_BLUE).move_to([-1.9, y, 0]))
        self.beat(
            FadeIn(title, sub),
            LaggedStart(
                *[
                    FadeIn(VGroup(r, c), shift=RIGHT * 0.2)
                    for r, c in zip(rows, column, strict=True)
                ],
                lag_ratio=0.06,
            ),
            share=0.85,
        )

        # s1  "But a backbone is not one number."
        box = Rectangle(
            width=column.width + 0.35, height=column.height + 0.3, color=APPLE_ORANGE
        ).move_to(column)
        # Beside the box, not above it: above is the title line.
        question = label("one number?", 26, APPLE_ORANGE).next_to(box, RIGHT, buff=0.3)
        self.beat(FadeIn(box), Write(question))

        # s2  "It is a representation, and different questions read different parts of it."
        grid, cells, columns = rank_grid()
        grid.move_to([2.3, top - (len(names) - 1) * step / 2, 0])
        # Align grid rows with the name rows: scale so row pitch matches `step`.
        grid.stretch_to_fit_height((len(names) - 1) * step + 0.34)
        grid.set_y(top - (len(names) - 1) * step / 2)
        bands = VGroup()
        for level in LEVEL_ORDER:
            members = [cells[(names[0], p)] for p in columns if DATA["levels"][p] == level]
            left, right = members[0].get_left()[0], members[-1].get_right()[0]
            bands.add(
                Line([left, 0, 0], [right, 0, 0], color=LEVEL_COLOUR[level], stroke_width=6).set_y(
                    top + 0.35
                )
            )
        sub2 = label("rank on each of the 20 tasks", 22, MUTED).move_to(sub, aligned_edge=LEFT)
        self.beat(
            self.swap(
                [box, question, column, sub],
                FadeIn(sub2),
                AnimationGroup(*[FadeIn(b) for b in bands]),
                LaggedStart(*[FadeIn(c, scale=0.6) for c in grid], lag_ratio=0.002),
            ),
            share=0.8,
        )

        # s3  "VisBench is built around that idea."
        everything = VGroup(title, sub2, rows, grid, bands)
        name = label("VisBench", 72, INK)
        tagline = label("probing frozen vision backbones", 28, MUTED).next_to(name, DOWN, buff=0.3)
        self.beat(everything.animate.set_opacity(0.12), FadeIn(name, tagline, shift=UP * 0.2))

        # s4  "Freeze the backbone, and ask it twenty different questions."
        backbone = RoundedRectangle(
            corner_radius=0.15, width=2.4, height=1.4, color=APPLE_BLUE, fill_opacity=0.15
        ).move_to([-3.6, -0.6, 0])
        frozen = label("frozen\nbackbone", 24, APPLE_BLUE).move_to(backbone)
        probes = probes_by_level()
        fan = VGroup()
        for k, probe in enumerate(probes):
            y = 2.4 - k * 0.27
            colour = LEVEL_COLOUR[DATA["levels"][probe]]
            dot = Circle(0.07, color=colour, fill_opacity=1, stroke_width=0).move_to([1.4, y, 0])
            fan.add(
                Line(backbone.get_right(), dot.get_center(), color=colour, stroke_width=1.5),
                dot,
            )
        self.beat(
            self.swap(
                [everything, name, tagline],
                FadeIn(backbone, frozen),
                LaggedStart(*[FadeIn(m) for m in fan], lag_ratio=0.03),
            ),
            share=0.8,
        )
        self.finish()
