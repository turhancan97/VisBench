"""The thumbnail and GitHub social preview: one still, white, in the palette.

    manim -s -r 1280,720 video/scenes/thumbnail.py Thumbnail   # YouTube (16:9)
    manim -s -r 1280,640 video/scenes/thumbnail.py Thumbnail   # GitHub (2:1)

Layout reads `config.frame_width`, so the same scene fits both ratios. The grid
is the real 14 x 20 rank grid from data.json, as in the video. On white the
palette's yellow is used only for outlines, never for text, which it would
make unreadable.
"""

from __future__ import annotations

from pathlib import Path

from common import DATA, LEVEL_ORDER, by_mean_rank, label, rank_grid
from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Line,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    SVGMobject,
    VGroup,
    config,
)
from palette import (
    APPLE_BLUE,
    APPLE_ORANGE,
    APPLE_PINK,
    APPLE_YELLOW,
    LEVEL_COLOUR,
)

WHITE = "#FFFFFF"
INK_ON_WHITE = "#1E2028"
MUTED_ON_WHITE = "#5F6370"
LOW_ON_WHITE = "#E4E8F2"  # the 14th-rank end of the grid's scale, on white

#: The logo's own icon, in its own colours: the brand is not recoloured into
#: the video palette. Manim's SVG import skips <text>, so the wordmark beside it
#: is set here in the logo's ink colour (the logo names Arial semibold; the
#: render nodes have no Arial, and DejaVu Sans Bold is the closest available).
ICON = Path(__file__).resolve().parents[2] / "assets" / "visbench-icon-light.svg"
LOGO_INK = "#262421"

config.background_color = WHITE


class Thumbnail(Scene):
    def construct(self) -> None:
        self.camera.background_color = WHITE
        half_w, half_h = config.frame_width / 2, config.frame_height / 2

        # The grid: right-hand side, as large as the height allows.
        grid, cells, columns = rank_grid(cell=0.3, gap=0.05, low=LOW_ON_WHITE)
        grid.scale_to_fit_height(config.frame_height * 0.56)
        grid.move_to([half_w - 0.55 - grid.width / 2, -0.25, 0])
        names = by_mean_rank()
        bands = VGroup()
        for level in LEVEL_ORDER:
            members = [cells[(names[0], p)] for p in columns if DATA["levels"][p] == level]
            bands.add(
                Line(
                    members[0].get_left(),
                    members[-1].get_right(),
                    color=LEVEL_COLOUR[level],
                    stroke_width=10,
                ).shift(UP * 0.32)
            )
        # The CLIP ConvNeXt row's extremes, as in the video: first and last.
        ranks = DATA["ranks"]
        row = "clip_convnext_base"
        marks = VGroup(
            *[
                SurroundingRectangle(
                    cells[(row, p)],
                    color=APPLE_YELLOW if ranks[p][row] == 1 else APPLE_ORANGE,
                    buff=0.03,
                    stroke_width=5,
                )
                for p in columns
                if ranks[p][row] in (1, len(names))
            ]
        )

        # The words: left-hand side.
        left = -half_w + 0.7
        wordmark = label("VisBench", 66, LOGO_INK, weight=BOLD)
        icon = SVGMobject(str(ICON)).scale_to_fit_height(wordmark.height * 1.45)
        title = VGroup(icon, wordmark).arrange(RIGHT, buff=0.3)
        question = label("Which vision backbone", 36, INK_ON_WHITE)
        question2 = label("is best?", 36, INK_ON_WHITE)
        answer = label("It depends on the question.", 31, APPLE_BLUE)
        words = VGroup(title, question, question2, answer).arrange(
            DOWN, aligned_edge=LEFT, buff=0.22
        )
        title.shift(UP * 0.15)
        answer.shift(DOWN * 0.2)
        words.move_to([left, 0.55, 0], aligned_edge=LEFT)

        facts = VGroup()
        for text, colour in (
            ("20 probes", LEVEL_COLOUR["high_level"]),
            ("14 backbones", LEVEL_COLOUR["mid_level"]),
        ):
            chip_text = label(text, 22, INK_ON_WHITE)
            chip = RoundedRectangle(
                corner_radius=0.22,
                width=chip_text.width + 0.55,
                height=chip_text.height + 0.4,
                stroke_color=colour,
                stroke_width=4,
                fill_color=colour,
                fill_opacity=0.12,
            )
            facts.add(VGroup(chip, chip_text.move_to(chip)))
        facts.arrange(RIGHT, buff=0.25).next_to(words, DOWN, buff=0.55, aligned_edge=LEFT)

        install = label("pip install visbench", 27, APPLE_PINK)
        install.next_to(facts, DOWN, buff=0.45, aligned_edge=LEFT)
        caption = label("rank on each task: 1st = blue, 14th = pale", 22, MUTED_ON_WHITE)
        caption.next_to(grid, DOWN, buff=0.22)

        self.add(grid, bands, marks, words, facts, install, caption)

        # Nothing may touch the edge, at either ratio.
        for mob in (grid, bands, marks, words, facts, install, caption):
            assert mob.get_left()[0] > -half_w + 0.2, mob
            assert mob.get_right()[0] < half_w - 0.2, mob
            assert mob.get_bottom()[1] > -half_h + 0.2, mob
            assert mob.get_top()[1] < half_h - 0.2, mob
            # Words and grid must not overlap.
        assert words.get_right()[0] < grid.get_left()[0] - 0.2, (
            f"text runs into the grid: words end at {words.get_right()[0]:.2f}, "
            f"grid starts at {grid.get_left()[0]:.2f}, title {title.width:.2f} wide, "
            f"question {question.width:.2f}, answer {answer.width:.2f}, "
            f"frame {config.frame_width:.2f}"
        )
        assert facts.get_right()[0] < grid.get_left()[0] - 0.2, "chips run into the grid"
