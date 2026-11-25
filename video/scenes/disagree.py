"""Scene 3: fourteen backbones, twenty boards, and rankings that do not agree.

    manim -qh video/scenes/disagree.py Disagree

The narration states counts ("first on three, last on five") and boards ("tops
scene and car recognition, last on figure-ground"). The scene computes each
from data.json and raises if it no longer matches what is spoken, so a corpus
change cannot leave the voice contradicting the picture.
"""

from __future__ import annotations

from common import (
    DATA,
    DISPLAY,
    LEVEL_ORDER,
    PROBE_NAMES,
    NarratedScene,
    by_mean_rank,
    label,
    rank_grid,
    stack,
)
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    RoundedRectangle,
    SurroundingRectangle,
    VGroup,
    Write,
)
from palette import (
    APPLE_ORANGE,
    APPLE_PURPLE,
    APPLE_YELLOW,
    INK,
    LEVEL_COLOUR,
    MUTED,
)

LAST = 14


def placements(name: str) -> tuple[list[str], list[str]]:
    ranks = DATA["ranks"]
    firsts = [p for p in ranks if ranks[p][name] == 1]
    lasts = [p for p in ranks if ranks[p][name] == LAST]
    return firsts, lasts


def spoken(condition: bool, what: str) -> None:
    if not condition:
        raise ValueError(f"the narration says {what}, and data.json no longer does")


class Disagree(NarratedScene):
    narration = "disagree"

    def construct(self) -> None:
        mae_first, mae_last = placements("mae_vitb16")
        spoken(len(mae_first) == 3 and len(mae_last) == 5, "MAE: first on 3, last on 5")
        clip_first, clip_last = placements("clip_convnext_base")
        spoken(
            set(clip_first) == {"scene_classification", "vehicle_classification"}
            and set(clip_last) == {"generic_segmentation", "orientation"},
            "CLIP ConvNeXt: first on scene + vehicles, last on figure-ground + orientation",
        )

        # s0  "Run fourteen backbones through all twenty, and the rankings do not agree."
        title = label("Rank on every board", 36).to_corner(UP + LEFT)
        key = label("1st = bright, 14th = dark", 20, MUTED).next_to(
            title, RIGHT, buff=0.35, aligned_edge=DOWN
        )
        grid, cells, columns = rank_grid(cell=0.3, gap=0.04)
        grid.move_to([1.35, -0.35, 0])
        names = by_mean_rank()
        row_labels = VGroup()
        for name in names:
            first_cell = cells[(name, columns[0])]
            text = label(DISPLAY[name], 18, INK)
            text.next_to(first_cell, LEFT, buff=0.25)
            row_labels.add(text)
        bands = VGroup()
        for level in LEVEL_ORDER:
            members = [cells[(names[0], p)] for p in columns if DATA["levels"][p] == level]
            band = Line(
                members[0].get_left(),
                members[-1].get_right(),
                color=LEVEL_COLOUR[level],
                stroke_width=6,
            )
            bands.add(band.shift(UP * 0.35))
        self.beat(
            FadeIn(title, key),
            FadeIn(row_labels),
            FadeIn(bands),
            LaggedStart(*[FadeIn(c, scale=0.6) for c in grid], lag_ratio=0.003),
            share=0.8,
        )

        # s1  "M.A.E. is first on three boards and last on five."
        def outline(name: str, probes: list[str], colour: str) -> VGroup:
            return VGroup(
                *[
                    SurroundingRectangle(cells[(name, p)], color=colour, buff=0.03, stroke_width=3)
                    for p in probes
                ]
            )

        mae_row = row_labels[names.index("mae_vitb16")]
        mae_marks = VGroup(
            outline("mae_vitb16", mae_first, APPLE_YELLOW),
            outline("mae_vitb16", mae_last, APPLE_ORANGE),
        )
        mae_note = label("MAE: 1st on 3, last on 5", 22, APPLE_YELLOW).to_corner(DOWN + LEFT)
        self.beat(mae_row.animate.set_color(APPLE_YELLOW), Create(mae_marks), FadeIn(mae_note))

        # s2  "A CLIP-trained Conv-Next tops scene and car recognition, ... last on figure-ground"
        clip_row = row_labels[names.index("clip_convnext_base")]
        clip_marks = VGroup(
            outline("clip_convnext_base", clip_first, APPLE_YELLOW),
            outline("clip_convnext_base", clip_last, APPLE_ORANGE),
        )
        clip_note = label(
            "CLIP ConvNeXt-B: 1st on scene + vehicles, last on figure-ground + orientation",
            22,
            APPLE_YELLOW,
        ).to_corner(DOWN + LEFT)
        self.beat(
            mae_row.animate.set_color(INK),
            FadeOut(mae_marks, mae_note),
            clip_row.animate.set_color(APPLE_YELLOW),
            Create(clip_marks),
            FadeIn(clip_note),
        )

        # s3  "Even the high-level tier splits into two clusters that barely correlate."
        board = VGroup(title, key, grid, row_labels, bands, clip_marks, clip_note)
        c = DATA["clusters"]
        image_box = self.cluster_box(
            "Image-level", [PROBE_NAMES[p] for p in c["image_level"]], [-3.6, 0.3, 0]
        )
        local_box = self.cluster_box(
            "Localised", [PROBE_NAMES[p] for p in c["localised"]], [3.6, 0.3, 0]
        )
        within_a = label(f"ρ = {c['within_image']:+.2f} within", 22, APPLE_PURPLE).next_to(
            image_box, DOWN, buff=0.25
        )
        within_b = label(f"ρ ≥ {c['within_localised_min']:+.2f} within", 22, APPLE_PURPLE).next_to(
            local_box, DOWN, buff=0.25
        )
        across = DashedLine(image_box.get_right(), local_box.get_left(), color=MUTED, buff=0.2)
        across_text = label(f"ρ ≤ {c['across_max']:+.2f} across", 22, MUTED).next_to(
            across, UP, buff=0.15
        )
        heading = label("The high-level tier: two clusters", 36, APPLE_PURPLE).to_corner(UP + LEFT)
        self.beat(
            FadeOut(board),
            FadeIn(heading),
            FadeIn(image_box, local_box),
            FadeIn(within_a, within_b),
            Create(across),
            FadeIn(across_text),
        )

        # s4  "Averaging across them produces a number that describes nothing."
        average = label("average = ?", 34, APPLE_ORANGE).move_to([0, -2.7, 0])
        strike = Line(average.get_left(), average.get_right(), color=APPLE_ORANGE, stroke_width=5)
        self.beat(Write(average), Create(strike))
        self.finish()

    def cluster_box(self, title: str, members: list[str], at: list[float]) -> VGroup:
        items = stack(members)
        head = label(title, 26, APPLE_PURPLE)
        body = VGroup(head, items).arrange(DOWN, buff=0.3)
        frame = RoundedRectangle(
            corner_radius=0.2,
            width=max(body.width + 0.7, 3.2),
            height=body.height + 0.6,
            color=APPLE_PURPLE,
        )
        return VGroup(frame, body).move_to(at)
