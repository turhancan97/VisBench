"""Scene 5: the scale control — two ConvNeXts swap places under a constant.

    manim -qh video/scenes/twist.py Twist

Every value comes from `data.json`: the published `generic_segmentation`
board, the scale control's x8.66 / x0.1155 arms, and the feature std table.
"""

from __future__ import annotations

from common import DATA, DISPLAY, NarratedScene, label
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    AnimationGroup,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    Rectangle,
    Succession,
    Transform,
    VGroup,
    Write,
)
from palette import APPLE_BLUE, APPLE_ORANGE, APPLE_YELLOW, DIM, INK, MUTED

CLIP, SUP = "clip_convnext_base", "convnext_base"
COLOUR = {CLIP: APPLE_YELLOW, SUP: APPLE_BLUE}

# Board geometry (frame is 14.2 x 8 units).
TOP, ROW = 2.55, 0.36  # 14 rows end above the subtitle band
NAME_X, BAR_X, BAR_MAX = -6.75, -4.05, 3.6
PANEL_X = 1.25  # left edge of the right-hand panel
AXIS_LO, AXIS_HI = 0.40, 0.80  # the axis is truncated, and labelled so


def bar_width(value: float) -> float:
    return max(0.02, (value - AXIS_LO) / (AXIS_HI - AXIS_LO) * BAR_MAX)


def ranked(board: dict[str, float]) -> list[str]:
    return sorted(board, key=board.get, reverse=True)


def both_swapped(task: str) -> tuple[dict[str, int], dict[str, int]]:
    """Ranks before and after moving *both* columns onto each other's scale."""
    board = dict(DATA["boards"][task])
    arms = DATA["scale_control"][task]
    before = {n: i + 1 for i, n in enumerate(ranked(board))}
    board[CLIP] = arms[f"{CLIP}@8.66"]
    board[SUP] = arms[f"{SUP}@0.1155"]
    after = {n: i + 1 for i, n in enumerate(ranked(board))}
    return before, after


class Twist(NarratedScene):
    narration = "twist"

    def row_y(self, rank: int) -> float:
        return TOP - (rank - 1) * ROW

    def make_bar(self, name: str, value: float, y: float, colour: str) -> Rectangle:
        bar = Rectangle(
            width=bar_width(value), height=0.26, fill_color=colour, fill_opacity=1, stroke_width=0
        )
        return bar.move_to([BAR_X + bar.width / 2, y, 0])

    def construct(self) -> None:
        task = "generic_segmentation"
        board = DATA["boards"][task]
        arms = DATA["scale_control"][task]
        order = ranked(board)

        # ---- the board ----------------------------------------------------
        title = label("Figure-ground segmentation", 34).to_corner(UP + LEFT).shift(DOWN * 0.05)
        unit = label("IoU, linear probe", 22, MUTED).next_to(
            title, RIGHT, buff=0.35, aligned_edge=DOWN
        )
        rows: dict[str, VGroup] = {}
        for rank, name in enumerate(order, start=1):
            y = self.row_y(rank)
            colour = COLOUR.get(name, APPLE_BLUE)
            text = label(DISPLAY[name], 20, INK).move_to([0, y, 0], aligned_edge=LEFT)
            text.set_x(NAME_X + text.width / 2)
            bar = self.make_bar(name, board[name], y, colour)
            value = label(f"{board[name]:.3f}", 18, MUTED).next_to(bar, RIGHT, buff=0.12)
            rows[name] = VGroup(text, bar, value)

        axis_y = self.row_y(len(order)) - 0.36
        axis = Line([BAR_X, axis_y, 0], [BAR_X + BAR_MAX, axis_y, 0], color=MUTED, stroke_width=2)
        ticks = VGroup(
            *[
                label(f"{t:.1f}", 18, MUTED).move_to(
                    [BAR_X + (t - AXIS_LO) / (AXIS_HI - AXIS_LO) * BAR_MAX, axis_y - 0.25, 0]
                )
                for t in (0.4, 0.6, 0.8)
            ]
        )

        # s0  "But a seed sweep only measures noise."
        self.beat(
            AnimationGroup(
                FadeIn(title, unit),
                Create(axis),
                FadeIn(ticks),
                LaggedStart(
                    *[
                        AnimationGroup(
                            FadeIn(rows[n][0]), GrowFromEdge(rows[n][1], LEFT), FadeIn(rows[n][2])
                        )
                        for n in order
                    ],
                    lag_ratio=0.08,
                ),
            ),
            share=0.9,
        )

        # s1  "Here is something it cannot see."  -- dim everything but the pair
        others = [n for n in order if n not in COLOUR]
        self.beat(
            *[rows[n][1].animate.set_fill(DIM) for n in others],
            *[rows[n][0].animate.set_color(MUTED) for n in others],
        )

        # s2  "...features are about nine times smaller than its supervised twin's."
        std = DATA["feature_std"]
        panel_x, unit_len = PANEL_X, 1.1
        panel_title = label("Feature scale (std of features)", 22, MUTED).move_to(
            [panel_x, 2.75, 0], aligned_edge=LEFT
        )
        scale_bars, scale_labels = {}, {}
        for i, name in enumerate((CLIP, SUP)):
            y = 2.1 - i * 0.75
            bar = Rectangle(
                width=std[name] * unit_len,
                height=0.34,
                fill_color=COLOUR[name],
                fill_opacity=1,
                stroke_width=0,
            )
            bar.move_to([panel_x + bar.width / 2, y, 0])
            scale_bars[name] = bar
            scale_labels[name] = label(f"{std[name]:.2f}", 20, INK).next_to(bar, RIGHT, buff=0.12)
        ratio = std[SUP] / std[CLIP]
        ratio_tag = label(f"~{ratio:.0f}x apart", 24, APPLE_YELLOW).move_to([panel_x + 2.2, 2.1, 0])
        self.beat(
            FadeIn(panel_title),
            *[GrowFromEdge(b, LEFT) for b in scale_bars.values()],
            *[FadeIn(t) for t in scale_labels.values()],
            Write(ratio_tag),
        )

        # s3  "Multiply each onto the other's scale ... the two swap places."
        factor = {CLIP: 8.66, SUP: 0.1155}
        new_value = {CLIP: arms[f"{CLIP}@8.66"], SUP: arms[f"{SUP}@0.1155"]}
        multipliers = VGroup(
            label("x 8.66", 22, APPLE_YELLOW).move_to([panel_x + 4.75, 2.1, 0]),
            label("x 0.1155", 22, APPLE_BLUE).move_to([panel_x + 4.75, 1.35, 0]),
        )
        rescaled = {}
        for name, bar in scale_bars.items():
            target = Rectangle(
                width=std[name] * factor[name] * unit_len,
                height=0.34,
                fill_color=COLOUR[name],
                fill_opacity=1,
                stroke_width=0,
            )
            rescaled[name] = target.move_to([panel_x + target.width / 2, bar.get_y(), 0])
        new_board = dict(board)
        new_board.update(new_value)
        new_rank = {n: i + 1 for i, n in enumerate(ranked(new_board))}

        moves = []
        for name in order:
            y = self.row_y(new_rank[name])
            text, bar, value = rows[name]
            if name in COLOUR:
                new_bar = self.make_bar(name, new_value[name], y, COLOUR[name])
                new_text = label(f"{new_value[name]:.3f}", 18, INK).next_to(
                    new_bar, RIGHT, buff=0.12
                )
                moves += [
                    text.animate.set_y(y),
                    Transform(bar, new_bar),
                    Transform(value, new_text),
                ]
            else:
                moves.append(VGroup(text, bar, value).animate.set_y(y))
        before_rank = {n: i + 1 for i, n in enumerate(order)}
        swap_tag = label(
            f"{before_rank[CLIP]}th  ↔  {before_rank[SUP]}th", 26, APPLE_ORANGE
        ).move_to([panel_x + 2.6, 0.45, 0])
        swap_note = label("a constant changed the ranking", 20, APPLE_ORANGE).next_to(
            swap_tag, DOWN, buff=0.15
        )
        self.beat(
            Succession(
                AnimationGroup(
                    FadeOut(ratio_tag),
                    FadeIn(multipliers),
                    *[Transform(scale_bars[n], rescaled[n]) for n in scale_bars],
                    *[FadeOut(t) for t in scale_labels.values()],
                ),
                AnimationGroup(*moves),
                AnimationGroup(Write(swap_tag), FadeIn(swap_note)),
            ),
            share=0.85,
        )

        # s4  "The same happens on semantic segmentation and on depth."
        right_panel = VGroup(panel_title, *scale_bars.values(), multipliers, swap_tag, swap_note)
        charts = VGroup(
            *[
                self.slope_chart(t, name, x)
                for t, name, x in (
                    ("semantic_segmentation", "Semantic seg.", PANEL_X + 0.55),
                    ("depth", "Depth (d1)", PANEL_X + 3.45),
                )
            ]
        )
        self.beat(self.swap(right_panel, FadeIn(charts, shift=UP * 0.2)))

        # s5  "The head is fitted with a fixed learning rate, so part of what..."
        caption = VGroup(
            label("Linear head, fixed learning rate:", 22, INK),
            label("the board reads feature scale,", 22, APPLE_ORANGE),
            label("not only representation.", 22, APPLE_ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        caption.move_to([PANEL_X, -0.95, 0], aligned_edge=UP + LEFT)
        self.beat(FadeIn(caption, shift=UP * 0.15))

        # s6  "Every seed shares that scale, so no amount of re-fitting..."
        seeds = VGroup(*[Dot(radius=0.07, color=APPLE_ORANGE) for _ in range(5)]).arrange(
            RIGHT, buff=0.22
        )
        seed_note = label("all five seeds share the same scale", 20, MUTED)
        foot = VGroup(seeds, seed_note).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        foot.move_to([PANEL_X, -2.1, 0], aligned_edge=UP + LEFT)
        self.beat(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in seeds], lag_ratio=0.15), FadeIn(seed_note)
        )
        self.finish()

    def slope_chart(self, task: str, title: str, x: float) -> VGroup:
        """Rank before -> after moving both columns, for one board."""
        before, after = both_swapped(task)
        top, step, width = 2.3, 0.205, 1.55

        def y(rank: int) -> float:
            return top - (rank - 1) * step

        head = label(title, 22, INK).move_to([x + width / 2, 2.75, 0])
        rails = VGroup(
            Line([x, y(1), 0], [x, y(14), 0], color=DIM, stroke_width=2),
            Line([x + width, y(1), 0], [x + width, y(14), 0], color=DIM, stroke_width=2),
        )
        foot = VGroup(
            label("before", 16, MUTED).move_to([x, y(14) - 0.3, 0]),
            label("after", 16, MUTED).move_to([x + width, y(14) - 0.3, 0]),
        )
        marks = VGroup()
        for name in (CLIP, SUP):
            a, b = [x, y(before[name]), 0], [x + width, y(after[name]), 0]
            marks.add(
                Line(a, b, color=COLOUR[name], stroke_width=4),
                Dot(a, radius=0.06, color=COLOUR[name]),
                Dot(b, radius=0.06, color=COLOUR[name]),
                label(str(before[name]), 16, COLOUR[name]).next_to(Dot(a), LEFT, buff=0.08),
                label(str(after[name]), 16, COLOUR[name]).next_to(Dot(b), RIGHT, buff=0.08),
            )
        return VGroup(head, rails, foot, marks)
