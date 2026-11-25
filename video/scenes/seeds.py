"""Scene 4: is a placement a fact or a coin flip? The paired seed re-fit.

    manim -qh video/scenes/seeds.py Seeds

Three adjacent rows of one board, picked by `extract_data.threshold_example`:
an ordered pair and a visibly tied pair, the tie with the *wider* gap. Dots are
the five seeds; a line joins the same seed across two rows, so an ordered pair
leans one way every time and a tie criss-crosses.
"""

from __future__ import annotations

from common import DATA, DISPLAY, NarratedScene, label
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    VGroup,
    Write,
)
from palette import APPLE_BLUE, APPLE_GREEN, APPLE_ORANGE, DIM, INK, MUTED

BOARD_TITLE = {"fine_grained_classification": "Fine-grained recognition (CUB), top-1"}


class Seeds(NarratedScene):
    narration = "seeds"

    def construct(self) -> None:
        ex = DATA["threshold_example"]
        ordered, tied = ex["ordered"], ex["tied"]
        rows = [ordered["better"], ordered["worse"], tied["worse"]]
        if ordered["worse"] != tied["better"]:
            raise ValueError("the scene needs three adjacent rows sharing a middle one")
        seeds = {**ordered["seeds"], **tied["seeds"]}

        values = [v for r in rows for v in seeds[r]]
        lo, hi = min(values) - 0.004, max(values) + 0.004
        x0, x1 = -3.2, 5.8

        def x(v: float) -> float:
            return x0 + (v - lo) / (hi - lo) * (x1 - x0)

        ys = {rows[0]: 1.3, rows[1]: 0.0, rows[2]: -1.3}

        # s0  "So is a placement a fact, or a coin flip?"
        question = label("A placement: fact, or coin flip?", 38).to_corner(UP + LEFT)
        board = label(BOARD_TITLE.get(ex["task"], ex["task"]), 22, MUTED).next_to(
            question, DOWN, aligned_edge=LEFT, buff=0.2
        )
        names = VGroup(
            *[
                label(DISPLAY[r], 24, INK).move_to([-4.0, ys[r], 0], aligned_edge=RIGHT)
                for r in rows
            ]
        )
        for n, r in zip(names, rows, strict=True):
            n.set_x(-3.7 - n.width / 2)
            n.set_y(ys[r])
        rails = VGroup(
            *[Line([x0, ys[r], 0], [x1, ys[r], 0], color=DIM, stroke_width=2) for r in rows]
        )
        self.beat(FadeIn(question, board), FadeIn(names), Create(rails))

        # s1  "Every trained board was re-fitted at five seeds, ... the paired difference."
        dots = {
            r: [Dot([x(v), ys[r], 0], radius=0.09, color=APPLE_BLUE) for v in seeds[r]]
            for r in rows
        }
        links_o = VGroup(
            *[
                Line(
                    dots[rows[0]][k].get_center(),
                    dots[rows[1]][k].get_center(),
                    color=APPLE_GREEN,
                    stroke_width=2.5,
                )
                for k in range(5)
            ]
        )
        links_t = VGroup(
            *[
                Line(
                    dots[rows[1]][k].get_center(),
                    dots[rows[2]][k].get_center(),
                    color=APPLE_ORANGE,
                    stroke_width=2.5,
                )
                for k in range(5)
            ]
        )
        caption = label("dots = 5 seeds; a line joins the same seed", 20, MUTED).to_corner(
            DOWN + LEFT
        )
        self.beat(
            LaggedStart(*[FadeIn(d, scale=0.4) for r in rows for d in dots[r]], lag_ratio=0.04),
            Create(links_o),
            Create(links_t),
            FadeIn(caption),
            share=0.75,
        )

        verdict_o = label(
            f"gap {ordered['gap']:.3f}: ordered, {ordered['wins']}/5 seeds", 22, APPLE_GREEN
        ).move_to([1.3, ys[rows[0]] + 0.5, 0])
        verdict_t = label(
            f"gap {tied['gap']:.3f}: tied, {tied['wins']}/5 seeds", 22, APPLE_ORANGE
        ).move_to([1.3, ys[rows[2]] - 0.5, 0])

        # s2  "Of two hundred and eight pairs, one hundred and fifty-two are ordered..."
        t = DATA["sweep_totals"]
        totals = VGroup(
            label(f"{t['ordered']} ordered", 30, APPLE_GREEN),
            label(f"{t['tied']} tied", 30, MUTED),
            label(f"{t['reversed']} reversed", 30, INK),
        ).arrange(RIGHT, buff=0.7)
        of = label(f"of {t['pairs']} adjacent pairs, {t['boards']} boards", 22, MUTED)
        tally = VGroup(totals, of).arrange(DOWN, buff=0.2).move_to([0.9, -2.75, 0])
        self.beat(FadeOut(caption), FadeIn(verdict_o, verdict_t), FadeIn(tally, shift=UP * 0.2))

        # s3  "And on most boards, no threshold on the gap could have told you which was which."
        point = label(
            f"the wider gap is the coin flip -- on {t['thresholdless']} of {t['boards']} boards,"
            " no gap threshold can sort the pairs",
            22,
            APPLE_ORANGE,
        )
        if point.width > 12.6:
            point.scale_to_fit_width(12.6)
        point.move_to([0, -2.75, 0])
        self.beat(FadeOut(tally), Write(point))
        self.finish()
