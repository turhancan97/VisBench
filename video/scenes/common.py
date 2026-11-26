"""What every scene shares: the stage, the data, and beats timed to narration.

A scene is a sequence of beats, one per narrated sentence. `NarratedScene.beat`
plays a beat's animations and then waits out the rest of that sentence, so the
picture can never run ahead of or behind the voice: the timing comes from the
audio Kokoro actually produced (`build/audio/<scene>/timing.json`), not from an
estimate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from manim import Scene, Text, config

VIDEO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VIDEO))

from palette import BACKGROUND, INK  # noqa: E402

DATA = json.loads((VIDEO / "data.json").read_text())
FONT = "DejaVu Sans"

config.background_color = BACKGROUND


def label(text: str, size: float = 28, colour: str = INK, **kwargs) -> Text:
    """One text style for the whole video."""
    return Text(text, font=FONT, font_size=size, color=colour, **kwargs)


def stack(lines: list[str], size: float = 22, colour: str = INK, spacing: float = 0.9) -> Text:
    """Lines as one Pango text block, so every line sits on the same baseline grid.

    Stacking separate Text objects spaces them by bounding box, and a line with
    descenders ("g", "p") or tall letters ("f", "l") gets a different gap from
    one without: the list looks uneven however it is aligned. Pango lays out a
    multi-line string on consistent baselines.
    """
    return Text("\n".join(lines), font=FONT, font_size=size, color=colour, line_spacing=spacing)


# Spoken and on-screen names differ only where the reader needs it to.
DISPLAY = {
    "clip_convnext_base": "CLIP ConvNeXt-B",
    "convnext_base": "ConvNeXt-B (IN1k)",
    "dinov2_vits14": "DINOv2-S/14",
    "dinov2_vitb14": "DINOv2-B/14",
    "dino_vitb8": "DINO-B/8",
    "dino_vitb16": "DINO-B/16",
    "clip_vitb16": "CLIP-B/16",
    "clip_vitb32": "CLIP-B/32",
    "siglip_vitb16": "SigLIP-B/16",
    "sam_vitb16": "SAM-B/16",
    "mae_vitb16": "MAE-B/16",
    "supervised_vitb16": "ViT-B/16 (IN1k)",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
}


#: The subtitle band: the bottom of the frame, below this line, belongs to the
#: subtitles. Every beat checks that no content has strayed into it.
SUBTITLE_TOP = -2.95
SUBTITLE_Y = -3.47
SUBTITLE_SIZE = 26
#: Characters per subtitle line and lines per cue; a longer sentence is split
#: into several cues timed in proportion to their length.
LINE_CHARS, CUE_LINES = 72, 2
#: Silent hold at the end of every scene, before the next one starts.
HOLD = 1.0
#: Safe margin inside the frame edge, in scene units.
MARGIN = 0.15


def subtitle_cues(texts: list[str], durations: list[float], gap: float) -> list[tuple]:
    """`(start, end, text)` per cue, from each sentence's measured duration.

    A sentence that wraps to more than CUE_LINES lines is split at word
    boundaries into cues of at most CUE_LINES lines, each shown for a share of
    the sentence proportional to its length. A cue stays up until the next one
    starts, so there is no flicker of empty band between sentences.
    """

    cues, t = [], 0.0
    for text, duration in zip(texts, durations, strict=True):
        # As few cues as fit, then words shared out evenly between them, so a
        # long sentence never ends on a two-word orphan cue.
        k = max(1, -(-len(text) // (LINE_CHARS * CUE_LINES)))
        chunks = ["\n".join(balanced_lines(c)) for c in split_evenly(text.split(), k)]
        total = sum(len(c) for c in chunks)
        start = t
        for chunk in chunks:
            length = duration * len(chunk) / total
            cues.append([start, start + length, chunk])
            start += length
        t += duration + gap
    for here, after in zip(cues, cues[1:], strict=False):
        here[1] = after[0]
    if cues:
        cues[-1][1] = t
    return [tuple(c) for c in cues]


def split_evenly(words: list[str], parts: int) -> list[str]:
    """`parts` runs of words of near-equal length, breaking at punctuation if close.

    Each break goes after the word nearest the ideal length; a word ending in
    , : ; within a quarter of a part of that point is preferred, since that is
    where a reader pauses anyway.
    """
    if parts <= 1:
        return [" ".join(words)]
    ends, total = [], 0
    for word in words:
        total += len(word) + 1
        ends.append(total)
    breaks, start = [], 0
    for p in range(1, parts):
        ideal = total * p / parts
        window = total / parts / 4
        candidates = range(start, len(words) - (parts - p))
        nearest = min(candidates, key=lambda i: abs(ends[i] - ideal))
        punctuated = [
            i for i in candidates if words[i][-1] in ",:;" and abs(ends[i] - ideal) <= window
        ]
        cut = min(punctuated, key=lambda i: abs(ends[i] - ideal)) if punctuated else nearest
        breaks.append(cut + 1)
        start = cut + 1
    bounds = [0, *breaks, len(words)]
    return [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:], strict=False)]


def balanced_lines(text: str) -> list[str]:
    """As few lines as LINE_CHARS allows, of near-equal width (no one-word last line)."""
    import textwrap

    lines = max(1, -(-len(text) // LINE_CHARS))
    width = -(-len(text) // lines)
    while True:
        wrapped = textwrap.wrap(text, width)
        if len(wrapped) <= lines or width >= LINE_CHARS:
            return wrapped
        width += 2


class NarratedScene(Scene):
    """A Scene whose beats are sized to the narration's sentences, with subtitles."""

    #: The `## n. name` of this scene in script.md.
    narration: str = ""

    def setup(self) -> None:
        timing_path = VIDEO / "build" / "audio" / self.narration / "timing.json"
        self.timing = json.loads(timing_path.read_text())
        self._beat = 0
        self.add_sound(str(timing_path.parent / "scene.wav"))
        self._add_subtitles()

    def _add_subtitles(self) -> None:
        from manim import DOWN, Mobject, Rectangle, VGroup
        from palette import BACKGROUND

        cues = subtitle_cues(self.timing["subtitles"], self.timing["durations"], self.timing["gap"])
        # The band is the frame below SUBTITLE_TOP, opaque enough to read over
        # anything that animates past it on the way out.
        band = Rectangle(
            width=config.frame_width + 0.2,
            height=config.frame_height / 2 + SUBTITLE_TOP,
            fill_color=BACKGROUND,
            fill_opacity=0.92,
            stroke_width=0,
        )
        band.move_to([0, (-config.frame_height / 2 + SUBTITLE_TOP) / 2, 0])
        texts = []
        for _, _, text in cues:
            # One Text per line, centred: a multi-line Text aligns left.
            mob = VGroup(*[label(line, SUBTITLE_SIZE, INK) for line in text.split("\n")])
            mob.arrange(DOWN, buff=0.12)
            if mob.width > config.frame_width - 1.2:
                mob.scale_to_fit_width(config.frame_width - 1.2)
            mob.move_to([0, SUBTITLE_Y, 0]).set_opacity(0)
            texts.append(mob)
        clock = Mobject()
        clock.shown = None
        renderer = self.renderer

        def tick(mob, dt):
            # The time of the frame being drawn, read from the renderer, which
            # counts every frame written. Summing `dt` instead loses a frame per
            # play(): Manim restarts each animation's clock and hands its first
            # frame dt=0, so the subtitles fell ~0.15 s behind within a scene.
            now_t = renderer.time
            now = next((i for i, (a, b, _) in enumerate(cues) if a <= now_t < b), None)
            if now != mob.shown:
                if mob.shown is not None:
                    texts[mob.shown].set_opacity(0)
                if now is not None:
                    texts[now].set_opacity(1)
                mob.shown = now

        clock.add_updater(tick)
        self.chrome = [band, *texts, clock]
        for mob in (band, *texts):
            mob.set_z_index(100)
        self.add(band, *texts, clock)

    def swap(self, outgoing, *incoming, out_share: float = 0.3):
        """Fade `outgoing` fully out, *then* play `incoming`: never both at once.

        A cross-fade puts the old and the new text on screen together, and
        where they share a place they overprint. Returned as one animation, so
        it can be handed to `beat()` beside others.
        """
        from manim import AnimationGroup, FadeOut, Succession

        outgoing = outgoing if isinstance(outgoing, (list, tuple)) else [outgoing]
        return Succession(
            FadeOut(*outgoing, run_time=out_share),
            AnimationGroup(*incoming, run_time=1 - out_share),
        )

    def beat(self, *animations, share: float = 0.6, **kwargs) -> None:
        """Play `animations` over `share` of the next sentence, then hold.

        With no animations the whole sentence is held. Raises if a scene asks
        for more beats than the narration has sentences, which is how a script
        edit that drops a sentence fails loudly instead of desynchronising. At
        the end of every beat, nothing may have left the frame or entered the
        subtitle band.
        """
        durations = self.timing["durations"]
        if self._beat >= len(durations):
            raise IndexError(
                f"{self.narration}: beat {self._beat} but the narration has "
                f"{len(durations)} sentences; re-run narrate.py or fix the scene"
            )
        length = durations[self._beat] + self.timing["gap"]
        self._beat += 1
        # Where this sentence ends in the audio, absolutely. Waiting *until* it,
        # rather than for `length`, stops each play()/wait() rounding to whole
        # frames from accumulating into a picture that lags the voice.
        end = sum(durations[: self._beat]) + self.timing["gap"] * self._beat
        if animations:
            run = max(0.3, share * length)
            self.play(*animations, run_time=run, **kwargs)
        remaining = end - self.renderer.time
        if remaining > 1 / config.frame_rate:
            self.wait(remaining)
        content = [m for m in self.mobjects if m not in self.chrome]
        assert_in_frame(content, f"{self.narration} beat {self._beat - 1}", floor=SUBTITLE_TOP)

    def finish(self) -> None:
        """Refuse to end a scene that left sentences un-animated; then hold."""
        unused = len(self.timing["durations"]) - self._beat
        if unused:
            raise RuntimeError(f"{self.narration}: {unused} sentence(s) never got a beat")
        self.wait(HOLD)


def assert_in_frame(mobjects, where: str, floor: float | None = None) -> None:
    """Raise if anything ends outside the frame, or below `floor` when given.

    Clipping, and content sliding under the subtitles, are easy to miss in a
    still and impossible to miss in this error.
    """
    half_w = config.frame_width / 2 - MARGIN
    half_h = config.frame_height / 2 - MARGIN
    lowest = -half_h if floor is None else floor
    for mob in mobjects:
        for part in mob.get_family():
            if not part.has_points():
                continue
            left, right = part.get_left()[0], part.get_right()[0]
            bottom, top = part.get_bottom()[1], part.get_top()[1]
            if left < -half_w or right > half_w or bottom < lowest or top > half_h:
                raise RuntimeError(
                    f"{where}: {type(part).__name__} at x[{left:.2f},{right:.2f}] "
                    f"y[{bottom:.2f},{top:.2f}] leaves the content area "
                    f"(x within +-{half_w:.2f}, y in [{lowest:.2f}, {half_h:.2f}])"
                )


# ---- shared pieces: probe names, rank colours, the rank grid ---------------

#: Short on-screen names, grouped by level in the order the grid shows them.
PROBE_NAMES = {
    "classification": "object cls.",
    "scene_classification": "scene cls.",
    "fine_grained_classification": "fine-grained",
    "vehicle_classification": "vehicles",
    "retrieval": "retrieval",
    "detection": "detection",
    "instance_segmentation": "instance seg.",
    "semantic_segmentation": "semantic seg.",
    "scene_parsing": "scene parsing",
    "depth": "depth",
    "surface_normal": "normals",
    "generic_segmentation": "figure-ground",
    "occlusion_edge": "occlusion",
    "correspondence": "correspondence",
    "similarity": "similarity",
    "relative_pose": "camera pose",
    "edge": "edges",
    "keypoints2d": "keypoints",
    "corner": "corners",
    "orientation": "orientation",
}
LEVEL_ORDER = ("high_level", "mid_level", "low_level")


def probes_by_level() -> list[str]:
    """Every probe, grouped high -> mid -> low, in PROBE_NAMES order."""
    return [p for level in LEVEL_ORDER for p in PROBE_NAMES if DATA["levels"][p] == level]


def by_mean_rank() -> list[str]:
    return sorted(DATA["mean_rank"], key=DATA["mean_rank"].get)


def rank_colour(rank: int, n: int = 14) -> str:
    """Rank 1 is full blue, rank n fades to the stage: one hue, so it reads as order."""
    from manim import ManimColor, interpolate_color
    from palette import APPLE_BLUE, DIM

    return interpolate_color(
        ManimColor(DIM), ManimColor(APPLE_BLUE), 1 - (rank - 1) / (n - 1)
    ).to_hex()


def rank_grid(cell: float = 0.34, gap: float = 0.04):
    """14 rows (by mean rank) x 20 columns (by level): each cell coloured by rank.

    Returns (grid, cells, columns) where cells[(backbone, probe)] is the square
    and columns is the probe order, so a scene can outline any cell it narrates.
    """
    from manim import Square, VGroup

    rows, columns = by_mean_rank(), probes_by_level()
    cells, grid = {}, VGroup()
    for i, name in enumerate(rows):
        for j, probe in enumerate(columns):
            rank = DATA["ranks"][probe][name]
            square = Square(cell, fill_color=rank_colour(rank), fill_opacity=1, stroke_width=0)
            square.move_to([j * (cell + gap), -i * (cell + gap), 0])
            cells[(name, probe)] = square
            grid.add(square)
    return grid, cells, columns
