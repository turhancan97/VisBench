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


class NarratedScene(Scene):
    """A Scene whose beats are sized to the narration's sentences."""

    #: The `## n. name` of this scene in script.md.
    narration: str = ""

    def setup(self) -> None:
        timing_path = VIDEO / "build" / "audio" / self.narration / "timing.json"
        self.timing = json.loads(timing_path.read_text())
        self._beat = 0
        self.add_sound(str(timing_path.parent / "scene.wav"))

    def beat(self, *animations, share: float = 0.6, **kwargs) -> None:
        """Play `animations` over `share` of the next sentence, then hold.

        With no animations the whole sentence is held. Raises if a scene asks
        for more beats than the narration has sentences, which is how a script
        edit that drops a sentence fails loudly instead of desynchronising.
        """
        durations = self.timing["durations"]
        if self._beat >= len(durations):
            raise IndexError(
                f"{self.narration}: beat {self._beat} but the narration has "
                f"{len(durations)} sentences; re-run narrate.py or fix the scene"
            )
        length = durations[self._beat] + self.timing["gap"]
        self._beat += 1
        if animations:
            run = max(0.3, share * length)
            self.play(*animations, run_time=run, **kwargs)
            self.wait(length - run)
        else:
            self.wait(length)

    def finish(self) -> None:
        """Refuse to end a scene that left sentences un-animated or ran off-frame."""
        unused = len(self.timing["durations"]) - self._beat
        if unused:
            raise RuntimeError(f"{self.narration}: {unused} sentence(s) never got a beat")
        assert_in_frame(self.mobjects, self.narration)


#: Safe margin inside the frame edge, in scene units.
MARGIN = 0.15


def assert_in_frame(mobjects, where: str) -> None:
    """Raise if anything on screen ends outside the frame.

    Clipping is easy to miss in a still and impossible to miss in this error,
    so every scene checks what is left on screen when it ends.
    """
    half_w = config.frame_width / 2 - MARGIN
    half_h = config.frame_height / 2 - MARGIN
    for mob in mobjects:
        for part in mob.get_family():
            if not part.has_points():
                continue
            left, right = part.get_left()[0], part.get_right()[0]
            bottom, top = part.get_bottom()[1], part.get_top()[1]
            if left < -half_w or right > half_w or bottom < -half_h or top > half_h:
                raise RuntimeError(
                    f"{where}: {type(part).__name__} at x[{left:.2f},{right:.2f}] "
                    f"y[{bottom:.2f},{top:.2f}] runs off the frame"
                )
