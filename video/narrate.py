#!/usr/bin/env python
"""Synthesise the narration with Kokoro-82M, one clip per sentence.

    python video/narrate.py                 # every scene in script.md
    python video/narrate.py twist           # one scene
    python video/narrate.py --voice am_michael twist

Reads the **Narration** blocks of `script.md` verbatim, splits each into
sentences, and writes `build/audio/<scene>/<nn>.wav` plus
`build/audio/<scene>/timing.json` with every sentence's duration. A scene
sizes each animation beat to its sentence, so picture and voice cannot drift.

Spelling in the script is already phonetic where Kokoro needs it ("M.A.E.",
"Conv-Next"); keep it that way rather than adding a substitution table here,
so the script reads exactly as it will sound.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "script.md"
AUDIO = HERE / "build" / "audio"
SAMPLE_RATE = 24_000  # Kokoro's output rate

#: Silence appended after each sentence, in seconds. Part of the timing, so a
#: scene's beat for sentence i lasts exactly duration[i] + GAP.
GAP = 0.35


def scenes() -> dict[str, list[str]]:
    """`{scene: [sentence, ...]}` in script order."""
    text = SCRIPT.read_text(encoding="utf-8")
    out: dict[str, list[str]] = {}
    for match in re.finditer(
        r"^## \d+\. (\w+)\s*$.*?\*\*Narration\.\*\*\n(.*?)(?=^## |\Z)", text, re.S | re.M
    ):
        name, body = match.group(1), " ".join(match.group(2).split())
        # Split after . ! ? followed by a capital, so "M.A.E. is" stays whole.
        out[name] = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", body) if s]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenes", nargs="*")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a")  # American English
    all_scenes = scenes()
    for name in args.scenes or list(all_scenes):
        out = AUDIO / name
        out.mkdir(parents=True, exist_ok=True)
        durations = []
        for i, sentence in enumerate(all_scenes[name]):
            chunks = [
                np.asarray(audio)
                for _, _, audio in pipeline(sentence, voice=args.voice, speed=args.speed)
            ]
            wave = np.concatenate(chunks)
            sf.write(out / f"{i:02d}.wav", wave, SAMPLE_RATE)
            durations.append(len(wave) / SAMPLE_RATE)
        # One continuous clip for the scene, with GAP after every sentence.
        silence = np.zeros(int(GAP * SAMPLE_RATE), dtype=np.float32)
        parts = []
        for i in range(len(durations)):
            wave, _ = sf.read(out / f"{i:02d}.wav", dtype="float32")
            parts += [wave, silence]
        sf.write(out / "scene.wav", np.concatenate(parts), SAMPLE_RATE)
        timing = {
            "voice": args.voice,
            "speed": args.speed,
            "gap": GAP,
            "sentences": all_scenes[name],
            "durations": durations,
            "total": sum(durations) + GAP * len(durations),
        }
        (out / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
        print(f"{name}: {len(durations)} sentences, {timing['total']:.1f}s ({args.voice})")


if __name__ == "__main__":
    main()
