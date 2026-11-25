# The VisBench explainer video

A 3–4 minute explainer for ML researchers who know backbones, built in the
style of 3Blue1Brown: [Manim](https://www.manim.community/) scenes, narration
synthesised with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).

| file | what it is |
| --- | --- |
| `script.md` | the narration and what is on screen, scene by scene. **Edit narration here only** |
| `extract_data.py` | pulls every on-screen number from the corpus and controls into `data.json` |
| `data.json` | committed; regenerate whenever the corpus changes |
| `palette.py` | the eight brand colours and the role each one plays |
| `narrate.py` | Kokoro, one clip per sentence, plus `timing.json` with each duration |
| `scenes/common.py` | `NarratedScene`: beats sized to the narration, and a frame-bounds guard |
| `scenes/<name>.py` | one scene per `## n. name` in the script |
| `render.sh` | narration → Manim → one still per beat, for one or more scenes |

Renders, audio and stills go to `video/build/`, which is gitignored.

## Two guarantees the pipeline keeps

- **Picture and voice cannot drift.** Narration is generated per sentence and
  each scene's `beat()` lasts exactly that sentence's measured duration plus a
  fixed gap. A scene that asks for more or fewer beats than its narration has
  sentences raises, so a script edit cannot silently desynchronise a scene.
- **Nothing on screen leaves the frame.** `finish()` checks every mobject still
  on screen against the frame minus a margin and raises with its coordinates.
  Clipping is easy to miss in a still and impossible to miss in this error.

## Setup

The tools live in their own conda environment, not the project `.venv`
(Manim, ffmpeg and Kokoro are not VisBench dependencies). No LaTeX is needed;
every label is Pango text.

```bash
conda create -p /shared/results/common/kargin/.cache/envs/visbench-video \
    -c conda-forge python=3.11 manim ffmpeg numpy
<env>/bin/python -m pip install kokoro soundfile   # espeak-ng arrives via pip
```

## Rendering

```bash
PYTHONPATH=. python video/extract_data.py            # project python: imports visbench
QUALITY=-ql sbatch -p cpu --cpus-per-task=4 --mem=16G --time=01:00:00 \
    -o logs/video_%j.log video/render.sh twist        # fast draft
sbatch ... video/render.sh twist                      # 1080p60 (the default)
```

Run renders on a compute node, never the login node. `VOICE=am_michael`
changes the narrator; `af_heart` is the default.
