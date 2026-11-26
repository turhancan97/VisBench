# The VisBench explainer video

**Published: [https://www.youtube.com/watch?v=giKxewO0-lo](https://www.youtube.com/watch?v=giKxewO0-lo)** (3:30).

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
| `assemble.sh` | joins the rendered scenes, in `script.md`'s order, into one file |

Renders, audio and stills go to `video/build/`, which is gitignored.

## Narration and subtitles

`script.md` is the one source for both. `[shown](spoken)` sends the first part
to the subtitle and the second to Kokoro, for names it needs spelled
phonetically (`[MAE](M.A.E.)`) and numbers it should say in words. Narration is
synthesised at speed 0.9, every scene ends on a one-second hold, and subtitles
are drawn by the scene itself in a reserved band at the bottom of the frame,
switched on the measured sentence timings. A long sentence is split into cues
of near-equal length, breaking at punctuation where there is one.

## Three guarantees the pipeline keeps

- **Picture and voice cannot drift.** Narration is generated per sentence and
  each scene's `beat()` lasts exactly that sentence's measured duration plus a
  fixed gap. A scene that asks for more or fewer beats than its narration has
  sentences raises, so a script edit cannot silently desynchronise a scene.
- **Nothing on screen leaves the frame or covers the subtitles.** At the end of
  every beat, every mobject on screen is checked against the frame minus a
  margin and against the top of the subtitle band, and a violation raises with
  its beat and coordinates. Clipping is easy to miss in a still and impossible
  to miss in this error.
- **Old text leaves before new text arrives.** `NarratedScene.swap()` fades the
  outgoing mobjects fully out and only then plays the incoming animations, so a
  transition never overprints two labels in one place.

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
