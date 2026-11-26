# VisBench explainer — script

Audience: ML researchers who know backbones. Target: 3–4 minutes, narrated
at Kokoro speed 0.9 with a one-second hold between scenes. Each `##` is one Manim scene and one narration clip; `narrate.py`
reads the **Narration** blocks verbatim, so edit them here and nowhere else.
`[shown](spoken)` puts one text in the subtitle and another in the voice, for
names Kokoro needs spelled phonetically ("[MAE](M.A.E.)") and numbers it
should say in words.
Numbers come from `video/data.json` (`extract_data.py`), never typed by hand.

Palette roles (from the brief): blue = backbones/features, teal = mid-level,
purple = high-level, mustard = low-level, pink = the probe head, orange =
warning / a reversal, green = "ordered" / holds, yellow = highlight.

---

## 1. hook

**On screen.** A familiar leaderboard: fourteen backbone names sorted by one
number. The single column splits into twenty columns; the sorted order
scrambles.

**Narration.**
Ask which vision backbone is best, and the answer usually comes as a
leaderboard: one number per model, sorted. But a backbone is not one number.
It is a representation, and different questions read different parts of it.
VisBench is built around that idea. Freeze the backbone, and ask it twenty
different questions.

## 2. probe

**On screen.** An image enters a frozen backbone (blue, padlock); a feature
grid comes out and is written to a cache. A small pink head sits on top and
reads it. The twenty probes fan out into three bands: high level (purple),
mid level (teal), low level (mustard).

**Narration.**
Each question is a probe. Features come out of the frozen network once, and
are cached. On top of them we fit the smallest head that can answer, usually
a single linear map. If a linear map can read the answer off the features, the
representation carries it. The twenty probes span three levels, following
Chen, Marks and Cheng: high-level semantics, mid-level geometry, and low-level
signal, from edges to corners.

## 3. disagree

**On screen.** Fourteen rows, twenty columns of ranks as a heatmap. Highlight
MAE's row: three firsts, five lasts. Then the CLIP ConvNeXt row: first on
scene and car recognition, last on figure-ground segmentation and orientation.

**Narration.**
Run fourteen backbones through all twenty, and the rankings do not agree.
[MAE](M.A.E.) is first on three boards and last on five. A CLIP-trained [ConvNeXt](Conv-Next)
tops scene and car recognition, and comes last on figure-ground segmentation
and on orientation.
Even the high-level tier splits into two clusters that barely correlate.
Averaging across them produces a number that describes nothing.

## 4. seeds

**On screen.** Two adjacent rows of one board as dots at five seeds. One pair
separates cleanly (green, "ordered"); another pair's dots interleave (grey,
"tied"). The counter: 152 ordered, 56 tied, 0 reversed, of 208.

**Narration.**
So is a placement a fact, or a coin flip? Every trained board was re-fitted at
five seeds, and each adjacent pair tested on the paired difference. Of [208](two
hundred and eight) pairs, [152](one hundred and fifty-two) are ordered, [56](fifty-six) are
ties, and none reverses. And on most boards, no threshold on the gap could
have told you which was which.

## 5. twist

**On screen.** The figure-ground board, fourteen bars. Two are highlighted:
CLIP's Conv-Next at the bottom, its supervised twin above. A scale bar shows
their feature magnitudes, about nine times apart. Multiply CLIP's features by
the ratio, and the twin's down by the same: the two bars cross. The same
reversal flashes for semantic segmentation and depth.

**Narration.**
But a seed sweep only measures noise. Here is something it cannot see. The
CLIP [ConvNeXt's](Conv-Next's) features are about nine times smaller than its supervised
twin's. Multiply each onto the other's scale, a constant that removes nothing
a linear head could use, and on figure-ground segmentation the two swap
places. The same happens on semantic segmentation and on depth. The head is
fitted with a fixed learning rate, so part of what those boards rank is
feature scale, not representation. Every seed shares that scale, so no amount
of re-fitting would have revealed it.

## 6. crop

**On screen.** A frame whose target spans all of it. The dataset crops to
224; then a dashed orange rectangle crops again to 87.5–95%, and everything
outside it dims. A table: four boards whose leader changed when the crop was
removed, all to DINO-B/8.

**Narration.**
It is not the first bias every seed shared. Until version [0.27](zero point
twenty-seven), nine of the corpus's columns saw a frame cropped twice: the
dataset cropped to [224](two hundred and twenty-four) pixels, and the timm
preprocessing cropped again, keeping only the central [87.5 to 95
percent](eighty-seven and a half to ninety-five percent) of an image whose
target spans all of it. Fixing it re-ran [180](a hundred and eighty) cells, and
moved four leaders, all to [DINO-B/8](DINO B-8). A comparable number needs every
step of its pipeline written down, and checked.

## 7. close

**On screen.** A single number grows its context around it: protocol,
pooling, layers, seeds, controls. Then `pip install visbench`.

**Narration.**
That is what VisBench is for. Not a winner, but a number you can compare,
with what it is comparable to written beside it: the protocol, the pooling,
the layers, the seed sweep, and the controls that say what a board is really
measuring.
