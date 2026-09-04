# Controls

Records that are **rankable against the corpus but must not be listed beside
it**, and one file per question they answer.

The distinction is usually not comparability. `resolution.jsonl`'s five records
land in the *identical* comparability group as the twelve backbones they were
run to explain, and are kept out anyway. It is what a row means.
(The two `dpt_head*.jsonl` files are the exception and are excluded twice over:
they change the head and the layers, both of which are in the key, so those
records form their own groups and *could not* be listed beside the corpus even
if it were desirable.) `results/corpus/visbench.jsonl` answers "what does this
backbone score", and every row in it is a model somebody might choose. A
control answers "what happens when one thing about one backbone changes", and
listing it as a thirteenth competitor invites exactly the reading it was built
to prevent.

This is the same instinct as the standing rule never to rank or average across
`finetune`: frozen and fine-tuned numbers are both valid and answer different
questions, so the schema keeps them apart rather than letting a table mix them.

Nothing here feeds a generated table. `scripts/render_tables.py` and
`LEADERBOARD.md` read the corpus only.

## `dpt_head.jsonl` and `dpt_head_cnn.jsonl` — is the oracle gate a bound on a DPT head?

**Two files, because they are two comparability groups and two questions.**
`dpt_head.jsonl` holds the five low-level probes against the corpus's **nine
twelve-block ViTs** at `--head dpt --layers 2 5 8 11`;
`dpt_head_cnn.jsonl` holds them against the **three CNNs**, each reading its
own last four feature stages. `layers` is in `comparability_key`, so they could
not share a file even if the readings were interchangeable — and they are not.
`scripts/build_dpt_control.sh` runs either group and
`scripts/analyse_dpt_control.py` reads them.

**Why it was needed.** `DenseTrainingTask.evaluate_oracle` has refused two
probes — photometric superpixels and BSDS500's — and it models a **linear** head
*exactly*: `LinearHead` is a 1x1 convolution per patch plus a bilinear upsample,
which is literally what the oracle computes. Whether it also bounds a DPT head,
which decodes progressively and could place structure *within* a patch, was
recorded as untested. A gate that decides whether work happens should not rest
on an assumption.

**The first answer, on two backbones, was right and much too general.** It read
"a DPT head reaches 70-104% of the linear oracle and exceeds it in two of ten
cases", which sounds like a property of decoders. Both exceedances were
`mae_vitb16`, and with two backbones there was no way to tell a property of the
head from a property of that row. Widening to nine settles it.

### The ViT group — nine backbones, one grid, only the head moves

All twelve blocks of a ViT share one grid, so `--layers 2 5 8 11` hands DPT four
maps at the **same** resolution: any gain is decoding rather than finer input,
and exceeding the oracle means structure was placed *within* a patch. Every
`ceiling_*` here is bit-identical to the corpus's linear one for the same
backbone — asserted, not assumed, and it is what makes this a clean control.

| | published (n=2) | measured (n=9) |
| --- | --- | --- |
| fraction of the linear oracle, magnitude probes | 70.6-103.7% | **60.4-103.8%**, median 84.5% |
| including `orientation` | not quoted | **53.9-103.8%**, median 83.1% |
| exceeded the oracle | 2 of 10 | **2 of 45** |
| boards whose leader changed | 1 of 5 | **2 of 5** |

**The gate bounds eight of the nine ViTs.** The only two cells that exceed it
are `mae_vitb16` on `edge` (103.8%) and `corner` (102.6%); not one of the other
eight exceeds it on any of the five probes. Mean fraction per backbone:

| backbone | mean of 5 | min | max |
| --- | --- | --- | --- |
| `mae_vitb16` | **94.0%** | 70.5% | 103.8% |
| `dino_vitb16` | 85.6% | 69.5% | 98.1% |
| `sam_vitb16` | 84.0% | 66.3% | 98.0% |
| `dinov2_vits14` | 83.7% | 70.4% | 95.1% |
| `dinov2_vitb14` | 83.0% | 68.9% | 94.4% |
| `supervised_vitb16` | 78.3% | 60.4% | 93.2% |
| `clip_vitb32` | 78.0% | 67.9% | 87.6% |
| `clip_vitb16` | 75.3% | 60.9% | 89.5% |
| `siglip_vitb16` | 69.1% | 53.9% | 84.2% |

So the correction to the gate's description stands — it is a bar for the head
VisBench reports, not a bound on what is achievable — but **the exception looks
backbone-specific rather than head-specific.** MAE is the one row trained by
masked *pixel* reconstruction, which is a plausible reason for sub-patch
structure to survive in its features, and it is 8 points clear of the next
backbone on this measure. Do not quote the 104% as what decoders do.

**Two boards change leader, not one.** `occlusion_edge` reverses as published
(`mae_vitb16` 0.3630 → `dinov2_vitb14` 0.4131), and **`keypoints2d` also
reverses** (`dino_vitb16` 0.5784 → `mae_vitb16` 0.6576) — which two backbones
could not show, because `dino_vitb16` was not in the control. The published
"the other four boards keep their order" is therefore wrong at nine.

### Reordering, counted only over pairs that are actually separable

A DPT head is **an order of magnitude less reproducible than a linear one**,
which this control measured on itself: re-running the original ten cells three
days later moved them by **2e-4 to 3.3e-3** relative, where the linear boards
reproduce at ~1e-7 on four of five probes. Every ceiling was bit-identical,
which again is by construction.

That matters for how far the ordering may be read. Every ViT board here has
adjacent pairs closer than that drift, so a raw discordant-pair count includes
coin flips. Counting only pairs **both** boards separate by more than their own
drift:

| board | discordant / decidable | too close to call |
| --- | --- | --- |
| `edge` | 4 / 35 | 1 |
| `keypoints2d` | 8 / 35 | 1 |
| `occlusion_edge` | 5 / 36 | 0 |
| `corner` | 3 / 33 | 3 |
| `orientation` | 4 / 35 | 1 |
| **all five** | **24 / 174 (14%)** | 6 of 180 |

**Both leader changes survive comfortably** — 0.0501 and 0.0792 against a drift
of ~0.002 — and so does the standing rule they support. What is *not* readable
is a DPT board's fine ordering: quote these to three decimals, the same
concession `detection` already carries for a different reason.

### The CNN group — the head moves *and* so does the bottleneck

A CNN's stages are at 56/28/14/7 on a ResNet at 224px, and `_grid_of` takes the
**finest** requested map, which is right: a DPT head is bounded by its finest
input. The consequence is that a CNN's DPT run is **not the same experiment**.
Its oracle moves with it — `edge` goes from 0.4977 at 7x7 to **0.8727** at
56x56 — so the two runs' fractions are not two readings of one scale, and
`gain` (DPT score over linear score) is the only comparable column. All 15
cells are marked `*` by the analysis script for this reason.

| probe | `convnext_base` | `resnet18` | `resnet50` |
| --- | --- | --- | --- |
| `edge` | 1.24x | 1.45x | 1.43x |
| `keypoints2d` | 1.80x | **2.80x** | 2.69x |
| `occlusion_edge` | 1.59x | 1.69x | 1.60x |
| `corner` | 1.18x | 1.38x | 1.42x |
| `orientation` | 1.17x | 1.61x | 1.72x |

**The gains are far larger than any ViT's** (1.08-1.31x there, up to 2.80x
here), and the reason is not that CNN features are better served by a decoder.
It is that **a linear probe on a CNN throws away spatial detail that exists in
the same forward pass**: it reads only the final 7x7 stage, while the 56x56
stage-1 map was computed and discarded. A ViT has nothing equivalent to discard.
Nothing exceeds its own oracle here — max 71.2%, and `orientation` reaches only
11.5-15.8% of a 2.76-degree oracle — which is the other half of the same fact:
the finer bottleneck is much more demanding.

**Three of five boards change leader, and two invert completely.**
`convnext_base` leads the CNNs on the linear `keypoints2d`, `corner` and
`orientation` boards and comes **last** on all three with a DPT head, with
`resnet50` taking every one; rho is **-1.000** on `corner` and `orientation`.
9 of 15 decidable pairs reorder, against 14% for the ViTs. Every gap involved is
far above the drift, so the flips are real — but **n=3**, so two of those rho
values are one swap each. Read this as "head choice can invert a CNN board",
not as a measured effect size.

### What each group licenses

- **The gate's description**, corrected: a bar for a linear head, exceeded only
  by `mae_vitb16` and only twice in 45 cells. State it over the ViT group; the
  CNN group cannot speak to it, because its oracle is not the gate the corpus
  boards are read against.
- **"A head is not a neutral magnifying glass"**, now with 24 reordered pairs
  out of 174 on ViTs and three inverted boards on CNNs, rather than one
  reversal. This is the demonstration behind CLAUDE.md's rule to report the
  linear number when comparing representations.
- **A new caveat on the corpus's dense boards**: they may understate CNNs by
  more than ViTs, and the reason is a choice of which stage a linear head reads
  rather than a property of the representation. See `CORPUS_FINDINGS.md`.

**It still does not reopen BSDS500.** That line closed on a linear ceiling of
0.4193 ODS at a 16x16 grid against Canny's published 0.60. Scaling by the best
ratio observed anywhere in the ViT group (1.038, up from 1.037) gives ~0.435 —
still below the weakest classical baseline, so the argument survives the
correction to its premise. See `visbench/tasks/low_level/README.md`.

**Why these are controls and not corpus boards.** `comparability_key` includes
`layers` and the head, so each group forms its own group; putting either in the
corpus would make those five tasks unrenderable, since `board_for` refuses a
task with more than one group.

## `resolution.jsonl` — is DINOv2's dense lead its grid?

Five dense boards for `dinov2_vitb14_196`: the same weights, the same hub ref,
at 196px instead of 224. That makes its token grid 14x14 rather than 16x16 —
matching every ViT-B/16 in the corpus.

**Why it was needed.** Feature resolution is the strongest correlate of every
dense board (rho +0.50 to +0.96, `scripts/analyse_board_correlates.py
--section structure`), and the only backbones carrying 256 tokens are the two
DINOv2s. So grid size, the DINOv2 objective and LVD-142M pretraining were one
variable, and no dense board could say which of the three it had ranked.

**What it found.** Matching the grid costs DINOv2-B under 3% on every board,
and it keeps its lead over the whole ViT-B/16 pack on both boards it led:

| board | 256 tok | 196 tok | change | rel |
| --- | --- | --- | --- | --- |
| `generic_segmentation` | 0.7556 | 0.7407 | -0.0149 | 2.0% |
| `depth` | 0.7851 | 0.7791 | -0.0060 | 0.8% |
| `surface_normal` (deg, lower better) | 30.1143 | 30.6556 | +0.5413 | 1.8% |
| `edge` | 0.4481 | 0.4363 | -0.0119 | 2.6% |
| `corner` | 0.6526 | 0.6349 | -0.0178 | 2.7% |

Resolution accounts for **21%** of DINOv2-B's lead on `generic_segmentation`
and **7%** on `depth`. On the other three DINOv2-B never led — `mae_vitb16` is
ahead on all of them — so there was no lead to explain.

**Two limits to carry with the number.** The control spans 256 to 196 tokens,
about 1.3x, while the corpus correlation spans 49 to 256, about 5x: it bounds
the slope where the DINOv2-versus-ViT-B/16 comparison actually lives, and says
nothing about the 49-token backbones. And 196px is slightly off DINOv2's
training resolution, so read a drop as "resolution or distribution", not as
resolution alone.

**It is one-sided by necessity.** DINOv2 interpolates its position embeddings
inside its own forward pass, so 196px is its intended use. open_clip does not
interpolate at all and timm needs `dynamic_img_size`, so no ViT-B/16 here can
be *raised* to 256 tokens without changing the model.

Reproduce with:

    RESULTS=results/controls/resolution.jsonl \
    BACKBONES=dinov2_vitb14_196 \
    scripts/build_corpus.sh generic_segmentation depth surface_normal edge corner
