# Controls

Records that are **rankable against the corpus but must not be listed beside
it**, and one file per question they answer.

The distinction is usually not comparability. `resolution.jsonl`'s five records
land in the *identical* comparability group as the twelve backbones they were
run to explain, and are kept out anyway. It is what a row means.
(`dpt_head.jsonl` is the exception and is excluded twice over: it changes the
head and the layers, both of which are in the key, so those records form their
own group and *could not* be listed beside the corpus even if it were
desirable.) `results/corpus/visbench.jsonl` answers "what does this
backbone score", and every row in it is a model somebody might choose. A
control answers "what happens when one thing about one backbone changes", and
listing it as a thirteenth competitor invites exactly the reading it was built
to prevent.

This is the same instinct as the standing rule never to rank or average across
`finetune`: frozen and fine-tuned numbers are both valid and answer different
questions, so the schema keeps them apart rather than letting a table mix them.

Nothing here feeds a generated table. `scripts/render_tables.py` and
`LEADERBOARD.md` read the corpus only.

## `dpt_head.jsonl` — is the oracle gate a bound on a DPT head?

Ten records: the five low-level probes on `dinov2_vitb14` and `mae_vitb16` with
`--head dpt --layers 2 5 8 11`, against the corpus's linear numbers over the
same frames.

**Why it was needed.** `DenseTrainingTask.evaluate_oracle` has refused two
probes — photometric superpixels and BSDS500's — and it models a **linear** head
*exactly*: `LinearHead` is a 1x1 convolution per patch plus a bilinear upsample,
which is literally what the oracle computes. Whether it also bounds a DPT head,
which decodes progressively and could place structure *within* a patch, was
recorded as untested. A gate that decides whether work happens should not rest
on an assumption.

**It does not bound a DPT head.** Fraction of the linear oracle reached:

| probe | backbone | grid | linear | DPT |
| --- | --- | --- | --- | --- |
| `corner` | `dinov2_vitb14` | 16 | 78.5% | 94.5% |
| `corner` | `mae_vitb16` | 14 | 82.8% | **102.4%** |
| `edge` | `dinov2_vitb14` | 16 | 70.7% | 90.8% |
| `edge` | `mae_vitb16` | 14 | 81.6% | **103.7%** |
| `keypoints2d` | `dinov2_vitb14` | 16 | 32.2% | 83.4% |
| `keypoints2d` | `mae_vitb16` | 14 | 39.3% | 98.3% |
| `occlusion_edge` | `dinov2_vitb14` | 16 | 59.7% | 78.1% |
| `occlusion_edge` | `mae_vitb16` | 14 | 63.5% | 70.6% |

`orientation` is the same story in degrees: 24.57 -> 16.03 on DINOv2-B and
18.82 -> 12.90 on MAE, against ceilings of 11.02 and 12.18.

**So the gate is a bar for the head VisBench reports, not a bound on what is
possible.** DPT exceeds it outright in two of ten cases and reaches a median of
94.5%. That is a real limit on how the gate should be quoted, and it is now
stated wherever the gate is described rather than left as a footnote.

**It does not reopen BSDS500.** That line closed on a linear ceiling of 0.4193
ODS at a 16x16 grid, against Canny's published 0.60. Scaling by the best ratio
observed here (1.037) gives ~0.435 — still below the weakest classical
baseline, so the argument survives the correction to its premise. See
`visbench/tasks/low_level/README.md`.

**A ranking flipped, which is the finding nobody asked for.** On
`occlusion_edge` the linear board puts `mae_vitb16` ahead (0.3273 against
0.3167) and DPT reverses it (0.4138 against 0.3634). The other four boards keep
their order. **A head is not a neutral magnifying glass** — this is the reason
CLAUDE.md says to report the linear number when comparing representations, now
with a demonstration attached rather than only an argument.

**The ceiling is a property of the grid, not the head**, and these records
confirm it: every `ceiling_*` here is bit-identical to the corpus's linear one
for the same backbone. That is by construction — `evaluate_oracle` never sees a
head — and it is asserted rather than assumed.

**Why this is a control and not a corpus board.** `comparability_key` includes
`layers` and the head, so these records form their own group; putting them in
the corpus would make those tasks unrenderable, since `board_for` refuses a task
with more than one group. n=2 backbones, chosen as the two 16x16-and-14x14 ViTs
that lead these boards.

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
