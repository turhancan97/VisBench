# Controls

Records that are **rankable against the corpus but must not be listed beside
it**, and one file per question they answer.

The distinction is not comparability. Every record here passes
`comparability_key` against its corpus board — `resolution.jsonl`'s five land
in the *identical* group as the twelve backbones they were run to explain. It
is what a row means. `results/corpus/visbench.jsonl` answers "what does this
backbone score", and every row in it is a model somebody might choose. A
control answers "what happens when one thing about one backbone changes", and
listing it as a thirteenth competitor invites exactly the reading it was built
to prevent.

This is the same instinct as the standing rule never to rank or average across
`finetune`: frozen and fine-tuned numbers are both valid and answer different
questions, so the schema keeps them apart rather than letting a table mix them.

Nothing here feeds a generated table. `scripts/render_tables.py` and
`LEADERBOARD.md` read the corpus only.

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
