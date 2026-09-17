**The release that broke a confound, and then found that the boards it
publishes cannot see what it broke.**

A thirteenth backbone, `dino_vitb8`, and the corpus goes from 204 board cells
to **221** — seventeen boards, thirteen backbones each, schema unchanged at v9,
and **no published value moves**.

It was added to settle one question. Feature resolution is the strongest
structural correlate of nearly every dense board, and until now every fine-grid
backbone in the corpus was also a DINOv2 — so "grid" and "the DINOv2 recipe"
could not be told apart. The existing control, `dinov2_vitb14_196`, could only
ever *lower* DINOv2's grid. `dino_vitb8` raises a **non**-DINOv2's: same
objective, data, width and depth as `dino_vitb16`, 784 tokens against 196.

**The answer reverses the obvious reading, and the first draft of it was
wrong.** Quadrupling the grid moves the published **linear** boards by a
rounding error or the wrong way — which reads as "resolution does not matter"
and was written up that way. The same five probes with a **DPT** head improve
**5 of 5**, by one to two orders of magnitude more. The ceilings say why: they
rise every time, and the share a linear head recovers of them falls every time.
So resolution is causal about what a representation **carries** and close to
invisible in what VisBench **reports**.

That claim then stopped resting on one sibling pair. Correlating token count
against each of the five boards twice — once with the published linear score,
once with the DPT score on the same features — across **all ten ViTs** makes
the grid correlation stronger under a DPT head on **4 of 5** probes (mean rho
+0.388 against **+0.701**; `keypoints2d` +0.096 against **+0.775**), and the
gap is largest exactly where the linear head recovers least of its own oracle
(Spearman **−0.900**).

**Never quote a linear board as evidence that resolution does not help.** That
rule is now in the public board-reading guide, which previously carried only
the control that could lower a grid.

Also here: a guard that control records cannot reach the corpus, after a merge
nearly re-imported three records a previous re-run had deliberately held out.

### Added

- **The resolution finding widened from one sibling pair to all ten ViTs**
  (`scripts/analyse_dpt_control.py --grid`). The `dino_vitb8` board showed that
  quadrupling the feature grid at fixed objective, data, width and depth moves
  the published linear boards by a rounding error and the DPT rows by one to
  two orders of magnitude more — on **n=2**. The DPT control already held a DPT
  score beside every linear one for ten ViTs spanning 49, 196, 256 and 784
  tokens, so the same question can be asked at n=10 over data already
  committed, with no new run.

  Correlate token count against each of the five boards twice, once with the
  published linear score and once with the DPT score on the same features, and
  **the grid correlation is stronger under the DPT head on 4 of 5 probes**,
  mean rho **+0.388 → +0.701**. `keypoints2d` moves +0.096 → **+0.775**: the
  difference between "this board does not rank by resolution" and "this board
  ranks by resolution and its head cannot read it".

  **The mechanism is visible in the same table.** Average each probe's linear
  score as a share of its own oracle across the ten backbones, and it runs
  opposite to how much the correlation moves — Spearman **−0.900**. `corner`
  already recovers 77.6% linearly and does not move at all (+0.000);
  `keypoints2d` recovers 35.5% and moves most (+0.679). n=5 probes, so that is
  a mechanism which fits rather than one established.

  The ViT group only: a CNN's DPT run reads a finer map than its linear one, so
  its two rows are not two readings of one grid. `TOKENS` is copied from
  `analyse_board_correlates.STRUCTURE` for the reason that file copies
  `HEADLINE_METRICS`, and a test pins the copy against both the source table
  and the backbones the control actually ran.

- **A guard that control records never reach the corpus**
  (`tests/results/test_controls_stay_out_of_the_corpus.py`). `results/controls/`
  holds records deliberately kept out of `results/corpus/visbench.jsonl`, and
  one merge nearly put three of them in: `scripts/merge_corpus.sh` merges
  everything in `results/corpus/parts/`, that directory still held the
  2026-09-10 files from the schema-v8 `training` re-run, and three of those are
  the A100 `fine_grained_classification` cells the re-run **held out** because
  publishing them drops `convnext_base` below `resnet50` on the CUB board.

  **The script's own protection is blind to this by construction.** It
  deduplicates by exact JSON line, which stops a record already *in* the corpus
  being added twice and does nothing about one deliberately kept *out*. A guard
  inside the merge could not help either: it would read the same `parts/`
  directory it is merging, and a directory containing an excluded record is
  self-consistent — the same shape as the corpus-matrix guard that could not see
  its own short probe list.

  So the guard asserts the **outcome**, not the route, at two strengths of
  match: exact JSON line (what a re-merge produces byte for byte) and
  `(task, backbone, timestamp)` (what a re-serialised or re-schema'd copy
  produces, which the exact check misses — verified by simulating both). Both
  were checked against all six control files before being allowed to reject
  anything.

  `merge_corpus.sh` now documents that `parts/` is an archive rather than a
  queue, and shows the staging-directory recipe that merges only one step's
  parts. No measurement changes.

- **`dino_vitb8`, a thirteenth backbone and its seventeen-cell board — added
  to break one confound, and it broke in the direction that costs something.**
  `vit_base_patch8_224.dino`: the same objective, pretraining set, width and
  depth as `dino_vitb16` at a patch of 8, so **784 tokens against 196** — the
  finest grid in the corpus and the only fine one that is not a DINOv2.

  **Why it was added.** Feature resolution is the strongest correlate of nearly
  every dense board, and until now *the only backbones carrying a fine grid
  were the two DINOv2s* — so grid size, the DINOv2 objective and LVD-142M
  pretraining moved as one variable. The existing resolution control could only
  lower a grid (DINOv2 at 196px) and said so in as many words: "it is one-sided
  because nothing else here can be raised." This raises the other side.

  **What it found, in two halves that say opposite-looking things.** On the
  published *linear* boards, quadrupling the grid at fixed objective, data,
  width and depth gains **+0.2861 on `correspondence`** — where a match can
  only land on a patch centre, so the grid genuinely is the floor — and moves
  every other dense board by a rounding error or the wrong way: `edge` +0.0007,
  `occlusion_edge` −0.0060, `detection` −0.0186, `keypoints2d` −0.0283.

  **The ceilings say why.** A ceiling is computed from the target and the grid
  with no weights involved, so it must rise with resolution, and does. The
  share a *linear* head recovers **falls every time, five of five**: `edge`
  78.9% → 65.1%, `corner` 82.7% → 72.5%, `keypoints2d` 42.7% → 31.0%,
  `occlusion_edge` 56.9% → 48.3%, `orientation` 56.8% → 31.9%.

  **The DPT control reverses the obvious conclusion**
  (`results/controls/dpt_head.jsonl`, 5 cells added). With a DPT head the same
  finer grid helps **5 of 5** rather than 3 of 5, by one to two orders of
  magnitude more — `edge` +0.1365, `keypoints2d` +0.1520, `corner` +0.0933,
  `occlusion_edge` +0.0237, `orientation` +2.96 deg — and that head sits at
  88–97% of the ceiling at *both* grids. So the resolution correlation **is**
  causal about what the representation makes available, and is simply
  **invisible to the readout VisBench reports**: one affine map per patch
  extracts a falling share of a growing target and the two cancel.

  Quoting a linear board as evidence that resolution does not help is therefore
  a mistake — one this changelog entry made in its first draft, before the
  control was run. It is the sharpest case of the DPT control's standing
  lesson: a head decides not just the ordering but whether an effect is visible
  at all. `CORPUS_FINDINGS.md` carries the reading and the caveats, including
  that the controlled half rests on a single sibling pair.

  Corpus **360 → 379 records**, `LEADERBOARD.md` **204 → 221 cells**, 17 boards
  at 13 backbones each; the DPT control goes 55 → 60 records and the split
  control 24 → 26, so both again cover every corpus backbone. No existing
  measurement changes.
