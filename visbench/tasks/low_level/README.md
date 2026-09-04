# Low-level tasks

Low-level = signal-level properties recoverable largely without object or scene
understanding.

This folder was a documented placeholder from v0.1 until **step 6d-1**, which
added its first entry. **Step 6d-2** added the second, and moved the
reconstruction-derived Taskonomy domains from "refused" to "supported" by wiring
up `mask_valid/`. The third, corner detection, is the first probe in VisBench
whose target is **computed from the image** rather than read from disk, so it
runs on any image folder with no download. Gradient orientation is the fourth —
also computed from the image, but a *direction* rather than a magnitude.

## Implemented

| Task | Module | Dataset | Protocol |
|---|---|---|---|
| Edge detection | `edge.py` (`EdgeTask`) | Taskonomy `edge_texture` | `visbench_edge_regression` |
| 2D keypoint detection | `keypoints.py` (`Keypoint2DTask`) | Taskonomy `keypoints2d` | `visbench_keypoint2d_regression` |
| Corner detection | `corner.py` (`CornerTask`) | **any image folder** — the target is computed | `visbench_shi_tomasi_regression` |
| Gradient orientation | `orientation.py` (`OrientationTask`) | **any image folder** — the target is computed | `visbench_structure_tensor_orientation_regression` |

The first three are dense magnitude regression and share
`visbench/tasks/magnitude_base.py` (`DenseMagnitudeTask`): one channel, identity
activation, L1 loss, scored by per-image Pearson correlation (plus RMSE and MAE).
Quote the correlation — it is invariant to scale and offset, so a probe
predicting the split's mean everywhere scores 0 on it while still achieving a
small RMSE.

They are three probes rather than one because an edge response fires along
intensity *contours* and a keypoint response at *corners and blobs*, and a
backbone can be good at one and weak at the other. Distinct metric keys and
distinct `protocol` strings are what stop the numbers being pooled.

**Gradient orientation is the odd one out.** Its target is a *direction*, not a
magnitude, so it cannot use `DenseMagnitudeTask`: it predicts a 2-channel unit
vector `(cos 2θ, sin 2θ)` (the angle is mod π, so the double angle handles the
wrap), the loss is a coherence-weighted angular error, and the metric
`orientation_error` is degrees of angular error, not a correlation. See
`orientation.py`. It measures phase — per-image `|r|` with the corner and edge
targets is under 0.09, where those two sit at 0.53 — which is exactly the gap
the DoG-blob candidate could not fill (it landed at 0.51 with `corner`).

## Corner detection, and the target that is computed rather than read

`corner.py` is the first probe here whose target needs **no dataset**. It is
Shi-Tomasi cornerness — the smaller eigenvalue of the Gaussian-windowed
structure tensor — computed from the RGB frame at read time by
`visbench/data/derived.py`. Any folder of photographs runs it.

Three things were settled by measurement while building it, over 60 Taskonomy
val frames at 224px.

**The operator is Shi-Tomasi, not Harris, and the tail is why.** Share of target
mass in the strongest 1% of pixels, against the ~0.10 of the two probes above
and the 0.46 that stopped `edge_occlusion` ranking:

| response | tail@1% |
|---|---|
| Harris `R`, clipped at 0 | 0.52 |
| Harris `\|R\|` | 0.33 |
| Shi-Tomasi λ_min | **0.27** |

All three are too concentrated raw. λ_min starts closest, is non-negative by
construction, and has no `k` — one fewer free parameter making "Harris corners"
a family rather than a definition. `log1p(1e4·λ_min)` lands the tail at **0.089**
and the frame mean at **0.593**, satisfying 6d-2's tail criterion and 6d-1's
"an L1 target must be of order 1" at the same setting.

**The target overlaps with the edge target, measurably, and this is stated
rather than hidden.** Per-image correlation:

| target pair | mean r |
|---|---|
| `edge_texture` vs `keypoints2d` — two probes shipped separately | **0.147** |
| corner vs `keypoints2d` | 0.271 |
| **corner vs `edge_texture`** | **0.519** |

So this target is *more* redundant with edges than the two Taskonomy probes are
with each other. The overlap is intrinsic, not an artifact of the compression:
it holds at 0.46–0.54 across eight transforms including near-linear ones,
because a corner is a pixel whose gradient is large in two directions and an
edge map is gradient magnitude. **A corner score and an edge score are not
independent evidence about a backbone.**

**Correlated targets can still rank differently, and that is what earns the
probe its place.** 600 train / 600 val, linear head, ten epochs, on the *same
frames* the edge probe uses so only the target differs:

| backbone | `corner_correlation` | rank | `edge_correlation` | rank |
|---|---|---|---|---|
| dinov2_vitb14 | **0.6526** | 1 | 0.4481 | 3 |
| dinov2_vits14 | 0.6512 | 2 | 0.4558 | 2 |
| clip_vitb16 | 0.6227 | 3 | **0.4565** | 1 |
| clip_vitb32 | 0.5367 | 4 | 0.3834 | 4 |
| resnet18 | 0.5014 | 5 | 0.3430 | 6 |
| resnet50 | 0.4923 | 6 | 0.3549 | 5 |

Spread **0.1603**, wider than the edge probe's 0.1136, and the ordering is not
the same one: **CLIP-B/16 is first on edges and third on corners**, and the two
ResNets swap. That is the "a backbone can be good at one and weak at the other"
claim demonstrated between two probes whose targets correlate at 0.52.

Those six numbers are in the committed corpus as of **8b**, and the frames they
ran on are pinned there: `scripts/stage_corner_frames.py` symlinks the first 600
rows of each Taskonomy `tiny` split list into the flat `<split>/images/` layout
a derived-target dataset reads. **That is the same 600 the edge rows above use,
verified set-equal rather than assumed** — which is the only thing that makes
the side-by-side ranking in this table a comparison rather than a coincidence.
A derived target has no data of its own, so a board for one has to name a set;
running the probe still needs no download.

**Do not read the DINOv2-S/B gap of 0.0014 as a failure to rank.** It nearly was
read that way here. The occlusion-edge case that "scored 0.088 and could not
separate two backbones" was flat *across all six*, which is what made it
meaningless; the edge probe's own top two differ by 0.0007. Ask about the spread
over the full set, not one pair.

**None of the three is BSDS500's protocol**, and a record saying so is the point of the
`protocol` field. BSDS is the canonical edge benchmark, but ODS/OIS/AP matches
predicted edge pixels to several annotators' by bipartite correspondence after
non-maximum suppression, swept over thresholds. Borrowing a protocol is only
worth it if borrowed exactly — see `NOTICE`, and the depth probe's 256-bin
expectation, which a from-memory reconstruction would have turned into scalar
regression. Adding BSDS properly is a step of its own.

**That step has started, and its first third is done.**
`scripts/fetch_bsds500.py` and `visbench.data.BSDS500Dataset` read the 500
images with *every* annotator's boundary map. Three things about that data decided the design, all
measured over all 500 rather than assumed:

- **Two orientations and nothing else** — 348 images are 481x321 and 152 are
  321x481 — so native-resolution batching is a grouping problem, not an
  arbitrary-size one.
- **The annotator count varies**, 4 to 9, mode 5. The stack is ragged *across*
  images, so no fixed `A` is promised anywhere.
- **The annotators disagree a lot**: per image the densest marks a median of
  **1.92x** as many boundary pixels as the sparsest, 4.70x at the 95th
  percentile. That spread is why the protocol credits a prediction matching
  *any* annotator, and why the dataset refuses to hand back one "true" boundary
  map. `target()` returns the mean as a training convenience and says in its own
  docstring that it is not the scoring ground truth.

**Nothing resizes or crops**, which is the one place this dataset breaks the
house rule. Every other dense dataset here centre-crops to 224 square because a
VisBench number only has to be comparable with other VisBench numbers; a BSDS
number exists to be comparable with the *published* ones, which are scored at
native resolution. There is no `image_size` argument and adding one would
forfeit the only reason to add the dataset. It also means `DenseTrainingTask`'s
square-map assumption has to be faced by the probe step.

**`bench/` and `grouping/` are deliberately not fetched.** Neither carries a
licence, so the ODS/OIS/AP implementation must come from the paper rather than
from `correspondPixels` — the position `NOTICE` already takes on probe3d's
CC BY-NC code. The fetch script enforces it by extension rather than leaving it
to intent. **That matcher was the remaining risk and it is now written**, in
`visbench/metrics/boundary.py` — ODS, OIS and AP, with Zhang-Suen thinning
because `scikit-image` is not a dependency.

**It reproduces the published human agreement**, which is the check that
matters: BSDS500 quotes human ODS at **0.80** on the test split, and scoring
each annotator against the others leave-one-out gives **0.8030** on test and
**0.7870** on val. Self-consistent unit tests could not have caught a
misunderstanding of what BSDS measures; this could.

Two things were measured rather than assumed:

- **The cheap matcher was refused on evidence.** Only cardinality reaches the
  score, so an arbitrary maximum-cardinality matching (Hopcroft-Karp, 725x
  faster) looks equivalent — but precision counts predictions matched by *any*
  annotator, and different maximum-cardinality matchings put different
  predictions in that union. It moved precision by up to **0.013** on real
  images, in both directions. Greedy is worse: not even maximum-cardinality.
- **The padding goes on the smaller side.** The matching is symmetric, but the
  solver needs one dummy column per row; predictions outnumber annotator pixels
  at nearly every threshold, and putting them in the rows measured **6.5x
  slower** for an identical answer. Component decomposition was tried first and
  abandoned — the components do not stay small, one real image had a single
  component of 16,348 nodes.

### Why there is no BSDS probe, and no board

**The line stops at the metric, on the oracle gate's evidence.** The gate above
asks what a probe could score *if the features contained the answer*; run on the
BSDS target it says a linear probe on the features every corpus backbone
produces cannot reach the published range. BSDS500 val, 100 images, the
published 99-level sweep at `max_dist` 0.0075, the consensus map pooled to a
square grid and bilinearly upsampled back:

| feature grid | input on a /14 ViT | ODS | AP | P@ODS | R@ODS |
|---|---|---|---|---|---|
| **16x16** | **224px — every corpus backbone** | **0.4193** | 0.2430 | 0.7441 | 0.2919 |
| 24x24 | 336px | 0.5810 | 0.4092 | 0.8567 | 0.4395 |
| 32x32 | 448px | 0.6966 | 0.5447 | 0.9216 | 0.5599 |
| 48x48 | 672px | 0.8105 | 0.6944 | 0.9664 | 0.6979 |
| 64x64 | 896px | 0.8725 | 0.7842 | 0.9804 | 0.7860 |
| 128x128 | — | 0.9534 | 0.9252 | 0.9885 | 0.9206 |
| *no grid* | — | *0.9999* | *0.9897* | *1.0000* | *0.9999* |

The last row is the sanity anchor and it is doing real work: scored with no
bottleneck at all the consensus map reaches **0.9999**, so the ladder above is a
property of the *grid* and not of this implementation failing to reach high ODS.

For scale: human agreement is **0.787** on this split (0.803 on test), and the
published detectors this benchmark exists to rank sit at roughly **0.60**
(Canny), **0.73** (gPb) and **0.79** (HED).

**So the ceiling at 224px is 0.42 — below the weakest classical baseline in the
literature.** An actual probe would land some way under that. The number would
be a real measurement, but it could not be compared with any published BSDS
number, and comparability with the published numbers is the *entire* reason this
dataset was worth adding rather than reusing the Taskonomy edge probe. A board
whose ceiling is beneath Canny's score would invite exactly the misreading the
`protocol` field exists to prevent.

**Two things this finding is not.** It is not the superpixel case: that target
scored 0.02-0.04 against probes scoring 0.18-0.65 *in the same metric*, whereas
this is a ceiling in ODS and the shipped probes' oracle range is in Pearson
correlation — the two do not compare, and an earlier draft of this section wrongly
set 0.42 against the 0.25 that rejected superpixels. And it is not a claim that
the representation lacks boundary information: `edge` and `corner` both rank
backbones perfectly well on the same 16x16 grid. It is a claim about a **1px
human-drawn target evaluated at native resolution**, which is a much finer thing
to ask of one vector per 14px patch.

The other obstacle, unchanged and now moot: BSDS is scored at native 321x481
while `DenseTrainingTask` assumes square target maps.

### If the BSDS line is picked up again

Two routes, in cost order.

- ~~**Test whether a DPT head beats the pooling oracle.**~~ **Done, and
  widened 2026-09-04 to the whole corpus. It does beat it — but only one
  backbone does, and not by enough.** Across the five low-level probes and the
  **nine** twelve-block ViTs, a DPT head reaches **54-104%** of the linear
  oracle (median 83%) and exceeds it in **2 of 45 cells**, both `mae_vitb16`
  (`results/controls/dpt_head.jsonl`). Not one of the other eight ViTs exceeds
  it anywhere. So the gate is a bar for the head VisBench reports rather than a
  bound on what is achievable, and that correction is made wherever the gate is
  described — but the exception looks specific to the one row trained by masked
  *pixel* reconstruction rather than general to decoders.

  **It does not reopen this line.** Scaling the 0.4193 linear ceiling by the
  best ratio observed anywhere in that group (1.038) gives ~0.435 ODS — still
  below Canny's published 0.60, so the argument for closing survives the
  correction to its premise. A DPT board here would also be a *different*
  comparability group from every published VisBench dense number, which is a
  second reason it would not be the board this line wanted.

  **The CNN half of that control cannot be read as licence either**
  (`dpt_head_cnn.jsonl`). Its DPT gains are much larger — up to 2.80x — but
  `_grid_of` takes the *finest* requested map, so a ResNet's DPT run is bounded
  by a 56x56 oracle rather than the 7x7 one its linear number was measured
  against. The bottleneck moved with the head, which is the opposite of what
  this route needed to establish, and nothing there exceeded its own oracle.
- **Run at a finer grid, and accept what it costs.** BSDS reaches the published
  range around 32x32, i.e. 448px on a /14 ViT. DINOv2 can do this — it
  interpolates its position embeddings inside its own forward pass. open_clip
  cannot interpolate at all and timm needs `dynamic_img_size`, both recorded in
  `CLAUDE.md`, so this buys a **DINOv2-only** board. A board that cannot compare
  backbones is not a corpus board, and it would confound resolution with
  representation, which is the confound `results/controls/` exists to separate.

What *is* shipped is a validated ODS/OIS/AP implementation that reproduces the
published human agreement. Anyone with a real edge detector can score against
it; it needs no probe to be useful.

### The mid-level twin

`visbench/tasks/mid_level/occlusion_edge.py` (`OcclusionEdgeTask`) shares every
line of this implementation and reads Taskonomy's `edge_occlusion` instead. It
is **mid-level** because an occlusion edge is a depth discontinuity and
recovering one needs scene geometry, where a texture edge is an intensity
discontinuity and does not. Running both on the same frames is about as clean a
comparison of the two tiers as VisBench offers.

It is also the cautionary case for anyone adding a third magnitude probe: its
target has 46% of its total mass in the strongest 1% of pixels, against ~0.10 for
the two above, and at that tail the L1 loss and the Pearson metric pull in
opposite directions. The linear-target probe scored 0.088 and could not tell
DINOv2-S from DINOv2-B; under a `log1p` target it scores 0.29/0.32 and ranks
them. **Check the tail before assuming this protocol transfers.**

## Still candidates

| Task | Notes |
|---|---|
| Optical flow | Needs image pairs and a flow head. `PairViewDataset` already expresses the pairing (see correspondence), so the flow head is the real cost. No flow dataset is assumed present. |
| Texture / reflectance | Intrinsic-image decomposition; ground truth is scarce outside synthetic data. **Taskonomy does not ship a reflectance domain**, so this is not unblocked by 6d-2. `principal_curvature` and `reshading` are present and are still refused, but no longer for want of a mask — see below. |
| Image quality assessment | No-reference IQA against human MOS ratings. Closest in shape to mid-level similarity, which is zero-shot; IQA is not. |
| ~~Edge detection on BSDS500~~ | **Dataset and metric shipped; the probe was refused by the oracle gate.** `scripts/fetch_bsds500.py` + `BSDS500Dataset` + `visbench.metrics.boundary`, reproducing the published human ODS. A linear probe's ceiling at 224px is 0.4193 ODS, below Canny's published 0.60, so no board. See "Why there is no BSDS probe" below. |
| ~~Superpixel / texture segmentation~~ | **Built and rejected**, 2026-08-28. Needs no dataset, passed every pre-measurement, and scored 0.021–0.043 — see "The superpixel rejection" below. |
| Color constancy / illuminant estimation | A per-image scalar/vector target rather than a dense one, and it needs measured illuminant ground truth. |
| Vanishing point / line detection | Published as a Taskonomy domain, but **not in the copy on this machine** — that download carries eight domains and this is not one. |

### Derived targets — what is done and what remains

`visbench/data/derived.py` is the machinery `corner` left behind. A magnitude
derived target is a `ShiTomasiResponse`-shaped class plus a `DenseMagnitudeTask`
subclass; a vector one — `OrientationResponse` — needs its own small task base,
which `orientation.py` now provides. **Photometric superpixels** was the last
candidate needing no dataset, and it was built and rejected — see below. Nothing
remains on this list that does not need a download first.

Five cautions. The first was paid for by the superpixel rejection and the next
three by `corner` and `orientation`; all four are checks to run **before**
writing the task, and between them they cost one pass over a split and an
afternoon of correlations rather than a per-backbone board.

- **Run the oracle gate.** `python scripts/oracle_ceiling.py --targets <yours>`
  — what the probe could score if the features contained the answer. The
  shipped targets sit at 0.53–0.83 on a 16x16 grid; photometric superpixels sat
  at 0.25 and was built anyway, because this check did not exist. See "The
  oracle gate" above. It is last of the cheap checks and first of the ones that
  can refuse a target outright, because a target nothing can recover cannot rank
  backbones however distinctive it is.
- **Check the tail first.** Every raw corner response was spikier than
  `edge_occlusion`'s 0.46, the case that stopped ranking. A compression is not
  optional, and which one is a measurement. (An *angle* has no tail — the
  orientation target needed no compression, which the pre-measurement confirmed
  before the task was written.)
- **Check the overlap with what already ships, before building.** It costs one
  afternoon of correlations rather than a probe run per backbone. **DoG-blob
  detection was rejected on exactly this**: its target correlated 0.50 with the
  edge target and **0.51 with `corner`**, i.e. as redundant with an existing
  probe as `corner` is with `edge`. `orientation` passed the same check with
  `|r|` under 0.09 against both, because it measures phase.
- **A correlated target can still rank differently** — that is what earned
  `corner` its place despite 0.52 with edge (spread 0.1603 vs 0.1136, CLIP-B/16
  first on edges and third on corners). It is a reason to build and measure, not
  a reason to skip the overlap check.
- **`protocol` must name the generator, not the family.** "Harris corners" is
  not a definition — the k parameter, window, smoothing and non-maximum
  suppression all move the target. Naming the operator is half the fix; the
  other half is that every setting travels in `dataset_params`, so two sigmas
  split into two comparability groups without anyone noticing.

### The superpixel rejection — a probe that passed every gate and measured nothing

Built in full on 2026-08-28 and reverted the same day. SLIC groups the frame by
colour and position; the probe regressed the boundary map of that partition. It
is recorded here because the *pre-measurement passed*, and the way it passed is
the transferable part.

**What it scored**, 200 frames per split, defaults, against the shipped probes
on the same three backbones:

| probe | dinov2_vits14 | clip_vitb16 | resnet50 |
| --- | --- | --- | --- |
| `corner` | 0.6512 | 0.6227 | 0.4923 |
| `edge` | 0.4558 | 0.4565 | 0.3549 |
| `occlusion_edge` | 0.2924 | 0.2558 | 0.1979 |
| `keypoints2d` | 0.2356 | 0.2175 | 0.1792 |
| **`superpixel`** | **0.0434** | **0.0209** | **0.0238** |

Five times below the weakest probe that ships, with a spread of 0.023 in which
ResNet-50 "beats" CLIP by 0.003. `train_loss` was *lowest* for the worst
scorers, which says what happened: the heads learned the mean boundary density
and nothing about where the boundaries are.

**Every check in the gauntlet passed.** Tail mass 0.055, nowhere near
`edge_occlusion`'s 0.46. Overlap with `edge_texture` 0.267 per image, well under
the 0.52 `corner` shipped with. Cross-image `|r|` 0.044 — *below* the edge
target's own 0.060 — so the boundaries genuinely followed the image rather than
SLIC's seeding lattice. Two rival formulations were measured and rejected on the
overlap rule (segment-mean residual at 0.509; log segment area degenerate,
because SLIC equalises segment sizes by construction).

**What the gauntlet did not ask, and now does.** Every check above is about the
*target*: its distribution, and its relationship to other targets. None asked
whether the target is **recoverable from patch features at all**. A SLIC
boundary is one pixel wide and, in a flat region, its position is set by the
lattice seeding and sub-threshold colour noise — information that is not in a
14px patch token in principle, not merely in practice.

A pooled-resolution overlap check was tried and is *not* the missing test: it
measures whether two targets agree at coarse scale, which is a different
question. It also produced a misleading near-veto — the boundary map reads 0.684
against `edge` at a 16x16 grid, but the shipped `corner` target reads **0.781**
there and ranks backbones differently anyway, so it cannot refuse anything.

**The missing check is the oracle gate below**, added afterwards and calibrated
against this rejection.

## The oracle gate — is the target recoverable at all?

`scripts/oracle_ceiling.py`, over
`DenseTrainingTask.evaluate_oracle`. Pool the target to the feature grid,
upsample it back, score it with the probe's own metric: that is what a *perfect*
backbone would make available to the head, because a dense probe sees one
feature vector per patch and its head interpolates up from there. Signal finer
than a patch is not merely hard to predict, it is **absent from the input**.

It needs no backbone, no features and no fitted head, so it costs one pass over
a split rather than a board — which is the whole argument for running it before
writing the task rather than after. `CorrespondenceTask.evaluate_ceiling` is the
same idea, arrived at for the same reason.

Measured over the pinned 600 val frames at 224px, `--limit 600`:

| target | verdict | 16 (ViT/14) | 14 (ViT/16) | 7 (ResNet) | best on its board |
|---|---|---|---|---|---|
| `corner` | ships | **0.8316** | 0.8053 | 0.6685 | 0.6669 |
| `keypoints2d` | ships | **0.6976** | 0.6674 | 0.4728 | 0.2850 |
| `edge` | ships | **0.6336** | 0.6106 | 0.4977 | 0.4982 |
| `occlusion_edge` | ships | **0.5301** | 0.5150 | 0.4336 | 0.3273 |
| `superpixel` | **rejected** | **0.2519** | 0.2133 | 0.1109 | 0.0434 |

Those are the three grids the corpus backbones actually hand a head at 224px.
The last column is read off the twelve-backbone corpus, not off the
six-backbone tables higher up this file, and it is `mae_vitb16` in three rows of
four. `superpixel` never had a board — 0.0434 is the best of the three
backbones it was measured on before being dropped.

`orientation` is on the same table in its own unit — 11.02° / 12.18° / 18.57°
against a 45° chance floor and a board spanning 18.8–31.2°.

**The gap between 16 and 7 is why the ceiling now travels with every score.**
Since 2026-09-01 these five probes emit `ceiling_*` alongside their metrics
through `context_metrics`, exactly as `correspondence` does: `corner`'s ceiling
is 0.8316 against a ViT/14 and 0.6685 against a ResNet, so ranking the two
without saying so invites a reader to attribute a grid difference to a
representation. It is computed from the run's own feature grid, read off the
features rather than from a declared patch size, and it costs a pass over the
split's *targets* — not its features, which are the expensive half.

**A ceiling is never a score.** Do not rank on one, average one, or divide by
one: it falls with the grid, so a board ordered by ceiling is a board ordered by
feature resolution. **The committed corpus now carries them**: the five boards
were re-run on 2026-09-01 so their records hold `ceiling_*` (and, since those
records predated schema v8, a `training` block too). They were *regenerated*,
not backfilled — every value in a record was produced by a run.

**The bar.** Every target that ships sits at 0.53–0.83; the one that was built
and thrown away sits at **0.25**, less than half the weakest of them, and at
grid 7 it is 0.11 against their 0.43–0.67. A candidate down there should not be
built. There is no magic constant here and there should not be — read it the way
the tail rule is read, as a working range with a known-ruinous case beside it.

**Do not turn it into a denominator.** The tempting next step is to score a
probe as a fraction of its oracle, and it does not work: `corner` reaches 80% of
its oracle and `keypoints2d` only 41%, yet both rank backbones perfectly well.
The gate is the oracle's own height, not the ratio. The oracle is in any case an
*achievable* score rather than a proven bound — it reconstructs the target from
its patch means, which is near-optimal and not provably maximal, unlike
`evaluate_ceiling`'s minimum over candidate patch centres.

**The upsample is bilinear because `LinearHead`'s is.** A nearest upsample would
raise every number and make the gate more permissive than the heads it protects
— the wrong direction for a check whose job is to reject. It is why even a
target built from hard grid cells scores about 0.88 rather than 1.0.

**A probe opts in.** `DenseMagnitudeTask` and `OrientationTask` declare
`oracle_prediction`; every other dense probe raises, naming the way to opt in.
Pooling is only the right bottleneck for a target that averages — the mean of
classes 1 and 15 is class 8, and a bin-expectation depth target is not the
quantity its head emits — and a silently defaulting oracle would return a
confident number about nothing. That is worse here than no number, because this
gate exists to *stop* work.

The rejected superpixel target is reconstructed inside the script rather than in
the package, so the gate keeps a known negative to be calibrated against. It is
the original: at SLIC's defaults it reproduces the published pre-measurements to
0.059 tail (against 0.055) and 0.271 overlap with `edge_texture` (against
0.267).

**What survived.** `DerivedTargetDataset` now memoises computed targets — see
`MEMO_LIMIT`. `CachedFeatures.__getitem__` calls `dataset.target(index)` on every
access, so a ten-epoch streaming run had been recomputing every target ten times;
`corner` and `orientation` had been paying that quietly.

### Why `principal_curvature` and `reshading` are still refused

Not the mask — that is wired up as of 6d-2 and these two would get it. Each is
blocked on a **task** decision instead, and `TaskonomyDataset` refuses them by
name with the reason rather than reporting an unknown domain:

- `principal_curvature` packs two principal curvatures plus an unused channel
  into RGB, and which is which, and in what units, is not established here.
  Reading it as a magnitude map would train and score against a quantity this
  library cannot name.
- `reshading` is a re-rendered shading image, so its target is an RGB frame
  rather than a per-pixel measurement, and no probe here consumes one.

Contributions welcome — see the head interface in `visbench/heads/`, which is a
deliberate extension point, `visbench/tasks/dense_base.py`, which supplies
everything a trained dense probe needs bar four methods, and
`visbench/tasks/magnitude_base.py`, which supplies the rest if the target is a
magnitude map.
