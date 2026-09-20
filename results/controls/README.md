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

**One subdirectory here is a family rather than a file.** `seeds/<probe>.jsonl`
holds a board re-fitted at several seeds (20b), one file per swept probe, and
those are the most dangerous records in this directory: they differ from their
board only in `seed`, which `comparability_key` does not read, so they would
merge *into* the published group rather than beside it.

**One file here is not records at all.** `pose_noise.json` holds a study whose
runs perturb cached features — something no flag expresses and no
`ResultRecord` could honestly describe — so it is JSON rather than a `.jsonl`
of records, and the distinction is deliberate: a record claims a run happened
under a stated configuration, and these runs had no such configuration to
state.

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

### The ViT group answers a second question it was not built for

Added 2026-09-16, no new records: `scripts/analyse_dpt_control.py --group vit
--grid`. The `dino_vitb8` board had just shown, on **one sibling pair**, that
quadrupling the feature grid at fixed objective, data, width and depth moves
the published linear boards by a rounding error while moving the DPT rows by
one to two orders of magnitude more. The ViT group here is ten backbones
spanning 49, 196, 256 and 784 tokens with a DPT score beside every linear one,
so it can ask the same question at n=10 instead of n=2: **does the DPT board
track the feature grid harder than the linear board does?**

| probe | Spearman(tokens, linear) | Spearman(tokens, DPT) | change |
| --- | --- | --- | --- |
| `edge` | +0.322 | **+0.555** | +0.233 |
| `keypoints2d` | +0.096 | **+0.775** | +0.679 |
| `occlusion_edge` | +0.480 | **+0.843** | +0.363 |
| `corner` | +0.665 | +0.665 | +0.000 |
| `orientation` | +0.377 | **+0.665** | +0.288 |
| **mean** | **+0.388** | **+0.701** | **+0.313** |

**Stronger under the DPT head on 4 of 5, never weaker.** The sibling pair's
direction, over ten backbones and four training objectives, so it is not one
pair restated.

**And the gap is where a linear head recovers least**, which is the sibling
pair's ceiling argument seen from the other side. Average each probe's linear
score as a share of its own oracle across the ten, and it runs opposite to how
much the correlation moves — Spearman **-0.900**:

| probe | linear recovers | rho change |
| --- | --- | --- |
| `corner` | 77.6% | +0.000 |
| `edge` | 73.0% | +0.233 |
| `occlusion_edge` | 51.8% | +0.363 |
| `orientation` | 49.8% | +0.288 |
| `keypoints2d` | 35.5% | +0.679 |

`corner` is the control inside the control: the head already recovers 77.6% of
what the grid offers, there is little left for a decoder to uncover, and the
correlation does not move at all. **n=5 probes**, so read the mechanism as one
that fits rather than one that is established.

**The CNN group is excluded deliberately, and not for sample size.** A CNN's
DPT run reads a finer map than its linear one (`_grid_of` takes the finest),
so its two rows are not two readings of one grid — the comparison would be
between two bottlenecks rather than between two heads, which is the same
reason only the DPT/linear *gain* is comparable there.

### What each group licenses

- **The gate's description**, corrected: a bar for a linear head, exceeded only
  by `mae_vitb16` and only twice in 45 cells. State it over the ViT group; the
  CNN group cannot speak to it, because its oracle is not the gate the corpus
  boards are read against.
- **"A head is not a neutral magnifying glass"**, now with 24 reordered pairs
  out of 174 on ViTs and three inverted boards on CNNs, rather than one
  reversal. This is the demonstration behind CLAUDE.md's rule to report the
  linear number when comparing representations.
- **How much of a dense board's grid correlation its own head can see**, which
  is a statement about the readout rather than about the representations: on
  four of five probes, less than a DPT head sees, and least where the linear
  head recovers least of its oracle.
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

## `detection_split.jsonl` — does a board's cluster come from its task or its split?

**Run 2026-09-09 (after 14a-4), 24 records: `detection` on VOC's
`ImageSets/Segmentation` — the *instance* probe's own images — in two sizes.**

### The question

14a-4 found `instance_segmentation` ranking with the mid-level geometry boards
(`occlusion_edge` +0.958, `surface_normal` +0.930, `depth` +0.902) rather than
its own high-level tier (`semantic_segmentation` +0.378, `retrieval` −0.217),
and showed the mask branch is **not** the reason: the `box_map_50` half of the
same runs agrees with the mask half at **+0.986** and is topped by
`occlusion_edge` too. So the box half alone ranks with geometry, while
`detection` — one implementation, one matcher, one metric, one dataset family —
sits at +0.804 with `semantic_segmentation`. What differs is the data.

### The control as first written down was impossible

`CORPUS_FINDINGS.md` named it as "the instance probe's own head on
`ImageSets/Main` at `--limit 600`". That cannot be run. `SegmentationObject`
exists for **2913 images only**, so **141 of the first 600 `Main` train stems
have an instance mask** and the other 459 have no target at all — checked, not
assumed. `Annotations/` carries **17125 XMLs** covering every devkit image, so
the direction has to invert: move the *published* probe onto the *new* probe's
images. That is also the better experiment, since `detection`'s baseline
reading is the one already published.

### The design

| | probe | stems | size |
| --- | --- | --- | --- |
| **A** `full` | `detection` | Segmentation | 1464 train / 1449 val |
| **B** `limit600` | `detection` | Segmentation | 600 |
| **C** *(corpus)* | `detection` | **Main** | 600 |
| **D** *(corpus)* | `instance_segmentation` | Segmentation | 1464 / 1449 |

Three comparisons exhaust the difference between C and D:

| pair | rho | what varies |
| --- | --- | --- |
| **A vs D** | **+0.958** | nothing but box provenance and the mask branch |
| A vs B | +0.818 | training size |
| B vs C | +0.818 | which images |
| **A vs C** | **+0.510** | both |

### The answer: the split, and it is not a small effect

**Moving `detection` onto the instance probe's images moves it into the
geometry cluster.** Nothing about the probe changed:

| | mean vs mid | mean vs high | top partner |
| --- | --- | --- | --- |
| `detection` (published, Main-600) | — | — | `semantic_segmentation` +0.804; `occlusion_edge` +0.483 |
| `detection` (Segmentation-full) | **+0.784** | **+0.018** | `occlusion_edge` **+0.965** |
| `detection` (Segmentation-600) | +0.723 | +0.287 | `generic_segmentation` +0.853 |

And **once images and size match, `detection` and `instance_segmentation` rank
the same board** (+0.958). Box provenance — VOC's hand-drawn XML against boxes
derived from the instance mask — plus the whole mask branch account for about
0.04 of rho, which is nothing.

So 14a-4's published negative claim is confirmed and can be stated positively:
**it is the split.** Neither half explains it alone (+0.818 each) and the two
compound (+0.510).

### The row that shows it most plainly

`mae_vitb16` scores **0.1296** on the published detection board — tenth of
twelve — and **0.3371** on the same probe over the segmentation split, where it
is **first**. This is the standing "a count over a corpus is a fact about that
corpus" caution with a mechanism attached: MAE's position on that board is
contingent on which 600 frames it read.

### What this does and does not license

It does **not** move a published number. Every corpus record stands; what is
contingent is the *reading*.

It **does** caveat the corpus's headline two-cluster finding. `detection` is a
member of the "localised high-level" cluster, and its membership is
split-contingent — so that cluster is a property of these boards as configured,
not of the tasks in the abstract. Mid- and low-level coherence is untouched
here, and the two clusters remain the stable structure; what weakens is any
claim that a *task* belongs to a cluster.

### The thing that could not be checked — now checked

This section used to end "unanswerable from the corpus": the published
detection records predated schema v8 and carried `training: null`, so whether C
underfits relative to A could not be asked. The v8 re-run gave those records a
`training` block, and the answer is **yes**, on every backbone:

| config | mean `train_loss` |
| --- | --- |
| A `seg/full` — 1464 train | **1.3071** |
| B `seg/limit600` — 600 train, segmentation images | 1.4012 |
| C `main/limit600` — the published board | 1.3500 |

A fits better than C on **12/12** backbones (mean −0.0428), so the published
board does underfit relative to the full-split config. The decomposition is
the same shape as the rho one: training *size* is worth −0.0940 (A against B,
12/12), and it is partly offset because the `Main` images are **easier to fit**
than the segmentation ones at equal size — B is worse than C on 12/12, mean
+0.0512.

`scripts/analyse_split_control.py` prints this under "THE FIT". The caveat that
applies to the rho decomposition applies here too: no pair varies one thing
alone, so the two lines bound the size effect rather than isolating it. And
`train_loss` is comparable across these three only because they share a probe,
a loss and a target — it is meaningless between boards.

`scripts/build_split_control.sh` holds the flags,
`slurm/split_control.sbatch` the array, `scripts/analyse_split_control.py` the
reading, and `tests/scripts/test_split_control_scripts.py` pins the two-file
matrix and that no record can reach the corpus.

## `hardware_a100.jsonl` — three cells a re-run could not reproduce

Three records: `fine_grained_classification` on `convnext_base`,
`dino_vitb16` and `dinov2_vitb14`, run on an **A100** during the schema-v8
`training` re-run, where every published corpus number was produced on a
**V100**. They are here rather than in the corpus because they are the three
cells of ninety-six that did not reproduce the value they were re-running.

The other nine cells of that board reproduce their published value **exactly**,
and each of these three was run **twice** on the A100 and gave an identical
number both times — so the A100 is deterministic and the disagreement is with
the silicon, not with the run.

| backbone | published (V100) | A100, twice | `train_top1` |
| --- | --- | --- | --- |
| `convnext_base` | 0.7311 | **0.6836** | 0.9892 |
| `dino_vitb16` | 0.7520 | 0.7587 | 1.000000 |
| `dinov2_vitb14` | 0.8683 | 0.8640 | 1.000000 |

**`train_top1` says which kind of disagreement each one is.** `convnext_base`
is the only cell on the board that does not interpolate — every other backbone
reaches 1.000000 — and it is also the largest mover by a factor of seven. The
other two interpolate and land on a different point of a zero-training-error
plateau, worth 25 and 39 images of 5794.

**Why they are not in the corpus.** Publishing them would drop `convnext_base`
from 0.7311 to 0.6836, below `resnet50` at 0.6943 — a **ranking change on a
published board caused by a variable no record carries**, since the schema has
never recorded which GPU produced a number. The rule the re-run followed was:
*replace a published record only where the re-run reproduces it; otherwise
leave the published record alone and say why.* That is not selecting the
convenient number — it is refusing to let unrecorded hardware move a board.

**The follow-up landed on 2026-09-11**, once `dgx2` came out of `DRAIN`. All
three were re-run on a V100 and **each reproduced its published value exactly**
— 0.731101, 0.751985, 0.868312 — so they merged into the corpus with their
`training` block, and the board now answers the underfitting question for
twelve of twelve. It reads **saturated**: every backbone reaches `train_top1`
1.0000, so its whole spread is generalisation.

**These records stay here, because the three-way comparison is the finding.**
Same code, same data, same seed, two silicons:

| backbone | V100 (published, and re-run) | A100, twice | V100 `train_top1` | A100 `train_top1` |
| --- | --- | --- | --- | --- |
| `convnext_base` | 0.731101 | 0.6836 | 1.000000 | **0.989156** |
| `dino_vitb16` | 0.751985 | 0.7587 | 1.000000 | 1.000000 |
| `dinov2_vitb14` | 0.868312 | 0.8640 | 1.000000 | 1.000000 |

**The largest disagreement is a fit that does not interpolate, not a metric
that wobbles.** `convnext_base` is the only cell on the board that fails to
reach `train_top1` 1.0 on the A100, and it is the only one whose score moves by
more than a point. So the fit diagnostics separate "different hardware" from
"different answer" here exactly as they separated a failing node from a weak
backbone — read `training` before attributing a cross-silicon gap to anything
else.

## `pose_linear.jsonl` — what the head this board departs from would have said

Thirteen records: `relative_pose` on the same NAVI pairs, the same seed and the
same pinned protocol as the published board, with **one affine map** in place of
probe3d's MLP (`--hidden-dims ""`). `scripts/build_pose_linear_control.sh` runs
it, and it is nearly free after the board because the features are the same
ones — what decides that is the *cache root*, not the machine.

**Why it exists.** Every other VisBench board is quoted with the least
expressive head that can express the task, because then a gap between two
backbones is a gap between two *representations*. In practice that has always
come out a linear map: an affine layer, a `LinearHead`, or the 1x1 convolutions
`DetectionHead` and `InstanceHead` are built from. `relative_pose` is the first
board that departs from it, and the honest form of "we used a bigger head" is a
measurement of what the smaller one did.

| backbone | MLP | linear | linear vs floor |
| --- | --- | --- | --- |
| `mae_vitb16` | 22.70 | 63.06 | +3.78 |
| `dinov2_vits14` | 25.37 | 62.94 | +3.91 |
| `dino_vitb8` | 27.10 | 62.94 | +3.91 |
| `dino_vitb16` | 29.04 | 63.27 | +3.58 |
| `dinov2_vitb14` | 29.08 | 63.41 | +3.43 |
| `sam_vitb16` | 35.02 | 64.09 | +2.75 |
| `resnet50` | 37.01 | **65.99** | **+0.85** |
| `convnext_base` | 37.60 | 63.44 | +3.41 |
| `siglip_vitb16` | 38.29 | 63.95 | +2.89 |
| `clip_vitb16` | 40.56 | 63.53 | +3.31 |
| `supervised_vitb16` | 41.43 | 63.71 | +3.14 |
| `resnet18` | 43.01 | 63.95 | +2.90 |
| `clip_vitb32` | 43.14 | 64.00 | +2.85 |

**A linear head is at or near chance on this task, for every backbone.** It
clears the no-feature floor by **0.85 to 3.91 degrees** where the MLP clears it
by 23.7 to 44.2, and `resnet50` at +0.85 is indistinguishable from predicting a
constant. Its whole spread is **3.06 degrees against the MLP's 20.44** — and
this board's reproducibility noise is about **one** degree, so a linear board
would be separating its thirteen rows by roughly three times the noise, where
the MLP board separates them by twenty.

**It underfits, and `train_loss` is what says so rather than the score.**
0.0680 to 0.0729 across all thirteen, flat — against 0.0020 to 0.0057 for the
MLP on the identical features. Flat training loss at 25x the alternative's, on
50,519 pairs against 1,536 input dimensions, is the function class rather than
the sample size.

**And its residual ordering is not the MLP's.** Spearman between the two is
**+0.681**, which sounds like agreement until the top is read: the linear board
puts `dino_vitb8` first and `mae_vitb16` **third**, and lifts `convnext_base`
from eighth to fifth. Ordering a board by differences of a degree or two, when a
degree is noise, is how a ranking gets manufactured.

**The cost of the departure, stated plainly.** A gap between two rows on the
published board is less purely a gap between two representations than elsewhere
in this corpus: a deeper head can compensate for a weaker feature vector, which
is the DPT control's lesson arriving on a board that ships. This file is what
lets a reader see how much, rather than take a sentence for it.

## `pose_seeds.jsonl` — which rows of the pose board are actually ordered?

Sixty-five records: all thirteen corpus backbones re-fitted at five seeds in the
**published** configuration — same pairs, same MLP, same flags, only `--seed`
moves. `scripts/build_pose_seed_sweep.sh` produces them and
`scripts/analyse_pose_seeds.py` reads them.

**Why these must not go near the corpus, and it is not the usual reason.** The
other controls here differ from the board in something that lands in
`comparability_key` — a head, a split, a layer set — so they form their own
group and *could not* be listed beside it. These carry the published
configuration exactly, so they land in the **identical** group, and
`latest_per_backbone` would happily let a seed-3 run evict a published seed-0
one. A board is one seed by construction; thirteen backbones at five seeds is
sixty-five rankable rows describing thirteen models.

**The harness check, which is the gate.** Every corpus pose cell was run at the
default seed 0, so all thirteen seed-0 rows here must reproduce their published
values. **All thirteen do, at delta 0.0 exactly.** Without that the other seeds
would be measuring some adjacent configuration, and nothing below would mean
anything.

### The noise, over every row rather than two

| | sd over 5 seeds | range |
| --- | --- | --- |
| median | 0.5160 | 1.3332 |
| min | 0.3235 (`siglip_vitb16`) | 0.8322 |
| max | 1.0860 (`resnet18`) | 2.4901 |

19b could reach only two backbones and they disagreed 3x, which left open
whether `clip_vitb16`'s 2.23 was typical or a freak. **It was neither.** Ranges
run 0.83 to 2.49, so a 2.2 is unremarkable — but `clip_vitb16` itself reads
**1.3332** here against 19b's **2.2302**, a third measurement of the same
quantity disagreeing with the second as the second disagreed with the first.
Different harness and five draws rather than three; both are samples of an
unstable statistic, which is the point rather than a discrepancy to resolve.

### The tie list, re-derived pairwise

Not by comparing a gap against a noise figure — that needs a distributional
assumption nothing here has earned, and **only 11% of the seed variance is
common-mode**, so the two rows of a pair move largely independently. The same
five seeds were run for both rows, so the **paired difference** is available and
is what the question actually needs.

| pair | board gap | mean paired diff | sd | t | verdict |
| --- | --- | --- | --- | --- | --- |
| `mae_vitb16` vs `dinov2_vits14` | 2.67 | 3.172 | 0.574 | 12.36 | ordered |
| `dinov2_vits14` vs `dino_vitb8` | 1.73 | 2.351 | 0.701 | 7.50 | ordered |
| `dino_vitb8` vs `dino_vitb16` | 1.94 | 1.185 | 0.579 | 4.57 | ordered |
| `dino_vitb16` vs `dinov2_vitb14` | 0.04 | **−0.800** | 0.567 | **−3.15** | **REVERSED** |
| `dinov2_vitb14` vs `sam_vitb16` | 5.94 | 6.950 | 0.698 | 22.26 | ordered |
| `sam_vitb16` vs `resnet50` | 1.99 | 2.753 | 0.488 | 12.62 | ordered |
| `resnet50` vs `convnext_base` | 0.59 | 0.649 | 0.679 | 2.14 | tied |
| `convnext_base` vs `siglip_vitb16` | 0.69 | 0.672 | 0.656 | 2.29 | tied |
| `siglip_vitb16` vs `clip_vitb16` | 2.28 | 1.643 | 0.620 | 5.92 | ordered |
| `clip_vitb16` vs `supervised_vitb16` | 0.87 | 1.078 | 0.897 | 2.69 | tied |
| `supervised_vitb16` vs `resnet18` | 1.58 | 2.726 | 1.026 | 5.94 | ordered |
| `resnet18` vs `clip_vitb32` | 0.12 | **−2.083** | 1.524 | **−3.06** | **REVERSED** |

`|t| >= 2.776` is the two-sided 95% critical value at df=4. **At five seeds this
test has little power**, so a "tied" verdict is often a pair nobody measured
enough times rather than a pair that is genuinely level.

### What this says, and it is not "widen the threshold"

**The published tie list named the right five pairs.** All five fail to
separate, and three of them — `resnet50`/`convnext_base`,
`convnext_base`/`siglip_vitb16`, `clip_vitb16`/`supervised_vitb16` — are
genuine ties.

**Two of the five are not coin flips; they lean the other way.** Across five
seeds `dinov2_vitb14` beats `dino_vitb16` by 0.80 on average and `clip_vitb32`
beats `resnet18` by **2.083** — seventeen times the 0.12 the board shows. Seed 0
is the minority outcome for both. A reader who treats a tie as "could go either
way" is right; one who reads the board's order as a weak preference is wrong for
exactly these two.

**And a gap threshold is the wrong instrument, which is the transferable part.**
The obvious fix after 19b was to widen the one-degree rule to the measured
median range of 1.33. That would be worse: it would newly call
`dino_vitb8`/`dino_vitb16` (gap 1.94, t=4.57) and
`supervised_vitb16`/`resnet18` (1.58, t=5.94) ties when both are solidly
ordered, while *still* not noticing that the pair 0.12 apart is reversed by
2.08. No threshold on the gap can express this, because the gap is one draw of a
quantity whose spread is not a function of the gap.

**No published number moves.** The board reports what seed 0 produced and
remains a correct record of it. What changes is the reading rule: **do not order
two adjacent rows of this board from the board alone — this file says which
pairs are ordered.**

## `seeds/*.jsonl` — which rows of which boards are actually ordered? (20b, 20c)

**910 records across fourteen boards**: every trained board except
`relative_pose` (whose sweep is `pose_seeds.jsonl`, below) and
`scene_classification` (held out; its own section follows), each re-fitted at
five seeds across all thirteen corpus backbones in the **published**
configuration — same flags, same head, only `--seed` moves.
`slurm/seed_sweep.sbatch` produces them through `scripts/build_corpus.sh`
(`SEEDS=5`, so a sweep re-fits the published flags rather than a second copy of
them) and `scripts/analyse_seeds.py <probe>` reads them.

**Why they must not go near the corpus**: these carry the published
configuration and differ only in `seed`, which `comparability_key` does not
read, so merged they would be sixty-five rankable rows inside each published
board's own group. `build_corpus.sh` refuses `SEEDS>1` against a corpus path,
and `tests/results/test_seed_sweeps.py` checks the committed files.

### The headline, over fifteen boards

| | |
| --- | --- |
| adjacent pairs **ordered** | 119 |
| **tied** — not separable at five seeds | 58 |
| **reversed** — the board's order is the minority outcome | **3** |
| boards where the largest *unordered* gap exceeds the smallest *ordered* one | **10 of 15** |

**Roughly a third of all adjacent pairs cannot be ordered**, and the fraction is
a property of the board rather than of the corpus: `occlusion_edge` orders only
**4** of its twelve pairs and `edge` and `detection` five, against **11 of 12**
for `generic_segmentation` and `scene_parsing`.

### 20b's own conclusion, corrected

20b swept three boards, found no reversal on any of them, and published this:
*"the reversals do not generalise — they were a property of the pose head."*
**That is wrong, and this step is what found it.** `surface_normal` — a linear
board, ten epochs, nothing like `PoseHead` — reverses
`siglip_vitb16`/`convnext_base`: the board shows siglip ahead by **0.0376**
degrees and across five seeds convnext is ahead by **0.157** (t −2.90, 1/5).

The corrected statement is that **reversals are rare and marginal, not absent
and not confined to a nonlinear head**. All three in this corpus sit at |t|
2.90 to 3.15 against a critical value of 2.776, so each is a 95% call at n=5
rather than an emphatic one; more seeds are what would firm them up.

| board | pair | board gap | paired diff | t |
| --- | --- | --- | --- | --- |
| `surface_normal` | `siglip_vitb16` vs `convnext_base` | 0.0376 | **−0.157** | −2.90 |
| `relative_pose` | `dino_vitb16` vs `dinov2_vitb14` | 0.0404 | **−0.800** | −3.15 |
| `relative_pose` | `resnet18` vs `clip_vitb32` | 0.124 | **−2.083** | −3.06 |

### Per board

`ordered`/`tied`/`reversed` are over the twelve adjacent pairs of a
thirteen-row board. `min ordered` is the smallest gap that *is* separable and
`max unordered` the largest that is *not*; wherever the second exceeds the
first, no threshold on the gap can sort that board's pairs.

| board | ord | tie | rev | min ordered | max unordered | median sd | board spread | common-mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `generic_segmentation` | 11 | 1 | 0 | 0.00508 | 0.00049 | 0.00205 | 0.2199 | 7% |
| `scene_parsing` | 11 | 1 | 0 | 0.00232 | 0.00350 | 0.00149 | 0.3540 | 14% |
| `depth` | 10 | 2 | 0 | 0.00025 | 0.00350 | 0.00086 | 0.2536 | 9% |
| `orientation` | 10 | 2 | 0 | 0.2851 | 0.0705 | 0.1050 | 12.32 | 5% |
| `semantic_segmentation` | 10 | 2 | 0 | 0.00884 | 0.00224 | 0.00160 | 0.4195 | 8% |
| `instance_segmentation` | 9 | 3 | 0 | 0.00291 | 0.00307 | 0.00247 | 0.2148 | 4% |
| `surface_normal` | 9 | 2 | **1** | 0.2571 | 0.0719 | 0.0816 | 10.96 | 12% |
| `classification` | 7 | 5 | 0 | 0.00051 | 0.00102 | 0.00046 | 0.0415 | 8% |
| `corner` | 7 | 5 | 0 | 0.00380 | 0.00580 | 0.00417 | 0.1783 | 8% |
| `fine_grained_classification` | 7 | 5 | 0 | 0.00880 | 0.03676 | 0.00141 | 0.3987 | 2% |
| `keypoints2d` | 7 | 5 | 0 | 0.00702 | 0.01074 | 0.00629 | 0.1273 | 3% |
| `relative_pose` | 7 | 3 | **2** | 1.585 | 0.866 | 0.516 | 20.44 | 11% |
| `detection` | 5 | 7 | 0 | 0.00730 | 0.01858 | 0.00789 | 0.1988 | 17% |
| `edge` | 5 | 7 | 0 | 0.00072 | 0.01571 | 0.00618 | 0.1552 | 8% |
| `occlusion_edge` | 4 | 8 | 0 | 0.01221 | 0.03039 | 0.00686 | 0.1532 | 6% |

Common-mode — the share of seed variance that moves every row together, and so
cancels in a difference — runs **2% to 17%**. On no board is it the majority,
which is why a gap is a poor instrument everywhere rather than only where the
sweep happens to disagree with it.

### The gate, and what "reproduces" means per board

Every corpus cell was run at the default seed 0, so every seed-0 row must
reproduce its published value. All fourteen boards pass — but *exactly* only for
`classification` and `fine_grained_classification` (and `relative_pose`). The
tolerances live in `tests/results/test_seed_sweeps.py` as measured floors, in
three groups:

| group | boards | worst observed |
| --- | --- | --- |
| exact | `classification`, `fine_grained_classification`, `relative_pose` | 0 |
| float32 reduction order | the nine dense boards | 1.7e-07 (`scene_parsing`) to 2.9e-05 (`depth`) |
| metric not smooth in the prediction | `detection`, `instance_segmentation`, `orientation` | 9.2e-04 to **2.5e-02** |

**Every one of those runs reported a `train_loss` identical to its published
cell's**, which is what says the sweep re-fitted the same head on the same
features and the *metric* is what moved. That distinction is only available
because schema v8 records the fit, and it is the whole reason the
`scene_classification` result below could be recognised for what it is.

`detection` and `instance_segmentation` score a **ranking**, so a near-tie
between two boxes flips and the AP curve moves; `orientation`'s angular error is
ill-conditioned, which `CORPUS_FINDINGS.md` already records from the run that
added its ceilings. These are exactly the boards this project already says to
quote to three decimals.

## `scene_classification_seeds.jsonl` and `scene_noise.json` — the board that amplifies

**65 records and one study.** `scene_classification` was swept with the other
fourteen boards (20c) and **eleven of its thirteen seed-0 rows do not reproduce
their published cells**, worst **−0.0102** on `resnet50`, *and* carry a
different `train_loss`. The sweep is kept here rather than under `seeds/`
because a sweep that does not reproduce its board cannot be read as that board's
noise — the same rule that held three cells out of the corpus in the v8 re-run.

20d asked why, and the answer is **amplification**: a disturbance far too small
to identify is sufficient to produce the whole disagreement. So the question
"what changed" is not worth answering, and the board gets a separability verdict
anyway.

### What was ruled out, in this order

* **Different data.** The dataset fingerprints match the published cells — and
  `scripts/measure_scene_noise.py` builds its splits by calling the CLI's own
  `_folder_split`, so the val fingerprint it reads is the board's, checked.
* **Nondeterministic training.** Two further repeats of `resnet50` and
  `mae_vitb16` agree **bit for bit** with each other and with the sweep.
* **Changed features.** Every backbone's cache directory has exactly **8,217**
  files written since — NAVI's frame count, i.e. the pose board's extraction
  under different image hashes. No Places365 entry was overwritten.
* **The silicon**, which was the last standing hypothesis and is refuted by the
  other boards: **five boards were published in the same 2026-09-10 batch**
  (`depth`, `semantic_segmentation`, `surface_normal`, `generic_segmentation`,
  `detection`) **and all five reproduce today on a V100.** A different machine
  underneath would have moved them too.

### The measurement: a trigger, not a dose — on a second board

`results/controls/scene_noise.json`, from `scripts/measure_scene_noise.py`.
Noise of a known absolute size injected into the cached features, refitted,
three draws per magnitude. **Both rows' unperturbed baselines reproduce the
sweep's seed-0 value at delta 0.0**, which is the gate and is pinned by
`tests/results/test_scene_noise_study.py`.

| perturbation | `resnet50` mean abs move | `mae_vitb16` mean abs move |
| --- | --- | --- |
| ±1e-06 | **0.00437** | 0.00000 |
| ±1e-05 | **0.00505** | 0.00003 |
| ±1e-04 | **0.00690** | 0.00007 |
| ±1e-03 | **0.00632** | 0.00016 |

**`resnet50` moves by the same amount at every size across a thousand-fold
range** — 19b's finding, reproduced on a board with a different head, a
different metric and a different dataset, so it is a property of a
near-interpolating fit rather than of `PoseHead`. **`mae_vitb16` responds
proportionally and negligibly**, 150x smaller at ±1e-06, where it does not move
at all.

**That is the prediction the study was built to test.** If amplification is the
mechanism, the rows that fail to reproduce should be the ones whose fit is
closest to interpolating — and `mae_vitb16`, at `train_top1` **0.916** against
`resnet50`'s **0.9998**, is one of the only two rows that *do* reproduce. It is.

**The sizes line up too.** `resnet50`'s amplified movement (~0.005) is its own
seed-to-seed sd (**0.00424**), and the published-versus-today gap (0.0102) is
about one seed *spread* (0.00984). The published cell is one draw from the
distribution today's runs draw from, not a different experiment.

**Corroborating it by accident**: this script's first version built the backbone
before seeding, where `run()` seeds first, and its baseline missed by **6e-3** on
`resnet50` and 5e-5 on `mae_vitb16` — a hundredfold difference between the two
rows from an RNG-ordering change alone, which is the same contrast the curves
show deliberately.

### So the board does get a verdict, and it is the same in both configurations

Taking the held-out sweep at face value — it is five seeds of *today's*
configuration, internally consistent — `scene_classification` orders **8 of its
12 adjacent pairs**, ties 4, and reverses none. The verdict survives the
unresolved disagreement, because the two configurations **differ on only 2 of
78 pairs, and both are pairs the sweep calls tied**: `resnet50`/`dino_vitb8` and
`sam_vitb16`/`dino_vitb16`. Every pair either configuration can order, both
order the same way.

**No published number moves**, and the file stays out of `seeds/`: it still
does not reproduce its board, and the guard that says so is the reason this was
investigated rather than averaged over.

## `pose_noise.json` — how much does a pose number move, and what moves it?

**Not records.** The runs behind this perturb cached features, which no flag
expresses and no `ResultRecord` could honestly describe, so the study is a JSON
file of its own and `scripts/measure_pose_noise.py --summarise` reprints it.
Everything else here is a `.jsonl` of real records; this one is deliberately not.

**Why it was needed.** `relative_pose` shipped with a reproducibility claim
resting on **one** comparison: the `mae_vitb16` cell scored 22.6984 on the
cluster and 21.7695 from a second extraction, so the board is quoted to whole
degrees and five adjacent pairs are called ties. One observation is thin for a
rule that decides which rows a reader may separate, and this project's own
standing instruction — the one `duration_seconds` bought, at the cost of three
files and a merged PR — is to repeat a measurement before concluding from it.

Three parts, each answering something the single observation only gestured at.

### 1. The observation, at n=2

Score `mae_vitb16` and `clip_vitb16` from each of the two cache roots on this
machine, one seed, one pair set, features the only difference. Only those two
are comparable: the local cache holds no entry for `dino_vitb16` or
`sam_vitb16` under the pooling this probe resolves.

| backbone | local | cluster | movement |
| --- | --- | --- | --- |
| `mae_vitb16` | 20.9461 | 22.0494 | −1.1033 |
| `clip_vitb16` | 40.4348 | 41.1144 | −0.6796 |

**About a degree survives contact with a second row**, bracketing the published
0.93. What one observation could not show is that the two differ by a factor of
1.6, so "about a degree" is the right precision to state it to and any tighter
figure is fitting one sample.

### 2. The mechanism, measured rather than inferred

The published write-up attributed the feature difference to extraction batch
size — the proof run used 64 where `build_corpus.sh` defaults to 32 — on the
evidence that a *later* probe whose two caches were built at the same batch size
came out bit-identical. That is consistent with the hypothesis and does not test
it. Extracting the same 455 frames into two fresh roots at the two batch sizes,
same GPU, same everything else:

| | |
| --- | --- |
| max \|difference\| | **2.38e-05** |
| mean \|difference\| | 2.12e-07 |
| bit-identical frames | 7/455 (2%) |

**The hypothesis is right.** Batch size alone reproduces the size of difference
the two caches carry (1.4–1.8e-05 sampled across them), and it is the batching
rather than the GPU, since both passes ran on the same card minutes apart.

### 3. The amplification — which is where the published story was wrong

Inject uniform noise of a known size into the cached features, refit, and see
what the score does. Three draws per magnitude, against the *unperturbed*
spread over three seeds:

| perturbation | `mae_vitb16` mean \|move\| | `clip_vitb16` mean \|move\| |
| --- | --- | --- |
| seed only, no perturbation (range) | 0.7162 | 2.2302 |
| ±1e-06 | 0.8215 | 0.2688 |
| ±1e-05 | 0.2061 | 0.7565 |
| ±1e-04 | 0.5585 | 1.2481 |
| ±1e-03 | 0.7927 | 1.0378 |
| every magnitude and draw (n=12) | 0.5947, max 1.3525 | 0.8278, max 1.7707 |

**There is no dose-response, and the published sentence describes one.** It
reads "their stored vectors differ by up to 1.1e-05 ... Thirty epochs of a
1,536-dimensional MLP turn that into 0.93 degrees", which is a causal chain from
a perturbation size to a movement size. A thousand-fold change in the
perturbation produces no trend: `mae_vitb16` is flat and non-monotonic, and the
two backbones do not even agree on the shape. **18 of the 24 perturbed scores
land inside the range three seeds produce with no perturbation at all.**

So the degree is **the width of this fit's run-to-run scatter**, reached by any
disturbance whatever — a different cache, a different seed, a perturbation four
orders of magnitude smaller than the one measured. Feature noise is a trigger,
not a dose.

**The board's reading rule is unchanged and better founded.** Quoting
`relative_pose` to whole degrees follows from the scatter being ~1 degree, which
is what both the n=2 cross-cache movement and the perturbation study say. What
changes is the explanation beneath it, and one thing a reader should now know:
the tie list is calibrated to what separates two *published cells* — same seed,
same code, two extractions — and **not** to what a re-fit at another seed would
do, which for `clip_vitb16` is 2.23 degrees.

### A three-draw range is not a noise estimate

`pose_protocol.jsonl` reports seed ranges of **0.21** and **0.41** for these two
backbones. This study measures **0.7162** and **2.2302** for the same backbones
and the same three seeds — 3x and 5x larger. Both are ranges over three draws,
both were computed correctly, and they disagree because the statistic is
unstable, not because either run is wrong.

That is the standing lesson arriving on the claim it was most needed for: this
file already says `spread / noise` has misled in **both** directions, and a
board's reading rule was built on the smaller of two such numbers. **Quote a
range over three draws as what it is — one sample of a noisy statistic — and
prefer a bar that several independent measurements agree on.**

## `pose_protocol.jsonl` — does the shipped pose probe measure what the pre-measurement measured?

Six records: `RelativePoseTask` on NAVI at the pinned protocol — eight partners
per training anchor, 50,519 training pairs against 1,740 validation ones,
probe3d's MLP — for `mae_vitb16` and `clip_vitb16` over three seeds each. The
task is deliberately **not registered** in 16a-2, so it cannot acquire a corpus
board, a CLI row or a `TARGET_STYLES` entry by accident; the board itself is
16a-3's.

**These are here because a published figure needs a surviving record.** Step 8a
produced six figures on an ad-hoc staging that was never committed and left
nothing behind to check them against; the numbers below are quoted in
`ENGINEERING_LOG.md` and in `visbench/tasks/low_level/README.md`, so they are
kept.

**They were re-run in 16a-3**, when the accuracy metrics were renamed to the
parametrised form (`rotation_acc@30deg`), so that the committed file carries
keys the library still emits rather than keys nothing produces. All six
reproduced to **+0.000000** — the same cache, the same seeds, bit for bit — so
the file's numbers are unchanged and only its metric *names* moved. That
exactness is also the other half of the reproducibility entry below: this board
moves by about a degree across two *extractions* and not at all within one.

| backbone | rot err (3 seeds) | vs floor | seed range | parked | delta |
| --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | 21.84 | +45.01 | 0.21 | 24.29 | −2.45 |
| `clip_vitb16` | 40.50 | +26.35 | 0.41 | 42.34 | −1.84 |

**The floor reads 66.85 in every record, which is the check that matters.** It
is the parked value to the digit, so the shipped `NaviPoseDataset` drew the
same pairs the pre-measurement drew — the claim 16a-1 makes and this is the
evidence for.

### What moved the scores, and what did not

The shipped probe differs from the pre-measurement in two ways at once, so
both were measured separately on the same features (single-backbone
`scripts/premeasure_pose.py` runs, which write no records and are quoted here
as script output):

| configuration | `mae` | `clip` |
| --- | --- | --- |
| parked — EXIF-naive frames, one fixed shuffle | 24.29 (range 0.73) | 42.34 (range 0.69) |
| EXIF-fixed frames, one fixed shuffle | 22.91 (range 1.01) | 42.32 (range 1.36) |
| shipped — EXIF-fixed, reshuffled every epoch | 21.84 (range 0.21) | 40.50 (range 0.41) |

**The EXIF fix does not move a pose number beyond seed noise**: −1.38 on `mae`
against seed ranges of 0.73 and 1.01, and −0.02 on `clip`. Turning 4.0% of the
frames the right way up was a correctness fix, not a numbers fix, and saying
otherwise from `mae`'s row alone would be reading a difference the size of the
noise. **Per-epoch reshuffling is the half that shows**: −1.07 and −1.82, the
same direction on both, and it cuts the seed range **three to five times**,
which is the more useful of the two effects for a board.

**Do not compare `train_loss` across those rows.** The pre-measurement reports
the last training batch in train mode and the shipped task reports a full-split
pass in eval mode, so 0.0020 and 0.0029 are two different statistics. Only the
first two rows of the table measure it the same way.

## `relative_depth.jsonl` — is the `depth` board measuring metric accuracy?

Five records: `RelativeDepthTask` on the whole NYUv2 split, the same frames,
crop and validity rule `probe_depth` reads, differing **only** in the readout.
`depth` predicts metres and is scored by delta accuracy; this predicts a
unitless score and is scored by how often a sampled point pair is *ordered*
correctly, so neither scale nor shift is supervised or scored.

`scripts/build_relative_depth_control.sh` runs it. The task is deliberately
**not registered**, so it cannot acquire a corpus board, a CLI row, a
`TARGET_STYLES` entry or a gallery figure by accident.

**It began as a candidate probe and was rejected on this measurement.** That
rejection is the finding.

| backbone | ordinal | its ceiling | band position | `depth` d1 |
| --- | --- | --- | --- | --- |
| `dinov2_vitb14` | 0.8466 | 0.8627 | 89.8% | 0.7851 |
| `dinov2_vits14` | 0.8407 | 0.8627 | 86.0% | 0.7652 |
| `mae_vitb16` | 0.8400 | 0.8583 | 88.0% | 0.6945 |
| `clip_vitb16` | 0.7757 | 0.8583 | 45.9% | 0.6321 |
| `resnet50` | 0.7523 | 0.8074 | 45.9% | 0.5395 |

**Spearman between the two readouts is +1.000.** Identical ordering, 0.0943 of
spread against `depth`'s 0.2456, and a smallest adjacent gap of **0.0007** --
`dinov2_vits14` against `mae_vitb16`, which `depth` separates by 0.0707. A
readout that cannot separate two backbones the shipped one separates by a
hundredfold has not earned a board; that is 6d-2's occlusion-edge test applied
to a readout rather than a target.

### What it says about a board that does ship

**`depth` is not ranking backbones by metric accuracy.** Discarding scale and
shift entirely -- never supervising them, never scoring them -- leaves the
ranking unchanged. Whatever separates these five on the depth board survives
the removal of every metric quantity, so it is *ordering plus feature
resolution*, and the delta accuracies are reporting it in metres rather than
being about metres.

That is consistent with what the resolution control already found (tokens
correlate +0.818 with `depth`) and sharpens it: the part of the depth board
that is not resolution is ordinal, not metric.

**Do not read this as "the depth board is wrong".** It reproduces probe3d's
published protocol, which is the only reason its numbers are comparable to
anything, and metric depth is the question that protocol asks. The finding is
about what the *ranking* is sensitive to, which is a different claim.

### Why the candidate failed, and the check that was missing

The oracle gate passed comfortably -- a patch-mean oracle orders 94.0% of pairs
correctly at 16x16 and 89.5% at a ResNet's 7x7, so unlike photometric
superpixels the signal survives the bottleneck a dense probe reads through.

What killed it is the other end. **"The lower point in the image is nearer"
scores 65.2%** with no features at all, because indoor depth increases with
height in the frame. So the usable band on the shipped runs is 0.7056 to 0.8627
-- **0.157 wide** -- and the five backbones' own ceilings already differ by
0.055 of it. A third of the room is grid size before a representation is
consulted, and the three strongest backbones then sit at 86-90% of what is
left, 0.0007 apart.

`corner` ranks backbones perfectly well at a comparable 0.83 ceiling, because
its trivial floor is near zero. **The gauntlet measured every candidate's
ceiling and never measured its floor.** That check now exists -- see the
gauntlet section of `visbench/tasks/low_level/README.md` -- and this is the
rejection that added it.

### Two things worth keeping about how it was measured

**A minimum depth-ratio threshold makes the task easier, not harder**, which is
the opposite of the intuition that prompted trying it. At ratio >= 2.0 the
vertical shortcut reaches 83.7% against a 99.9% ceiling; unrestricted pairs
leave 65.2% against 94.0%. The widest band is at no threshold at all, so pairs
are sampled unrestricted. `scripts/premeasure_ordering.py` reproduces the
sweep, and it costs one pass over a split -- no backbone and no fitted head.

**`depth` re-ran to ~1e-6 of the corpus** on all four shared backbones while
this was measured, and the `dinov2_vitb14` ordinal number reproduced to four
decimals across two different entry points (the CLI and `visbench.run()`
through `scripts/run_relative_depth.py`). So the comparison above is against
published numbers rather than against a re-run that might have drifted.

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
