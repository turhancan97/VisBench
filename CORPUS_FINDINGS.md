# What the corpus says, and how to read a board

**Read this before quoting any number from
[`results/corpus/visbench.jsonl`](results/corpus/visbench.jsonl) or from
`LEADERBOARD.md`.** These are the findings that decide what a board means --
lifted out of `CLAUDE.md` on 2026-08-20 because that file is loaded into every
session's context whole and had grown past the limit it is read under. The
*claims* are still summarised there; the evidence is here.

Every one of these was measured on the committed corpus. Several correct an
earlier published reading, and each says which -- that is the point of keeping
them rather than only the conclusions.

**The single most important one, if you read nothing else:** "which backbone is
best" is not a well-formed question against this corpus. `mae_vitb16` is first
on six of the sixteen boards and last on four. A summary that picks a winner
is discarding the result.

Two standing cautions apply to everything below:

- **A count over a corpus is a fact about that corpus, not about a backbone.**
  Counts here moved twice without any backbone's features changing, purely
  because a column was added. Re-read them off `LEADERBOARD.md` rather than
  quoting these paragraphs.
- **n=12, so every correlation here has wide error bars**, and the backbone
  properties are correlated with each other.
  [`scripts/analyse_board_correlates.py`](scripts/analyse_board_correlates.py)
  reproduces the correlational findings, and its `--drop` re-runs without any
  row so you can see which conclusions survive.

---


- **Quote `orientation` to two decimals, and treat its bottom two rows as
  tied** (2026-09-01). Re-running the five low-level boards to add their
  ceilings gave a reproducibility measurement for free: four came back at
  **~1e-7 relative**, and `orientation` at **1e-3** — a thousand times worse.

  | probe | max relative drift over 12 backbones |
  | --- | --- |
  | `corner` | 3.1e-07 |
  | `edge` | 9.4e-07 |
  | `keypoints2d` | 1.5e-06 |
  | `occlusion_edge` | 1.0e-05 |
  | **`orientation`** | **1.8e-03** |

  In degrees: max drift **0.0210**, median **0.0051**. The smallest adjacent gap
  on that board is **0.0048** — `clip_vitb32` 29.9626 against `resnet50`
  29.9675 — which is *below the median drift*. **Those two rows are not
  separable**, and a claim that either beats the other is noise. Every other
  adjacent gap is at least 0.0676, more than three times the worst drift, so the
  rest of the board is solid and the published ordering did not change.

  **The cause is the metric, not the probe.** Features come from cache and are
  byte-identical and the head is seeded; what differs is float-level
  non-determinism in training, which the other four probes also show — at 1e-7.
  `orientation_error` is a coherence-weighted `acos` of a cosine similarity, and
  `acos` has infinite derivative at its endpoints, the same ill-conditioning
  `OrientationTask._loss_eps` exists to keep out of the *loss*. The metric needs
  no gradient and so carries no such guard, and amplifies the same 1e-7 weight
  difference into 1e-4 degrees.

  This is `detection`'s lesson by a different mechanism: there a *discrete*
  metric made non-determinism visible, here an *ill-conditioned* one amplifies
  it. **Check a gap against the drift before reading a rank off a board.**

- **Feature resolution is the strongest correlate of every dense board, and on
  the two boards where that mattered it explains under a quarter of the gap**
  (the resolution control, 2026-08-21).
  [`results/controls/resolution.jsonl`](results/controls/resolution.jsonl) is
  the measurement; that directory's README is the write-up.

  The confound was real and this file was built without noticing it. Tokens
  correlate +0.958 with `generic_segmentation`, +0.867 with `surface_normal`,
  +0.818 with `depth` — the strongest structural correlate of any dense board,
  where width correlates with essentially nothing. And **the only backbones
  carrying 256 tokens are the two DINOv2s**, so grid size, the DINOv2 objective
  and LVD-142M pretraining were one variable. No dense board could say which of
  the three it had ranked.

  `dinov2_vitb14_196` separates them: the same weights and the same hub ref at
  196px, whose 14x14 grid matches every ViT-B/16 in the corpus.

  | board | 256 tok | 196 tok | change | rel |
  | --- | --- | --- | --- | --- |
  | `generic_segmentation` | 0.7556 | 0.7407 | -0.0149 | 2.0% |
  | `depth` | 0.7851 | 0.7791 | -0.0060 | 0.8% |
  | `surface_normal` (deg, lower better) | 30.1143 | 30.6556 | +0.5413 | 1.8% |
  | `edge` | 0.4481 | 0.4363 | -0.0119 | 2.6% |
  | `corner` | 0.6526 | 0.6349 | -0.0178 | 2.7% |

  **Matching the grid costs under 3% on every board**, and DINOv2-B keeps its
  lead over the entire ViT-B/16 pack on both boards it led — 0.7407 against
  `dino_vitb16`'s 0.6838, and 0.7791 against `mae_vitb16`'s 0.6945. Resolution
  accounts for **21%** of the `generic_segmentation` lead and **7%** of the
  `depth` one; between 79% and 93% survives.

  **The confound was narrower than it first looked, and saying so is half the
  finding.** On `surface_normal`, `edge` and `corner` DINOv2-B never led at all
  — `mae_vitb16` is ahead on all three — so there was no lead for resolution to
  explain. A first reading of the correlation table treated all five as
  confounded. Check who actually leads a board before explaining their lead.

  **Two limits travel with the number.** The control spans 256 to 196 tokens,
  about 1.3x, while the corpus correlation spans 49 to 256, about 5x — so it
  bounds the slope where the DINOv2-versus-ViT-B/16 comparison lives and says
  nothing about the 49-token backbones, where `clip_vitb32` (0.6019) against
  `clip_vitb16` (0.6787) suggests the large jumps are. And 196px is slightly
  off DINOv2's training resolution, so read a drop as "resolution or
  distribution", not resolution alone.

  **It is one-sided because nothing else here can be raised.** DINOv2
  interpolates its position embeddings inside its own forward pass, so 196px is
  its intended use. open_clip does not interpolate at all
  (`RuntimeError: tensor a (257) must match tensor b (197)`) and timm needs
  `dynamic_img_size` (`Input height (256) doesn't match model (224)`), so no
  ViT-B/16 in this corpus can be given a 16x16 grid without changing the model.

  **The oracle gate corroborates the mechanism without using a backbone at
  all** (2026-09-01), which is worth having because every number above comes
  from one. `DenseTrainingTask.evaluate_oracle` pools a target to a feature grid
  and scores the reconstruction, so it measures how much of a *target* survives
  a grid with no weights involved. Over the pinned 600 frames, going from a
  16x16 grid to a ResNet's 7x7: `corner` 0.8316 → 0.6685, `keypoints2d` 0.6976 →
  0.4728, `edge` 0.6336 → 0.4977, `occlusion_edge` 0.5301 → 0.4336. The dense
  targets really do lose a fifth to a third of themselves to a coarse grid, and
  that is a fact about the targets rather than about DINOv2.

  It does **not** widen the control's claim. The control's point is that
  matching the grid costs under 3% between 256 and 196 tokens; this says the
  much larger 256-to-49 drop the corpus correlation spans has real headroom
  behind it, which is the same "says nothing about the 49-token backbones"
  caveat from the other side. Do not read an oracle drop as a predicted score
  drop: it bounds what is available, not what a backbone recovers — `corner`
  reaches 80% of its oracle and `keypoints2d` 41%.

  **The control is deliberately not in the corpus**, though it passes
  `comparability_key` against every board it ran on — the five records land in
  the *identical* group. The corpus answers "what does this backbone score",
  and every row in it is a model somebody might choose; a control answers "what
  happens when one thing changes". Mixing them makes a board answer two
  questions, which is the same reason the schema never ranks across `finetune`.

- **The `depth` board is not ranking backbones by metric accuracy — discarding
  scale entirely leaves the ranking unchanged** (the relative-depth control,
  2026-09-04).
  [`results/controls/relative_depth.jsonl`](results/controls/relative_depth.jsonl)
  is the measurement; `results/controls/README.md` is the write-up.

  A readout that predicts a *unitless score* per pixel and is scored only by
  how often a sampled point pair is **ordered** correctly — no scale, no shift,
  neither supervised nor scored — was run on the same NYUv2 frames, crop and
  validity rule `probe_depth` reads.

  | backbone | ordinal | `depth` d1 |
  | --- | --- | --- |
  | `dinov2_vitb14` | 0.8466 | 0.7851 |
  | `dinov2_vits14` | 0.8407 | 0.7652 |
  | `mae_vitb16` | 0.8400 | 0.6945 |
  | `clip_vitb16` | 0.7757 | 0.6321 |
  | `resnet50` | 0.7523 | 0.5395 |

  **Spearman between the two readouts is +1.000.** So whatever separates these
  five on the depth board survives the removal of every metric quantity: the
  board is ranking them by **ordering plus feature resolution**, and its delta
  accuracies report that in metres rather than being about metres.

  That sharpens the resolution control rather than contradicting it (tokens
  correlate +0.818 with `depth`): the part of the depth ordering that is *not*
  resolution is ordinal, not metric.

  **This is not "the depth board is wrong."** It reproduces probe3d's published
  protocol, which is the only reason its numbers compare to anything, and
  metric depth is the question that protocol asks. The claim is about what the
  *ranking* is sensitive to, which is a different thing.

  **The control exists because the probe was rejected.** Identical ordering at
  38% of the spread (0.0943 against 0.2456), with `dinov2_vits14` and
  `mae_vitb16` landing **0.0007** apart where `depth` separates them by 0.0707.
  A readout that cannot separate two backbones the shipped one separates by a
  hundredfold has not earned a board. **n=5**, and the four shared `depth`
  numbers re-ran to ~1e-6 of the corpus, so the comparison is against published
  values.

- **The dense boards understate a CNN by more than they understate a ViT, and
  the reason is which stage a linear head reads rather than anything about the
  representation** (the DPT control, widened 2026-09-04).
  [`results/controls/dpt_head_cnn.jsonl`](results/controls/dpt_head_cnn.jsonl)
  is the measurement; `results/controls/README.md` is the write-up.

  Every dense board in the corpus is a linear head on a backbone's **last**
  feature map. For a ViT that is the whole story — all twelve blocks share one
  grid, so nothing finer was computed and discarded. For a CNN it is not: a
  ResNet's stage-1 map is **56x56** and exists in the same forward pass as the
  7x7 stage-4 map the board reads.

  Handing a DPT head all four stages instead is worth far more to a CNN than
  four blocks are worth to a ViT:

  | | DPT gain over the linear score |
  | --- | --- |
  | nine ViTs, `2 5 8 11` | 1.08x - 1.31x |
  | three CNNs, own last four stages | **1.17x - 2.80x** |

  `keypoints2d` on `resnet18` goes 0.1659 to 0.4648. That is not a claim that
  CNN features are better than the board says in some absolute sense — the
  board's protocol is the published one and every backbone is read the same way.
  It is a claim about **what a cross-architecture dense comparison is
  measuring**: for a ViT the linear number is close to all the spatial detail
  the backbone has, and for a CNN it is a deliberate discard.

  **Do not "fix" this by giving the boards a DPT head.** Three of the five CNN
  boards change leader under one, and two *invert* (`convnext_base` first to
  last on `corner` and `orientation`, rho -1.000), so a DPT board would be a
  different ranking rather than a better-resolved one — and it would be a
  different comparability group from every published VisBench number. The point
  is the caveat, not a replacement protocol.

  **n=3 CNNs**, so two of those rho values are one swap each, and the gains are
  the more solid half of this finding.

- **`mae_vitb16` is first on six of the sixteen boards and last on four, and
  this is the corpus finally demonstrating what the taxonomy claims** (10b,
  2026-08-14; **counts re-read off the board at twelve backbones, 10e**; scene
  board added 2026-08-28, `orientation` and `fine_grained_classification`
  boards 2026-08-28). Read
  this before quoting any board. MAE leads edge (0.4982), corner (0.6669),
  correspondence (0.3577), occlusion edges (0.3273), surface normals
  (27.52° mean) and **orientation** (18.82° error) — and comes **last** on
  classification (0.9582), retrieval
  (0.1883), mid-level similarity (0.6897) and **fine-grained recognition**
  (0.4696), with semantic segmentation
  (0.3350) now *eleventh* rather than last, because `sam_vitb16` scores 0.3339
  beneath it. The sixteenth board is what took its last-place count from three
  to four, and it is a semantic one, so the tier pattern is unchanged — but the
  *count* moved again without MAE's features moving, which is the standing
  warning two paragraphs down. `scene_classification`, the fourteenth board, is another semantic
  one and MAE places tenth of twelve there — same tier pattern, not a new last. It led keypoints2d too until `dino_vitb16` (0.2850) and
  `sam_vitb16` (0.2696) both passed it. **A count over a corpus is a fact about
  that corpus, not about the backbone**: both of those counts moved without
  MAE's features changing, which is why they are re-read off `LEADERBOARD.md`
  rather than carried forward as prose. No other backbone here is simultaneously best and worst — though see
  the `dino_vitb16` bullet for why that shape is not the price of low-level
  strength.

  Before 10b the boards mostly reproduced one capacity ordering (DINOv2 > CLIP >
  ResNet on nearly everything), which made the three tiers look like a
  taxonomy the numbers merely tolerated. One pixel-reconstruction backbone
  separates them outright. **"Which backbone is best" is not a well-formed
  question against this corpus**, and a summary that picks a winner is
  discarding the result.

  **MAE's retrieval 0.1883 is barely above the 0.1 chance floor and is not a
  broken run** — the obvious reading, and wrong. The check is internal to the
  corpus: the *same* features score 0.9582 top-1 under a **trained** linear
  probe. A learned projection recovers category structure that cosine
  similarity on the raw CLS token cannot, which is MAE's documented behaviour
  without fine-tuning, and a genuine extraction bug would have taken the linear
  probe down with it. **Two probes over one feature set is the cheapest
  available test of whether a shocking number is a bug**, and it needed no new
  run — both records were already in the corpus.

  **Its correspondence win is not the grid**, which is the first objection that
  board invites: 0.3577 against a `ceiling_recall@5px` of 0.3932, where
  `dinov2_vits14` scores 0.3049 against a *higher* 0.4123. Read against the
  ceiling the gap widens. This is what `context_metrics` was for.

  **`convnext_base` is the mirror image and carries `resnet50`'s caveat.** It
  tops classification (0.9997) and retrieval (0.9890) and places seventh or
  lower on every dense geometric board. It is ImageNet-1k *supervised* and
  Imagenette's classes are ImageNet-1k wnids, so those two scores are close to
  in-distribution recall rather than transfer. `docs/tasks.md` names both
  backbones now, not just the ResNet.


- **`supervised_vitb16` is the corpus's first controlled experiment, and the
  split lands on the tier boundary** (10c, 2026-08-19). It is
  `vit_base_patch16_224.augreg_in1k`: the *same architecture* and the *same
  pretraining set* as `mae_vitb16`, so the only variable between those two rows
  of any board is the training objective. Every other pair of backbones here
  varies at least two things at once, which is why no earlier board could say
  what a gap belonged to.

  Supervised wins **all four high-level** boards — classification 0.9972 v
  0.9582, retrieval **0.9947 v 0.1883**, semantic segmentation 0.5791 v 0.3350,
  detection 0.1669 v 0.1296. MAE wins **all three low-level** ones and **five of
  six mid-level** ones — depth, normals (27.52° v 36.72° mean), correspondence,
  occlusion edges, generic segmentation, plus edge, keypoints2d and corner.
  Thirteen boards, one variable, five against eight.

  **The single crossing is mid-level similarity** (supervised 0.8202 v 0.6897),
  and it must not be rounded away — this bullet said "the winner changes exactly
  at the tier boundary" for one commit, which was wrong because it counted
  `similarity` as high-level. It is **mid-level**, and this file says two
  sections down that conflating it with high-level retrieval is the one thing
  not to do with that probe. **Count the tiers from `record.level`, not from
  which boards feel semantic.** A tidy claim about a taxonomy is exactly the
  kind that gets written before it is counted.

  A hypothesis for the crossing, untested: NIGHTS' images are Stable Diffusion
  generations prompted with ImageNet, CIFAR, Flowers-102, Food-101 and SUN397
  categories, so its 2AFC may reward category familiarity — the in-distribution
  caveat once more — rather than the resemblance the probe means to isolate.

  **`augreg_in1k`, never a 21k recipe**, and this is the whole entry: the 21k
  models are stronger and would be the more flattering number, and they move the
  pretraining data as well as the objective, which destroys the comparison. A
  fast test pins the tag against `mae_vitb16`'s so the tempting upgrade fails
  the suite rather than quietly ruining the experiment.

  **Its classification and retrieval rows carry `convnext_base`'s caveat and
  more strongly.** Imagenette's classes are ImageNet-1k wnids and this backbone
  was trained on ImageNet-1k *with labels*, so 0.9972 and 0.9947 are close to
  in-distribution recall. Do not read the retrieval gap against MAE as a
  measurement of transfer.


- **`dino_vitb16` completes an objective family, and the answer is that
  high-level structure comes from a semantic training signal rather than from
  labels** (10d, 2026-08-19). 10d took the corpus to **eleven backbones, 143
  records** — twelve and 156 once 10e's control landed — thirteen groups all
  holding every column; the merge was purely additive.
  `vit_base_patch16_224.dino` is the same architecture and the same pretraining
  set as `mae_vitb16` and `supervised_vitb16` — 12 blocks, 768 wide, patch 16,
  one prefix token, `global_pool='token'`, ImageNet-1k — so the corpus now has
  three values of one variable instead of two. What the pair could not do is
  separate **"trained with labels" from "trained toward semantics"**; DINO is
  label-free and semantic, so it sits on whichever side that distinction falls.

  **It sits with the supervised model, and the claim rests on retrieval
  alone.** `dino_vitb16` scores **0.9192** mAP against supervised 0.9947 and
  MAE **0.1883** — MAE last of twelve, DINO fifth. So a label-free objective
  that learns *categories* recovers nearly everything labels buy, while one
  that learns *pixels* recovers almost none of it, which the supervised/MAE
  pair could not tell apart because it moved both at once. Part of the residual
  supervised margin is the in-distribution caveat, since Imagenette's classes
  are ImageNet-1k wnids.

  **This bullet originally cited semantic segmentation as a second board, and
  10e's recipe control refuted that half** — see the `sam_vitb16` bullet. On
  semseg a *supervised* backbone trained under a different recipe lands within
  0.001 of MAE, so that board cannot separate objectives at all. The sentence
  was written before the control existed, read as well-supported, and was
  wrong. **Retrieval is the board that carries this result; do not re-add the
  second one.**

  **The high-level sweep is three boards and a tie.** Supervised beats DINO on
  detection by **0.0009** (0.1669 against 0.1660), which is under the ~1e-3
  spread measured on that probe, so at the three decimals detection is quoted
  to those rows are equal. Classification separates nothing either — eleven of
  twelve backbones are above 0.98. Do not report "supervised wins all four".

  **DINO does not pay MAE's price at the other end, which is the second
  finding.** MAE leads five boards and is last on three; DINO's worst placing
  anywhere is **ninth of twelve**, and it takes two overall firsts —
  mid-level similarity **0.9019**, beating the 0.8701 that had stood since v0.2,
  and keypoints2d **0.2850**, taking that board off MAE — plus second on edge,
  corner and correspondence. **The high-versus-low trade-off the MAE row looks
  like it demonstrates is therefore not a law.** One objective is strong at both
  ends, and any summary drawn from the nine-backbone corpus that reads MAE's
  shape as "the price of low-level strength" is over-reading it.

  **The existing pair varies one and a half things, and this is worth stating
  rather than rounding away.** `augreg_in1k` normalises with mean/std 0.5 while
  `mae` and `dino` both use ImageNet statistics. Each checkpoint must be
  preprocessed with the statistics it was trained under — that is what
  `resolve_data_config({}, model=model)` does per model, and there is no correct
  alternative — so the supervised-against-MAE comparison carries an input
  normalisation difference alongside its objective difference. **DINO-against-
  MAE holds the normalisation fixed**, which makes it the tighter of the two.
  Pinned by a test on the resolved transform, not by a comment, since a timm
  release is free to move that metadata.

  **`deit3_base_patch16_224.fb_in1k` is the near-miss and is excluded by
  structure, not by name.** It is supervised ImageNet-1k on twelve 768-wide
  blocks with patch 16 and one prefix token — every property the family is
  pinned on — and it carries `LayerScale` where the three carry `Identity`, so
  it moves the architecture as well as the recipe. A fast test refuses any
  `deit3` entry outright.

  **What timm does *not* have, checked rather than assumed** (timm 1.0.28): no
  MoCo v3, iBOT, MSN or SimMIM at `vit_base_patch16_224`. Those need a
  checkpoint fetched outside the registry and loaded through `CustomBackbone`,
  which is a materially larger step and would have to answer how a
  non-registry backbone reaches the CLI and the corpus at all. Of the tags that
  *do* exist, only `.dino` qualifies; `.sam_in1k` is a legitimate second
  control but varies the **optimiser** rather than the objective, so it belongs
  in its own step if it is wanted.


- **A recipe control was added, and it refuted half of an already-published
  claim** (10e, 2026-08-19). `sam_vitb16` is
  `vit_base_patch16_224.sam_in1k`: the same architecture, the same pretraining
  set, the same labels and the same input normalisation as `supervised_vitb16`,
  differing only in **how it was trained to that objective** — sharpness-aware
  minimisation with light augmentation, against AugReg's AdamW with heavy
  augmentation. The corpus is **twelve backbones, 168 records** across fourteen
  boards.

  It exists to supply a denominator nothing else in this corpus had: **a gap
  between two objectives means nothing until you know how large a gap two runs
  of the *same* objective can produce.** Every other pair here differs in
  architecture, data or objective; this one differs in none of the three.

  **On semantic segmentation the recipe gap is larger than the entire objective
  spread**, which is the finding:

  | board | augreg | sam | recipe gap | objective spread | ratio |
  | --- | --- | --- | --- | --- | --- |
  | retrieval | 0.9947 | 0.9912 | 0.0034 | 0.8064 | **234x** |
  | correspondence | 0.3232 | 0.3298 | 0.0066 | 0.0345 | 5.2x |
  | depth (d1) | 0.6195 | 0.6356 | 0.0161 | 0.0750 | 4.7x |
  | similarity | 0.8202 | 0.8695 | 0.0493 | 0.2122 | 4.3x |
  | detection | 0.1669 | 0.1797 | 0.0128 | 0.0373 | 2.9x |
  | keypoints2d | 0.2573 | 0.2696 | 0.0123 | 0.0277 | 2.2x |
  | corner | 0.6204 | 0.6454 | 0.0250 | 0.0465 | 1.9x |
  | occlusion edge | 0.1996 | 0.2680 | 0.0684 | 0.1277 | 1.9x |
  | edge | 0.4420 | 0.4734 | 0.0314 | 0.0562 | 1.8x |
  | generic seg | 0.6195 | 0.6667 | 0.0472 | 0.0643 | 1.4x |
  | **semantic seg** | 0.5791 | **0.3339** | **0.2452** | 0.2441 | **1.0x** |

  **`sam_vitb16` scores 0.3339 mIoU and `mae_vitb16` scores 0.3350** — a
  *supervised* backbone with labels, landing within 0.0011 of the pixel-
  reconstruction one, and **last of twelve**. Whatever that board measures, it
  is not the training objective. Every setting was identical across all twelve
  runs (same schedule, same head, same split), so this is not a tuning
  artefact of one run.

  **What it costs and what survives.** The 10d write-up cited retrieval *and*
  semantic segmentation as the two boards showing that high-level structure
  comes from a semantic signal. Retrieval survives enormously — the objective
  effect there is 234x the recipe effect. The semseg half is dead, and it was
  already written into `CHANGELOG.md`, `CLAUDE.md` and an open pull request
  when the control landed.

  **The general rule this establishes: quote an objective gap against the
  recipe gap on the same board, not against zero.** Seven of the thirteen
  boards have objective spreads under 3x their recipe gap, so "objective X
  beats objective Y" is a weak claim on most of this corpus and a strong one on
  four boards. Nothing in a record says which, and no comparability rule can —
  the two runs are perfectly comparable, which is the point.

  **It also softens 10c's mid-level story.** `sam_vitb16` beats
  `supervised_vitb16` on ten of the thirteen boards, several substantially
  (occlusion edge 0.2680 against 0.1996, generic segmentation 0.6667 against
  0.6195, similarity 0.8695 against 0.8202, normals 34.56° against 36.72°). So
  "supervised training is weak at mid- and low-level" was partly a statement
  about the **AugReg recipe**, not about supervision.

  **The two classification rows are identical to four decimals** (0.9972 both),
  which is a saturated board rather than a duplicated checkpoint: the weights,
  cache keys and features are all distinct and a slow test pins that.


- **The semantic segmentation board is the one dense board that does not rank
  by feature resolution, and what it rewards instead is not any single
  structural variable** (2026-08-20, `scripts/analyse_board_correlates.py`).
  This closes the question 10e left open. That step showed the board cannot
  separate training objectives — a supervised backbone lands within 0.0011 of a
  pixel-reconstruction one — without saying what it *does* separate.

  Spearman of each board's ranking against feature-grid area, over the twelve
  backbones already in the corpus:

  | board | rho vs grid |
  | --- | --- |
  | generic_segmentation | **+0.958** |
  | surface_normal | +0.867 |
  | corner | +0.860 |
  | occlusion_edge | +0.853 |
  | depth | +0.818 |
  | edge | +0.734 |
  | **semantic_segmentation** | **+0.545** |

  And the +0.545 is carried by DINOv2, which has both the finest grid and the
  largest pretraining corpus: **drop those two rows and it falls to +0.212**;
  hold the pretraining data fixed (the eight IN1k backbones) and it is **exactly
  0.000**. Width explains nothing anywhere on this board (-0.035).

  **The control was already in the corpus, which is the part worth copying.**
  `generic_segmentation` runs on the *same 1449 VOC images* at the same
  resolution with the same linear head and the same schedule; the only thing
  that differs is whether the target has 2 classes or 21. Grid +0.958 against
  +0.545, pretraining size +0.238 against +0.615. Same pixels, same probe,
  opposite behaviour — so this is a property of **what the target asks for**,
  not of the dataset, the head or the protocol. No new run was needed to
  establish it.

  **What it tracks is weak and must be quoted as weak.** Pretraining corpus
  size is the best single correlate at **+0.615**, the highest of all thirteen
  boards, robust to the WebLI size assumption (+0.615 to +0.650) and to
  dropping DINOv2 (+0.539). But within the six *identical* ViT-B/16s — same
  shape, same width, same 196 tokens — it is only +0.314, and the spread there
  is **0.3207 mIoU** (clip_vitb16 0.6546 down to sam_vitb16 0.3339), larger
  than the spread across all four CNNs. Most of this board's variance is not
  architectural at all.

  **Nothing published needs retracting**: every run used identical settings, so
  the rankings are comparable and the corpus is unaffected. What changes is the
  *reading* — do not present this board as evidence about a training objective,
  and do not assume a dense board ranks by resolution because the other six do.
  **n=12, so these coefficients have wide error bars and the properties are
  correlated with each other.** It is a lead sized for the corpus, not a proof;
  `--drop` exists so the next person can check which conclusions survive.


- **The high-level tier is two clusters, not one — and the tier-mean test flips
  sign with the corpus, so read the clusters, not the mean** (2026-08-20,
  updated 2026-08-28, `analyse_board_correlates.py --section agreement`). The
  taxonomy's claim, tested as "probes within a tier agree with each other more
  than probes across tiers":

  | | n=9, 13 boards | n=12, 13 boards | n=12, 14 boards | n=12, 15 boards | n=12, 16 boards |
  | --- | --- | --- | --- | --- | --- |
  | within low_level | +0.761 | +0.825 | +0.825 | +0.839 | **+0.839** |
  | within mid_level | +0.666 | +0.666 | +0.666 | +0.666 | **+0.666** |
  | within high_level | +0.497 | +0.296 | +0.297 | +0.297 | **+0.373** |
  | across tiers | +0.340 | +0.304 | +0.265 | +0.266 | **+0.276** |

  At nine backbones every within-tier mean cleared the cross-tier mean. At
  twelve it did not — high-level landed below. Adding `scene_classification` as
  the fourteenth board moved it marginally back above, and **the within-high
  mean barely changed** (+0.296 → +0.297): what moved was the *cross-tier*
  mean, dragged down because the image-level high-level boards disagree sharply
  with the low-level tier — the most negative pair in the corpus is now
  `orientation` / `scene_classification` at **−0.51**. The fifteenth board,
  `orientation`, then *tightened* the low-level tier (+0.825 → +0.839): despite
  a target that is near-independent of every other probe's per pixel (it is a
  phase, `|r|` under 0.09 with `edge` and `corner`), its board ranks backbones
  almost exactly like `keypoints2d` (+0.95), `corner` (+0.82) and `edge`
  (+0.79). Mid-level is identical to three decimals across all four columns.

  The sixteenth board, `fine_grained_classification`, lifted the within-high
  mean the most any single board has (+0.297 → +0.373) and left the other two
  tiers unmoved to three decimals. That is not the high-level tier becoming
  coherent: it is a sixth board joining the *larger* of its two clusters, which
  raises the mean of a set that is still bimodal.

  So the sign of "high-level mean minus cross-tier mean" is within noise and
  has been on both sides. **The stable finding is the shape**: high-level is not
  one loose tier, it is two tight clusters that ignore each other; low-level is
  one cluster and neither `orientation` nor the CUB board changed that.

  | pair | rho |
  | --- | --- |
  | detection / fine_grained_classification | **+0.860** |
  | detection / semantic_segmentation | **+0.804** |
  | classification / retrieval | **+0.769** |
  | detection / scene_classification | **+0.720** |
  | fine_grained_classification / scene_classification | +0.671 |
  | fine_grained_classification / semantic_segmentation | +0.643 |
  | scene_classification / semantic_segmentation | +0.524 |
  | classification / fine_grained_classification | +0.343 |
  | classification / scene_classification | +0.161 |
  | classification / detection | +0.140 |
  | classification / semantic_segmentation | +0.140 |
  | fine_grained_classification / retrieval | +0.112 |
  | detection / retrieval | -0.035 |
  | retrieval / semantic_segmentation | -0.042 |
  | retrieval / scene_classification | -0.217 |

  Image-level categorisation on one side, localised prediction on the
  other, and **nothing between them** — and the two probes that ought to sit
  with `classification`, because they *are* `classification` with a different
  folder, both sit with the localised cluster instead.
  `scene_classification` was the first (+0.72 with detection, −0.22 with
  retrieval). **`fine_grained_classification` is the replication, and a
  sharper one**: its strongest partner anywhere in the corpus is `detection`
  at **+0.860** — the highest high-level pair there is, above
  detection/semseg — while it reaches only +0.343 with the object board it
  shares every line of its implementation with, and +0.112 with `retrieval`.

  Two independent probes now show the same thing, which is what moves this from
  a curiosity about Places365 to a property of the cluster. A place category is
  read from layout and spatial context; a species is read from localised parts —
  a beak, a wing bar — against a shared body plan. Both are what the
  VOC-dense probes reward and what single-object Imagenette does not.
  The ImageNet-1k-supervised backbones that top the object board are near-bottom
  on scene classification, on dense VOC, **and on CUB**: all four of them
  (`convnext_base`, `resnet50`, `supervised_vitb16`, `resnet18`) take places
  8-11 of twelve there, above only `mae_vitb16`.

  **What the mechanism is not.** It is not that these boards are simply harder,
  and it is not shared images: `fine_grained_classification` reads CUB, which no
  other board touches, and still lands nearest `detection` on VOC. The
  cross-source check two bullets down covers this generally.

  It also explains the semseg result in the bullet below — that board's nearest
  neighbour anywhere in the corpus is `detection`, not the two semantic boards
  it shares a tier with.

  **What first pulled high-level below the line was a controlled axis, not
  rows.** The three backbones between n=9 and n=12 are `supervised_vitb16`,
  `dino_vitb16` and `sam_vitb16` — all ViT-B/16 on ImageNet-1k, differing only
  in objective and recipe. The within-high mean fell from +0.497 to +0.296 the
  moment the corpus varied *objective* with capacity held down, and adding a
  fourteenth board did not move it back.

  **Do not read this as the taxonomy being wrong.** It comes from Chen, Marks &
  Cheng and it is what this library is organised around; what fails is the
  narrower claim that these five boards measure one thing. Treat `high_level`
  as a folder, not as a quantity to average over. **n=12, so the coefficients
  are wide** — `--drop` re-runs without any row, and the n=9 column above is
  exactly `--drop supervised_vitb16 --drop dino_vitb16 --drop sam_vitb16`.

  **The n=9 column reproduces a previous session's ad-hoc analysis to three
  decimals** (0.50 / 0.76 / 0.67 / 0.34, recorded then and never committed),
  which is the check that the script is measuring what that analysis measured
  rather than something new that happens to look similar.


- **The board clustering is not an artefact of shared datasets, and the check
  is one the corpus already contained** (2026-08-20,
  `analyse_board_correlates.py --section sources`). The obvious objection to
  the tier result below: `detection` and `semantic_segmentation` correlate at
  +0.804 and *both read VOC*, so the pairing might be about the images rather
  than the task. Two things refute it.

  **The identical-image pair is the weakest of the three VOC pairs.**
  `semantic_segmentation` and `generic_segmentation` read the **same 1449
  images** at the same resolution through the same head and schedule;
  `detection` reads 600 different VOC frames.

  | pair | rho | |
  | --- | --- | --- |
  | detection / semantic_segmentation | **+0.804** | different frames |
  | detection / generic_segmentation | +0.720 | different frames |
  | generic_segmentation / semantic_segmentation | **+0.538** | *same 1449 images* |

  Shared pixels as the cause predicts the opposite ordering. And
  `generic_segmentation`'s five nearest neighbours anywhere in the corpus —
  surface_normal +0.881, depth +0.867, corner +0.853, occlusion_edge +0.832,
  edge +0.769 — read **NYUv2 and Taskonomy**, not VOC.

  **Imagenette is the second, independent counterexample.** `classification`,
  `retrieval` and `correspondence` all read it and average **+0.128** — the
  lowest within-source figure in the corpus. Sharing a dataset is plainly not
  sufficient for agreement.

  **What the pooled numbers do say, and it is a real caveat.** Within-source
  agreement is +0.674 against +0.298 across sources, which is *comparable* to
  the tier split (within-tier means well above the +0.266 cross-tier). So
  dataset is about as good a
  predictor as tier — but that is carried by Taskonomy (10 pairs, +0.810) and
  NYUv2 (1 pair, +0.902), which are groups of dense geometric boards that
  would be expected to agree whatever they read. Neither grouping is the real
  structure; **what a board's target asks for** is, which is why
  `generic_segmentation` sits with the resolution-driven dense boards and
  `semantic_segmentation` does not.

  `SOURCE_IMAGES` is hand-written because the records cannot supply it —
  Imagenette, NYUv2 and the staged corner frames are **all called `val`** in
  the `dataset` field, so grouping on it would merge boards sharing nothing.


- **`scene_classification` ranks backbones almost independently of the object
  `classification` board — the two "classification" boards are not one
  measurement** (2026-08-28, 12 backbones, Places365 val, `--limit 100`).
  Spearman between the two orderings is **+0.16**.

  | | object (Imagenette) | scene (Places365) |
  | --- | --- | --- |
  | 1st | `convnext_base` 0.9997 | `siglip_vitb16` 0.4035 |
  | top of board | ImageNet-1k supervised CNNs | image-text ViTs (SigLIP, both CLIPs) |
  | `convnext_base` | 1st | 9th |
  | `supervised_vitb16` | 5th | 11th |
  | `mae_vitb16` | 12th | 10th |
  | spread | 0.041 (saturated, 11/12 above 0.988) | **0.132** |

  Three things are going on and they reinforce each other. **The object board is
  saturated** — a spread of 0.04 with eleven backbones clustered above 0.988
  cannot rank. **Imagenette's classes are ImageNet-1k wnids**, so the supervised
  CNNs' object numbers are in-distribution recall, not transfer, and Places365
  removes that advantage. And **scene category is a spatial-context task**:
  `scene_classification` correlates +0.72 with `detection` and +0.52 with
  `semantic_segmentation` but only +0.16 with object `classification` and −0.22
  with `retrieval` — it sits with the localised-prediction cluster of the
  high-level tier, not the image-level-categorisation one it nominally belongs
  to (see the tier finding above).

  Practical consequence: quote scene classification as a *separate* semantic
  result, and do not read a high object-classification number as saying
  anything about scene understanding — for the supervised CNNs it says close to
  the opposite. **n=12, one seed.**

- **`orientation` — a target that is independent per pixel and a board that is
  not** (2026-08-28, 12 backbones, the same pinned Taskonomy `tiny` frames as
  `corner` and `edge`). The orientation probe measures local gradient *phase*,
  and its target's per-image correlation with the `edge` and `corner` targets
  is **under 0.09** — where those two sit at 0.53. That was the pre-measurement
  criterion for building it at all: DoG-blob was rejected for the same check at
  0.51 with `corner`.

  But the *board* is not independent. Over the twelve backbones it ranks them
  almost exactly like `keypoints2d` (**rho +0.95**, one of the strongest pairs
  in the whole corpus), `corner` (+0.82), `edge` (+0.79) and `correspondence`
  (+0.85), and it *anti*-correlates with the image-level semantic boards
  (`scene_classification` **−0.51**, the corpus's most negative pair;
  `classification` −0.15). The spread is 18.8°–31.2° against a 45° chance
  floor, so every backbone is well clear of chance. `mae_vitb16` leads (as it
  does across the low-level tier), the image-text ViTs `siglip_vitb16` and
  `clip_vitb32` are *last* — the opposite of a semantic board — and DINOv2-S
  beats DINOv2-B (22.1° vs 24.6°), the "bigger is not better on low-level"
  pattern again.

  The lesson is that **target independence and board independence are separate
  properties**. Adding a probe whose target overlaps an existing one (like
  `corner` at 0.52 with `edge`) can still be worth it if the ranking differs;
  adding one whose target is *orthogonal* (like `orientation`) can still produce
  a board that says nothing new about the ordering. What `orientation` adds is
  not a new capability axis — it is one more backbone-ranking that agrees with
  the geometry cluster, which is itself the finding: a backbone good at
  localised structure is good at all of it, magnitude and phase alike. **n=12,
  one seed.**


- **`detection` does not reproduce to four decimals *on DINOv2*, it never did,
  and there is nothing to fix** (2026-08-13, narrowed 2026-08-14). It was
  recorded as an open mystery through two releases. The answer is that *every*
  probe's training is non-deterministic on GPU and detection is the only one
  whose metric can see it — **but whether it sees it depends on the backbone,
  and the first write-up of this bullet got that wrong.** See the "not a
  probe-wide property" paragraph below before quoting a spread.

  Measured on one V100, in this order, each step ruling out the obvious answer
  before reaching for the next:

  | measurement | result |
  | --- | --- |
  | head weights, same seed, twice | differ by **7.5e-09** — CUDA `conv2d` backward accumulates atomically |
  | the same under `use_deterministic_algorithms(True)` | **bit-identical** |
  | real detection probe, corpus flags, run twice | `map_50` **0.228834** then **0.229836** |
  | the same with deterministic kernels | **0.229867 both times**, every metric identical to six decimals |

  **`detections_per_image` is the tell**: 83.0033 against 83.0200 against the
  corpus's 83.0333. That is a *count* — about ten detections out of 49,800
  appear or vanish between runs as borderline cells cross the 0.05 score
  threshold. Nothing drifts smoothly; individual detections flip.

  **Why detection alone.** mIoU, Pearson correlation and angular error are
  continuous averages over ~10^5 pixels, so an independent 1e-8 perturbation
  averages *down* and never reaches the fourth decimal. Average precision is a
  **discrete ranking** over ~50,000 detections with hard thresholds at 0.05
  (score) and 0.5 (IoU). A ranking has no averaging to do. So the same noise
  every probe pays is attenuated by twelve metrics and amplified by one.

  **The corpus number is not wrong.** For `dinov2_vits14`, 0.229080 is one draw
  from a distribution roughly 1e-3 wide, and the two fresh runs bracket it. Do
  not "correct" the corpus toward any single rerun.

  **It is not a probe-wide property, and the first version of this bullet said
  it was** (2026-08-14). That claim was measured on `dinov2_vits14` alone and
  written as though it described `detection`. Both CLIP backbones were then run
  twice with the corpus flags on the same V100, and both are **bit-identical**
  — not to four decimals, to every digit, on `map_50`, `map_50_95` *and*
  `detections_per_image`, and identical to the committed corpus record as well,
  which was produced months earlier under v0.5.0. So detection reproduces
  exactly on some backbones and drifts on others:

  | backbone | dense width | grid at 224 | reproduces? |
  | --- | --- | --- | --- |
  | `dinov2_vits14` | 384 | 16x16 | **no** — 0.228834 / 0.229836 / corpus 0.229080 |
  | `dinov2_vitb14` | 768 | 16x16 | **no** — 0.2897 against the corpus's 0.2895 |
  | `clip_vitb16` | 768 | 14x14 | yes, 0.18940807014166364 twice + the corpus |
  | `clip_vitb32` | 768 | 7x7 | yes, 0.188608609616858 twice + the corpus |
  | `resnet18` | 512 | 7x7 | yes, 0.091190732340 **three times** + the corpus |
  | `resnet50` | 2048 | 7x7 | yes, 0.137981235614 **three times** + the corpus |

  **The correlate is the grid, and all six rows now agree** (the ResNets
  measured 2026-08-14, three repeats each, bit-identical on `map_50`,
  `map_50_95` *and* `detections_per_image`, and equal to their committed corpus
  records to every digit). Only the two 16x16 rows drift.

  Width is excluded twice over: DINOv2-B shares CLIP's 768 channels and drifts,
  while `resnet50`'s 2048 and `resnet18`'s 512 are both stable. Architecture
  family is excluded too — the CNNs behave like the 7x7 ViT, not like the other
  CNN-shaped thing. The mechanism that fits is cuDNN choosing an atomics-based
  backward algorithm at one spatial size and a deterministic one at another.

  This was recorded as "a lead on n=4, not a finding" until the two ResNet rows
  were measured. They were the cheap half of the experiment and they sat
  undone through two releases, which is the useful lesson: **the run that
  closes an open question is usually smaller than the write-up explaining why
  it is still open.**

  **What this buys concretely**: the corpus detection board's smallest adjacent
  gap is `clip_vitb16` 0.1894 against `clip_vitb32` 0.1886, i.e. **0.0008** —
  *below* the 1e-3 spread measured on DINOv2, which is why the pair was worth
  checking at all. Both of those rows turn out to be the exactly reproducible
  ones, so the ordering is verified rather than lucky and **no tie marking is
  needed**. Every other adjacent gap on that board is 0.04-0.06.

  Four things it is **not**, each excluded by measurement rather than argument:
  *version* (the corpus record is v0.5.0 and both reruns are v0.8.0, the only
  differing field — but the two v0.8.0 runs disagree with each other by more
  than either disagrees with the corpus); *TF32 or GPU generation* (`.venv`
  resolves only on `dgx1`/`dgx2`, which are V100s with no TF32 hardware, so
  every run this venv has ever done used identical arithmetic); *the metric*
  (both sorts are already `stable=True` and the matching follows `VOCevaldet.m`
  with no tie-break left to chance); and *seeding* (that was the `--push-to`
  bug, already fixed, and it moves a number far more than 1e-3).

  **Do not set the determinism flags to "fix" this.** They would make one
  machine agree with itself while still not making two machines agree, and
  buying that costs a one-time change to every published detection number in a
  committed corpus. Quote detection to **three** decimals instead — that is the
  safe floor on every backbone, including the ones that happen to reproduce
  exactly.

  A methodological note worth keeping, because it nearly ended the
  investigation early: a synthetic proxy for this said average precision was
  *insensitive* to perturbations below 1e-4, which read as a refutation. The
  toy task was simply too well-behaved — real DINOv2 features give a far
  stiffer loss landscape and a much larger divergence over 750 steps. **When a
  synthetic experiment contradicts a real disagreement, suspect the proxy
  before the hypothesis.**
