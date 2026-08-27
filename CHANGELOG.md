# Changelog

All notable changes to VisBench are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

Each released section is written to be pasted straight into a GitHub release,
so it stands on its own rather than assuming you have read the ones above it.

## [Unreleased]

### Added

- **`scene_classification` — a fourteenth probe, scene category rather than
  object category.** Mechanically it is the object-classification linear probe
  (`SceneClassificationTask` subclasses `ClassificationTask`); what differs is
  the question — the category of the *place*, its layout and context, which a
  backbone can be good at while being weak at object identity. It is a distinct
  probe with its own leaderboard board rather than a dataset flag on
  `classification`, because `board_for` renders one comparability group per
  task and records on a second dataset under one task name would make the
  object board unrenderable.

  The canonical dataset is Places365-standard, which
  `ImageFolderDataset` reads with no loader code (`train/<class>/` +
  `val/<class>/`, 365 classes). Ships with
  [`examples/scene_classify.py`](examples/scene_classify.py), a `visbench run
  scene_classification` / `visbench show scene_classification` CLI row, and a
  gallery figure.

  **The 12-backbone corpus board is committed** (`build_corpus.sh` runs it at
  `--limit 100`: the full 100/class official val scored, 100 training images
  per class). The corpus is now **168 records across fourteen boards**. Top-1,
  against the Imagenette object-`classification` board:

  | | object 1st → | scene 1st → |
  | --- | --- | --- |
  | | `convnext_base` 0.9997 (supervised CNN) | `siglip_vitb16` 0.4035 (image-text ViT) |
  | `convnext_base` | 1st | 9th |
  | `supervised_vitb16` | 5th | 11th |
  | `mae_vitb16` | 12th | 10th |
  | spread | 0.041 (saturated) | **0.132** |

  Spearman between the two orderings is **+0.16** — the two "classification"
  boards rank backbones almost independently. The object board is saturated
  (eleven of twelve above 0.988) and, for the ImageNet-1k-supervised backbones,
  measures in-distribution recall rather than transfer (Imagenette's classes
  are ImageNet-1k wnids). `scene_classification` correlates +0.72 with
  `detection` and +0.52 with `semantic_segmentation` but +0.16 with object
  `classification` — it joins the localised / spatial-context cluster of the
  high-level tier, not the image-level-categorisation one. See the
  `scene_classification` and updated tier findings in `CORPUS_FINDINGS.md`.

- **A resolution control, and the finding that feature resolution is not what
  DINOv2's dense lead is made of.** Feature resolution is the strongest
  structural correlate of every dense board (rho +0.50 to +0.96, where width
  correlates with nothing), and the only backbones carrying 256 tokens are the
  two DINOv2s — so grid size, the DINOv2 objective and LVD-142M pretraining
  were one variable, and no dense board could say which of the three it ranked.

  `dinov2_vitb14_196` separates them: the same weights and hub ref at 196px, a
  14x14 grid matching every ViT-B/16 in the corpus. **Matching the grid costs
  under 3% on all five dense boards**, and DINOv2-B keeps its lead over the
  entire ViT-B/16 pack on both boards it led — 0.7407 against `dino_vitb16`'s
  0.6838 on `generic_segmentation`, 0.7791 against `mae_vitb16`'s 0.6945 on
  `depth`. Resolution accounts for 21% and 7% of those two gaps respectively.

  On `surface_normal`, `edge` and `corner` DINOv2-B never led at all, so there
  was no lead to explain — the confound was narrower than the correlation table
  suggested, which is half the finding.

  The five records are in
  [`results/controls/resolution.jsonl`](results/controls/resolution.jsonl),
  **deliberately outside the corpus** although they pass `comparability_key`
  against every board they ran on. The corpus says what a backbone scores; a
  control says what changes when one thing about one backbone moves. Nothing in
  `results/controls/` feeds a generated table.

- **The board clustering survives a shared-dataset audit**
  (`--section sources`). The obvious objection to the tier result below is that
  `detection` and `semantic_segmentation` correlate at +0.804 and both read
  VOC. Two things in the corpus refute it. `semantic_segmentation` and
  `generic_segmentation` read the **same 1449 images** through the same head
  and agree *least* of the three VOC pairs (+0.538, against +0.804 and +0.720
  for pairs reading different frames), and `generic_segmentation`'s nearest
  neighbours are NYUv2 and Taskonomy boards. Independently, the three probes
  sharing Imagenette average **+0.128** — sharing a dataset is not sufficient
  for agreement.

  Pooled, within-source agreement is +0.634 against +0.345 across, comparable
  to the tier split — but that is carried by groups of dense geometric boards
  that would agree whatever they read. Neither dataset nor tier is the real
  structure; what the target asks for is.

- **Boards correlated against each other, and the high-level tier does not
  survive it.** `analyse_board_correlates.py --section agreement` ranks every
  board against every other and averages within and across tiers, which is the
  taxonomy's own claim stated as a number. Two tiers hold and one does not:
  low-level +0.825, mid-level +0.666, **high-level +0.296 against a cross-tier
  +0.304**.

  **The mean hides the shape.** High-level is not uniformly loose, it is two
  tight pairs that ignore each other — `detection`/`semantic_segmentation` at
  +0.804 and `classification`/`retrieval` at +0.769, with all four pairs
  between them from −0.042 to +0.140. Image-level categorisation on one side,
  localised VOC prediction on the other. A failing tier therefore prints pair
  by pair rather than only its mean, because a uniformly loose tier would
  question the probes while two clusters question the tier.

  It also explains the semantic segmentation result below: that board's
  nearest neighbour anywhere in the corpus is `detection`, not the semantic
  boards it shares a tier with.

  **What broke it was a controlled axis, not more rows.** At nine backbones
  every tier cohered (high-level +0.497). The three added since are all
  ViT-B/16 on ImageNet-1k differing only in objective and recipe, and the tier
  stopped cohering exactly when the corpus stopped varying only capacity. The
  n=9 column reproduces a previous ad-hoc analysis to three decimals, which is
  the check that this measures what that measured.

  **This is not the taxonomy being wrong** — it is the narrower claim that
  those four boards measure one thing. Do not average a backbone's high-level
  results into a figure of merit. n=12, so the coefficients are wide.

- **`scripts/analyse_board_correlates.py`, and the answer to what the semantic
  segmentation board actually ranks by.** Step 10e showed that board cannot
  separate training objectives — a supervised backbone lands within 0.0011 mIoU
  of a pixel-reconstruction one — and left "then what does it separate?" open.
  The script correlates each board's ordering against the structural properties
  of the backbones it ranked, using only records already in the corpus, so the
  answer cost no GPU time.

  **It is the only dense board that does not rank by feature resolution.** Every
  other one scores between +0.73 and +0.96 Spearman against feature-grid area;
  semantic segmentation is +0.545, which falls to +0.212 without DINOv2 and to
  exactly 0.000 with the pretraining data held fixed.

  **The control was already in the corpus.** `generic_segmentation` runs on the
  same 1449 VOC images at the same resolution with the same linear head and
  schedule, differing only in whether the target has 2 classes or 21 — and it
  ranks by grid at +0.958. Same pixels, same probe, opposite behaviour, so this
  is a property of what the target asks for rather than of the data or the
  protocol.

  What it rewards instead is weak: pretraining corpus size is the best single
  correlate at +0.615, the highest of the thirteen boards, but within six
  *identical* ViT-B/16s the spread is 0.3207 mIoU at only +0.314. **Nothing
  published needs retracting** — every run used identical settings, so the
  rankings are comparable. What changes is the reading. At twelve backbones the
  coefficients have wide error bars, which is what `--drop` is for.

### Fixed

- **A reconfigured backbone could silently delete the corpus row it was built
  to be compared against.** `latest_per_backbone` keys on `record.backbone` and
  keeps the newest — right for a re-run, catastrophic for a reconfiguration.
  `DINOv2.__init__` set `self.name = variant`, so the same weights at a
  different `image_size` reported `dinov2_vitb14`, and the newer record would
  have evicted the 224px number from every board it appeared on with nothing
  rendered saying the configuration had moved.

  Latent since v0.1 and unreachable until something was actually run at a
  second resolution, so no published number ever moved. `backbone_key` carried
  the resolution the whole time, so the **cache** was never at risk — two
  mechanisms that look like one.

  Three changes: `DINOv2` takes a `name=` so a configuration can declare what it
  calls itself; **`latest_per_backbone` now raises** when one name arrives under
  two `backbone_key`s, the posture `METRIC_DIRECTIONS` and `style_for` already
  take; and `register_backbone`/`register_task` take `name` positional-only, so
  a decorator parameter cannot shadow a constructor argument of the same name.

- `CLAUDE.md`'s registered-backbone list was missing `sam_vitb16`, which shipped
  in v0.11.0. The generated tables and `LEADERBOARD.md` had it throughout.

## [0.11.0] — 2026-08-20

**Two more backbones, and with them the corpus stops being a set of comparisons
and becomes a set of controls.** v0.10.0 made one comparison honest by holding
architecture and pretraining set fixed across a supervised ViT-B/16 and an MAE
one. v0.11.0 finishes the job at both ends: `dino_vitb16` adds a *third* value
of that same variable, and `sam_vitb16` varies something else entirely — the
training recipe — so that an objective gap finally has a denominator. Thirteen
probes against **twelve** backbones, **156 records**, every one of the thirteen
comparability groups holding all twelve. Both merges were purely additive, so
every number v0.10.0 published is unchanged here.

**The control refuted half of the family's own published claim, before either
shipped.** `sam_vitb16` differs from `supervised_vitb16` only in how it was
trained to the same objective — sharpness-aware minimisation with light
augmentation against AugReg's AdamW with heavy augmentation. On semantic
segmentation that recipe change costs 0.2452 mIoU, which is *larger than the
entire spread across training objectives* on the same board: a supervised
backbone trained with labels lands at 0.3339, within 0.0011 of the
pixel-reconstruction one, and last of twelve. Whatever that board measures, it
is not the objective. Retrieval survives the same test enormously — the
objective effect there is 234x the recipe effect — so the family's high-level
result stands on retrieval and no longer cites semantic segmentation.

**The rule this leaves behind: quote an objective gap against the recipe gap on
the same board, not against zero.** Seven of the thirteen boards have objective
spreads under 3x their recipe gap. Nothing in a record says which, and no
comparability rule can — the two runs are perfectly comparable, which is the
point.

**`dino_vitb16` answers what the supervised/MAE pair could not.** DINO is
label-free and semantic, so it separates "trained with labels" from "trained
toward semantics", which the earlier pair moved together. It scores 0.9192 mAP
on retrieval against supervised's 0.9947 and MAE's 0.1883: a label-free
objective that learns categories recovers nearly everything labels buy, while
one that learns pixels recovers almost none of it. It also declines to pay
MAE's price at the other end — its worst placing anywhere is ninth of twelve,
and it takes mid-level similarity (0.9019, beating the 0.8701 that had stood
since v0.2) and 2D keypoints (0.2850) outright. **The high-versus-low
trade-off the MAE row looks like it demonstrates is therefore not a law.**

Also here: `examples/custom_backbone.py`, closing a gap between a capability
this project documented and one it had ever demonstrated.

### Added

- **`examples/custom_backbone.py` — the capability that was documented and
  never demonstrated.** `CustomBackbone` appeared in the README and in tests and
  nowhere in `examples/`, while all thirteen probes had one. It wraps
  torchvision's ResNet-18, probes it through ordinary `visbench.run()` calls,
  shows the cache key moving when the weights change, and registers a named
  subclass — on generated data, so it needs no dataset and no download beyond a
  45 MB core-dependency checkpoint.

  **Measuring it turned up something not previously written down.** A wrapped
  model is constructed *before* `run()` seeds; a registry name is constructed
  *after*. So a trained probe's head is initialised from a different RNG state
  on the two paths, from features that are otherwise bit-identical (max absolute
  difference 0.0): classification top-1 0.9125 wrapped against 0.9062 named.
  Each path's own spread across five seeds is 0.0062, **the same size**, so this
  is RNG jitter and not a cost of wrapping — and the wrapped path is *perfectly*
  reproducible, including under RNG consumed before `run()` is called, which is
  the opposite of what the hazard sounds like. Zero-shot probes are identical
  bit for bit, since no head is fitted.

  Writing it also found three bugs in itself that only running it could show: a
  `FeatureCache(str)` that mypy rejects, a `backbone.cache_key()` call that is a
  string on the registered path, and a module returning pooled `(B, C)` where
  the probe wants a conv map. **The rule that a capability wants a real run
  behind it earned itself again.**

- **`sam_vitb16`, a recipe control — and it refuted half of the entry below
  before either shipped.** `vit_base_patch16_224.sam_in1k` is the same
  architecture, pretraining set, labels and input normalisation as
  `supervised_vitb16`, differing only in how it was trained to that objective:
  sharpness-aware minimisation with light augmentation, against AugReg's AdamW
  with heavy augmentation. Twelve backbones now, **156 records**.

  It supplies a denominator nothing else in this corpus had. **A gap between
  two objectives means nothing until you know how large a gap two runs of the
  same objective can produce** — and every other pair here differs in
  architecture, data or objective, while this one differs in none of the three.

  **On semantic segmentation the recipe gap is larger than the whole objective
  spread.** `sam_vitb16` scores **0.3339** mIoU against `supervised_vitb16`'s
  0.5791 — and `mae_vitb16`'s 0.3350. A supervised backbone with labels lands
  within 0.0011 of the pixel-reconstruction one and comes **last of twelve**,
  under settings identical across all twelve runs. Whatever that board
  measures, it is not the training objective, and the entry below no longer
  cites it.

  **Retrieval survives the same test enormously**: the objective spread there
  is 0.8064 against a recipe gap of 0.0034, a factor of **234**.

  **The rule this establishes is to quote an objective gap against the recipe
  gap on the same board, never against zero.** Seven of the thirteen boards
  have objective spreads under 3x their recipe gap. No comparability rule can
  flag this — the two runs are perfectly comparable, which is exactly the
  point.

  It also softens v0.10.0's mid-level story: `sam_vitb16` beats
  `supervised_vitb16` on ten of the thirteen boards, so "supervised training is
  weak at mid- and low-level" was partly a statement about the AugReg recipe
  rather than about supervision.

- **`dino_vitb16`, which turns the objective comparison from a pair into a
  family.** `vit_base_patch16_224.dino` is the same architecture and the same
  pretraining set as `mae_vitb16` and `supervised_vitb16` — 12 blocks, 768 wide,
  patch 16, one prefix token, ImageNet-1k — so one variable now takes three
  values rather than two: supervised labels, masked pixel reconstruction, and
  self-distillation.

  What the pair could not separate is **"trained with labels" from "trained
  toward semantics"**. DINO is label-free and semantic, so whichever side of the
  high-level boards it lands on answers that.

  **It is also the tighter of the two comparisons.** `augreg_in1k` normalises
  with mean/std 0.5 where `mae` and `dino` both use ImageNet statistics, and
  each checkpoint is preprocessed with the statistics it was trained under —
  the only correct handling — so supervised-against-MAE varies the input
  normalisation alongside the objective, while DINO-against-MAE does not. A
  test asserts that on the resolved transform rather than leaving it to a
  comment.

  `deit3_base_patch16_224.fb_in1k` is the near-miss and is refused: it matches
  every property the family is pinned on and carries `LayerScale` where these
  three carry `Identity`, so it would move the architecture as well as the
  recipe.

  **Measured: thirteen records, so the corpus is eleven backbones and 143.**
  Purely additive — 13 lines added, none removed — so every number v0.10.0
  published is byte-identical.

  **The answer is that high-level structure comes from a semantic training
  signal, not from labels.** On retrieval, DINO scores **0.9192** against
  supervised 0.9947 and MAE 0.1883 — MAE last of eleven, DINO fourth. A
  label-free objective that learns *categories* recovers nearly all of what
  labels buy, and one that learns *pixels* recovers almost none of it, which
  the earlier pair could not separate because it varied both at once.

  **Retrieval is the board that carries this, and semantic segmentation is
  not** — see `sam_vitb16` below, which was added afterwards and refuted that
  half before it shipped.

  **And DINO does not pay MAE's price at the other end.** MAE leads five boards
  and comes last on four; DINO's worst placing anywhere is eighth of eleven, on
  a classification board where nine of the eleven score above 0.98. It also
  takes **two overall firsts** — mid-level similarity at 0.9019, ahead of the
  0.8701 that had stood since v0.2, and 2D keypoints at 0.2850 — plus second on
  edges, corners and correspondence. The high/low trade-off the MAE row appears
  to demonstrate is therefore not a law: one objective is strong at both ends.

  **The high-level sweep is three boards and a tie, not four.** Supervised beats
  DINO on detection by 0.0009 (0.1669 against 0.1660), which is below the ~1e-3
  spread this project has measured on detection, so those two rows are a tie at
  the three decimals detection is quoted to. Classification is saturated and
  separates nothing either.

### Changed

- **The published counts the two new backbones moved.** The generated tables
  and `LEADERBOARD.md` were regenerated with each merge; the prose counting
  them was not, so the README and `docs/tasks.md` still described a nine- or
  ten-backbone board. MAE is first on **five** boards and last on **three**,
  not six and four: it lost 2D keypoints to `dino_vitb16` (0.2850) and
  `sam_vitb16` (0.2696), and it is no longer last on semantic segmentation
  because `sam_vitb16` scores 0.3339 beneath it. Both corrections have the same
  shape — **a count over a corpus is a fact about that corpus, not about the
  backbone**, and neither number moved because MAE changed.
- **`docs/roadmap.md` no longer claims the winner changes at the tier
  boundary.** Mid-level similarity crosses it, so the tiers have to be counted
  from `record.level` rather than from which boards feel semantic, and 10e
  weakened the claim further by showing the semantic-segmentation board cannot
  separate objectives at all.
- **`CLAUDE.md` is split, and `ENGINEERING_LOG.md` is new.** The contributor
  file had passed the 150k-character limit it is loaded under, at 203k, so the
  closed v0.3 step write-ups and the superseded release histories moved to an
  archive at the repo root while the rules they established stayed behind.
  203k -> 132k. No user-facing behaviour changes; the file is internal.


## [0.10.0] — 2026-08-19

**Four more backbones, and with the last of them the corpus stops being a table
and becomes an experiment.** `TimmBackbone` learned to read a ViT's own
structure rather than assuming a CNN's, which added ConvNeXt-B, MAE ViT-B/16 and
SigLIP-GAP ViT-B/16 in one change; a supervised ViT-B/16 followed. The record
corpus is **thirteen probes against ten backbones, 130 records**, every one of
the thirteen comparability groups holding all ten.

**The three tiers of the task taxonomy visibly separate for the first time.**
Through v0.9.0 the boards mostly reproduced a single capacity ordering — DINOv2
above CLIP above ResNet on nearly everything — which made the high/mid/low split
look like a taxonomy the numbers merely tolerated. `mae_vitb16` is first on six
of the thirteen boards and last on four, which no backbone here had done before:
it leads all three low-level probes plus correspondence, occlusion edges and
surface normals, and comes last on classification, retrieval, semantic
segmentation and mid-level similarity. **"Which backbone is best" is not a
well-formed question against this corpus**, and a summary that picks a winner is
discarding the result.

**`supervised_vitb16` is what turns that observation into a controlled
experiment.** It is `vit_base_patch16_224.augreg_in1k` — the same architecture
and the same pretraining set as `mae_vitb16`, so the only variable between those
two rows of any board is the training objective. Every other pair of backbones
in this corpus varies at least two things at once, which is why no earlier board
could say what a gap belonged to. Supervised wins all four high-level boards;
MAE wins all three low-level ones and five of the six mid-level ones. Five
boards to eight, one variable.

**The documentation gallery is drawn on real photographs**, Open Images frames
under a per-image licence check rather than generated scenes — the same
redistribution rule as before, satisfied by better sourcing rather than waived.

**Purely additive on the measurement side.** Every number v0.9.0 published is
byte-identical here; the corpus gained 52 records and lost none, and no ranking
already quoted moved.

### Added

- **A supervised ViT-B/16, and with it the corpus's first controlled
  experiment.** `supervised_vitb16` is `vit_base_patch16_224.augreg_in1k`: the
  *same architecture* and the *same pretraining set* as `mae_vitb16`, differing
  only in training objective. Ten backbones now, 130 records, thirteen
  comparability groups all holding all ten.

  `augreg_in1k` rather than the stronger 21k recipes precisely because of that.
  Every other pair of backbones in this corpus varies at least two things at
  once — architecture and data, or data and objective — so no board could say
  which of them a gap belonged to. This pair varies one.

  **The result tracks the taxonomy's tiers, with one crossing.** Supervised
  wins **all four high-level** boards — classification 0.9972 against 0.9582,
  retrieval **0.9947 against 0.1883**, semantic segmentation 0.5791 against
  0.3350, detection 0.1669 against 0.1296. MAE wins **all three low-level**
  ones — edge, keypoints2d, corner — and **five of the six mid-level** ones:
  depth, surface normals (27.52° against 36.72° mean), correspondence,
  occlusion edges, generic segmentation. Five boards to eight.

  **The exception is mid-level similarity**, where supervised wins 0.8202
  against 0.6897, and it is worth naming rather than rounding away. One
  plausible reading, offered as a hypothesis rather than a result: NIGHTS'
  images are Stable Diffusion generations prompted with categories drawn from
  ImageNet, CIFAR-10/100, Flowers-102, Food-101 and SUN397, so a 2AFC over them
  may reward category familiarity more than the "perceptual and geometric
  resemblance" the probe is meant to isolate. That would make it the
  in-distribution caveat again rather than a genuine mid-level win, and it is
  untested either way.

  **The classification and retrieval rows carry the in-distribution caveat**
  that `resnet50` and `convnext_base` already do, and more strongly: Imagenette's
  classes are ImageNet-1k wnids and this backbone was trained on ImageNet-1k
  with labels, so 0.9972 and 0.9947 are closer to recall than to transfer. The
  gap against MAE on those two boards is not a measurement of transfer quality.

- **The documentation gallery is drawn on real photographs.** All thirteen
  figures in `docs/_static/gallery/` are now Open Images frames rather than
  generated scenes, fetched by a new `scripts/fetch_gallery_frames.py`.

  **The licence is checked, not assumed.** Every frame is CC BY 2.0, verified
  per image against an allowlist at fetch time, and refused if its metadata
  carries no author or no landing page — an unattributable CC BY image is one
  this repository may not redistribute. `assets/gallery_frames/CREDITS.md` is
  generated beside the frames, and `tests/test_gallery_licences.py` fails if a
  committed photograph has no credit or a credit names no photograph. `NOTICE`
  records the whole arrangement.

  **What each figure can show is decided by that constraint, and each says
  which it is.** `corner` and `correspondence` have exact ground truth computed
  from the frame itself; `detection`, `generic_segmentation` and
  `semantic_segmentation` draw Open Images' own boxes and instance masks;
  `classification` and `retrieval` need only which folder a photograph is in.

  **`depth`, `surface_normal`, `keypoints2d` and `occlusion_edge` drop the
  target column entirely** and show what a published VisBench head predicts,
  named in the footer. Those four need sensor or reconstruction geometry that no
  redistributable photograph carries, and a three-column figure with an invented
  middle column would teach the wrong convention to exactly the reader who came
  to learn it. They are drawn on interior frames, because their heads were
  fitted on NYUv2 rooms and Taskonomy buildings and a photograph filled by an
  animal's face shows domain shift rather than the probe. They render at 224
  rather than the gallery's 160: a trained head's `output_size` is fitted state,
  so these heads emit 224x224 whatever they are fed, and beside a 160 crop the
  two panels are different sizes *and* different framings.

  The datasets the probes are actually scored on — VOC, ImageNet, NYUv2,
  Taskonomy, NIGHTS — still appear nowhere in this repository, which is the
  constraint that produced the generated gallery in the first place.

- **Three more backbones: ConvNeXt-B, MAE ViT-B/16 and SigLIP-GAP ViT-B/16.**
  Nine registered now, across four families. `TimmBackbone` had refused
  transformers outright, and rightly for what it then was: `has_cls_token` and
  `patch_size` were *class* attributes declaring "CNN" for every model, and a
  false `has_cls_token` discards the CLS token while the record claims there was
  none to keep. It reads both per instance now, so any timm ViT is usable — the
  three registered here are examples rather than special cases.

  **`pooling="default"` is read from timm's `global_pool`** rather than inferred
  from whether a CLS token exists. The base class's "CLS if there is one, mean
  otherwise" is a good default and only a proxy: a ViT can carry a CLS token and
  still be trained to average. MAE reports `token` and SigLIP-GAP reports `avg`,
  so two models of identical shape resolve `default` differently — each to what
  it hands its own classifier, the rule the ResNets already followed.

  **SigLIP is the `_gap_` variant deliberately.** Canonical SigLIP pools with an
  `AttentionPoolLatent` (`global_pool='map'`) — a trained module, not a
  reduction over tokens, so it cannot be a pooling *mode* over the features the
  cache stores. That is refused by name, with a message saying which sibling to
  use. The dense features are the same SigLIP features either way.

  **ConvNeXt is a documented exception to a rule this module has stated since
  v0.2.** Its head is `avg -> LayerNorm2d`, so what the model hands its
  classifier is `norm(mean(x))` while VisBench returns `mean(x)` — they differ
  by 27.5 at most on one frame. Both invariants cannot hold, since LayerNorm
  across channels does not commute with a spatial mean. The one kept is that
  **`pooled` is always a reduction of `dense`**, because that is what the cache
  stores and what every pooling task reduces. A test pins which backbones match
  their own head and that ConvNeXt does not.

- **The corpus at 13 probes x 9 backbones — 117 records, and the first time the
  three tiers visibly separate.** The three backbones above are measured now,
  not merely registered: 39 new records, every one of the thirteen groups
  holding all nine. **Purely additive** — 39 lines added, none removed, so every
  number published before this is byte-identical and no ranking already quoted
  moved.

  **`mae_vitb16` is first on six of the thirteen boards and ninth on four**,
  which no backbone in the corpus has done before. It leads all three low-level
  probes (edge, 2D keypoints, corner) plus correspondence, occlusion edges and
  surface normals, and comes **last** on classification, retrieval, semantic
  segmentation and mid-level similarity. Six of the thirteen boards are now
  headed by a model that is last on the other four, so "which backbone is best"
  is not a well-formed question against this corpus — which is what the task
  taxonomy claims and what the previous six-backbone corpus could only hint at,
  because its rows mostly reproduced one capacity ordering.

  **Retrieval's 0.1883 for MAE is barely above the 0.1 chance floor and is not
  a broken run.** The check is internal to the corpus: the same features score
  0.9582 top-1 under a *trained* linear probe. A learned projection recovers
  category structure that cosine similarity on the raw CLS token cannot, which
  is MAE's documented behaviour without fine-tuning — and an extraction bug
  would have taken the linear probe down with it.

  MAE's correspondence win is **not** its grid, which is the first question that
  board invites: it scores 0.3577 against a `ceiling_recall@5px` of 0.3932,
  where `dinov2_vits14` scores 0.3049 against a *higher* ceiling of 0.4123. Read
  against the ceiling the gap widens rather than closing.

  `convnext_base` tops classification (0.9997) and retrieval (0.9890) and then
  places seventh or lower on every dense geometric board. It carries the same
  caveat `resnet50` already did and `docs/tasks.md` now names both: it is
  ImageNet-1k **supervised**, and Imagenette's ten classes are ImageNet-1k
  wnids, so those two scores are close to in-distribution recall rather than a
  transfer result.

### Fixed

- **`slurm/corpus.sbatch` could never schedule the `corner` probe.** Its
  `PROBES` array had twelve entries where `build_corpus.sh`'s `ALL_PROBES` has
  thirteen; `corner` shipped in 8a/8b and was added to one file and not the
  other, so it sat unschedulable through v0.9.0. The sbatch's own guard could
  not catch it — that guard multiplies its *own* probe list by the backbone
  list, so a twelve-probe matrix is self-consistently the wrong size, and the
  corpus it produces looks complete because every group present still holds
  every backbone. `tests/scripts/test_corpus_scripts.py` now pins the two lists
  equal to each other and to `list_probes()`.

- **The same guard accepted any array range ending at the right index.** It
  checked only `SLURM_ARRAY_TASK_MAX`, so `--array=3-38` on a 39-task matrix
  passed while silently omitting the first probe, and a strided or
  comma-separated range passed however much it skipped. It checks
  `SLURM_ARRAY_TASK_COUNT` as well now, which is the number of tasks actually
  submitted and the only one of the two that can see a hole in the middle.

## [0.9.0] — 2026-08-14

**Every probe can now be looked at, not only scored.** `visbench show` writes a
grid of image / target / prediction panels for all thirteen probes, across four
renderers, and a test asserts `show_probes() == list_probes()` so a new probe
cannot ship undrawable. It measures nothing and records nothing: it exists
because a dense target that has drifted from the image it belongs to fails
*silently* — the probe trains, and the number merely comes out mediocre, which
reads as a hard task or a weak representation. Those are the two explanations
this library exists to tell apart.

Three of this project's own bugs are the argument, and each renderer states the
diagnostic its history calls for as a **figure** rather than leaving it to the
eye: **coherence** separates a broken geometry from a weak backbone, **class
balance** catches a split collapsed to one class scoring 1.0, and **vote
balance** reveals a vote read from the wrong CSV column. None is a score and
none is recorded.

Alongside it: `visbench run --save-probe` writes a trained head locally with no
Hub account, `--push-to` publishes one, and a rendered figure per probe now
ships in the README and on the docs site — generated from synthetic scenes with
exact ground truth, so the gallery rebuilds from one command with no downloads
and no dataset's redistribution terms to honour.

**Every measurement v0.8.0 reported, v0.9.0 reports identically**, with one
correction and one precision change, both below: a seeding bug in `--push-to`
that moved trained numbers, and detection quoted to three decimals rather than
four.

### Added

- **`visbench show` — the first visualisation anywhere in the package.** It
  writes a grid of image / target / prediction panels to a file, measures
  nothing and records nothing. Nine probes have a spatial target to draw: the
  eight dense ones plus `detection`, whose boxes are drawn straight onto the
  crop in the post-transform pixel coordinates the dataset returns.

  It exists because a dense target that has drifted from the image it belongs
  to fails **silently** — nothing raises, the probe trains, and the number
  merely comes out mediocre, which reads as a hard task or a weak
  representation. Those are the two explanations this library exists to
  separate. Both the correspondence misalignment that scored `recall@1px =
  0.003` and VOC's palette PNGs read through `convert("L")` (classes
  `[0, 1, 15, 255]` becoming `[0, 38, 147, 220]`) were found by reading code,
  and both are obvious in one frame.

  Three rules, each with its own guard in `tests/viz/`. **The viewer applies no
  geometry of its own** — no resize, no re-read, no second crop; a viewer that
  did could make a misaligned pipeline look fine and a correct one look broken,
  which is worse than no viewer at all. **An invalid pixel is drawn magenta per
  that probe's own convention** — there are four conventions across the nine
  probes (`0`, the zero vector, negative, `NaN`) and none is visible in a
  tensor's shape, so `TARGET_STYLES` is a listed table that raises on an
  unlisted probe, the posture `METRIC_DIRECTIONS` already takes. **A prediction
  is scaled by the target's range**, since scaling each panel to its own
  extremes makes a prediction at half the target's magnitude render identically
  to a correct one.

  Pillow and numpy only — no matplotlib, and no dependency change. New
  `visbench/viz/` (`styles`, `colour`, `panels`) and `docs/show.md`.

- **`visbench show` now covers every probe.** `classification`, `retrieval` and
  `similarity` were the three left out, because none has a spatial target —
  nothing to lay beside the image at the same resolution. What they have is a
  *decision*, and the new `visbench/viz/gallery.py` draws that: a contact sheet
  of frames and their labels, a query with its nearest neighbours, and a triplet
  with the human vote marked. `show_probes() == list_probes()` is now asserted,
  so a new probe cannot ship undrawable by accident.

  Each carries the diagnostic its own history calls for, stated as a figure
  following `error_coherence`'s precedent:

  - **class balance** — `subset(n)` on a labelled folder takes a prefix and the
    file list is grouped by class, so an Imagenette prefix is entirely class 0
    and the run **scores 1.0 while measuring nothing**. The footer reads
    `1 class, 8 items ... any score here is an artefact` whichever frames were
    drawn. Frames are picked *spread across* the split for the same reason:
    drawing a prefix would reproduce the artefact the sheet exists to reveal.
  - **vote balance** — the NIGHTS candidates are presented in arbitrary order,
    so the human vote should sit near 50%. Far from it means the vote was read
    from the wrong CSV column, which otherwise surfaces only as a mediocre
    accuracy. Drawn, it is unmistakable: the "preferred" candidate is visibly
    the more distorted one.

  **Retrieval loads the whole split** regardless of `--frames`, because
  leave-one-out retrieval over four images ranks each against three
  alternatives — shortening the split would not shorten the drawing, it would
  destroy what is drawn. `--limit` is now an explicit flag for capping it, and
  is distinct from `--frames`, how many rows to draw.

  `--backbone` now defaults to `None` and is required only where something must
  be computed: `correspondence` and `retrieval`, whose content *is* the
  features, and anywhere `--predict-from` is used. `similarity` draws the human
  vote without one.

- **`visbench show correspondence` — the pair renderer.** Two views side by
  side with the matches drawn between them, green where a match landed within
  `--threshold` pixels of where the geometry says it should have and red
  otherwise, plus an amber segment from the expected position to the actual one.
  Tenth drawable probe, and the one the panel grid could not express: it has two
  images and a geometric relation, not an image and a target of the same shape.

  **This is the panel that would have caught `recall@1px = 0.003`**, and the
  reason is that the bug does not look like noise. A homography in original
  pixels against features from a 224 centre crop makes every match wrong *in the
  same direction*; a coherent field of long errors is a broken pipeline where
  scattered short ones are a weak backbone, and no recall figure separates them.

  So the row states it as a number: **`error_coherence`**, the mean resultant
  length of the error directions. Measured on 224px homography pairs with
  ResNet-18 features — **0.40 and 0.29** for correctly-scored pairs against
  **0.98 and 1.00** for the same pairs with the homography in the wrong pixel
  frame, while the median error moved 10-23px → 227-294px. The median alone
  cannot tell "broken" from "hopeless"; the coherence can. It is a diagnostic,
  never a score, and is not recorded.

  Matches are sampled **evenly** across the kept set, not taken from the front:
  they arrive sorted by similarity, so a prefix would draw a systematically
  better picture than the score describes. The probe always needs a backbone —
  the matches do not exist until features do — and `--predict-from` is refused
  by name, since correspondence is zero-shot and has no saved head.

- **`CorrespondenceTask.match_details`** returns every kept match as points and
  errors in the working frame. `_pair_errors` now calls it, so a drawn panel and
  a reported number come from one code path by construction — a second copy of
  the geometry would put a drawing that *vouches for* a wrong number one edit
  away.

- **`visbench run --save-probe PATH`** writes the trained head, with the
  backbone identity beside it, to a local file. It wraps the existing
  `save_probe` and needs no Hub account — before it, the only way to get an
  artifact from a shell was `--push-to`, so `show`'s prediction column had no
  CLI-producible input. A zero-shot probe is refused before the run, as
  `--push-to` already was.

- **A rendered figure for every probe, in the README and on the docs site.**
  Thirteen `visbench show` pages under `docs/_static/gallery/`, generated by
  `scripts/render_gallery.py`.

  **Every frame is synthetic, and that is a licensing decision before it is a
  convenience one.** VisBench ships to PyPI under MIT, and the datasets these
  probes normally run on — VOC, ImageNet, NYUv2, Taskonomy, NIGHTS — each
  restrict redistribution to some degree while none clearly grants it, so
  committing panels containing their photographs would put third-party imagery
  in an MIT package. That is the line this project already took when it declined
  to vendor probe3d's CC BY-NC code.

  Generating buys three things beyond the licence. The gallery **rebuilds from
  one command with no downloads**, so a figure cannot drift from what the code
  draws. The **ground truth is exact** — sphere normals analytic, depth from the
  z-order, masks and boxes by construction — so each panel shows its probe's
  convention rather than an approximation. And **invalid pixels are placed
  deliberately**: every scene carries a sensor dropout, so the magenta marker
  appears where it should, which a lucky real frame might never show.

  Only the pixels are synthetic. Every figure is produced by the real command
  over a real dataset class, through the real renderers.

  The figures live under `docs/_static/` rather than `assets/` because Sphinx
  cannot follow a path that escapes its source tree, and MyST does not warn
  about one — so `-W` would not catch it. They are excluded from the sdist,
  which they would otherwise nearly triple.

- **The Hub integration has an example and a documentation page.** `push_probe`
  and `load_probe_from_hub` shipped in v0.6.0 with tests but no runnable
  demonstration, so the half of the feature that touches the network was
  reachable only by reading source. New `examples/push_probe.py` covers the
  round trip on real DINOv2 weights and **does not upload unless `--push` is
  passed** — the default prints the identity block and the generated model card
  so you can see what would go out, and `--pull` does the other direction with
  `--revision` for pinning a commit.

  New `docs/hub.md` is the reference: what the four identity fields prevent,
  why a download is read with `weights_only=True`, why a push is private by
  default, and what is refused (zero-shot probes, unfitted probes, artifacts
  from a newer `ARTIFACT_VERSION`) before anything is created.

- **`visbench run --push-to REPO_ID` publishes the head it just trained**, with
  `--public` to override the private default. One command produces the record
  and the artifact, which is the point: a head is only meaningful against the
  features it was fitted on, and those are decided by the run's own flags. A
  separate publishing step is free to drift from them, and a head trained under
  drifted flags uploads, loads and scores without complaint.

  A zero-shot probe is refused **before** the run rather than by `save_probe`
  afterwards — the same error either way, but one of them costs the whole run.

- **`scripts/build_corpus.sh` takes `PUSH_TO`**, so a whole board can be
  published from the file that already holds every probe's flags rather than a
  second copy of them, and `scripts/publish_collection.py` groups the results
  into one Hugging Face collection. Neither uploads without an explicit flag.

### Changed

- **The CLI's flag helpers were re-cut so `show` and `run` share one source.**
  Every probe's flags are now `<view flags> + _schedule_flags`, and `show`
  composes its surface from the first half rather than a parallel copy — a
  second copy could build a *different dataset* than `run` would from the same
  command line. `--image-size` moved from the head group to the data group,
  where it belongs: it decides the dataset's resize and centre crop.
  `run`'s surface is unchanged, flag for flag and default for default, pinned
  by `test_run_flags_are_unchanged_by_the_split`.

- **Detection's reported precision is three decimals, not four.** The probe has
  never reproduced to four on DINOv2, and the reason turned out not to be a
  defect. Every probe's training is non-deterministic on GPU — `conv2d` backward
  accumulates atomically, giving head weights that differ by ~7.5e-09 between
  two runs from the same seed — but detection is the only probe whose metric can
  see it. Twelve probes report continuous averages over ~10^5 pixels, where
  independent noise averages down and never reaches the fourth decimal. Average
  precision is a discrete ranking over ~50,000 detections with hard thresholds
  at 0.05 and 0.5, and a ranking has no averaging to do.

  Measured on one V100 with the corpus flags: two back-to-back runs on
  DINOv2-S/14 scored `map_50` 0.228834 and 0.229836, with `detections_per_image`
  moving 83.0033 to 83.0200 — roughly ten of 49,800 detections crossing the
  score threshold. Under `torch.use_deterministic_algorithms(True)` two runs are
  identical to six decimals on every metric, which is what pins the cause to the
  kernels.

  **Whether the drift appears depends on the backbone, and it tracks the feature
  grid.** Measured on all six: only the two 16x16 rows (DINOv2-S and B) drift.
  `clip_vitb16` (14x14), `clip_vitb32` (7x7), `resnet18` (7x7) and `resnet50`
  (7x7) are *bit*-identical between runs and equal to their committed corpus
  records to every digit — the ResNets across three repeats each.

  Channel width is excluded twice over: DINOv2-B shares CLIP's 768 and drifts,
  while ResNet-50's 2048 and ResNet-18's 512 are both stable. So is architecture
  family — the CNNs behave like the 7x7 ViT. What fits is cuDNN selecting an
  atomics-based backward kernel at one spatial size and a deterministic one at
  another.

  It also settles the one board question this raised. The smallest adjacent gap
  on the detection board is CLIP-B/16 over B/32 at 0.0008, below the DINOv2
  spread; both of those rows reproduce exactly, so the ordering stands and
  neither is marked as tied.

  **No number changed and the corpus is untouched.** DINOv2-S's 0.229080 is one
  draw from a distribution about 1e-3 wide, and both reruns bracket it. The
  determinism flags are deliberately *not* set: they would make one machine
  agree with itself without making two machines agree, at the cost of a
  one-time change to every published detection number.

- `docs/index.md` said twelve probes in two places. There are thirteen.

### Fixed

- **`--push-to` silently changed the number it published.** The CLI built the
  backbone itself when that flag was given, so it could hand the same object to
  `push_probe` — which put construction *before* `run()`'s `set_seed()` instead
  of after. A backbone's random init draws from the global RNG (DINOv2 and timm
  both initialise randomly before loading the state dict), so the head was
  seeded from a different state and every trained probe scored differently,
  while every recorded field — seed included — stayed identical.

  Caught by publishing a full board and diffing it against the corpus: 20 of 26
  records differed, and **the 6 that reproduced were exactly the zero-shot
  probes**, which train no head. Taskonomy edge on DINOv2-S read 0.4407 against
  the corpus's 0.4558, and two published rankings flipped.

  `run()` now returns the backbone it used on `RunResult.backbone`, and the CLI
  passes the name in every case. The regression test pins the backbone's
  *weights* rather than the metrics: the CLI fixtures are colour-separable and
  score 1.0 whichever way the RNG is threaded, so a metric comparison there
  passes with the bug in place — it was written that way first, and mutation
  testing is what showed it proved nothing.

- **A saved `detection` probe was unusable when loaded back.** `grid_hw` is
  fitted state living outside `self.head`, and `probe_state()` did not carry
  it, so `load_probe` returned a probe whose `predict` raised "this probe has
  not been fitted". This is the case `probe_state` was added for — the same one
  `ClassificationTask`'s standardiser was — and detection was missed when it
  arrived. Latent since v0.6.0, and reachable only through the Hub artifact
  path, which is why no measurement moves.

## [0.8.0] — 2026-08-07

**The thirteenth probe, and the first whose target VisBench computes rather than
downloads.** `corner` is Shi-Tomasi cornerness derived from the RGB frame at
read time, so it runs on any folder of photographs with no dataset and no
extras — and it ships with its six numbers in the committed record corpus,
behind a frame set a script pins and reconstructs.

Every measurement v0.7.0 reported, v0.8.0 reports identically. The corpus grows
from 72 records to 78 and gains no revisions.

### Added

- **The corner probe is in the record corpus, on a pinned frame set.** Six new
  records take `results/corpus/visbench.jsonl` to 78 — thirteen probes against
  six backbones, thirteen comparability groups each holding all six — and the
  hand-written table in `docs/tasks.md` is replaced by a generated board that
  the fast suite fails on if it drifts.

  A probe whose target is computed runs on any folder, which is its selling
  point and, for a leaderboard, its problem: two corner numbers are comparable
  only if they ran the same images. New `scripts/stage_corner_frames.py` pins
  which — the first 600 rows of each Taskonomy `tiny` split list, symlinked into
  the flat `<split>/images/` layout, **verified set-equal to the frames the edge
  probe reads**. That equality is what makes the published cross-probe claim
  exact: the two targets correlate at 0.52, and the reason the corner probe
  earns its place is that they nonetheless rank backbones differently.

  Symlinks rather than copies, because `cache_identity` keys on path, size and
  mtime and a symlink reports its target's — so staged and original frames share
  one feature-cache entry. `scripts/build_corpus.sh` gained `probe_corner`,
  which skips with an actionable message when the staged folder is absent,
  following `generic_segmentation`'s precedent.

  All six regenerated numbers reproduce the previously published ones to four
  decimals, through a staging path that shares no code with the ad-hoc one that
  produced them.

- **Corner detection (`corner`), the thirteenth probe and the first whose target
  needs no dataset.** Shi-Tomasi cornerness — the smaller eigenvalue of the
  Gaussian-windowed structure tensor — computed from the RGB frame at read time
  by the new `visbench/data/derived.py`. Any folder of photographs runs it:

  ```bash
  visbench run corner --data /path/to/any/images --limit 600
  ```

  It reuses `DenseMagnitudeTask` unchanged, so the identity activation, the L1
  loss and the per-image Pearson correlation are the ones `edge` and
  `keypoints2d` already use. What is new is `DerivedTargetDataset`, which pairs
  an image folder with a target generator.

  **Computing the target after the crop removes the alignment hazard
  structurally.** Every other dense dataset here pays separately to keep image
  and target under one geometry — nearest-neighbour resampling for depth, an
  achieved-ratio rescale for boxes — and the correspondence probe paid for
  getting it wrong once, at `recall@1px` = 0.003. A derived target is generated
  from the exact cropped array the backbone sees, so there is no second geometry
  and no resampling of the response at all.

  **The operator is Shi-Tomasi rather than Harris, and the tail decided it.**
  Share of target mass in the strongest 1% of pixels, against ~0.10 for the two
  probes that work and the 0.46 that stopped `edge_occlusion` ranking: Harris
  `R` clipped at 0 gives 0.52, `|R|` gives 0.33, λ_min gives 0.27. All three are
  too concentrated raw; `log1p(1e4·λ_min)` lands the tail at 0.089 and the frame
  mean at 0.593, meeting 6d-2's tail criterion and 6d-1's "an L1 target must be
  of order 1" at one setting. λ_min is also non-negative by construction and
  carries no `k`, one fewer free parameter making "Harris corners" a family
  rather than a definition.

  **The target overlaps with the edge target and the docs say so.** Per-image
  correlation is 0.52 against `edge_texture` and 0.27 against `keypoints2d`,
  where the two Taskonomy probes correlate at 0.147 with each other. The overlap
  is intrinsic rather than an artifact of the compression — it holds at
  0.46–0.54 across eight transforms including near-linear ones. A corner score
  and an edge score are therefore not independent evidence about a backbone.

  **They rank differently, which is what earns it a place.** Over six backbones
  on the same frames the spread is 0.1603 against the edge probe's 0.1136, and
  the ordering is not the same: CLIP-B/16 is first on edges and third on
  corners, and the two ResNets swap.

  Every setting of the generator lands in `dataset_params`, so two sigmas split
  into two comparability groups without anyone noticing. No schema bump —
  that field is what it was added for.

- **A DOI.** v0.7.0 is archived on Zenodo, and the concept DOI
  [10.5281/zenodo.21822684](https://doi.org/10.5281/zenodo.21822684) now
  appears in `CITATION.cff`, the README badge row and BibTeX block, and the
  documentation landing page.

  It is the *concept* DOI deliberately: Zenodo mints one of those per project
  and a version DOI per release, and the concept one always resolves to the
  newest archive, which is what someone citing "VisBench" wants. A paper
  reporting measured numbers should pin the version DOI instead, for the same
  reason every result record carries its schema, pooling and protocol.

  `tests/test_citation.py` pins the literal across all three files and fails on
  any other `10.5281/zenodo.*`, since a version DOI substituted for the concept
  one resolves fine and looks correct.

## [0.7.0] — 2026-08-06

**The contributor-facing release.** No new probe, backbone or metric — every
number v0.6.1 reported, v0.7.0 reports identically. What changed is everything
around them: the shortest path from `pip install` to a number is now
`visbench demo`, thirty seconds and no dataset; the reference material lives on
a documentation site instead of 500 lines down a README; the project's
conventions are written down for contributors rather than kept for an
assistant; and the work can be cited and archived.

If you are upgrading to get a measurement, there is nothing here for you. If
you are arriving for the first time, this is the release that makes that
possible.

### Added

- **Citation metadata, so VisBench can be cited and archived.** `CITATION.cff`
  drives GitHub's "Cite this repository" button and is read by Zenodo when it
  archives a release; `.zenodo.json` supplies the deposit metadata Zenodo
  prefers, including an ORCID. The README and the documentation landing page
  both gained a **Citing VisBench** section with BibTeX.

  Enabling the Zenodo GitHub integration mints a DOI for every subsequent
  release, plus a concept DOI that always resolves to the newest one. The
  concept DOI is the one to cite for the software; a version DOI is what a
  paper reporting measured numbers should pin, since a VisBench number is
  reproducible only against the release that produced it.

  Three rules are tested rather than trusted (`tests/test_citation.py`): the
  cited version must equal `visbench.__version__`, the two metadata files must
  agree on title and licence, and the author must carry an ORCID in both — in
  CFF's resolvable-URL form and Zenodo's bare-identifier form, which are not
  interchangeable.

  `pyproject.toml` gained the `authors` field it never had, so the PyPI page
  attributes the package rather than leaving it anonymous.

- **A documentation site**, at <https://turhancan97.github.io/VisBench/>, built
  with Sphinx and deployed from `main`. This first piece is the infrastructure:
  `docs/conf.py`, a landing page, the theme, and
  `.github/workflows/docs.yml` (build on every pull request, deploy on merge).
  The API reference, user guide and design pages follow.

  Sphinx rather than MkDocs because the docstrings were already written for it —
  33% of the package is docstrings, in numpydoc style, with 371 `:meth:`/
  `:func:`/`:class:` cross-references and 462 `#:` attribute comments that
  render as-is.

  The theme uses the project palette (`#3A7EAB`, `#CF4832`, `#D1D3D4`) taken
  from the logo SVGs, which already carry a dark variant. Body links use a
  darkened `#2F6A91`: the brand blue is 4.42:1 against white, just under the
  WCAG AA threshold for body text.

  A new `docs` extra carries Sphinx, Furo, MyST and copybutton. It is
  deliberately not part of `all`, which is the runtime-capability set.

- **`CONTRIBUTING.md`**, plus GitHub issue and pull-request templates. The
  project's conventions were real but lived in `CLAUDE.md`, which is written for
  an assistant rather than for a contributor: setup, the five commands CI runs,
  the two jobs they do not cover, how to add a probe, and the rules that exist
  because this library reports numbers — protocols are not claimed unless
  implemented, metric directions are listed rather than inferred, a probe has to
  *rank* rather than merely score.

  Three tests keep the guide honest rather than merely written: the lint
  commands it documents must be the ones `ci.yml` runs, every gating CI job must
  be mentioned, and the extras it tells you to install must exist. A setup
  document that has drifted is worse than none — it sends someone to run the
  wrong commands, watch them pass, then fail review for a reason it told them
  not to expect.

  `pyyaml` joins the `dev` extra, because the issue-template test parses those
  files. It was available locally via timm and **not** in CI's `.[dev]` install,
  which is the same trap the `[hub]` tests hit in v0.6.0.

- **`visbench demo`** — a real probe run that needs no dataset, no
  configuration and no large download. Thirty seconds from `pip install` to an
  interpretable number:

  ```
  $ visbench demo
  drawing 20 images per class for 4 shapes...
  loading resnet18 (torchvision, ~45 MB on first run)...
  running the classification probe...

    top1         0.8125

    chance is 0.25 — the shapes differ in outline only.
  ```

  All thirteen `examples/` require `--data`, so the shortest path to a first
  number went through finding a dataset, laying it out correctly and fetching a
  1.7 GB backbone. This removes that.

  Nothing is special-cased: the same `visbench.run`, the same cache, the same
  result record as any other run, so what it demonstrates is the actual
  library. The backbone is torchvision's ResNet-18 — a **core** dependency,
  ~45 MB — wrapped in `CustomBackbone`, which is also the path a user takes for
  their own model.

  **The score is deliberately not 1.0.** Colour, size, position and rotation are
  randomised so only geometry identifies a class, and a first pass without that
  scored a flat 1.0 — the saturation this project rejects elsewhere. At the
  defaults it reads 0.812 against a chance of 0.25, and `--noise` walks it into
  chance: 0.975 / 0.812 / 0.550 / 0.438 / 0.312 across five settings. A probe
  whose score does not move when the signal is destroyed is not measuring the
  signal, and that slide is the demo's real lesson.

  New public helpers in `visbench.demo`: `synthesise` and `demo_backbone`.

### Changed

- **The README is reorganised around a reader, not around the project's
  history.** It was 1,020 lines with `Install` at line 390 — a newcomer scrolled
  past build order, roadmap and future directions before learning how to
  install anything. It is now **397 lines**, ordered: run something, install it,
  understand it, then the interfaces.

  Two large sections moved out rather than being deleted:

  - **`docs/tasks.md`** — the per-probe reference: every probe's data layout,
    the example that runs it, and its measured numbers. All nine generated
    tables live here now.
  - **`docs/roadmap.md`** — build order, roadmap and the candidate task
    backlog. These answer "what is the plan", not "how do I use this".

  A new **Where to go next** table at the foot of the README points at both,
  plus the leaderboard, the changelog and NOTICE.

- **`scripts/render_tables.py` takes a list of marked files**, since the
  generated boards no longer live in the README. A file in that list with no
  markers is an **error**, not a silent skip — moving a table without updating
  the list fails loudly instead of quietly leaving a stale copy behind.

  Two new tests: the README must actually link to the pages it split into, and
  relative links inside `docs/` must resolve. The second is deliberately the
  *opposite* rule to the README's — `docs/` is not package metadata, so relative
  links are correct there.

## [0.6.1] — 2026-08-02

**A correction.** v0.6.0's correspondence board was ranked upside down, because
its error threshold was measured in a unit that means a different physical
distance on every backbone. Nothing else in v0.6.0 is affected; if you do not
read the correspondence board, this release changes nothing for you.

### Fixed

- **Correspondence thresholds are measured in pixels, not patch widths.**
  `CorrespondenceTask(threshold_units=...)` and the CLI's `--units` both default
  to `"pixel"`, the headline metric is `recall@5px`, and the six correspondence
  records in the corpus were re-run.

  A patch width is a property of the *backbone*, not of the protocol: at 224px
  it is 14px on DINOv2/14, 16px on CLIP ViT-B/16 and 32px on ViT-B/32 or a
  ResNet's last stage. Scoring in patch widths therefore asked a coarse-grid
  backbone to land within 32px and a fine-grid one within 14px, and printed both
  under one metric name.

  On the same 200 Imagenette pairs, with only the unit changed:

  | backbone | `recall@1p` (v0.6.0) | `recall@5px` (now) |
  | --- | --- | --- |
  | resnet18 | **0.8927** | 0.0973 |
  | dinov2_vits14 | 0.7834 | **0.3049** |

  First and last place swap, and the pixel ordering is the one every other dense
  probe produces — DINOv2 > CLIP > ResNet.

  The quantisation floor that motivated patch widths is real and is now stated
  rather than divided out: `ceiling_recall@5px` is ~0.10 on a 7x7 grid against
  ~0.41 on a 16x16 one, and it already travelled beside every score.
  `threshold_units="patch"` is kept for single-backbone studies, where it
  answers a real question — the README's `max_warp` sweep is one.

  **No v0.6.0 number can be silently compared against a v0.6.1 one.**
  `threshold_units` lives in `task_params`, which `comparability_key` includes
  wholesale, so the two units already formed different groups. Nothing needed a
  special case.

### The finding worth carrying forward

**When a board looks wrong, check what the threshold *means* on each row before
reaching for the denominator.** v0.6.0 shipped this board with a caveat blaming
`num_matches`, the per-backbone match count. That difference is real — 4,911 for
ResNet-18 against 27,590 for DINOv2-B — and it is a *consequence* of grid
resolution, not the cause of the inversion. Normalising by the ceiling was tried
and did not fix it either, which should have been the clue that the score was
not the problem.

The old docstring argued the opposite case and argued it well, citing the true
fact that `recall@1px` has a ceiling of 0.015 on DINOv2-S. That fact is an
argument for choosing a sensible *pixel* threshold, not for a backbone-dependent
unit.

## [0.6.0] — 2026-08-02

**The leaderboard release.** Every VisBench number published before this was
produced ad hoc and hand-copied into a markdown table; most of the records
behind them no longer existed. This release replaces all of that with a
committed corpus of result records, comparability rules that decide which of
them may be ranked together, and generated tables that a test refuses to let
drift. It also makes a trained probe something you can hand to someone else.

Twelve probes against six backbones — DINOv2-S/B, CLIP-B/16 and B/32, ResNet-18
and ResNet-50 — in twelve comparability groups, each holding all six. The eight
dense probes rank them DINOv2 > CLIP > ResNet, with B/16 > B/32 and RN50 > RN18.

Result schema moves to **v7**, additively as always: `pooling_requested` joins
the resolved `pooling`, because keying comparability on the resolution alone
could never rank a CNN against a ViT.

### Added

- **`visbench/results/leaderboard.py`** (step 6e-1), the comparability rules as
  code. The schema has carried the fields a leaderboard needs since v0.1; what
  it never carried is the *rules* for which records answer the same question.
  Those lived as prose, which is why every published table was assembled by
  hand — and why one of them had drifted by the time it was noticed.
  - `comparability_key` — everything that must agree before two records may be
    ranked. The backbone is deliberately excluded, since it is the thing being
    compared; `finetune`, `protocol`, `dataset_fingerprint`, `task_params` and
    `dataset_params` are all included.
  - `rank`, which refuses rather than ranks: incomparable records, a metric
    missing from any record, a context metric, a diagnostic, or a metric with
    no recorded direction.
  - `ranking_disagreements`, which reports metric pairs that order the same
    backbones differently — because a task can disagree with itself.
  - `latest_per_backbone`, `shared_metrics`, `group_comparable`.

Validated against the 16 real records on disk. It reproduces every published
number exactly — VOC frozen 0.7328/0.7533, fine-tuned 0.7758/0.7992, Taskonomy
edge 0.4558/0.4481 — and separates them into four groups no two of which are
rankable against each other.

### The finding worth carrying forward

**`METRIC_DIRECTIONS` lists names instead of inferring them, and surface
normals are why.** `mean` and `median` are angular error in degrees, where
lower is better. Nothing about either word says so. Any heuristic that read
"mean" as a score would rank that leaderboard precisely upside down and the
result would read as a surprising finding rather than a bug — so an unknown
metric raises instead of defaulting to higher-is-better.

The same reasoning made three other things refusals rather than conveniences:
a metric absent from one record (ranking the rest presents a partial comparison
as a complete one), a `classes_scored` mismatch (two mAPs over different
denominators are averages of different quantities), and a `ceiling_` context
metric (correspondence's ceiling describes the split, so ranking on it ranks
the data).

- **`results/corpus/visbench.jsonl`** (step 6e-2), the record corpus — 26
  records, schema v6, twelve probes against both DINOv2 backbones, every one of
  them frozen. Twelve comparability groups, and all twelve hold both backbones,
  which is the check that matters: identical commands with only `--backbone`
  varying, so nothing split into two groups of one. Every published VisBench
  number to date was produced ad hoc and hand-copied; this is the first set that
  exists as records anyone can re-rank.

  The corpus is **tracked**. `results/*.jsonl` stays ignored for ad-hoc runs,
  but `results/corpus/` is negated in `.gitignore`, because a benchmark whose
  records nobody else can see is not a benchmark.

- **`scripts/build_corpus.sh`**, one function per probe with every flag in one
  place. `comparability_key` requires `task_params` and `dataset_params` to
  match exactly, so a stray `--limit` does not produce a wrong number — it
  produces two groups of one, and the run is wasted rather than misleading.
  Keeping the flags in a single file makes that structural instead of a matter
  of care.
- **`slurm/corpus.sbatch`**, a 24-task array (one per probe/backbone), and
  **`scripts/merge_corpus.sh`**, an idempotent rebuild from the per-task parts
  with validation. One JSONL per task rather than one shared file: this
  repository lives on NFSv4, and **NFS has no atomic `O_APPEND`**, so two tasks
  finishing together can interleave lines. A corrupted corpus is worse than no
  corpus, because which runs were lost is not recoverable from what remains.
- **`scripts/binarise_voc_masks.py`**, turning VOC's 21-class label maps into
  0/1/255 binary masks so `generic_segmentation` has a real dataset.
- **`PARAMETRISED_METRIC_DIRECTIONS`** in `leaderboard.py`, and
  `detections_per_image` / `num_matches` added to `DIAGNOSTIC_METRICS`.

### What running it actually found

**Two of the twelve probes ranked nothing, silently.** `retrieval` and
`correspondence` emit only *parametrised* metrics — `recall@1`, `recall@10`,
`auc@0.5p` — and not one of those names was in `METRIC_DIRECTIONS`.
`shared_metrics` skips a name it cannot direct, so both probes produced an empty
leaderboard section **rather than an error**. Every test fixture had used
unparametrised names, so only a real corpus could surface this.

The fix keys on the stem before `@`, and is still a listed table rather than the
name heuristic this module refuses everywhere else: those names are *generated*,
by `f"recall@{k}"` and `f"auc@{format_threshold(...)}"`, so the stem **is** the
metric's identity and the suffix is only which setting of it. An unlisted stem
still raises.

**The corpus reproduces every published number to four decimals** — VOC
segmentation 0.7328/0.7533, classification 0.9939, similarity 0.8701/0.8580,
edge 0.4558/0.4481, keypoints2d 0.2356/0.2248, occlusion edge 0.2924/0.3167,
correspondence `recall@1p` 0.7834 against a ceiling of 0.9509 — from a compute
node, against numbers measured months ago on a different machine.

**Detection is the one exception, and is recorded as unverifiable rather than
contradicted.** 0.2302/0.2882 `map_50` against step 6c-3's 0.2127/0.2616. Every
recorded field matches what was documented for that run, so the difference is
not in any field a record carries, and the original command was never committed
so it cannot be diffed. The ordering is unchanged and the corpus number is the
reproducible one.

**`edge` disagrees with itself three ways.** `edge_correlation` ranks DINOv2-S
first, `mae` ranks DINOv2-B first, and `rmse` ranks DINOv2-S first again. One
probe, three metrics, three orderings — which is why a renderer must never pick
a headline metric silently.

**Semantic segmentation ran twice by accident, and the duplicates are kept.**
The smoke test and the full array appended to the same part file. The metrics
are identical to six decimals; the durations are 137.6 s against 123.5 s, and
104.9 s against 115.9 s. That is "a wall clock is not a metric" demonstrated
rather than asserted.

### Changed

- **Result schema is at v7**, adding `pooling_requested` — what the task asked
  the backbone for, beside the `pooling` it resolved to. Additive as always:
  `None` on every earlier record, and v6 files still read.

  `pooling` has always been recorded **resolved**, because the literal word
  `"default"` does not say what produced a number. That is right for reading a
  record and wrong for comparing two: `default` resolves to `cls` on a ViT and
  `mean` on a CNN, so a leaderboard keyed on the resolution **split every
  pooled-feature probe along an architectural line**. Widening the corpus to six
  backbones is what exposed it — classification, retrieval, correspondence and
  similarity each became two groups, and `resnet50`, which tops two of those
  boards, could not be ranked against DINOv2 at all.

  The request is the protocol; the resolution is a property of the backbone.
  `comparability_key` now uses the request, falling back to the resolution for
  v6 and earlier. Two runs that both asked for `default` are comparable; two
  that named `cls` and `mean` explicitly are still not, which is the behaviour
  worth keeping.

- **The record corpus covers six backbones**, not two: DINOv2-S/B, CLIP-B/16 and
  B/32, ResNet-18 and ResNet-50, across all twelve probes. Twelve comparability
  groups, each holding all six. `slurm/corpus.sbatch` takes `VISBENCH_BACKBONES`
  so the matrix can be widened without editing it, and refuses an `--array`
  range that does not match the matrix unless `VISBENCH_PARTIAL=1` says the gap
  is deliberate — an incomplete corpus is invisible afterwards, because every
  group it *does* contain still holds every backbone it ran.

- **`visbench/results/render.py`** (step 6e-3), markdown tables built from
  records, plus `scripts/render_tables.py` and a generated
  [`LEADERBOARD.md`](LEADERBOARD.md) holding all twelve boards.

  Nine tables in `README.md` are now delimited by HTML comment markers and
  regenerated from `results/corpus/visbench.jsonl`. `tests/test_readme.py` runs
  the generator's `--check` in the **fast** suite, so a table that drifts from
  the records fails a build rather than shipping — which matters because a
  version number on PyPI can never be reused.

  The module formats; it never decides. `leaderboard.py` remains the only place
  comparability and ordering are determined, and `render.py` may not relax a
  rule from it. In practice that means two listed dicts and nothing else:
  `HEADLINE_METRICS`, so a board's ordering is declared rather than picked, and
  `CAVEATS`, so what a reader must know travels with the board it qualifies.

  What a table is not allowed to hide: the winner is bolded **per column**, so a
  task that ranks its backbones three different ways shows that instead of
  implying an outright winner; diagnostics and ceilings are rendered as columns
  even though ranking on them is refused, because a board that drops
  `num_matches` presents a comparison whose terms differ as though they did not;
  and narrowing a board for width cannot drop a denominator, drop the ceiling of
  a metric it keeps, or suppress a disagreement note.

- **`visbench.hub`** (step 6e-4), serialising a trained probe head together with
  the backbone identity that makes it meaningful: `save_probe`, `load_probe`,
  `probe_metadata`, `IncompatibleProbe`. Plus `head_spec()`, `probe_state()` and
  `load_probe_state()` on `BaseTask`, and `examples/save_probe.py`. **No network
  dependency** — the Hub transport is 6e-5, behind a `[hub]` extra, and saving
  and loading a probe works in a core install.

  Nothing about probe heads was serialisable before this: no `save`, `load` or
  `state_dict` anywhere in `visbench/tasks/` or `visbench/heads/`.

  **A head is only meaningful against the exact features it was fitted on, and
  almost every way of getting that wrong is shape-compatible.** Measured on real
  DINOv2-S weights over Imagenette: a linear head fitted on CLS tokens and then
  fed *mean-pooled* tokens from the same backbone scores **0.9620 against
  0.9820**. It does not crash and it does not produce garbage — it produces a
  number nobody would question.

  So `load_probe` checks four things and refuses by default: `backbone_key`
  (a fine-tuned checkpoint differs from its parent in nothing else), resolved
  `pooling` (the case above), `feature_mode`, and `layers`. `strict=False` warns
  and loads, because deliberately probing how far a head transfers is a real
  experiment — but a number produced that way is comparable with nothing.

  The artifact is read back with `torch.load(weights_only=True)`, and a test
  asserts it still can be. That is not a detail: 6e-5 fetches these from a hub,
  where an unrestricted load is arbitrary code execution.

- **`visbench.hub.remote`** (step 6e-5), pushing and pulling probes through the
  Hugging Face Hub: `push_probe`, `load_probe_from_hub`, `probe_card`. Behind a
  new **`[hub]` extra** — `pip install 'visbench[hub]'`. Saving a probe to a
  local file and loading it back still needs nothing but a core install, because
  `huggingface_hub` is imported inside the functions that use it.

  `load_probe_from_hub` is `load_probe` with a download in front of it, and
  `push_probe` calls `save_probe`. Neither adds a rule: a downloaded probe gets
  the same backbone identity checks and the same `weights_only=True` load as a
  local one, and a test asserts the uploaded bytes match what the local saver
  writes, so the two cannot drift into different formats.

  Pushing creates a **private** repository unless asked otherwise — a push is
  not reversible the way a local write is — and writes a generated model card
  beside the weights, from the same metadata the artifact carries. The refusal
  for an unfitted or zero-shot probe happens *before* the repository is created,
  so a rejected push leaves nothing behind.

- **Depth and surface normals are measured on NYUv2, not Taskonomy.** Their
  Taskonomy numbers came from uncommitted code and were unreachable from any
  entry point. probe3d's own NYUv2 copy has exactly the
  `<root>/<split>/{images,targets}` layout the CLI already expects — 795/654,
  the canonical split — so both probes joined the corpus with **no code
  change**: depth d1 0.7652/0.7851, normals mean 29.48°/30.11°.

  Two hazards, recorded inline in `build_corpus.sh`. `--target-scale 1.0` is
  load-bearing, because these targets are already in metres while NYUv2's PNG
  distribution is millimetres — passing 1000 would make RMSE look superb and
  mean nothing. And these normals are **dense**: not one zero-length vector
  across 40 sampled frames, including across the ~28% of pixels where the depth
  map has no ground truth, so the probe is scored on filled geometry and is not
  comparable with a masked normals probe.

### Known, and unresolved

**The correspondence board ranks on an average whose denominator each backbone
picks for itself.** `recall@t` is `(errors <= t).mean()` over the matches that
backbone's own features proposed, and `num_matches` is that denominator —
4,911 for ResNet-18 against 27,590 for DINOv2-B. ResNet-18 tops the board while
proposing 5.6x fewer, easier candidates from a coarse 7x7 grid, and normalising
by the per-backbone ceiling does not change the ordering.

This is the `classes_scored` situation the leaderboard already refuses, except
that every backbone differs, so guarding on it would make correspondence
unrankable rather than comparable. It is a protocol question, recorded here
rather than papered over, and it is the first thing the renderer has to settle.

## [0.5.0] — 2026-08-01

**Two probes, and a mask that was missing.** v0.4.0 refused six of Taskonomy's
domains because they are derived from its 3D reconstruction and carry holes that
no in-band value marks. This release reads `mask_valid/` and unblocks four of
them — and in doing so found that one, `keypoints3d`, had never been refused at
all: it was listed as a known domain and omitted from the refusal set, so v0.4.0
would read it as though every pixel were a real measurement. Nothing raised.
That fix is the reason to upgrade even if neither new probe interests you.

The two new probes are `keypoints2d` (low-level) and `occlusion_edge`
(mid-level). They share every line of their implementation with the existing
edge probe and sit one tier apart, which is the sharpest statement of what the
tiers mean this codebase has: recovering a depth discontinuity needs scene
geometry, recovering an intensity one does not.

### Added

- **`mask_valid/` support in `TaskonomyDataset`** (step 6d-2), which unblocks
  the domains derived from Taskonomy's 3D reconstruction. Four of the six
  refused domains are now supported: `depth_zbuffer`, `normal`,
  `edge_occlusion` and `keypoints3d`. Each declares how it marks an invalid
  pixel, and the dataset writes *that* marker rather than exposing a mask, so
  nothing downstream needs to know a mask was involved.
- **2D keypoint detection** (`keypoints2d`), the second low-level probe, on
  Taskonomy's `keypoints2d` response maps. Dense magnitude regression scored by
  per-image Pearson correlation, recorded as
  `protocol: "visbench_keypoint2d_regression"`.
- **Occlusion-edge detection** (`occlusion_edge`), a mid-level probe on
  `edge_occlusion`. It shares every line of implementation with the low-level
  edge probe and differs only in what it reads, which makes running both on the
  same frames a direct comparison of the two tiers: recovering a depth
  discontinuity needs scene geometry, recovering an intensity one does not.
- `visbench/tasks/magnitude_base.py` (`DenseMagnitudeTask`), lifted out of a
  working `EdgeTask` now that there are three magnitude probes — the same move
  that produced `DenseTrainingTask` from `DepthTask` and `tasks/schedule.py`
  from `DetectionTask`.
- `magnitude_metrics`, `load_valid_mask`, `TASKONOMY_SUPPORTED_DOMAINS`, and a
  `_load_raw_target` hook on `DenseFolderDataset` so a subclass can consult a
  second file for the same item without reimplementing the geometry.
- Twelve probes now: `visbench run keypoints2d` and `visbench run
  occlusion_edge`, plus `examples/keypoints.py` and
  `examples/occlusion_edges.py`.

Measured on Taskonomy, 600 train / 600 val frames at 224px, linear head, ten
epochs, one V100:

| probe | level | DINOv2-S/14 | DINOv2-B/14 |
| --- | --- | --- | --- |
| `keypoints2d` | low | **0.2356** | 0.2248 |
| `occlusion_edge` | mid | 0.2924 | **0.3167** |

The low-level probe favours the smaller backbone and the mid-level one the
larger, which is the direction the taxonomy would predict — but the standing
rule that bigger is not better on every task is not suspended because an
ordering came out convenient, and the keypoint gap is 0.011. These are floors to
re-measure against, not results.

### Fixed

- **`keypoints3d` was silently unmasked in v0.4.0.** It was listed as a known
  Taskonomy domain and was absent from the refusal set, so it constructed and
  read like an image-derived target while actually coming from the 3D
  reconstruction, holes and all. Nothing raised — it would simply have trained
  against fabricated readings. It is now masked like the other three.
- **A Taskonomy probe could be pointed at another probe's domain.**
  `visbench run edge --domain keypoints2d` loaded, trained and recorded a
  keypoint number as `visbench_edge_regression`, which is the exact
  mislabelling the `protocol` field exists to prevent. `--domain` is now
  restricted per probe to what its recorded protocol describes.

### Changed

- `edge_metrics` is now a thin wrapper over `magnitude_metrics`, which takes the
  correlation's key and masks non-finite target pixels. On an all-finite target
  it reduces exactly to the previous computation, which a test pins — no
  published `edge_correlation` moves.
- `TaskonomyDataset.target_scale` defaults **per domain** rather than to one
  number. `depth_zbuffer` is fixed at 512 and refuses to be changed: that
  divisor is what puts the target in metres, and depth metrics report RMSE in
  whatever unit they are given, so rescaling it would quietly change what the
  number means rather than rescaling a free parameter.
- `keypoints2d`'s default scale is 30 rather than 100, measured (0.176 → 0.210
  on a 400-frame split).

### The finding worth carrying forward

**The magnitude protocol does not transfer on tail weight alone, and the
occlusion-edge probe is where that showed up.** L1 was chosen for these probes
precisely so a handful of strong pixels cannot dominate the loss; Pearson
correlation is *dominated* by those same pixels. On a moderately heavy tail the
two coexist. On `edge_occlusion`, where 46% of the total target mass sits in the
strongest 1% of pixels (against ~0.10 for `edge_texture` and `keypoints2d`),
they pull in opposite directions and the probe measures nothing: it scored 0.088
and was flat under four target scales spanning 30x, four times the training
budget, and a ten times higher learning rate. Its DINOv2-S-versus-B gap was
0.0035 — noise — so it could not have ranked two backbones, which is the one
thing a probe has to do.

A `log1p` on the target brings the tail to 0.09, the correlation to 0.29/0.32,
and the gap to 0.024 in DINOv2-B's favour. So `edge_occlusion` is loaded in log
space and **no other domain is**: their tails are already mild, `edge_texture`'s
published v0.4.0 number is a linear-target number, and a log-space correlation
is not the same measurement as a linear-space one. `dataset_params` records
`target_transform` so the two can never be pooled.

The general form, for whoever adds the fourth magnitude probe: **check the tail
before assuming the protocol transfers.** A probe that is flat under every
hyperparameter is not underfitting.

## [0.4.0] — 2026-07-31

**The third level.** VisBench's task taxonomy has had three tiers since v0.1 —
high-level, mid-level, low-level — and `visbench/tasks/low_level/` has been a
folder containing a README explaining why it was empty. This release fills it.

Edge detection is dense edge-magnitude regression on Taskonomy's `edge_texture`
maps: the tenth probe, and the first whose target has **no invalid value at
all**. Depth's holes and normals' zero-length vectors mean "no ground truth";
an edge map's zeros mean *no edge*, which is a real reading covering most of a
frame. That is the third validity convention in the codebase, and inheriting
the nearest one would have scored the probe only where an edge already is.

Two of the decisions behind it were reached by measurement overturning the
obvious answer, which is the more useful thing to carry forward than the score.
Rectifying the output looks plainly correct — an edge magnitude cannot be
negative — and destroys the probe: ReLU dies outright and softplus collapses to
a constant, against 0.9997 for the identity. And the probe's first real run
scored 0.047, which `train_loss` identified as non-convergence rather than a
representation without edges; the cause was L1's sign-valued gradient not
shrinking to match a target whose mean is 0.011 of its container's range.

Also here: dataset construction got **up to 475x faster** — indexing VOC's
17,125-file `Annotations` fell from 76 s to 0.16 s — by asking `os.scandir` for
what `readdir` already returned instead of paying `Path.is_file()` a stat round
trip per entry. No result record ever showed that cost, because it is paid
before `run()` starts its timer.

### Added

- **Edge detection — the first low-level task** (step 6d-1), filling a folder
  that had been a documented placeholder since v0.1. `EdgeTask` is a dense
  edge-magnitude regression over frozen features: one channel, identity
  activation, L1 loss, scored by per-image Pearson correlation. Registered as
  `edge`, reachable as `visbench run edge`, with `examples/edges.py`.

  Also new: `visbench.data.taskonomy` (`TaskonomyDataset`, `load_taskonomy_split`,
  `TASKONOMY_DOMAINS`), `visbench.data.dense.load_edge_map`, and
  `visbench.metrics.dense.edge_metrics`.

  Measured on Taskonomy `edge_texture`, 600 train / 600 val frames at 224px,
  linear head, ten epochs, one V100:

  | backbone | `edge_correlation` | `rmse` | `mae` | `train_loss` |
  | --- | --- | --- | --- | --- |
  | DINOv2-S/14 | **0.4558** | 0.9226 | 0.5028 | 0.5721 |
  | DINOv2-B/14 | 0.4481 | 0.9265 | 0.4972 | 0.5631 |

  DINOv2-S edges out DINOv2-B by 0.008 — the same ordering as mid-level
  similarity, the opposite of segmentation and detection. Small enough to report
  as consistent with the level taxonomy, not as evidence for it.

  Four decisions worth keeping:

  - **`protocol: "visbench_edge_regression"` — not BSDS500's.** BSDS is the
    canonical edge benchmark, but ODS/OIS/AP matches edge pixels by bipartite
    correspondence after non-maximum suppression, swept over thresholds, against
    several annotators. These numbers must not share a table with published BSDS
    ones. BSDS500 is also not on the build machine, whereas Taskonomy is.
  - **Nothing is masked.** Depth's holes and normals' zero-length vectors mean
    "no ground truth"; an edge map's 0 means *no edge*, a real reading covering
    most of most frames. Reusing the earlier convention would have scored the
    probe only where an edge already is.
  - **The activation is the identity, and rectifying destroys the probe.** On
    features that encode the answer, ReLU scores 0.0000 (dead, zero prediction
    variance) and softplus -0.9851 (collapsed to a constant) against 0.9997 for
    the identity. Non-negativity is learned from the targets rather than imposed.
  - **`target_scale` defaults to 1000, not the container's 65535.** L1's
    gradient is sign-valued, so its step size does not shrink with the target;
    at 65535 a frame's mean is 0.011 and the optimiser oscillates rather than
    converging. Correlation goes 0.047 → 0.285 → 0.456 → 0.467 as the scale
    goes 65535 → 6553.5 → 1000 → 100, so 1000 sits at the knee. Scaling the
    target rather than raising the learning rate keeps this number under the
    same training budget as every other dense probe.

### Fixed

- **`TaskonomyDataset(target_scale=...)` had no effect** in its first form:
  `DenseFolderDataset.target()` applies `target_scale` only on its default depth
  path, so a custom `target_loader` silently dropped it — while `describe()`
  still reported the value and the fingerprint still folded it in. Caught by a
  scale sweep that returned four identical numbers, and now bound into the
  loader with a test asserting the target actually changes.

### Changed

- **Folder datasets list directories with `os.scandir` instead of `iterdir()` +
  `Path.is_file()`**, which on a network filesystem is the difference between
  **0.05 s and 5.69 s** over a cold 2,913-file directory. `Path.is_file()`
  cannot reuse what `readdir` already returned, so it costs one stat round trip
  per entry; `scandir` reads the file type the listing carried anyway.

  On VOC over NFSv4.2, first call in a fresh process: indexing the 17,125-file
  `Annotations` went from **76 s to 0.16 s**, and building a 600-stem
  `DetectionFolderDataset` from **5.86 s to 0.32 s**.

  The pattern was in all three folder datasets — `DetectionFolderDataset`,
  `DenseFolderDataset` and `ImageFolderDataset` — so VOC segmentation and
  Imagenette paid it too. All three now share `visbench.data.base.list_files`.

  **Timing only.** The same paths come back in the same sorted order, with
  directories and broken symlinks excluded and real symlinks followed;
  `tests/data/test_list_files.py` pins that, including an equality test against
  the expression it replaced.

  Worth recording *why this was not found sooner*: the cost is paid before
  `run()` starts its timer, so a result record reported `duration_seconds: 124`
  inside a wall clock of roughly twenty minutes. Step 6c-3 noted the symptom,
  named the right mechanism and predicted the wrong fix — resolving only the
  named stems, which would have changed a merged constructor and bought less.
  Profiling before optimising is what kept the cheaper fix in view, the same way
  it did in 6b.

## [0.3.0] — 2026-07-31

Two things v0.2 could not do: **adapt a backbone**, and **detect an object**.

v0.3 opens with opt-in fine-tuning of the last N backbone blocks, which is the
first change to challenge an assumption the feature cache has rested on since
v0.1: that features depend on the image and the weights alone. Step 6a bypassed
the cache to preserve it; step 6b restores caching for the half of the network
the assumption still holds for. It closes with detection — the ninth probe and
the first task whose target *transforms* rather than resamples, built dataset
first, then metric, then head, so the head was judged by a scorer already
cross-checked against `VOCevaldet.m`.

A fine-tuned number and a frozen one are **different measurements**, and the
new `finetune` field on the result record is what keeps them apart. It is
`None` for every number VisBench has published to date. Do not rank or average
across it.

### Added

- **The frozen prefix is cached** (step 6b), so fine-tuning recomputes only the
  blocks it is training. On by default when fine-tuning;
  `--no-prefix-cache` / `use_prefix_cache=False` opts out.

  The blocks *below* the cut never train, so their output is as fixed as a
  frozen backbone's — the property 6a's bypass was protecting, restricted to
  the half of the network that still has it. `PrefixCache` stores the token
  sequence after those blocks; the run resumes from it.

  Measured on VOC 2012 val, DINOv2-B/14, two unfrozen blocks, ten epochs, all
  in one session:

  | run | wall clock | mIoU |
  | --- | --- | --- |
  | recompute (6a's path) | 320.6 s, 368.9 s | 0.7992 |
  | prefix cache, cold | 379.5 s | 0.7992 |
  | prefix cache, warm | **268.2 s, 276.4 s** | 0.7992 |

  **The metric is identical in every run**, and identical to 6a's — this
  changes the clock and nothing else. The resumed forward pass is bit-identical
  to a whole one (max abs diff 0.0 against DINOv2's own
  `get_intermediate_layers`), and a fast test asserts the same equality through
  `run()` so CI keeps it. Disk cost: 2,913 entries, 2.30 GB.

  **~21%, and the profile explains the rest.** Per ten epochs over 1,464
  images, a prefix hit still pays 128.3 s of image decoding and 8.5 s of
  content hashing, and saves only the 24.3 s preprocess plus the frozen blocks'
  compute. The ~126 s floor that sized this step was measured on the *frozen*
  path, which streams precomputed features and never opens an image; a
  fine-tuning loop is image-driven by construction and cannot reach it. The
  frozen blocks were not the largest remaining term.

- **`PrefixCache`**, deliberately not a mode on `FeatureCache`. The two entry
  kinds are not interchangeable and confusing them is silent — a prefix resumed
  as features, or features handed to a resumption, both produce plausible
  numbers rather than errors. Three independent things keep them apart:
  separate classes, separate directories (`_prefix` beneath the shared root),
  and a key namespace where `prefix@10` cannot collide with any layer index.

  Consequently `FeatureCache.clear()` no longer removes its own root wholesale:
  that would delete prefix entries while reporting a count that excluded them.
  `stats()` excludes them too.

- **`BaseBackbone.forward_prefix` / `extract_features_from_prefix` /
  `can_use_prefix_cache`**, with `supports_prefix_cache` declaring the family.
  DINOv2 only. A backbone that cannot resume refuses by name rather than
  approximating.

  Layers below the cut are **refused, not approximated**: one block-k
  activation cannot serve a shallower depth, so a DPT run over `[2, 5, 8, 11]`
  with two blocks unfrozen declines the prefix cache and recomputes.
  `can_use_prefix_cache()` is the question to ask before choosing a path.

- **A chunked DINOv2 is refused by name.** `block_chunks > 0` makes
  `model.blocks` a sequence of chunks, so `unfreeze_last` and the cut would
  both slice at the wrong depth and still run. Unreachable through the hub
  entrypoints, which is why it is a guard and not a comment.

- `finetune.prefix_cache` in the result record says whether the cache was
  actually **used**, not whether one was offered — a declined run claiming the
  saving would misattribute its own cost.

- **Fine-tuning: unfreeze the last N backbone blocks** (step 6a), opt-in and
  off by default. `finetune_blocks=2` on any dense probe, `--finetune-blocks 2`
  on its CLI subcommand. **DINOv2 only** for now; every other family raises a
  refusal naming itself rather than silently doing nothing.

  Measured on Pascal VOC 2012 val, linear head, the same ten-epoch schedule and
  the same command as the frozen run, on one V100:

  | backbone | run | mIoU | mIoU/image | pixel acc | mean class acc | wall clock |
  | --- | --- | --- | --- | --- | --- | --- |
  | DINOv2-S/14 | frozen (v0.2 baseline) | 0.7328 | 0.6841 | 0.9267 | 0.8271 | 156 s / 126 s |
  | DINOv2-S/14 | fine-tuned, 2 blocks | **0.7758** | 0.7527 | 0.9405 | 0.8542 | 200 s |
  | DINOv2-B/14 | frozen (v0.2 baseline) | 0.7533 | 0.7161 | 0.9316 | 0.8403 | 126 s |
  | DINOv2-B/14 | fine-tuned, 2 blocks | **0.7992** | 0.7813 | 0.9465 | 0.8708 | 279 s |

  Both frozen runs reproduce v0.2's recorded numbers exactly — 0.732 and 0.753
  — which is what makes the +4.3 and +4.6 mIoU comparisons worth anything. The
  gain holds at both scales.

  **The two numbers are not comparable and the record says so.** A frozen probe
  measures what a representation already carries; a fine-tuned one measures what
  it can be adapted into. Schema v6's `finetune` field is what keeps them apart
  — see below.

- **What fine-tuning costs, and a timing that had to be retracted.** The first
  version of this entry reported fine-tuning as *free* — 238 s against a fully
  cached frozen run's 252 s on DINOv2-S — and concluded the frozen path was
  I/O-bound on streaming 1.3 GB of features per epoch. That was one measurement
  on a shared machine and it did not hold. Re-running the identical commands
  reproduced every metric to four decimals and none of the timings: 156 s
  frozen, 126 s on an immediate repeat, 200 s fine-tuned. **Fine-tuning is
  slower** — 1.3-1.6x on ViT-S, 2.2x on ViT-B.

  What replaced that conclusion is more useful. **The frozen path costs ~126 s
  regardless of backbone width**: ViT-S and ViT-B land within 0.2 s of each
  other even though ViT-B streams 2.3 GB against ViT-S's 1.3 GB. The cost is
  per-file overhead across 2,913 files, not bytes moved. Fine-tuning meanwhile
  tracks compute, 200 s to 279 s for the same two unfrozen blocks.

  For the planned step 6b this is the sizing, not a veto: caching the frozen
  prefix trades the frozen blocks' forward compute for a per-file read the
  frozen path already pays, so the margin **grows with backbone size and shrinks
  as more blocks are unfrozen**. It now has a floor and a baseline to be
  measured against rather than a FLOP count to be argued from.

- **`BaseBackbone.unfreeze_last(n)`** and `extract_features_trainable`. The
  trainable forward pass is a **separate entry point**, not a flag on
  `extract_features`: every existing caller — the cache above all — depends on
  getting detached tensors, and a keyword whose default preserved that would put
  the expensive mistake one typo away. `extract_features` keeps its
  `@torch.no_grad()`, and a test asserts it still returns detached tensors after
  an unfreeze, since that is what makes it safe to cache.

  Three things it refuses rather than doing quietly:

  - **a backbone family that cannot support it**, by name, rather than
    unfreezing a plausible-looking wrong set of parameters;
  - **an unfreeze that makes zero parameters trainable** — that run would train
    exactly like a frozen probe and report itself as fine-tuned, which is the
    CLIP QuickGELU failure in a new place;
  - **a trainable forward pass on a still-frozen backbone**, which would build a
    graph carrying no gradients.

  The backbone **stays in `eval()` when unfrozen**; only `requires_grad` flips.
  Train mode would start BatchNorm updating its running statistics and activate
  dropout, so a fine-tuned number would differ from its frozen baseline for two
  reasons at once with only one of them in the record.

- **Two learning rates.** The head keeps probe3d's 5e-4; the backbone defaults
  to `lr / 100`. Pretrained weights at the head's rate are destroyed inside the
  first epoch, and the symptom is a fine-tuned score *below* the frozen
  baseline rather than an error. Passing `backbone_lr` without
  `finetune_blocks` raises instead of being ignored.

- **Result schema v6 adds `finetune`** — `None` for a frozen probe, which is
  every v0.1 and v0.2 run and what a pre-v6 record carries by absence, so no
  reader needs a version check to ask the question. Otherwise `blocks`,
  `backbone_lr` and `trainable_params` (3,550,464 for two ViT-S blocks).
  `protocol` is unchanged at `visbench_semantic_seg`: the loss and metric are
  the same, only the trainable set differs.

- **The cache is bypassed on the fine-tuning path, not keyed differently.**
  Cache keys name the weights through `cache_key()`, and fine-tuned weights
  differ at every optimiser step: an entry written from them would be stale on
  arrival *and* indistinguishable from a frozen one, so every later frozen run
  of that backbone would silently read it. Keying on a per-step weights digest
  would grow the cache without bound and still never hit. A test asserts a
  fine-tuning run through `run()` writes **zero** cache entries, and the real
  VOC run left a 1.3 GB cache byte-for-byte the same size.

- `--finetune-blocks` / `--backbone-lr` on every dense CLI subcommand, and on
  `examples/segment_semantic.py`.

- **Detection metrics** (step 6c-2) — `box_iou`, `average_precision`,
  `detection_metrics` and `COCO_IOU_THRESHOLDS` in `visbench.metrics`, following
  the Pascal VOC protocol as `VOCevaldet.m` defines it.

  **Cross-checked against a literal transcription of that MATLAB**, over 3,060
  randomly generated APs at three IoU thresholds: zero mismatches, maximum
  absolute difference 0.0. The transcription is kept as a fast test rather than
  run once, because the obvious future change is vectorising the per-detection
  loop and the subtlety most likely to be lost is the one no analytic test
  covers — see the matching note below.

  Validated end to end on the real VOC val split, ground truth fed back as
  predictions over 500 images (1,249 boxes, 115 difficult):

  | predictions | mAP@50 | mAP@50:95 |
  | --- | --- | --- |
  | oracle (ground truth) | **1.0000** | **1.0000** |
  | boxes jittered 3 px | 0.9224 | 0.6731 |
  | half the objects dropped | 0.5270 | 0.5270 |
  | nothing detected | 0.0000 | — |

  An exact 1.0000 is the check that matters: an off-by-one in the matching, the
  recall denominator or the interpolation would land near 1 without reaching it.

  **`difficult` objects are ignored, not dropped — measured, not argued.** VOC
  removes a detection matching a difficult object from the tally entirely,
  neither true positive nor false positive. Dropping those objects from the
  ground truth instead makes a *correct* detection of one a false **positive**.
  Same oracle predictions, same 500 images, scored both ways:

  | protocol | mAP@50 |
  | --- | --- |
  | VOC's rule (ignore) | **1.0000** |
  | dropped from ground truth | 0.9567 |

  **4.3 mAP points, and the wrong one is lower** — so it reads as a weaker
  detector rather than a scoring bug. Only the first can claim VOC's protocol,
  which is why `average_precision` takes a `difficult` mask and why a run headed
  for scoring must build its dataset with `include_difficult=True`. 6c-1's
  `include_difficult=False` default is correct for training targets and is not
  sufficient here.

  **AP is dataset-level, and this is the one place the codebase's "per image,
  then averaged" rule does not apply.** Every other metric scores each image and
  averages so uneven coverage cannot reweight the split; AP cannot, being the
  area under one curve built by ranking every detection in the split. A test
  constructs a case where the global answer is 2/3 and the per-image mean is
  0.75, so the two cannot be quietly confused.

  Other conventions, each stated because each moves the number:

  - **Matching follows `VOCevaldet.m` including its order of checks** — a
    detection matches the box it overlaps *most*, and only then is that box's
    state consulted, difficult before already-claimed. There is deliberately no
    fallback to the second-best box: a greedy variant that reassigned duplicates
    scores higher than the reference and stops being comparable, while passing
    every hand-computed test.
  - **All-points interpolation** (VOC2010+ and COCO), not VOC2007's 11-point
    sampling, which is systematically higher and must not share a table with it.
  - **`map_50_95` is COCO-*style*, not a COCO number**: COCO's ten thresholds,
    but all-points integration at each where COCO quantises recall to 101 points.
    `map_50` is directly VOC-comparable.
  - **A class with no non-difficult objects is `None`, not 0** — recall has no
    denominator, and a 0 would drag mAP down in proportion to how many categories
    a split omits. `classes_scored` reports the real denominator, which is not
    always `num_classes`. A class that is present but entirely missed scores 0.0,
    and the two stay distinct.

- **The detection probe** (step 6c-3) — `DetectionTask`, registered as
  `detection`, plus a `DetectionHead` registered as `detection` and a
  `visbench run detection` subcommand. This completes step 6c: dataset (6c-1),
  metric (6c-2), head (6c-3), in that order and for the reason that order was
  chosen — the head is judged by a scorer that was cross-checked against
  `VOCevaldet.m` to zero difference *before* it existed, so a low mAP here says
  something about the head or the features and nothing about the metric.

  **Anchor-free and single-scale**, deliberately. One 1x1 convolution for class
  logits and one for box distances, over the backbone's patch grid at its native
  stride. FCOS's centre-inside-box assignment reduced to one level (smallest area
  wins an ambiguous cell, which is FCOS's own within-level tie-break — with no
  pyramid there is nothing else left of the rule), sigmoid focal loss on
  classification, GIoU loss on the positives.

  Focal because a dense anchor-free grid is overwhelmingly background and plain
  BCE there converges to predicting nothing while its loss falls. GIoU because
  the plain IoU loss has **zero gradient when the boxes do not overlap**, which
  is the state every prediction starts in. The classification bias is
  initialised to the focal prior, without which the schedule is mostly spent
  discovering that background is common.

  **The absolute mAP is low and that is the design, not a defect.** A
  single-scale linear head has no feature pyramid, so small objects fall between
  cells and are unrecoverable. Records say
  `protocol: "visbench_anchor_free_det"` — not `probe3d` (that paper has no
  detection task) and not VOC's (the *metric* is VOC's, the head is not). Read
  the number against another backbone, never against published detectors.

  **Not a `DenseTrainingTask` subclass, and not a close call.** That base assumes
  a stackable `(B, C, H, W)` target and recovers a split metric by weighting
  per-image metrics by batch size. Detection has neither: its target is a
  variable-length box list, and average precision is a dataset-level ranking
  that no weighted mean of per-batch numbers reproduces. What the two *do* share
  — probe3d's warmup/cosine schedule — was lifted into a new
  `visbench/tasks/schedule.py` and is now used by both, so a detection number and
  a segmentation number differ in the head and the loss rather than in the
  optimisation. `DenseTrainingTask` behaviour is unchanged.

  **The scored split keeps `difficult` objects; the training split drops them.**
  6c-2 measured that asymmetry at 4.3 mAP: VOC *ignores* a detection matching a
  difficult object, and dropping those boxes from the ground truth instead reads
  as a weaker detector rather than a changed protocol. The CLI builds the two
  splits accordingly and `DetectionTask` drops them from assignment as well, so
  one dataset with `include_difficult=True` can serve both halves.

  **`--image-size` reaches the dataset and the probe from one flag**, in the CLI
  and in `examples/detect.py`. Box targets are absolute post-transform pixels, so
  two different values put every grid cell at the wrong coordinate — and the run
  trains, scores badly, and reads as a weak backbone. The probe range-checks its
  targets, but that only catches one direction of the mismatch; sharing the flag
  is what catches both.

  Two tests carry the correctness claim. `_decode` applied to a hand-built
  "perfect" head output must reproduce the exact box — the detection counterpart
  of 6c-2's oracle check, since any off-by-one in the cell centres, the stride or
  the corner arithmetic lands *near* the box without reaching it. And a probe
  trained on features that literally encode the answer must reach **1.0 mAP**,
  which it does: assignment, both losses, the exp/stride decoding and VOC's AP
  all have to describe the same box for that to be reachable at all.

  Proved on real weights against VOC 2012 Detection (`ImageSets/Main`), 600
  train / 600 val at 224px, linear head, ten epochs:

  | backbone | map_50 | map_50_95 | classes_scored | dets/image | train_loss |
  | --- | --- | --- | --- | --- | --- |
  | DINOv2-S/14 | 0.2127 | 0.0722 | 20 of 20 | 84.6 | 1.2076 |
  | DINOv2-B/14 | **0.2616** | **0.0930** | 20 of 20 | 88.5 | 1.1124 |

  DINOv2-B leads by 4.9 mAP@50 — recorded as an observation, not as a check the
  probe passed; mid-level similarity still ranks the two the other way. The
  split is 600/600 rather than the full 5,717/5,823, so these establish that the
  probe runs end to end on real features, not a headline number.

- **A bounding-box dataset** (step 6c-1) — `DetectionFolderDataset`,
  `load_voc_boxes` and `VOC_CLASSES` in `visbench.data`. The first target in this
  codebase that **transforms rather than resamples**.

  Every dense target so far has been resized and cropped by the same loader that
  resizes and crops its image, so the two cannot drift. A box is four numbers
  that must be rescaled and shifted *by hand* to follow the image, and when that
  is skipped **nothing raises** — the boxes describe the original 500x375 frame
  while the tensor is 224x224, and the probe trains against supervision that is
  wrong everywhere. That is the correspondence misalignment bug (recall@1px =
  0.003) with a new coordinate convention.

  So the convention is fixed and asserted: **`xyxy`, absolute pixels,
  0-indexed, in post-transform space.** VOC is 1-indexed (its minimum
  `xmin`/`ymin` over all 17,125 files is 1), so one is subtracted at the loader
  boundary and nowhere else. The load-bearing test runs a non-square image
  through a known geometry, where a missed rescale, a missed shift and a swapped
  axis each produce a visibly different answer — fake backbones cannot show any
  of that.

  Boxes are rescaled by the **achieved** ratio rather than the nominal one: the
  resize rounds and applies a floor, so `image_size / min(w, h)` is not quite
  the factor actually used, and the difference is a sub-pixel error that grows
  with box size and hides in any single image. The image half is byte-identical
  to `DenseFolderDataset`'s crop, asserted by a test, because the box transform
  is derived from that geometry.

  A box outside the centre crop is **dropped**, not kept at zero area — scoring a
  detector against an object absent from its input measures nothing — while a
  straddling box is clipped, since its visible part is the correct target.
  `boxes`, `labels` and `difficult` are indexed by one mask so they cannot
  drift, and `num_original` keeps "this image has no objects" distinguishable
  from "this image's objects were all dropped". An image with nothing left is
  legitimate and does not raise.

  **`difficult` is returned by the loader and filtered by the dataset**, not
  filtered on read. VOC's protocol excludes those 4,462 objects from evaluation,
  but hiding them inside the loader would make the exclusion invisible to the
  result record, which is exactly what the `protocol` field exists to prevent.
  `include_difficult=False` is the default, matching VOC, and appears in
  `describe()`.

  Verified on the real split, not only on fixtures: 5,823 val images, and **all
  17,125 annotation files parse with zero failures**, giving 40,138 objects of
  which 4,462 are difficult — a count that independently matches what `grep`
  reports, which is the cross-check that the parser reads what the files say. No
  box in the first 300 images falls outside its tensor or comes back degenerate.

### Changed

- **CLAUDE.md said the result schema was at v5 in one place and v6 in another.**
  `SCHEMA_VERSION` has been 6 since 6a added `finetune`; the summary line in
  "Current state" still named v5 and credited 5j's `dataset_params` as the most
  recent addition. The two statements sat about 250 lines apart, so nothing
  forced them to be read together. Corrected to v6, keeping the 5j attribution
  as the previous bump. The failure mode is a session bumping to 7 for a field
  6a already shipped, which is not additive — it repurposes a version number.
- Recorded test counts refreshed to **1024 fast / 76 slow** (were 999 / 73,
  which predate 6a and 6b). Re-verified on 2026-07-29: all five commands green,
  no open issues on the tracker.

### Tidy-up after the v0.2.0 release

From a full audit of the repository on 2026-07-29 — all five verification
commands green, no open issues.

#### Added

- **`tests/test_readme.py` — every link and image in the README must be
  absolute**, checked in the fast suite. The README is package metadata:
  `pyproject.toml` names it as the long description, so PyPI renders it from
  `pypi.org`, where a relative path resolves against nothing and 404s. GitHub
  resolves the same path against the repository and looks perfect, so the
  mistake is invisible everywhere it is normally read.

  CI already runs `twine check dist/*` in its `build` job, which is
  `readme_renderer` — the renderer PyPI itself uses — and it **cannot catch
  this**: a relative link is valid markdown and renders without complaint. It
  simply points nowhere. The two checks are for different failures and neither
  substitutes for the other.

  Every link in `README.md` was made absolute by hand while preparing v0.2.0,
  and nothing stopped it drifting back; a PyPI version can never be reused, so
  a broken link would ship until the next release. The test carries a second
  assertion that the extraction pattern actually finds both link syntaxes and
  roughly the expected number of them — a regex that matched nothing would pass
  the guard forever, which is the failure the QuickGELU warning filter shipped
  with for its whole life.

#### Changed

- CLAUDE.md's release note said to check the README with `readme_renderer`
  before an upload — a manual step, with the tool in no extra and no
  automation behind it. It now points at the two checks that actually run:
  CI's `twine check` for rendering, and the new test for relative paths.
- The recorded fast-test count in CLAUDE.md was 932, six behind the 938 the
  v0.2.0 work left, and is now 969.
- Removed the stale `visbench-0.2.0` wheel and sdist from `dist/`. Gitignored,
  so nothing tracked changed, but `twine upload dist/*` at the next release
  would have tried to re-push a version PyPI will refuse.

## [0.2.0] — 2026-07-29

**v0.2 — dense mid-level tasks, broader backbone support, and a command line.**

Eight probes across three levels, three backbone families, four trained dense
probes, and a `visbench` command that reaches all of them. Everything below is
measured on real data, not fixtures.

```bash
uv sync --all-extras
visbench run correspondence --data /path/to/images --split val --limit 200
```

```python
import visbench
from visbench.data import DenseFolderDataset, load_label_map

result = visbench.run(
    "dinov2_vits14", "semantic_segmentation",
    DenseFolderDataset("voc/val", target_loader=load_label_map),
    train_dataset=DenseFolderDataset("voc/train", target_loader=load_label_map),
    num_classes=21,
)
result.metrics["miou"]     # 0.732
```

### What v0.2 measured

Every number below came from an `examples/` script or the CLI on a real
checkpoint, and each is the reason to have built the probe rather than a
by-product of it.

| task | dataset | metric | DINOv2-S/14 | DINOv2-B/14 |
| --- | --- | --- | --- | --- |
| semantic segmentation | VOC 2012 val | mIoU | 0.732 | **0.753** |
| mid-level similarity | NIGHTS test | 2AFC accuracy | **0.870** | 0.858 |
| correspondence | Imagenette val | recall@1p (ceiling 0.951) | 0.783 | — |

**The small model wins one and loses the other**, which is the case for probing
more than one level rather than assuming a single ranking — and the reason not
to sanity-check a new task by asking whether the larger backbone came out
ahead. Splitting NIGHTS by whether the reference image came from ImageNet gives
0.882 against 0.854 for DINOv2-S, so some of that 0.870 is contamination rather
than perceptual alignment.

### Added

- **The `visbench` command line**, the last piece of v0.2 and deliberately the
  last: a CLI freezes an API into strings that end up in shell scripts, and
  every one of the eight probes changed shape at least once while the Python
  side was being built. Three commands — `visbench list`, `visbench run <probe>`
  and `visbench cache stats | clear`.

  ```bash
  visbench run semantic_segmentation --data VOCdevkit/VOC2012 \
      --image-dir JPEGImages --target-dir SegmentationClass \
      --stems ImageSets/Segmentation/val.txt \
      --train-stems ImageSets/Segmentation/train.txt \
      --num-classes 21 --backbone dinov2_vits14
  ```

  Each probe is its own subcommand, because they do not take the same data:
  `visbench run depth --help` shows the folder layout depth expects and only
  depth's flags. All eight are reachable, and a test asserts that
  `visbench.list_probes()` and the CLI's table are the same set — a probe that
  ships without a way to run it from a shell should be a deliberate omission,
  not a drift.

  `run` is a thin wrapper over `visbench.run()`, which was written to be exactly
  that. What the CLI adds is dataset *construction* — which layout a probe
  expects, which loader reads its targets — held as a table of `ProbeSpec` rows
  in `visbench.cli.datasets` rather than a hierarchy. Adding a probe is a row.

  Verified against the numbers the Python API already produced, on the same
  data: NIGHTS similarity **0.8701** (identical to the figure recorded in step
  5i) and VOC val semantic segmentation **0.733 mIoU** against the recorded
  0.732.
- **`visbench.run()` now covers correspondence**, which it had refused since
  v0.1 because the task takes image pairs plus geometry rather than images plus
  labels. A task declares `uses_pairs` — the same kind of declaration
  `uses_dense` already was, and for the same reason: the shape is not
  inferable from anything else the task exposes, and guessing wrong is silent.

  The resolution is the one `TwoAFCDataset` reached for triplets: **do not
  widen the cache, flatten the structure and put it back by index.**
  `PairViewDataset` presents a pair dataset's two views as one dataset of `2N`
  images, so extraction, batching, streaming and the identity memo all work
  unchanged, and `regroup` restores the pairing. It is lazy in both directions:
  scoring a pair loads two feature maps, which go out of scope immediately,
  rather than materialising a whole split of dense features that had just been
  streamed to disk.
- **`PairDataset.view_identity` finally has a caller.** It has existed, with
  tests, since v0.1 and nothing in the library used it — `examples/correspond.py`
  handed the cache a bare list of PIL images, which has no identity at all, so a
  *fully cached* correspondence run still decoded, cropped and warped every
  image to discover it had the features already. Through `run()` a second run
  now re-extracts nothing. Measured on 200 Imagenette pairs with DINOv2-S/14:
  16.4 s cold, 8.2 s warm, identical scores.
- **A correspondence run records its ceiling.** `recall@1px` on DINOv2 ViT-S/14
  at 224px has a *ceiling* of 0.015 — matches can only land on patch centres —
  so a score logged without it invites the conclusion that the backbone failed
  when the grid is merely coarse. CLAUDE.md has required reporting the two
  together since v0.1 and only `examples/correspond.py` obeyed;
  `BaseTask.context_metrics` is the hook that makes `run()` and the CLI do it,
  under `ceiling_*` keys. Empty for every other task. Measured on 200 Imagenette
  pairs: `recall@1p` 0.783 against a ceiling of 0.951.
- **Result schema v5 adds `dataset_params`** — the dataset's counterpart to
  `task_params`, filled from whatever `BaseDataset.describe()` returns beyond
  the fields the record already has. A correspondence run's `max_warp` and a
  dense split's `image_size` decide what the number means and were recorded
  nowhere: they changed the fingerprint, so two such runs were distinguishable,
  but only as "not the same data", with nothing saying how they differed. Open
  rather than a column per setting, so a new dataset type never forces another
  bump.
- **`ImageFolderDataset.balanced_subset(n)`** — at most `n` images from *each*
  class. `subset(n)` takes a prefix, and the file list is grouped by class, so a
  prefix of an Imagenette split is entirely class 0 and a single-class retrieval
  run scores 1.0 while measuring nothing. Both examples that needed this carried
  their own copy with the same warning attached, which is the signal it belonged
  on the dataset; the CLI's `--limit` would have been a third.
- **`--stems` / `--train-stems`** on every dense subcommand, for splits named by
  a file rather than by a directory — how every real benchmark expresses an
  official split. Passing one switches the layout: `--data` becomes the dataset
  root, because the file *is* the split, and mixing the two would be asking the
  same question twice and letting the answers disagree. Without it the CLI could
  not run VOC, which is the one dataset semantic segmentation is proved on.
- `HomographyPairDataset` and `PairViewDataset` are exported from
  `visbench.data`, and `DEFAULT_CACHE_DIR` from `visbench.cache`. The first was
  an oversight: tests had to reach into `visbench.data.pair_dataset` for a class
  the README uses.

- **Mid-level image similarity** (`similarity`) — zero-shot two-alternative
  forced choice, following Chen, Marks & Cheng (arXiv:2411.17474). A reference
  and two candidates; the probe compares `cos(ref, left)` against
  `cos(ref, right)` in frozen pooled feature space and is scored against the
  human vote as binary classification (accuracy, F1, precision, recall).
  Deliberately kept separate from high-level retrieval: the ground truth is
  perceptual, not categorical, and merging them would conflate two different
  questions.
  Nothing is trained. That paper's README describes "training a similarity
  estimator" while its code builds a test loader and freezes the backbone — the
  code is what VisBench follows, so `fit()` is a no-op like retrieval's.
  Measured on the NIGHTS test split (1,824 triplets, `min_votes=6`), pooled
  features at 224px, against a ~51% chance baseline: DINOv2-S/14 **0.870**,
  DINOv2-B/14 0.858, CLIP-B/16 0.828, ResNet50 0.827. The small DINOv2 beats the
  base one here and loses to it on semantic segmentation, which is the case for
  probing more than one level rather than assuming one ranking. Splitting the
  test set by whether the reference came from ImageNet gives 0.882 against 0.854
  for DINOv2-S — a contamination signal worth reading before the headline
  number.
- **`TwoAFCDataset`** for NIGHTS-style triplets. A triplet is three images while
  the cache works one image at a time, so rather than widen the cache the
  dataset presents itself as a flat collection of *unique* images and puts the
  triplet structure in `labels()` as indices into itself. The cache, the
  fingerprint and `run()` all work unchanged, a shared reference is extracted
  once, and the pairing travels by index. `subset()` is refused there — slicing
  images would silently repoint every triplet — and `max_triplets=` on the
  constructor shortens both together instead.
  Columns are read **by name**; the reference implementation indexes them
  positionally (`iloc[idx, 2]` for the vote), which would score against the
  wrong column if the CSV were ever reordered, and would look like a mediocre
  result rather than an error.
- `two_afc_metrics`, verified to agree with scikit-learn to 1e-12 — the
  reference scores with sklearn, so matching it exactly removes any doubt about
  averaging conventions. Accuracy is computed as an exact integer ratio rather
  than a float32 mean, which differed in the third decimal.
- `tie_rate` is reported alongside. A forced choice has to break an exact tie
  somehow, and how often that happened is the difference between a real score
  and one propped up by a coin flip.
- `examples/similarity.py`, including the `test_imagenet` / `test_no_imagenet`
  splits — a backbone pretrained on ImageNet has seen those references, so a gap
  between the two is a contamination signal rather than a similarity result.
- **`BaseDataset.subset(n_or_indices)`** — the public way to shorten a split.
  Every example had been doing it by hand: two reached into the private
  `_labels`, and the four dense ones sliced three parallel lists in step, each
  carrying the same comment warning that dropping one would pair a target with
  the wrong image. A hazard that needs the same warning copy-pasted four times
  is a missing method. Subclasses declare `_parallel_attrs` and one tested
  implementation reindexes them together; `PairDataset` overrides it, since it
  delegates to a source rather than holding sequences. The original is left
  untouched, and `fingerprint()` follows automatically, so a `--limit` run can
  never be mistaken for a full one in the cache or the record. An `int` clamps
  ("use at most N"); an explicit index list is validated strictly, because a
  silently shorter split is the failure the method exists to prevent.
- **Semantic (multi-class) segmentation** (`semantic_segmentation`) — the
  high-level counterpart to the mid-level binary task, on the same base class
  and schedule so a difference between the two numbers is a difference in what
  is asked of the representation, not in how it was trained. Cross-entropy over
  class indices, masked at `IGNORE_INDEX = -1`; `_activate` is deliberately the
  identity, because cross-entropy needs logits and `argmax` is indifferent to
  any monotone transform, so loss, metrics and `predict` cannot disagree.
  `predict` returns `(B, C, H, W)` scores and `predict_labels` their argmax.
  `num_classes` is **required**: it sizes the head, and a wrong value does not
  raise, it trains a head that cannot express some categories.
  Measured on Pascal VOC 2012 val (1449 images), linear head at 224px, default
  ten-epoch schedule: DINOv2-S/14 **mIoU 0.732** (pixel accuracy 0.926, mean
  class accuracy 0.831, `train_loss` 0.193); DINOv2-B/14 **0.753** (0.931,
  0.838, 0.166).
- **mIoU is reported both ways, because the two reductions disagree.**
  `miou` accumulates one confusion matrix over the split and divides once —
  what VOC, ADE20K and Cityscapes define, and the only version comparable to
  published numbers. `miou_per_image` is this codebase's per-image rule. On VOC
  they differ by five points (0.732 against 0.683), so both are reported under
  distinct names rather than one being chosen silently. `SemanticSegmentationTask`
  overrides `evaluate` to accumulate both in one pass, since no weighted mean of
  per-batch ratios equals the ratio of the sums. New in `visbench.metrics.dense`:
  `confusion_matrix`, `metrics_from_confusion`, `semantic_metrics`.
- **`load_label_map`** reads a label map **without mode conversion**. VOC's
  `SegmentationClass` PNGs are palette images whose raw bytes are the class
  indices; `convert("L")` resolves the palette and turns classes
  `[0, 1, 15, 255]` into `[0, 38, 147, 220]` — which loads, trains and scores
  against labels that mean nothing. 255 becomes -1 by default, since leaving it
  is not neutral: it would become a class the probe is trained and scored on.
- **`DenseFolderDataset(stems=...)`** takes an official split list. VOC ships
  17k images beside 2.9k segmentation labels and names split membership in
  `ImageSets/Segmentation/*.txt`; without this the folders look like a
  catastrophic mismatch and pairing rightly refuses. Order is preserved, since
  targets travel by index, and a stem missing from either folder raises.
- **`DenseTrainingTask.target_dtype`** — targets were coerced to float in three
  places, which is right for a measurement and wrong for a class index. The
  coercion is now one attribute, so training, evaluation and `predict` cannot
  disagree about what a target is.
- `examples/segment_semantic.py`, which reads the Pascal VOC devkit directly
  with `--voc` as well as the folder-pair layout the other examples use.

- **timm CNN backbones** (`resnet18`, `resnet50`, or any timm CNN via
  `TimmBackbone(model_name=...)`) — the first non-ViT family. Dense features
  are the last conv map before global pooling, flattened to a token sequence so
  `extract_features` needs no branch on architecture. Mean-pooling those tokens
  reproduces a ResNet's own `global_pool` output exactly, so the pooled vector
  means the same thing for a CNN as a CLS token does for a ViT. Behind a `timm`
  extra.
- Cache keys carry the timm pretrained tag: `resnet50.a1_in1k` and
  `resnet50.a3_in1k` are different weights under one architecture name.
- **`CustomBackbone`** — wrap any `nn.Module` plus a preprocessing callable.
  The grid is read from the module's output shape, `embed_dim` from the first
  forward pass, and the cache key from a hash of the weights, so a fine-tuned
  checkpoint cannot reuse its parent's cached features. Ambiguous output shapes
  raise rather than guess: a square token *count* from a non-square layout
  would otherwise misplace every patch silently.
- `visbench.register_backbone` / `register_task` are public, so a
  `BaseBackbone` subclass outside this package can claim a registry name.
- `extract_features` takes **`feature_mode`**, so `dense_cls_broadcast` and
  `dense_plus_cls` are reachable through the public API. They were declared,
  implemented and tested in v0.1 but `apply_feature_mode` had zero callers and
  no parameter exposed them — a DPT head is exactly the consumer that wants
  `dense_plus_cls`, so this had to exist before heads were designed against it.
  `dense_plus_cls` returns the global vector under a new `cls` key, and the
  cache both keys on the mode and stores `cls`.
- **Pluggable task heads**, selectable by name per run (`visbench.heads`):
  `LinearHead` (1x1 convolution over the dense grid, upsampled) and `DPTHead`
  (RefineNet-style multiscale fusion, following probe3d and Ranftl et al.).
  `register_head` makes this a real extension point. A head declares which
  feature modes it consumes and `check_feature_mode` rejects a mismatch at
  construction rather than as a shape error partway through training.
  `DPTHead` refuses a single feature map: fed one layer it is not multiscale,
  and duplicating the input would report a single-layer result as a DPT number.
- `DPTHead(cls_dim=...)`, for when a backbone's CLS width differs from the
  channel count of the layer the vector is injected alongside.
- **Multi-layer feature extraction.** `extract_features(layers=[2, 5, 8, 11])`
  returns `dense_layers` — one map per requested depth, from a **single**
  forward pass — plus the resolved `layer_indices`. Declared in the interface
  since v0.1 and wired up now that the single-layer path is proven; this is
  what `DPTHead` has been waiting for.

  `dense`, `pooled` and `cls` still describe the last requested layer, so a
  multi-layer call is a strict superset of a single-layer one and a task
  reading only `dense` is unaffected. `dense_layers` is a separate key rather
  than `dense` sometimes being a list: a type that depends on how many layers
  were requested would break every existing consumer the moment a layer list
  was widened.

  Layer indices are resolved once, in `BaseBackbone.resolve_layers`, instead of
  in each backbone: negatives count from the end, and the list must be strictly
  increasing, since a multiscale head reads the first layer it is given as the
  coarsest. A descending or repeated list is rejected rather than reordered.
- Each layer gets **its own cache entry**, keyed on the resolved index.
  Widening `[3, 7]` to `[3, 7, 11]` re-extracts one layer rather than three,
  and a later single-layer run at layer 7 reads what the multi-layer run
  stored. `layers=[-1]` and `layers=[11]` name the same entry on a 12-block
  model rather than storing identical features twice.
- `TimmBackbone.layer_channels([1, 2, 3, 4])`, because a CNN's stages differ in
  width — which is exactly why `DPTHead` accepts per-layer `in_channels`.
- `CustomBackbone(layer_feature_fn=..., num_layers=...)`. An arbitrary
  `nn.Module` has no `get_intermediate_layers` to call, so this is where a user
  says how their model exposes depth. Without it `num_layers` stays 1 and a
  multi-layer request is refused — returning the final map several times would
  let a multiscale head report a single-layer result.
- Result schema v4 adds `layers`. A record for a run over four depths is not
  the same run as one over the last, and widening `layer`'s type would have
  changed how every v1–v3 record on disk parses.
- **Depth estimation** (`get_probe("depth")`) — the first dense task, and the
  first thing to use heads, multi-layer extraction and the cache together.
  Reproduces probe3d's configured protocol rather than re-deriving it: 256
  uniform depth bins with the prediction as their expectation (AdaBins'
  parameterisation), a loss of 10x scale-invariant log plus 0.5x gradient, and
  AdamW at 5e-4 for 10 epochs with 1.5 warmup and cosine decay. Its
  `metrics.py` and `losses.py` are separately MIT-headered, so these follow the
  reference closely enough for the numbers to be comparable — see NOTICE.
- `visbench.metrics.dense.depth_metrics` — `d1`/`d2`/`d3`/`rmse` per probe3d,
  plus `abs_rel` (flagged as an addition, since probe3d does not report it).
  Valid pixels only, averaged per image and then across images: pooling every
  pixel of a split instead would weight images by how much valid depth they
  happen to contain. `scale_invariant=` and `nyu_crop=` are available and off
  by default, because a number computed with either is not comparable to one
  computed without.
- **`DenseFolderDataset`** — images and per-pixel targets paired by filename
  stem, with the resize and centre-crop applied to **both together**. This is
  the module's whole reason to exist: a target cropped differently from its
  image trains a probe against misaligned supervision, and the only symptom is
  that the numbers come out bad. Targets resample nearest-neighbour, never
  bilinear, so holes cannot bleed into valid depth and reappear as plausible
  wrong values the valid mask no longer excludes.
- `BaseTask.layers`, carried into extraction by `visbench.run()`, so a task
  with a multiscale head gets the depths it needs from one forward pass. The
  record stores them resolved against the backbone.
- `examples/depth.py`.
- **Streaming features from disk**, lifting the memory ceiling that made dense
  tasks unable to run their own benchmark datasets.
  `FeatureCache.materialise(...)` runs the same extraction as
  `extract_dataset(...)` but keeps nothing in memory, returning a
  `CachedFeatures` — an ordinary `torch.utils.data.Dataset` over the per-image
  files the cache already writes. Hand it to a `DataLoader` and batching,
  per-epoch shuffling and worker processes come for free.

  Random access rather than a generator, deliberately: training reshuffles
  every epoch, and a generator yielding batches in dataset order can only
  shuffle *within* a batch, which would quietly make a probe worse than the
  representation it is meant to measure.

  Measured on 1,200 images whose features are 0.63 GB: **10.8 GB peak RSS in
  memory against 1.7 GB streaming**, and the 1.7 is mostly torch itself.
- Targets stream through the same index, so `dataset.labels()` no longer stacks
  every depth map (~4.8 GB for NYUv2). Reading features and supervision by one
  index also makes it structurally impossible for them to drift apart — a test
  shuffles the loader and checks every feature still arrives with its own
  target.
- `DepthTask` trains and evaluates from either source through one loop, and
  `evaluate` scores batch by batch rather than collecting every prediction
  first. `visbench.run()` streams automatically for any task declaring
  `uses_dense`.
- **Surface normal estimation** (`get_probe("surface_normal")`), the second
  dense mid-level task, following probe3d's `snorm_dpt.yaml` and its ten-epoch
  schedule: three direction channels plus an optional kappa, Bae et al.'s
  uncertainty-aware angular loss, and `evaluate_surface_norm`'s metrics —
  within-11.25/22.5/30-degree fractions and angular RMSE, plus the mean and
  median the wider literature reports. Predictions come back L2-normalised, so
  `predict()` hands over an actual unit normal rather than an unscaled
  direction.

  `normal_source` is recorded in every result: NYU normals are *derived*, from
  GeoNet's extraction or Ladicky's rather than from a sensor, and the sources
  disagree enough that a run which does not say which is not comparable to one
  that does.
- `SurfaceNormalTask.fit` **detects kappa collapse and warns**. probe3d's
  uncertainty-aware loss lets kappa settle wherever the head's accuracy puts it
  (3.5 at 30 degrees of error, 1.2 at 60, 0.05 at chance), which is the
  intended behaviour until the head is near chance — there kappa scales the
  direction's gradient by 1/20, weak supervision keeps accuracy at chance, and
  the two hold each other down. A real DINOv2 linear probe on a small split
  does exactly this and reports a chance-level score with no error at all.
  Whether a run falls in depends on head initialisation, so it is measured per
  run rather than predicted. The loss is left as probe3d wrote it: switching
  silently to the plain angular loss would make VisBench's numbers
  incomparable with the published ones, which is the only reason to have
  borrowed the protocol.
- **`DenseTrainingTask`**, the shared body of every trained dense probe —
  feature sources, batching, head construction, the optimiser and its schedule,
  the training loop, batch-wise prediction and metric averaging. A subclass
  supplies four things: `out_channels`, `_activate`, `_loss`, `_batch_metrics`.
  Lifted out of the working `DepthTask` rather than designed up front, because
  the second dense task is the first point at which the shared part is
  knowable. Depth's behaviour is unchanged and its tests pass untouched.
- `DenseFolderDataset` handles **vector targets**: a `target_loader` returning
  `(C, H, W)` is resized, cropped and stacked exactly like a scalar map, so
  surface normals travel the same geometry path depth does. `max_target` now
  raises on a multi-channel map rather than capping each component
  independently, which would zero the x component of every steep normal.
- `load_normal_map` reads `.npy` in either `(3, H, W)` or `(H, W, 3)` layout,
  and 8-bit RGB under the usual `2 * v / 255 - 1` encoding. Output is
  L2-normalised, and a pixel with no direction becomes exactly `(0, 0, 0)` —
  which is what marks it invalid, the role a 0 plays in a depth map.
- `examples/normals.py`.
- **Generic (binary) object segmentation** (`get_probe("generic_segmentation")`),
  the third dense task and the first whose protocol is *not* probe3d's — that
  paper has no binary segmentation task, so there was nothing to borrow. What is
  kept is its optimiser schedule, so a backbone's segmentation number sits
  alongside its depth and normal numbers under one training budget; the loss is
  masked binary cross-entropy and the metrics are foreground IoU, Dice and pixel
  accuracy. Records say `protocol: "visbench_binary_seg"` rather than
  `"probe3d"`, so no reader mistakes the two.

  Foreground IoU is the number to quote. Objects are a minority of most frames,
  so pixel accuracy alone looks excellent for a probe that predicts background
  everywhere — on the example dataset that is 87% accuracy at 0 IoU. All three
  are reported precisely because they disagree there.
- `binary_iou`, previously a `NotImplementedError` stub, with the signature it
  always had. Per-image then averaged, like every other metric in
  `visbench.metrics.dense`, so object size cannot reweight the split. An image
  with neither predicted nor ground-truth foreground scores 1.0 rather than 0/0;
  one with no labelled pixels at all contributes zero, matching how depth treats
  an image with no valid ones.
- A **validity convention for label maps**: a pixel is unlabelled where the
  target is *negative*. Depth and normals read 0 as "no ground truth", and
  reusing that here would have discarded every background pixel and trained the
  probe to answer foreground everywhere. The loss and the metric mask
  identically, so the pixels trained on and the pixels scored are one set.
- `load_mask` — reads `.npy` or an image, **non-zero is foreground**, covering
  both the 0/1 and 0/255 conventions without guessing a scale, and never
  rescaling (dividing by 255 would turn every foreground pixel into 1/255).
  `ignore_index=` maps a dataset's explicit ignore value — 255 in VOC-style
  palette masks — to -1. Off by default: every pixel of a plain
  foreground/background mask is labelled, and inventing an ignore region would
  quietly shrink what the probe is scored on. Do not pass `max_target` for a
  mask; it exists to invalidate out-of-range *sensor* readings and against a
  label map would erase the foreground class.
- `examples/segment.py`.

### Fixed

- **"A backbone whose extra is missing is skipped in the registry" was never
  true.** CLAUDE.md had said so since v0.1 and the CLI's listing repeated it.
  Both CLIP and timm import their dependency lazily *inside* `__init__`, so the
  registration module imports cleanly and the skip logic in
  `_REGISTRATION_MODULES` never fires for either — all six backbones are listed
  on a core-only install. The behaviour is right and the documentation was
  wrong: `get_backbone("clip_vitb16")` raises `ImportError: ... pip install
  visbench[clip]`, which is far more useful than the registry's "Unknown
  backbone 'clip_vitb16'" would be. So the docs are corrected rather than the
  imports moved, and `registry.missing_extra(name)` now answers the question
  without importing anything — `visbench list backbones` marks the ones that
  need an extra instead of a footer claiming they are absent. **Found by
  installing the v0.2.0 wheel into an empty venv**, which is a check this
  project had never run.
- **`run()` could not configure any dense probe's `batch_size`**, and nothing
  said so. `run()` owns `batch_size` (extraction) and `device` (the backbone's),
  and forwards everything else to the probe constructor — where all four dense
  probes take a `batch_size` meaning their *training* batch. Passing it was a
  bare `TypeError: got multiple values for keyword argument`. Found by writing
  the CLI, which is exactly the sort of thing a second caller finds; the fix is
  the one the examples had already reached for independently, building the probe
  as an object and passing that. The CLI keeps the two separate as `--batch-size`
  and `--train-batch-size`.
- **`CorrespondenceTask.evaluate_ceiling` silently scored a prefix** when given
  more feature pairs than geometries — nine geometries against ten pairs
  produced a number computed from nine and reported as covering the split.
  `evaluate` had always checked lengths explicitly; the ceiling path never did,
  which meant the two entry points disagreed about what a valid call was. Found
  by working through `zip(strict=)` site by site ([#4]).
- **`load_mask` was documented as handling VOC-style palette masks and does
  not.** Its `convert("L")` resolves the palette, so VOC's void value 255
  arrives as a light grey — non-zero, therefore *foreground* — and
  `ignore_index=255` never matches, because it compares against the resolved
  value rather than the index. Nothing raises; the masks are simply wrong at
  every object boundary. The docstring now says so and points at
  `load_label_map`. Found while building the semantic task on the same files.
- **`zip(strict=)` is now explicit at every call site**, and `B905` is enforced
  rather than ignored ([#4]). 12 sites take `strict=True` — features to targets,
  keys to cache entries, requested layers to backbone outputs; `zip(resolved,
  resolved[1:])` in `backbones/base.py` takes `strict=False`, since pairing a
  list with its own tail is meant to be ragged. Most are backstops for checks
  that already existed a few lines above, which is the point: a silently
  truncating zip is the failure mode CLAUDE.md warns about for index-paired
  targets, and it still trains.
- `examples/` is type-checked in CI as well as linted, which caught four real
  signature bugs there (`limit: int = None`) and one in the new `subset()`:
  it returned `BaseDataset`, so a `DenseFolderDataset` stopped being one after
  a `--limit`. Now generic over the caller's type.
- Workflow actions moved off the deprecated Node 20 runtime
  (`checkout@v5`, `setup-python@v6`, `setup-uv@v6`).
- `examples/` is linted in CI. It was outside both ruff steps, so two
  `zip(strict=)` sites there survived the sweep that fixed the rest ([#4]).
- **CI now runs the slow suite** ([#2]), in `.github/workflows/slow.yml`:
  on every push to `main`, nightly at 03:00 UTC, and on demand. `addopts`
  deselects `slow` and the gating workflow runs a plain `pytest`, so until now
  nothing on `main` had ever executed a real backbone forward pass — which is
  how both [#1] and [#3] shipped under a green tick, one of them for three days
  while the very job meant to prove the 3.9 floor reported success. Weights
  (~1.7 GB across `~/.cache/torch`, `~/.cache/clip` and `~/.cache/huggingface`)
  are cached against `HUB_REF`, since changing that ref makes an old download
  the wrong code rather than merely stale. Kept out of the gating workflow and
  off pull requests so the download never blocks ordinary work.
- **The CLIP QuickGELU guard never fired** ([#3]). It promoted open_clip's
  warning to an error by filtering on `message=".*QuickGELU mismatch.*"`, a
  phrase open_clip has never emitted — so the filter never matched and the guard
  was dead code from the day it was written. No shipped number was wrong, since
  both registered variants pair `-quickgelu` configs with OpenAI weights
  correctly, but the one check standing between a user and a silently
  wrong-activation model could not fire. Detection now matches the single token
  common to both directions open_clip warns in, lives in
  `_promote_quickgelu_warning` so it is testable without downloading a
  checkpoint, and re-emits unrelated warnings instead of swallowing them. Its
  only test was `slow`, and CI does not run `-m slow` ([#2]), which is why this
  survived; the replacement tests are in the fast suite.
- `FeatureCache.extract_dataset` refused nothing when handed a `PairDataset`:
  it read `item[0]` and silently discarded the second view and the geometry,
  returning features for half the data. It now raises.
- `cls` was produced by extraction with `dense_plus_cls` but never stored, so it
  existed on a cache miss and vanished on the next hit.
- `DPTHead(use_cls=True)` sized its CLS projection from the *last* layer's width
  while injecting the vector at the *first*, so any head built with per-layer
  `in_channels` raised a matmul shape error. It now follows the stage the vector
  actually reaches, and checks the vector's width with a message that names the
  expected one.
- `DPTHead` read `head((stage0, stage1))` — a tuple of two layers rather than a
  list — as one dense map plus a CLS vector, and reported it as "got a single
  tensor" when the caller had passed two. A `(stages, cls)` pair is now
  identified by its first element being a sequence.
- CI's mypy step had been failing since it was made gating, and failing in the
  worst way: it never checked a line of visbench. mypy parses the
  *dependencies'* stubs under `python_version` too, and they use newer syntax
  than this package does — torch has `match` statements (3.10+), numpy 2.x's
  `__init__.pyi` has PEP 695 `type` statements (3.12+). At `"3.9"` mypy hit a
  syntax error inside torch and stopped with "errors prevented further
  checking". Now `"3.12"`, matching the lint job's interpreter; the setting
  tracks the newest syntax any dependency uses, not this package's floor, and
  will need raising again as they move.

  3.9 support is still enforced, by two more direct checks: ruff's
  `target-version = "py39"` and the CI test matrix, which runs the whole suite
  on 3.9.
- `load_image` rebound the `with Image.open(...) as img` target, assigning a
  plain `Image` to a name typed `ImageFile`. Real, and invisible until mypy
  started running: Pillow 12 types `exif_transpose` precisely enough to catch
  it, Pillow 11.3 did not.
- `run()` now passes the task's `feature_mode` into extraction. It never had,
  which no task noticed only because none had yet overridden the default.

### Changed

- **The minimum supported Python is now 3.10** (was 3.9). This is a fix, not
  housekeeping: the pinned DINOv2 `HUB_REF` uses `float | None` at class-body
  scope, which 3.9 evaluates at import and rejects, so the flagship backbone,
  six of seven `examples/` scripts and the entire slow suite were broken on the
  floor the package advertised ([#1]). The alternative — repinning `HUB_REF` to
  a 3.9-compatible commit — would have invalidated every cached DINOv2 feature
  on every machine, since `HUB_REF` feeds `cache_key()`. Raising the floor keeps
  the ref and the caches; keys verified identical before and after. `pytest
  -m slow` goes from 8 failed / 19 errors to **73 passed**.
  - `requires-python`, the 3.9 classifier, ruff's `target-version`, the CI test
    matrix, the README badge and `uv.lock` all move together.
  - mypy's `python_version` stays **3.12** — it tracks the newest syntax any
    dependency stub uses, not this package's floor.
  - Annotations modernised to PEP 604 (`X | None`) across 153 sites, which is
    what ruff's `UP` rules require once the target is 3.10. Mechanical; no
    behaviour change.
  - `B905` (`zip()` without `strict=`) is newly reachable and is **ignored with
    a comment rather than fixed** ([#4]). Each of the 13 sites needs its own
    answer — `zip(resolved, resolved[1:])` is intentionally ragged, while
    `zip(self.image_paths, self.target_paths)` wants `strict=True` and would
    convert a silent misalignment into an error. Behaviour changes do not belong
    in a floor raise.
- mypy is **gating** in CI. It had `continue-on-error` from when everything was
  stubs, which made it a check that could never fail; 19 errors had accumulated,
  including the `PairDataset` variance violation above. Now clean.
- Removed the unused `visbench.utils.device.batched` helper.
- `BaseBackbone._forward_features` now returns a **list** of
  `(patch_tokens, cls, grid_hw)`, one per requested layer, and receives layer
  indices already resolved. Only affects code subclassing `BaseBackbone`
  directly; `extract_features` is unchanged for single-layer callers.
- A timm ViT is rejected when the backbone is constructed rather than at the
  first extraction. `forward_intermediates` reshapes a ViT's tokens into a grid
  when asked for NCHW, so from that point the output is indistinguishable from
  a conv map and nothing would notice the CLS token had been dropped while
  `has_cls_token` stayed False.
- The README's development section now lists the three lint commands verbatim.
  Running mypy with different flags reads the same `[tool.mypy]` config but
  checks something else, which is how the above went unnoticed.

### Install

**The first VisBench release on PyPI.**

```bash
pip install visbench                 # core: DINOv2, every task, the CLI
pip install 'visbench[clip,timm]'    # + CLIP and timm CNN backbones
```

From source, for development or to reproduce the numbers above exactly:

```bash
git clone https://github.com/turhancan97/VisBench && cd VisBench
uv sync --all-extras              # exact locked versions — what the numbers above used
# or
pip install -e ".[dev,clip,timm]" # ranges, for day-to-day work

visbench list                     # the CLI comes with either
pytest                            # 938 fast tests, no weights downloaded
pytest -m slow                    # 73 more, against the real DINOv2 and CLIP checkpoints
```

The core install works on its own; `clip` and `timm` are optional extras. A
backbone whose extra is missing stays listed — `visbench list backbones` marks
it — and constructing one tells you which extra to install rather than
pretending the name does not exist.

### Known limits, carried into v0.3

- No fine-tuning. Every probe reads frozen features; unfreezing the last N
  blocks is v0.3's first item, and it is the first change to challenge the
  cache's founding assumption that features depend on the image and the weights
  alone.
- No detection, and nothing under `tasks/low_level/` but a README.
- Retrieval ranks with an N×N score matrix — ~10 GB at 50k images, still no
  chunked path.
- Correspondence ground truth is synthetic homographies, so it measures
  viewpoint robustness on a **plane**. probe3d's ScanNet/NAVI protocol was
  scoped for v0.2 and did not land; it needs those datasets downloaded, which is
  the actual blocker.
- The CLI covers all eight probes but assumes a folder layout per probe. Anything
  stranger than `--stems` can express needs the Python API, which takes any
  `BaseDataset`.
- **`pip install visbench` takes the newest torch in `>=2.0,<3.0`, whose CUDA
  build may not have kernels for an older GPU.** Verifying this release on a
  clean venv pulled `torch 2.13.0+cu130`, which fails on a V100 (compute
  capability 7.0) with `GET was unable to find an engine to execute this
  computation`. That is torch's packaging, not VisBench's, and `--device cpu`
  works — but if you are on pre-Turing hardware, install torch yourself first,
  or use `uv sync` against the lockfile, which pins the exact versions every
  number here was measured with.

## [0.1.0] — 2026-07-24

**v0.1 — prove the abstraction.**

The first release: two backbones, three tasks, and the infrastructure they
share, all running end-to-end on a local image folder.

```python
import visbench
from visbench.data import ImageFolderDataset

result = visbench.run(
    "dinov2_vitb14",
    "retrieval",
    ImageFolderDataset("data/tiny", split="val"),
    results="results/visbench.jsonl",
)
result.metrics    # {"recall@1": 0.99, "recall@5": 1.00, "mAP": 0.91}
result.record     # the ResultRecord saying exactly how they were produced
```

### Backbones

| Name | Weights | Patch | Grid @224 |
| --- | --- | --- | --- |
| `dinov2_vits14`, `dinov2_vitb14` | torch.hub, pinned to a fixed upstream commit | 14 | 16×16 |
| `clip_vitb16`, `clip_vitb32` | open_clip, OpenAI weights (QuickGELU-correct) | 16 / 32 | 14×14 / 7×7 |

One method covers both: `extract_features(image, pooling=..., layers=...)`
returns `{"dense": (B,C,H,W), "pooled": (B,C), "grid_hw": (H,W)}` from a single
forward pass, with the same shape for every architecture family. Tasks choose
pooling; backbones just execute it.

CLIP returns the **pre-projection** CLS token by default. The 512-d image-text
projection is trained to discard whatever does not help match a caption —
exactly the visual detail a mid-level probe measures — and DINOv2 has no
equivalent head to compare against. `use_projection=True` gets the projected
vector, under its own cache key.

### Tasks

| Level | Task | Training |
| --- | --- | --- |
| high | `classification` | linear probe, AdamW on cached features |
| high | `retrieval` | none — leave-one-out cosine |
| mid | `correspondence` | none — dense feature matching + Lowe ratio test |

Correspondence reports error in **patch widths**, not pixels. A match can only
land on a patch centre, so patch spacing is a hard floor on achievable error —
in pixels that floor moves with resolution and patch size, and `recall@1px` on
DINOv2 ViT-S/14 at 224px has a *ceiling* of 0.015. It also reports that ceiling
beside every score, so a low number reads as "coarse grid" rather than "bad
backbone".

### Infrastructure

- **Feature cache**, mandatory rather than an optimisation. Content-addressed,
  so one forward pass per image per backbone; a cached image is never decoded
  again.
- **`ResultRecord` / JSONL** under one additive schema, carrying the weights
  ref, dataset fingerprint, resolved pooling, seed, duration and task
  hyperparameters — enough to reproduce the number, not just read it.
- **`visbench.run()`** — resolve pooling, extract, fit if the task trains,
  evaluate, append the record.
- **`uv.lock`** pinning 116 packages with hashes, extras included; CI fails if
  it drifts from `pyproject.toml`.
- MIT licence, plus a `NOTICE` recording the CC BY-NC parts of probe3d that are
  deliberately **not** reused.

### Measured on Imagenette

3,925-image val split, one V100. Correspondence on 50 pairs at `max_warp=0.2`.

| Task | Metric | DINOv2 ViT-S/14 | CLIP ViT-B/16 |
| --- | --- | --- | --- |
| classification | top1 | 0.9939 | **0.9954** |
| retrieval | recall@1 | **0.9921** | 0.9893 |
| retrieval | mAP | 0.8893 | **0.9102** |
| correspondence | recall@1p | **0.7650** | 0.6993 |
| correspondence | ceiling | 0.9408 | 0.9505 |

CLIP leads on both semantic tasks and trails on the geometric one despite a
*higher* ceiling — the high-level / mid-level split the task taxonomy exists to
expose.

Caching, 13,394 images: cold run 208 s, fully cached 26 s, 107 MB on disk.

### Install

Not on PyPI yet.

```bash
git clone https://github.com/turhancan97/VisBench && cd VisBench
uv sync --all-extras            # exact locked versions
# or
pip install -e ".[dev,clip]"    # ranges, for day-to-day work

pytest                          # 286 fast tests, no weights downloaded
pytest -m slow                  # 37 more, against the real checkpoints
```

### Known limits

- `run()` does not cover correspondence — it takes pairs plus geometry rather
  than images plus labels. See `examples/correspond.py`.
- Retrieval ranks with an N×N score matrix; ~10 GB at 50k images, and there is
  no chunked path yet.
- Correspondence ground truth comes from synthetic homographies, so it measures
  viewpoint robustness on a **plane**, not 3D correspondence across parallax.
  probe3d's ScanNet/NAVI protocol is the real test and lands in v0.2.

### Deferred to v0.2

CLI · ResNet/timm and user-supplied custom backbones · depth, surface normals,
generic segmentation, mid-level similarity · pluggable task heads (linear +
DPT) · multi-layer extraction

### Prior art

Protocols are reused and cited at the point of use, not re-derived —
[probe3d](https://arxiv.org/abs/2404.08476) (El Banani et al., CVPR 2024),
[Probing the Mid-level Vision Capabilities of Self-Supervised Learning](https://arxiv.org/abs/2411.17474)
(Chen, Marks & Cheng), and [vismatch](https://github.com/gmberton/vismatch) for
API philosophy.

[#1]: https://github.com/turhancan97/VisBench/issues/1
[#2]: https://github.com/turhancan97/VisBench/issues/2
[#4]: https://github.com/turhancan97/VisBench/issues/4
[#3]: https://github.com/turhancan97/VisBench/issues/3
[Unreleased]: https://github.com/turhancan97/VisBench/compare/v0.11.0...HEAD
[0.11.0]: https://github.com/turhancan97/VisBench/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/turhancan97/VisBench/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/turhancan97/VisBench/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/turhancan97/VisBench/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/turhancan97/VisBench/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/turhancan97/VisBench/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/turhancan97/VisBench/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/turhancan97/VisBench/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/turhancan97/VisBench/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/turhancan97/VisBench/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/turhancan97/VisBench/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/turhancan97/VisBench/releases/tag/v0.1.0
