# Engineering log — the v0.3 build, step by step

This is the **archive of closed build steps**, moved out of `CLAUDE.md` on
2026-08-20 because that file is loaded into every session's context and had
grown past the 150k-character limit at 203k. Nothing here is obsolete; it is
simply not needed by *every* session. The rules from these steps that still
constrain new code were lifted back into `CLAUDE.md`'s "decisions already paid
for" list, each one pointing here for the measurement behind it.

**Step labels resolve here.** When `CLAUDE.md`, `CHANGELOG.md` or a docstring
says "see 6c-2" or "6d-1's order-1 rule", the section is in this file.

**Read this when** you are about to touch the code a step built -- detection,
the leaderboard, the Hub artifact, the Taskonomy probes, the prefix cache --
and want to know what was already measured and rejected. Do not re-derive from
the code; several of these decisions look wrong until you read the numbers.

Sections are in step order, which is *not* the order they were written in.

New closed steps belong here, not in `CLAUDE.md`: that file carries the rule,
this one carries the derivation.

---

## v0.3 — fine-tuning + detection groundwork

### Step 6a — **done**. What it built, and what fine-tuning actually costs

Shipped: `BaseBackbone.unfreeze_last(n)` + `extract_features_trainable`,
`finetune_blocks`/`backbone_lr` on every dense probe and its CLI subcommand,
schema v6's `finetune` field, and the cache bypassed on that path.

Measured on VOC 2012 val, linear head, ten epochs, one V100 — the same command
each way, only `--finetune-blocks 2` added:

| backbone | run | mIoU | mIoU/image | pixel acc | wall clock |
| --- | --- | --- | --- | --- | --- |
| DINOv2-S/14 | frozen, cached | 0.7328 | 0.6841 | 0.9267 | 156 s / 126 s |
| DINOv2-S/14 | fine-tuned, 2 blocks | **0.7758** | 0.7527 | 0.9405 | 200 s |
| DINOv2-B/14 | frozen, cached | 0.7533 | 0.7161 | 0.9316 | 126 s |
| DINOv2-B/14 | fine-tuned, 2 blocks | **0.7992** | 0.7813 | 0.9465 | 279 s |

Both frozen runs reproduce v0.2's numbers exactly — 0.732 and 0.753 — which is
what makes +4.3 and +4.6 mIoU worth quoting rather than a difference between
two environments. **The gain holds at both scales; so does the cost.**

**A single timing on a shared machine is not a measurement, and this cost a
wrong conclusion.** 6a originally recorded 252 s frozen against 238 s
fine-tuned on ViT-S, and concluded from that one pair that fine-tuning was
*free* — that the frozen path was I/O-bound on streaming 1.3 GB per epoch and
6b's premise was therefore dead. Re-running the identical commands reproduced
every metric to four decimals and none of the timings: frozen came in at 156 s,
then 126 s on an immediate repeat, and fine-tuned at 200 s. The original pair
was inflated by machine contention. **Fine-tuning is slower, at both scales**,
by 1.3-1.6x on ViT-S and 2.2x on ViT-B. Repeat a timing before drawing a
design conclusion from it; the metrics are deterministic and the clock is not.

**What replaced it is the more useful finding: the frozen path costs ~126 s
regardless of backbone width.** ViT-S and ViT-B land within 0.2 s of each other
despite ViT-B streaming 2.3 GB against ViT-S's 1.3 GB. That cost is dominated by
per-file overhead across 2,913 files, not by bytes moved. Fine-tuning, by
contrast, tracks compute: 200 s to 279 s for the same 2 unfrozen blocks. So
**neither the FLOP argument nor the I/O argument was the right model** — reads
are per-file, and only the recompute scales.

That is what sizes 6b. Caching the frozen prefix removes the frozen blocks'
forward compute and adds back a per-file read of the same shape the frozen path
already pays. The margin therefore **grows with backbone size and shrinks as
more blocks are unfrozen**, and is worth roughly 279 s → the 126 s floor plus
two blocks' compute at ViT-B. Measure it against these numbers rather than
re-deriving them.

### Step 6a — the decisions, settled before it was built; do not re-derive them

Fine-tuning is opt-in and off by default. What makes it hard is not the
unfreezing, it is that **three separate things assume the backbone is frozen**:

- `BaseBackbone._finalize()` calls `requires_grad_(False)` and `eval()`, and
  `extract_features` is `@torch.no_grad()`. Gradients cannot flow at all.
- **The cache keys on `backbone.cache_key()`**, i.e. on the weights. Fine-tuned
  weights differ at every optimiser step, so a cached feature is stale the
  instant it is written — and written under the *frozen* key it is **poison**,
  served to every later frozen run of that backbone with nothing to say why the
  number moved.
- `run()` extracts before it fits, and `DenseTrainingTask` optimises
  `self.head.parameters()` alone. Fine-tuning needs *images* in the loop.

**Decided, 2026-07-29, after reading all three:**

- **The fine-tuning path does not touch the cache. Not "keyed differently" —
  untouched.** Keying on a per-step weights digest would grow the cache without
  bound and still never hit.
- **The backbone stays in `eval()` even when unfrozen**; only `requires_grad`
  flips. Unfreezing a ResNet stage in train mode starts BatchNorm updating its
  running statistics and activates dropout, which silently makes the v0.2
  numbers incomparable. This is also standard for small-batch fine-tuning.
- **A no-op unfreeze must raise.** If N resolves to zero trainable parameters,
  the run trains exactly like a frozen probe and reports it as fine-tuned —
  the QuickGELU failure in a new place. Guard it in the **fast** suite.
- **Two learning rates.** The backbone needs roughly 10–100x below the head's
  5e-4, or the pretrained features are destroyed in the first epoch. Both
  recorded.
- **Schema v6 adds one `finetune` field** (`blocks`, `backbone_lr`,
  `trainable_params`), `None` for every run to date. A leaderboard mixing
  frozen and fine-tuned numbers under one task name is meaningless, and this is
  what stops it. `protocol` is unchanged — same loss, same metric, different
  trainable set.
- **DINOv2 only.** CLIP, timm and `CustomBackbone` raise "not supported yet".
- **Proved on VOC semantic segmentation**, because measured frozen baselines
  exist there (DINOv2-S 0.732, DINOv2-B 0.753 mIoU) and there is headroom.
  Imagenette classification was rejected as the first proof: at 0.9939 top-1 it
  is saturated and could not show an effect either way.

**Why the cache is bypassed rather than made to work in 6a.** The right answer
is 6b — the frozen prefix below the cut *is* deterministic given the image, so
it can be cached and only the unfrozen suffix recomputed. 6a deferred it
because all three families tap layers through a whole-model API
(`get_intermediate_layers`, `forward_intermediates`) and none obviously resumes
a forward pass from block k. So 6b was lifted out of a working 6a and measured
against it, the same way `DenseTrainingTask` was lifted out of a working
`DepthTask`.

### Step 6b — **done**. It works, it saves 21%, and it found the next bottleneck

**The blocker was smaller than it looked.** DINOv2's
`_get_intermediate_layers_not_chunked` is a plain loop over `model.blocks` from
`prepare_tokens_with_masks`, so the cut is clean: prefix = patch embed + pos
encoding + blocks `[0:k]`, suffix = blocks `[k:]` + `norm` + the
`1 + num_register_tokens` split. Verified **bit-identical** to
`get_intermediate_layers` (max abs diff 0.0), which is what makes this a
caching change and not a numerical one.

Measured on VOC 2012 val, DINOv2-B/14, 2 unfrozen blocks, ten epochs, all in
one session so the comparison is internal to it:

| run | wall clock | mIoU |
| --- | --- | --- |
| recompute (6a's path) | 320.6 s, 368.9 s | 0.7992 |
| prefix cache, cold | 379.5 s | 0.7992 |
| prefix cache, warm | **268.2 s, 276.4 s** | 0.7992 |

**Every run reports 0.7992**, identical to 6a's. That is the whole correctness
claim, and a fast test asserts the same equality on fake backbones so CI keeps
it. Cost: 2,913 prefix entries, 2.30 GB — one per image, ~772 KB at ViT-B.

**The saving is ~21%, not the ~2x the 126 s floor suggested, and the profile
says why.** Per ten epochs over 1,464 training images:

| component | cost | on a prefix hit |
| --- | --- | --- |
| image decode | **128.3 s** | still paid |
| content hashing for the key | 8.5 s | still paid |
| preprocess (resize, normalise) | 24.3 s | skipped |

**The ~126 s frozen floor was never reachable from the fine-tuning path.** It
was measured on the frozen path, which streams precomputed features and never
opens an image. A fine-tuning loop is image-driven by construction, so it pays
a 128 s decode that no amount of prefix caching removes. The frozen blocks'
forward compute was simply not the largest remaining term — which is the same
shape of error as 6a's I/O conclusion, caught this time by profiling before
concluding.

**So the next optimisation is not more caching.** It is keying on *dataset
identity* (path + mtime) rather than image content, so a hit never decodes at
all — `FeatureCache` already has that machinery in its identity memo. That
needs the dataset to yield identities instead of images when a prefix is
available, which reaches into the data layer; do not start it without
measuring, and note that it would make the cache miss on an edited file under
an unchanged name, which content hashing catches today.

### Step 6b — decisions, settled while building it

- **A separate `PrefixCache`, not a mode on `FeatureCache`.** The entries are
  not interchangeable and a mix-up is silent: a prefix resumed as features, or
  features handed to a resumption, both produce plausible numbers. Three
  independent things keep them apart — different classes, different
  directories (`_prefix` under the same root), and a key namespace where
  `prefix@10` cannot collide with any layer index.
- **`FeatureCache.clear()` no longer `rmtree`s its own root**, because the
  prefix store nests inside it: a whole-root delete would remove entries it
  then failed to count, telling the caller it had removed fewer things than it
  did. `stats()` excludes them for the same reason.
- **Layers below the cut are refused, not approximated.** A single block-k
  activation cannot serve a shallower depth, so a DPT run over `[2, 5, 8, 11]`
  with two blocks unfrozen declines the prefix cache and recomputes.
  `can_use_prefix_cache()` is the question a caller asks *before* choosing a
  path; `extract_features_from_prefix` raises if asked anyway.
- **The record says whether the cache was actually used**, not whether one was
  offered (`finetune.prefix_cache`). A declined run that claimed the saving
  would misattribute its own cost.
- **`--no-prefix-cache` takes 6a's code path**, rather than using a cache that
  always misses. Otherwise the flag would measure the resumption path with the
  cache disabled, which is not the thing it exists to compare against.
- **A chunked DINOv2 is refused by name.** `block_chunks > 0` makes
  `model.blocks` a sequence of *chunks*, so every index into it means something
  else — `unfreeze_last` and the cut would both slice at the wrong depth and
  still run. The hub entrypoints pass `block_chunks=0`, so this is unreachable
  today, which is exactly why it is a guard: the day it becomes reachable there
  is no symptom.

### Step 6c — detection groundwork. Scope decided 2026-07-29, before any code

Expect this to take longer than any other single addition — it's the hardest
task to do cheaply on limited compute. **Build order is dataset and metric
first, head second**, decided rather than discovered:

1. `visbench/data/detection.py` — a box dataset and `load_boxes`.
2. `visbench/metrics/detection.py` — `average_precision`, mAP@50, mAP@50:95.
3. `visbench/tasks/high_level/detection.py` — the head, last, against a metric
   that is already trusted.

**Why this order, and it is not a preference.** Every dense task in this
codebase pays for target geometry, and boxes are strictly worse than masks
there: a box must survive the same resize and crop as its image, and unlike a
depth map *it does not resample* — it transforms, so the scale and offset are
applied by hand rather than by the loader. That is the correspondence
misalignment bug (recall@1px = 0.003) with a new coordinate convention, and a
head trained against silently shifted boxes still trains and merely scores
badly. Getting a number early from a head on fixtures would prove nothing,
because fake backbones cannot show it.

**The internal box convention is `xyxy` in absolute pixels, 0-indexed.**
Decided 2026-07-29, before the dataset was written, because both halves of it
are silent when wrong. Convert at the loader boundary and nowhere else, and
assert the choice in a fast test — a swapped pair loads, trains and scores.

- **`xyxy`**, matching what VOC's XML already stores, so the common path does no
  conversion and cannot get one backwards.
- **0-indexed**, so subtract 1 from VOC's `xmin`/`ymin`/`xmax`/`ymax` on read.
  VOC is 1-indexed (verified below) and every other coordinate in this codebase
  is not; keeping VOC's origin would make boxes the one array indexed
  differently from the tensor they describe.
- **Absolute pixels**, not normalised `[0, 1]`.

The hazard absolute pixels carry, and the reason it is written down rather
than assumed: **an absolute box is meaningless without the resolution it refers
to.** A normalised box survives the resize and crop untouched; an absolute one
must be transformed alongside its image, and if it is not, nothing raises — the
boxes simply describe the original 500x375 frame while the tensor is 224x224.
So `DetectionFolderDataset` must return boxes in **post-transform** pixel
space, matching the image tensor it returns in the same item, and a test must
assert exactly that on a non-square image where a missed rescale is visible.
This is the same rule dense targets already follow ("image and target must
survive the *same* resize and crop"), applied to a target that transforms
rather than resamples.

Prove it on **VOC2012 Detection**. Verified present on this machine
2026-07-29, no download needed:

```text
/shared/sets/datasets/pascal_voc_2021/VOCdevkit/VOC2012/
  Annotations/        17,125 XML          ImageSets/Main/  train 5,717 / val 5,823
  JPEGImages/         17,125 JPEG         ImageSets/Segmentation/  the 1,464 / 1,449 5h used
```

Note the detection split is **~4x the segmentation split** — `ImageSets/Main`
is not `ImageSets/Segmentation`, and the 1,464/1,449 figures quoted throughout
this file are the segmentation ones. A ten-epoch schedule sized on those is not
sized on these.

Three properties of the VOC XML, all verified, all silent when mishandled:

- **Boxes are `xyxy` and 1-indexed.** Minimum `xmin`/`ymin` across the whole
  set is 1, not 0 — which is why the loader subtracts 1, per the convention
  decided above. A one-pixel shift moves mAP slightly and looks like a weak
  backbone rather than an off-by-one.
- **4,462 objects are flagged `<difficult>1</difficult>`.** The VOC protocol
  *excludes* these from evaluation. Counting them as false negatives depresses
  mAP against every published number, which is the failure the `protocol` field
  exists to prevent — so if they are kept, the record must not claim VOC's
  protocol.
- The XML carries `<size>` per image, which is what the box rescale needs, so
  the original dimensions never have to be re-read from the JPEG.

`DetectionTask` is currently a `NotImplementedError` stub with
`level`/`feature_mode`/`zero_shot` already declared — extend it, do not rewrite
it.

### Step 6c-1 — **done**. The dataset, and what it settled

`visbench/data/detection.py`: `DetectionFolderDataset`, `load_voc_boxes`,
`VOC_CLASSES`. 27 fast tests. Verified against the real split — 5,823 val
images, and **all 17,125 annotation files parse with zero failures**, yielding
40,138 objects of which 4,462 are `difficult`. That count matching the one
measured independently by `grep` is the cross-check that the parser reads what
the files say.

Decisions made while building it, so they are not re-opened:

- **Rescale by the *achieved* ratio, not the nominal one.** `_resized_size`
  rounds and applies a `max()` floor, so the width actually used is not exactly
  `image_size / min(w, h)` times the original. Using the nominal factor leaves a
  sub-pixel error that grows with box size and is invisible in any one image.
  The dataset divides the post-resize dimension by the original, per axis.
- **The image half is byte-identical to `DenseFolderDataset`'s crop**, and a
  test asserts it. The box transform is *derived* from that geometry, so if the
  two ever diverge the boxes shift and nothing raises.
- **A box outside the crop is dropped, not kept at zero area.** A centre crop
  genuinely removes objects, and scoring a detector against an object absent
  from its input measures nothing. Straddling boxes are clipped, since the
  visible part is the correct target. `boxes`, `labels` and `difficult` are
  indexed by one mask so they cannot drift, and `num_original` keeps "no
  objects" distinguishable from "all objects dropped" — an image with zero
  surviving boxes is legitimate and must not raise.
- **`load_voc_boxes` returns `difficult`; the dataset filters it.** Filtering in
  the loader would make VOC's exclusion invisible to the result record, which is
  the one thing `protocol` exists to prevent. `include_difficult=False` is the
  default and is recorded in `describe()`.
- **Corners are treated as continuous coordinates**, so width is `x2 - x1`. VOC's
  `xmax` is an inclusive *index*, so a literal reading gives `x2 - x1 + 1`. One
  pixel, chosen rather than inherited.
- Coordinates are parsed as float then rounded, because some VOC
  redistributions write `174.0` where `int()` would raise.
- `NotADirectoryError` for a missing directory, matching `DenseFolderDataset`
  rather than inventing a second convention. Found by a test that compared the
  two.

**Resolved in 6c-2**, and it went the way that note predicted: the metric takes
the `difficult` mask and ignores those objects, so a run headed for scoring
constructs `DetectionFolderDataset(include_difficult=True)`. 6c-1's
`include_difficult=False` default is right for *training targets* and is **not
sufficient for scoring** — see below for what the difference costs.

### Step 6c-2 — **done**. The metric, and what each convention costs

`visbench/metrics/detection.py`: `box_iou`, `average_precision`,
`detection_metrics`, `COCO_IOU_THRESHOLDS`. 29 fast tests.

**Cross-checked against a literal `VOCevaldet.m` transcription** over 3,060
randomly generated APs at three IoU thresholds: **zero mismatches, maximum
absolute difference 0.0.** That transcription is kept as a fast test rather than
run once, because the obvious future change here is vectorising the
per-detection loop, and the subtlety most likely to be lost is the one thing no
analytic test covers — see below.

Validated end to end against the real VOC val split, ground truth fed back as
predictions over 500 images (1,249 boxes, 115 difficult):

| predictions | mAP@50 | mAP@50:95 |
| --- | --- | --- |
| oracle (ground truth) | **1.0000** | **1.0000** |
| boxes jittered 3 px | 0.9224 | 0.6731 |
| half the objects dropped | 0.5270 | 0.5270 |
| nothing detected | 0.0000 | — |

An exact 1.0000 is the check that matters: any off-by-one in the matching, the
recall denominator or the interpolation would land near 1 without reaching it.

**The `difficult` decision, measured rather than argued.** Same oracle
predictions, scored two ways on the same 500 images:

| protocol | mAP@50 |
| --- | --- |
| VOC's rule — a detection matching a difficult object is **ignored** | **1.0000** |
| difficult objects dropped from the ground truth | 0.9567 |

**4.3 mAP points**, and the wrong one is *lower*, so it looks like a weaker
detector rather than a scoring bug. Only the first can claim VOC's protocol.

Decisions settled while building it:

- **AP is dataset-level, and this is the one place "per image, then averaged"
  does not apply.** Every other metric here scores each image and averages, so
  uneven coverage cannot reweight the split. AP cannot: it is the area under one
  curve built by ranking *every* detection in the split. A test constructs a case
  where the global answer is 2/3 and the per-image mean is 0.75, so the two
  cannot be confused. Semantic segmentation resolves the same tension by
  reporting both; here there is no defensible per-image version, so there is one
  number.
- **Matching follows `VOCevaldet.m` exactly, including its order of checks**: a
  detection is matched to the box it overlaps *most*, and only then is that box's
  state consulted — difficult first, then already-claimed. There is deliberately
  **no fallback to the second-best box**. A greedy variant that reassigned
  duplicates scores higher than the reference and stops being comparable, while
  passing every hand-computed test. That is what the transcription test exists
  to catch.
- **All-points interpolation** (VOC2010+, COCO), not VOC2007's 11-point
  sampling, which is systematically higher and must not share a table.
- **`map_50_95` is COCO-*style*, not a COCO number.** It averages COCO's ten
  thresholds but integrates all points at each, where COCO quantises recall to
  101 points. `map_50` is directly VOC-comparable; the docstring says so.
- **A class with no non-difficult objects scores `None`, not 0.** Recall has no
  denominator there, and scoring 0 would drag mAP down in proportion to how many
  categories a split happens to omit. `detection_metrics` excludes them and
  reports `classes_scored`, which is the actual mAP denominator and is **not
  always `num_classes`** — a caller comparing two runs should check it matches.
- **A class present but entirely missed scores 0.0**, which is distinct from
  `None` and must stay so.
- `box_iou` uses continuous corners (width `x2 - x1`), matching the dataset. The
  two disagreeing about box size would shift every IoU and therefore which
  detections match.

### Step 6c-3 — **done**. The head, and the decisions behind it

`visbench/heads/detection.py` (`DetectionHead`), `visbench/tasks/high_level/
detection.py` (`DetectionTask`), a `detection` CLI subcommand,
`examples/detect.py`, and `visbench/tasks/schedule.py`. 43 fast tests for the
probe plus 3 for the CLI row.

**Anchor-free and single-scale.** Two 1x1 convolutions over the patch grid at
its native stride: `num_classes` logits and 4 box distances. FCOS's
centre-inside-box assignment reduced to one level, sigmoid focal loss on
classification, GIoU on the positives' distances, then threshold → per-class
NMS → cap.

**Measured on VOC 2012 Detection**, 600 train / 600 val images from
`ImageSets/Main` at 224px, linear head (`hidden_dim=0`), ten epochs, one V100:

| backbone | map_50 | map_50_95 | classes_scored | dets/image | train_loss |
| --- | --- | --- | --- | --- | --- |
| DINOv2-S/14 | 0.2127 | 0.0722 | 20 of 20 | 84.6 | 1.2076 |
| DINOv2-B/14 | **0.2616** | **0.0930** | 20 of 20 | 88.5 | 1.1124 |

**Do not read the +4.9 as a pass criterion.** It is the same direction as
semantic segmentation and the opposite of mid-level similarity, and the
standing rule above ("bigger is not better on every task") is not suspended
because it happened to hold here. What the numbers *are* good for is a floor to
re-measure against if the head changes. The absolute level is low by design —
see the first bullet below — and the split is 600/600, not the full
5,717/5,823, so it is a proof that the probe runs end to end on real weights,
not a headline number.

Decisions settled while building it, so they are not re-opened:

- **The low absolute mAP is the design.** A single-scale head has no feature
  pyramid; small objects fall between cells and are unrecoverable. The number
  ranks representations, which is what VisBench is for. `protocol:
  "visbench_anchor_free_det"` — not `probe3d` (no detection task there) and not
  VOC's (the *metric* is VOC's, the head is not). **Do not "fix" a low number by
  adding an FPN** without deciding first that VisBench wants to measure necks.
- **It does not subclass `DenseTrainingTask`, and that was not close.** That
  base assumes a stackable `(B, C, H, W)` target and recovers a split metric by
  weighting per-image metrics by batch size. Detection has neither — a
  variable-length box list, and an AP that is a split-level ranking (6c-2). The
  *shared* part, probe3d's warmup/cosine schedule, was lifted into
  `tasks/schedule.py` and both now call it, the same move that produced
  `DenseTrainingTask` from a working `DepthTask`. So a detection number and a
  segmentation number differ in head and loss, not in optimisation.
- **Focal loss, not BCE, and the prior bias init is not optional.** A dense
  anchor-free grid is overwhelmingly background; plain BCE converges to
  predicting nothing while its loss falls, which reads as a dead representation.
  The classification bias starts at `-log((1-0.01)/0.01)` so the schedule is not
  spent discovering that background is common.
- **GIoU, not IoU loss.** Plain IoU loss is flat at 1.0 for every disjoint pair,
  so it has **no gradient in the state every box starts in**. A test asserts
  GIoU keeps rising as boxes separate; that is the whole reason for the choice.
- **`exp(raw) * stride` for the distances, with the exponent clamped at 8.**
  Unclamped `exp` can reach `inf` in one bad step, and every later loss is then
  `nan` while the run reports 0.0 mAP as though the features were useless.
- **The scored split keeps `difficult`; the training split drops them.** Not an
  inconsistency: 6c-2 measured VOC's ignore rule at 4.3 mAP above dropping them
  from the ground truth, so scoring needs them present. Training against an
  object the annotators called unreasonable is a separate question. The task
  drops them in assignment regardless of what the dataset kept, so one
  `include_difficult=True` dataset can serve both halves.
- **`--image-size` reaches the dataset and the probe from one flag**, in the CLI
  and the example. Boxes are absolute post-transform pixels, so two values put
  every cell centre at the wrong coordinate — trains, scores badly, reads as a
  weak backbone. The probe range-checks its targets but that catches only the
  direction where boxes exceed the frame; sharing the flag catches both.
- **No schema bump.** `task_params` and `dataset_params` are open dicts, so the
  protocol, the three decoding settings and `include_difficult` all land in a
  record without touching `SCHEMA_VERSION`. That is what those two fields were
  added for (v3 and v5); resist adding a column.
- **Fine-tuning is not wired up here.** `finetune_blocks` stays 0 — the
  trainable-backbone path lives on `DenseTrainingTask` and detection does not
  inherit it. A test asserts `finetune()` is `None`, so the probe cannot claim a
  record it never produced.

**Observed while proving it: `DetectionFolderDataset` construction was slow on
a network mount, and it was not the images.** Measured and fixed afterwards —
see the next section. 6c-3's guess at the cause was right and its guess at the
*fix* was wrong, which is why it was recorded as a lead rather than acted on.

### Step 6d-0 — the dataset listing, measured then fixed

**`Path.is_file()` was the whole cost, and the fix is `os.scandir`.** Not the
constructor change 6c-3 predicted. `Path.is_file()` cannot reuse what `readdir`
already returned, so it is one stat round trip per entry; `scandir` answers the
same question from the file type the listing carried anyway.

Measured over NFSv4.2, one strategy per fresh process on cold directories:

| listing | 2,913 files, cold |
| --- | --- |
| `iterdir()` + `Path.is_file()` | **5.69 s** |
| `os.scandir()` + `entry.is_file()` | **0.05 s** |

And on VOC itself, first call in a fresh process: `_index_directory` over the
17,125-file `Annotations` went from **76 s** to **0.16 s**; the whole 600-stem
constructor from 5.86 s to 0.32 s, or 0.53 s to 0.22 s once warm.

- **The cost is invisible to every result record.** It is paid before `run()`
  starts timing, which is why a `duration_seconds` of 124 s sat inside a
  twenty-minute wall clock and nobody looked. When something feels slow and the
  record disagrees, the gap is *outside* the timer.
- **One helper, `visbench.data.base.list_files`, and three call sites.** The
  same `iterdir` + `is_file` pattern was in `DetectionFolderDataset`,
  `DenseFolderDataset` and `ImageFolderDataset` — i.e. in every folder dataset
  the library has, so VOC segmentation and Imagenette paid it too, not just
  detection.
- **6c-1's constructor is unchanged, and deliberately so.** Resolving only the
  named stems would have worked, but it costs an API change and a rule that a
  stem implies its extension — and once the stat is gone, indexing all 17,125
  entries costs 0.16 s. Measuring first is what kept the cheaper fix in view.
  This is the second time in v0.3 that profiling overturned the obvious
  optimisation; see 6b's closing note for the first.
- **This changes timing only.** `list_files` returns the same paths in the same
  sorted order, with directories and broken symlinks still excluded and a real
  symlink still followed — pinned by `tests/data/test_list_files.py`, including
  a test asserting equality with the `iterdir` expression it replaced.

**Two tests carry the correctness claim, both fast.** `_decode` applied to a
hand-built perfect head output must reproduce the exact box — 6c-2's oracle
check in a new place, because any off-by-one in the cell centres, the stride or
the corner arithmetic lands *near* the box without reaching it. And a probe
trained on features that literally encode the answer must reach **1.0 mAP**:
assignment, both losses, the exp/stride decoding and VOC's AP all have to
describe the same box for that to be reachable at all.

**Known deferred**: keying the prefix cache on dataset identity (path + mtime)
rather than image content, which 6b's profile identified as the 128.3 s
remaining cost. Explicitly not part of 6c — see 6b's closing note for why it
reaches into the data layer and what it would break.

### Step 6d-1 — **done**. Edge detection, and why it is not on BSDS500

`visbench/data/taskonomy.py` (`TaskonomyDataset`, `load_taskonomy_split`,
`TASKONOMY_DOMAINS`), `load_edge_map` in `data/dense.py`, `edge_metrics` in
`metrics/dense.py`, `visbench/tasks/low_level/edge.py` (`EdgeTask`), an `edge`
CLI subcommand and `examples/edges.py`. 35 fast tests. **`low_level/` was a
documented placeholder from v0.1 until this**, so all three levels of the
taxonomy now have entries.

Measured on Taskonomy `edge_texture`, 600 train / 600 val frames at 224px,
linear head, ten epochs, one V100:

| backbone | `edge_correlation` | `rmse` | `mae` | `train_loss` |
| --- | --- | --- | --- | --- |
| DINOv2-S/14 | **0.4558** | 0.9226 | 0.5028 | 0.5721 |
| DINOv2-B/14 | 0.4481 | 0.9265 | 0.4972 | 0.5631 |

DINOv2-S wins by 0.008 — same ordering as mid-level similarity, opposite to
segmentation and detection. **Do not read that as a pass criterion**; the
standing rule that bigger is not better on every task is not suspended because
it happened to point the way the taxonomy would predict, and 0.008 is small.
The value of these numbers is as a floor to re-measure against.

**BSDS500 is not on this machine, and that is not why it was skipped.** The
`bsds300` directory under `/shared/sets/datasets/` is the MAF *density
estimation* benchmark — `bsds300.hdf5` beside `gas` and `hepmass`, 8x8 patches
flattened to vectors, no images and no annotations. BSDS500 itself downloads in
67.5 MB (only the `www2.eecs.berkeley.edu` host still serves it; the canonical
`www.` URL 404s). The real cost is the **protocol**: ODS/OIS/AP matches
predicted edge pixels to several annotators' by bipartite correspondence after
non-maximum suppression, swept over ~99 thresholds, canonically via Goldberg's
CSA solver in C. That is a step of its own, it would need `scipy` promoted from
a transitive dependency to a declared one for the `.mat` ground truth, and the
rule about borrowing protocols exactly applies in full. Taskonomy was on disk,
dense, and describable honestly.

### Step 6d-1 — decisions, settled while building it

- **`edge_texture` and not the other five Taskonomy domains on disk.** It is
  computed from the RGB frame, so every pixel is a real measurement.
  `depth_zbuffer`, `edge_occlusion`, `normal`, `principal_curvature` and
  `reshading` are derived from the 3D reconstruction and have invalid regions in
  `mask_valid/` (whose files are confusingly named `_domain_depth_zbuffer.png`).
  `TaskonomyDataset` **refuses them by name** rather than silently scoring
  against reprojection holes, which would depress every backbone equally.
- **Zero is a real reading, and this is the third validity convention here.**
  Depth: 0 means no ground truth. Label maps: 0 is a real class, negative is
  unlabelled. Edges: **0 means "no edge"** and there is no invalid value at all,
  so nothing is masked. Reusing depth's rule would have scored the probe only
  where an edge already is — the easy half — and it would not have raised.
- **The metric is per-image Pearson correlation, not RMSE.** Edge magnitude is
  concentrated near zero, so a probe predicting the split's mean everywhere gets
  a *small* RMSE while having learned nothing. Correlation is scale- and
  offset-invariant, so it asks only *where* the edges are and scores that probe
  0 by construction. RMSE and MAE ride along because correlation is blind to the
  complementary failure (right shape, wrong magnitude). A fast test pins the
  constant-predictor case.
- **The activation is the identity, and both ways of imposing non-negativity
  destroy the probe.** Measured on features that encode the answer, ceiling 1.0:
  ReLU **0.0000** (dead — prediction variance exactly 0), softplus **-0.9851**
  (collapsed to a constant, whose residual noise anti-correlates), identity
  **0.9997**. Softplus is the more dangerous of the two because it looks
  reasonable: to emit 0.065 it needs a raw value near -2.7, where its own
  gradient is `sigmoid(-2.7)` ~ 0.06, so it attenuates ~16x in exactly the band
  this target occupies. Non-negativity is **learned from the targets**. A test
  asserts `_activate` is the identity, because reinstating a rectifier is the
  natural tidy-up and costs only a mediocre score.
- **`target_scale` is 1000, not the container's 65535, and this is optimisation
  not honesty.** L1's gradient is `sign(pred - target)`, magnitude 1 regardless
  of target size, so the step size does not shrink to match a small target and
  the optimiser oscillates. Measured, DINOv2-S, probe3d's schedule:

  | `target_scale` | frame mean | `edge_correlation` |
  | --- | --- | --- |
  | 65535 | 0.011 | 0.047 |
  | 6553.5 | 0.109 | 0.285 |
  | **1000** | **0.717** | **0.456** |
  | 100 | 7.165 | 0.467 |

  It plateaus once the target is order 1, so 1000 is the knee. **Scale the
  target, not the learning rate**: the scale is arbitrary (65535 is just uint16
  max) and the headline metric is invariant to it, while the learning rate is
  what keeps this number under the same training budget as every other dense
  probe. Raising the rate is also worse — lr 5e-3 reaches 0.348, lr 5e-2
  collapses to 0.066.
- **`TaskonomyDataset` subclasses `DenseFolderDataset` for its geometry**, via a
  new `_init_geometry()` lifted out of that constructor. Taskonomy is nested per
  building and its two halves never share a filename, so the *indexing* cannot
  be reused — but the resize, crop and nearest-neighbour target resampling must
  be identical, and sharing the code is a stronger guarantee than 6c-1's
  "byte-identical, and a test asserts it".
- **The split lists are not stat-ed at construction.** They name up to 272k
  frames; confirming them is exactly the per-file stat 6d-0 removed. A frame
  named but absent raises when read, naming the stem — late, but not silent.
  `fingerprint()` is overridden for the same reason: a Taskonomy partition is a
  fixed published release, so `(partition, split, building, point, view)`
  identifies the bytes without stat-ing 2N files.
- **`--limit` reaches the constructor as `max_images`**, not `subset()`
  afterwards, since building 272k paths to discard all but 600 is waste. Safe as
  a prefix here, unlike a labelled image folder, because the rows are already
  interleaved across buildings.
- **The splits are disjoint by building** — 25 / 4 / 5, verified — so a val
  number comes from rooms the probe never saw. Two buildings on disk
  (`taskonomy`, `wiconisco`) appear in no split.
- **`_dense_flags` was split in two.** The head and schedule half is now
  `_head_schedule_flags`, because Taskonomy is indexed from a split list and
  `--image-dir` / `--target-dir` / `--stems` would appear in `visbench run edge
  --help` promising something they cannot do.

**A parameter that is recorded but does nothing is the QuickGELU failure again.**
`target_scale` was accepted, returned by `describe()` and folded into the
fingerprint while having no effect: `DenseFolderDataset.target()` applies it only
on its *default* depth path, and `TaskonomyDataset` always passes a custom
`target_loader`. Nothing raised; the sweep above simply returned four identical
rows. It is now bound into the loader with `functools.partial`, and a fast test
asserts two scales give two different targets. **When a dataset takes a
numeric parameter, test that changing it changes the data.**

### Step 6d-2 — **done**. `mask_valid`, two more probes, and one that had to be fixed

`load_valid_mask` + `_DOMAIN_SPECS` in `data/taskonomy.py`, a
`_load_raw_target` hook on `DenseFolderDataset`, `magnitude_metrics` in
`metrics/dense.py`, `tasks/magnitude_base.py` (`DenseMagnitudeTask`),
`tasks/low_level/keypoints.py`, `tasks/mid_level/occlusion_edge.py`, two CLI
rows and two examples. Fast suite 1176 → 1216.

Four of the six previously-refused Taskonomy domains are supported:
`depth_zbuffer`, `normal`, `edge_occlusion`, `keypoints3d`.
`principal_curvature` and `reshading` are still refused, but **no longer for
want of a mask** — each is blocked on a task decision, and the error says which.

Measured on Taskonomy tiny, 600 train / 600 val at 224px, linear head, ten
epochs, one V100. The two new probes:

| probe | level | DINOv2-S/14 | DINOv2-B/14 |
| --- | --- | --- | --- |
| `keypoints2d` | low | **0.2356** | 0.2248 |
| `occlusion_edge` | mid | 0.2924 | **0.3167** |

And the two existing probes, on domains they could not read before:

| probe | domain | DINOv2-S/14 | DINOv2-B/14 |
| --- | --- | --- | --- |
| `depth` | `depth_zbuffer` | d1 0.5832, RMSE 0.7947 m | **d1 0.5986**, RMSE 0.7876 m |
| `surface_normal` | `normal` | **mean 26.66°**, d1 0.2727 | mean 27.37°, **d1 0.2787** |

Note the normals row **disagrees with itself** — DINOv2-S wins on mean angular
error and DINOv2-B on the 11.25° threshold — which is a useful reminder that
"which backbone won" is not always a well-formed question. Do not quote one of
those two and drop the other.

### Step 6d-2 — the findings, in the order they overturned something

- **Two of the five "blocked" domains were never blocked.** `depth_zbuffer`'s
  invalid region is *exactly* `depth == 65535` and `normal`'s is exactly what
  `load_normal_map`'s length threshold already zeroes — verified pixel for
  pixel against `mask_valid/` on 150 frames across 10 buildings. So the depth
  and surface-normal probes reached Taskonomy with **no change to either**, and
  the conventions they have carried since v0.2 turned out to be Taskonomy's
  conventions too. The mask file is still read for both, because "exact on 150
  frames" is evidence rather than a guarantee and an `or` of two agreeing tests
  costs nothing.
- **`keypoints3d` was silently unmasked in shipped v0.4.0.** It was in
  `TASKONOMY_DOMAINS` and *absent* from `_NEEDS_VALID_MASK`, so it constructed
  and read like an image-derived target while actually coming from the
  reconstruction. Nothing raised. This is the shape of bug a per-domain table
  prevents and a hand-maintained exclusion set invites: the set has to be
  right about every member, and being wrong about one is invisible.
- **The magnitude protocol does not transfer on tail weight alone, and this
  nearly shipped a probe that measures nothing.** L1 was chosen for these
  probes so a handful of strong pixels cannot dominate the loss; Pearson
  correlation is *dominated* by those same pixels. Share of total target mass
  in the strongest 1% of pixels: `edge_texture` 0.10, `keypoints2d` 0.11,
  **`edge_occlusion` 0.46**. At 0.46 the two pull apart and the probe scored
  0.088 — flat under four target scales spanning 30x, four times the training
  budget, and a ten times higher learning rate, with an S-versus-B gap of
  0.0035. **A probe flat under every hyperparameter is not underfitting.**
  `log1p` on the target brings the tail to 0.09, the score to 0.29/0.32 and the
  gap to 0.024. So `edge_occlusion` loads in log space and **nothing else
  does**: the other tails are mild, `edge_texture`'s published number is a
  linear-target number, and a log-space correlation is not the same
  measurement. `dataset_params` records `target_transform`.
- **The ranking check is what caught it, not the absolute number.** 0.088 looks
  like "a hard task", which detection's 0.21 mAP already established as
  acceptable by design. What is *not* acceptable is failing to separate two
  backbones, and that is the question to ask of a new probe whose score comes
  out low. Detection's 0.21 ranks; the linear occlusion probe did not.
- **`NaN` is the fourth validity convention, and it is deliberately loud.** A
  magnitude map has no spare in-band value — 0 means "no edge here" — so
  validity for the reconstruction-derived magnitude domains travels out of
  band. `NaN` makes an unmasked loss `NaN` on the first step, where a
  fabricated 0 would train quietly and merely score badly. Both
  `magnitude_metrics` and `DenseMagnitudeTask._loss` mask on `isfinite`, and a
  test asserts the masked metric equals scoring the unmasked crop, so the two
  cannot drift.
- **Masking happens before the geometry, in `_load_raw_target`.** The marker
  survives nearest-neighbour resampling unchanged (a 0 stays 0, a `NaN` stays
  `NaN`), whereas masking afterwards would need the mask resized in lockstep —
  a second geometry to keep in agreement, which is the failure
  `DenseFolderDataset` exists to prevent. Overriding that hook rather than
  `target()` keeps the resize, crop and `max_target` shared.
- **`target_scale` is per domain now, and one of them is not free.**
  `depth_zbuffer` is fixed at 512 and *raises* if changed: that divisor is what
  puts the target in metres, and `depth_metrics` reports RMSE in whatever unit
  it is handed, so rescaling it would quietly change what the number means. The
  other scales are arbitrary and the correlation metric is invariant to them.
- **`--domain` is restricted per probe.** In v0.4.0 `visbench run edge --domain
  keypoints2d` loaded, trained and recorded a keypoint number as
  `visbench_edge_regression`. The flag survives with `choices` of exactly what
  the probe's protocol describes, so a probe that grows a second honest domain
  has somewhere to put it.
- **Every mask file is named `..._domain_depth_zbuffer.png`** whatever it masks.
  Taskonomy derived one mask per frame from the depth render and never renamed
  it, so mask paths are built from the `mask_valid` directory with that suffix
  hard-coded rather than from the requested domain. Reading one as depth would
  give a map of 0 and 255 that loads, trains and scores.
### Step 6e-1 — **done**. The comparability rules, and two things that surprised it

`visbench/results/leaderboard.py`, 41 fast tests. Pure functions over records:
no rendering, no I/O beyond `writer.py`, no network. A number that should not
have gone in a table is wrong long before anyone formats it.

**Two prerequisites for this whole track were discovered by looking, not
assumed, and both change the plan:**

- **There is no record corpus.** `results/*.jsonl` is gitignored, so nothing is
  committed, and what is on disk locally is **16 records covering 2 of the 12
  probes** — leftovers from 6a/6b's timing work. Every published number was
  produced ad hoc and hand-copied; most of the records behind them are gone.
  That is what 6e-2 is for, and it is the expensive step.
- **Probe heads cannot be serialised at all.** No `save`, `load`, `state_dict`
  or `torch.save` anywhere in `visbench/tasks/` or `visbench/heads/`. "HF Hub
  probe sharing" means distributing trained head weights, so 6e-4 is a
  prerequisite for 6e-5 rather than part of it. **The Hub work and the
  leaderboard work are two projects** that were bundled under one roadmap
  bullet; they share almost no code.

Validated against the 16 real records: it reproduces VOC frozen 0.7328/0.7533,
fine-tuned 0.7758/0.7992 and Taskonomy edge 0.4558/0.4481 exactly, and splits
them into four mutually unrankable groups.

Decisions settled while building it, so they are not re-opened:

- **Directions are a listed table, never inferred from the name.** `mean` and
  `median` are surface-normal *angular error in degrees*, so lower is better,
  and nothing about either word says so. A heuristic reading "mean" as a score
  ranks that board upside down, and the output reads as a finding rather than a
  bug. An unrecorded metric therefore **raises** instead of defaulting.
- **Four things are refused rather than handled.** Incomparable records; a
  metric missing from any one record (ranking the rest presents a partial
  comparison as a complete one); a `classes_scored` mismatch (mAP over 18
  classes and over 20 are averages of different quantities — that field is the
  real denominator and is not always `num_classes`); and a `ceiling_` context
  metric (correspondence's ceiling describes the split, so ranking on it ranks
  the data).
- **`trainable_params` is excluded from the key, everything else is in.** It
  differs between ViT-S and ViT-B for the *same* `blocks` setting, so including
  it would make the one comparison fine-tuning exists to support look
  incomparable. `task_params` and `dataset_params` are otherwise included
  wholesale — conservative on purpose, because enumerating "the settings that
  matter" means editing this module every time a task grows one. `ignore=` is
  the escape hatch, and every name passed to it is a claim.
- **A task can disagree with itself, so `ranking_disagreements` is not
  optional.** Taskonomy normals: DINOv2-S wins on mean angular error, DINOv2-B
  on the 11.25° threshold. The real corpus produced a second case unprompted —
  on Taskonomy edges, `edge_correlation` ranks S first and `mae` ranks B first.
  A renderer that picks a headline metric silently will manufacture a result;
  an empty dict from this function is a real answer, not an absence.
- **`describe()` is lossy and `short_id()` exists because of it.** The corpus
  holds two `edge` groups identical in task, dataset, split, protocol and
  frozen-ness, differing only in `target_scale` (65535 against 1000, from
  6d-1's sweep, scoring 0.047 and 0.456). Described alone they read as one
  group listed twice. A digest rather than a diff, because which field differs
  depends on which two keys you hold.

### Step 6e-2 — **done**. The corpus, and the two probes it found ranking nothing

`results/corpus/visbench.jsonl` — **26 records, schema v6, all frozen, twelve
comparability groups and all twelve hold both backbones.** Produced by
`scripts/build_corpus.sh` (one function per probe, flags in one place),
`slurm/corpus.sbatch` (a 24-task array, one task per probe/backbone) and
`scripts/merge_corpus.sh` (idempotent rebuild from parts, plus validation).
This is the first set of VisBench numbers that exists as records anyone can
re-rank rather than as a hand-copied markdown table.

**The corpus is tracked, deliberately.** `results/*.jsonl` stays ignored for
ad-hoc runs; `results/corpus/` is negated in `.gitignore` because a benchmark
whose records nobody else can see is not a benchmark.

**It reproduces every published number to four decimals** — VOC segmentation
0.7328/0.7533, classification 0.9939, similarity 0.8701/0.8580, edge
0.4558/0.4481, keypoints2d 0.2356/0.2248, occlusion edge 0.2924/0.3167,
correspondence recall@1p 0.7834 against ceiling 0.9509 — **with one exception**,
below.

- **`retrieval` and `correspondence` ranked nothing, and did so silently.**
  Every metric they emit is parametrised — `recall@1`, `recall@10`, `auc@0.5p` —
  and none was in `METRIC_DIRECTIONS`. `shared_metrics` skips a name it cannot
  direct, so two of twelve probes produced an **empty leaderboard section rather
  than an error**. Fixed with `PARAMETRISED_METRIC_DIRECTIONS`, keyed on the
  stem before `@`. Still a listed table, not the name heuristic this module
  refuses elsewhere: those names are *generated* by `f"recall@{k}"` and
  `f"auc@{format_threshold(...)}"`, so the stem **is** the metric identity and
  the suffix is only which setting of it. An unlisted stem still raises. **Only
  a real corpus could have found this** — every fixture used unparametrised
  names.
- **`detections_per_image` and `num_matches` are diagnostics now.** Both say how
  much a probe *emitted*, not how good it was: a head that fires everywhere
  scores higher on them and worse on mAP.
- **Detection does not reproduce, and it is recorded as unverifiable rather than
  contradicted.** 0.2302/0.2882 map_50 against 6c-3's 0.2127/0.2616. Every
  recorded field matches what this file documents for that run — `hidden_dim` 0,
  ten epochs, 224px, 600 images, `classes_scored` 20 both sides — so the
  difference is **not in any field a record carries**, and the original command
  was never committed so it cannot be diffed. The ordering (B > S) is unchanged
  and the corpus number is the reproducible one. Do not "correct" the corpus
  toward the published pair. **Partly explained as of 2026-08-13** — detection
  on DINOv2 is non-deterministic on GPU at roughly 1e-3 in `map_50` (on CLIP it
  is bit-exact), see the bullet in
  "decisions already paid for". That accounts for a fourth-decimal wobble but
  **not** for the 0.0175 gap against 6c-3, which is an order of magnitude
  larger and remains unexplained; the difference there is still in something no
  record carries.
- **`edge` disagrees with itself three ways, not two.** 6e-1 knew about
  `(edge_correlation, mae)`; the corpus added `(mae, rmse)` — MAE ranks
  DINOv2-B first while RMSE ranks DINOv2-S first. One probe, three metrics,
  three orderings. This is the case `ranking_disagreements` exists for, and
  6e-3's renderer must not pick a headline metric silently.
- **Semantic segmentation ran twice by accident and the duplicates are kept.**
  The smoke test and the full array appended to the same part file. Metrics are
  identical to six decimals; durations are 137.6 s vs 123.5 s and 104.9 s vs
  115.9 s. That is the "a wall clock is not a metric" rule demonstrated rather
  than asserted, so it earns its two lines. `latest_per_backbone` handles them.

**Depth and surface normals are NYUv2 now, not Taskonomy.** Their Taskonomy
numbers came from uncommitted code and were unreachable from any entry point.
`/shared/sets/datasets/vision/probing_3D/nyuv2_new` has exactly the
`<root>/<split>/{images,targets}` layout the CLI already expects — 795/654, the
canonical split — so both probes joined with **no code change**: depth d1
0.7652/0.7851, normals mean 29.48°/30.11°. Two hazards, both recorded inline in
`build_corpus.sh`:

- **`--target-scale 1.0` is load-bearing.** These targets are `.npy` already in
  metres; NYUv2's *PNG* distribution is millimetres. Passing 1000 divides a
  3-metre reading to 3 mm, and `depth_metrics` reports RMSE in whatever unit it
  is handed — the number would look superb and mean nothing.
- **The normals are dense**, with not one zero-length vector across 40 sampled
  frames, including across the ~28% of pixels where the depth map has no ground
  truth. So `load_normal_map`'s validity rule marks nothing and the probe is
  scored on GeoNet's *filled* geometry. That is what probe3d's own files
  support, but it is **not comparable with a masked normals probe** such as the
  Taskonomy one.

**Cluster notes, each of which cost a failed submission.** `sbatch` copies the
script to `/var/spool/slurmd/job<id>/slurm_script`, so `${BASH_SOURCE[0]}` does
**not** locate the repo — use `$SLURM_SUBMIT_DIR`. And **submit to `-p dgx`, not
`dgxh100`**: the H100 image is Ubuntu 24.04 with only `python3.12`, while
`.venv/bin/python` symlinks to `/usr/bin/python3.10`, and the failure surfaces
as "cannot execute: required file not found" against the *console script*, which
points at the wrong file entirely. The sbatch guards both. The feature cache
lives at `/shared/results/common/kargin/visbench_cache` via `VISBENCH_CACHE`,
because `/home` is under a 60 GB quota and exhausting it mid-array surfaces as
an unrelated exception several steps later.

**Four more, each of which cost a submission on 2026-08-13.** They are cheap
individually and none of them fails in a way that names its own cause.

- **`dgx1`/`dgx2` are the *only* nodes where `.venv` resolves.** The rule above
  is narrower than it reads: it is not "avoid dgxh100", it is "dgx and nothing
  else". `rtx4090`'s `c22` fails identically (`No such file or directory`, exit
  127), and the login shell can be on `dgxh100` too, so `.venv/bin/python`
  fails *interactively* before any job is submitted. Those nodes are **V100s**,
  which is also why TF32 can never explain a number this project has produced.
- **`srun -p dgx` inside an existing allocation silently ignores the
  partition.** Within a job, `srun` creates a *step* in that job; the flag is
  accepted and disregarded, and it sits printing "Requested nodes are busy"
  against whatever node you are already on. `sbatch` is the only way out of an
  allocation.
- **`/tmp` is node-local.** A script or an `--output` path under `/tmp` exists
  only on the submitting host, so the job fails **with no log at all** — the
  one failure mode that leaves nothing to read. Put both under `$HOME`.
- **`build_corpus.sh` needs `.venv/bin` on `PATH`, not just `.venv/bin/python`.**
  It invokes the `visbench` console script, and the miss surfaces as
  `visbench: command not found` from inside the script's own per-probe error
  handler — which reports it as a *failed probe*, not a missing environment.

`generic_segmentation` reads binary masks built by
`scripts/binarise_voc_masks.py` from VOC's `SegmentationClass`. **Do not point
it at `SegmentationClass` directly**: `load_mask` converts a palette PNG, so
void 255 resolves to light grey and is scored as foreground — measured 0.078
foreground against a true 0.0399, and **0 void pixels recovered out of 5,355**.

### Step 6e-2b — **done**. Six backbones, and the field that made them comparable

The corpus covers **DINOv2-S/B, CLIP-B/16 and B/32, ResNet-18 and ResNet-50
across all twelve probes** — twelve comparability groups, each holding all six.
The eight dense probes rank them in the order the taxonomy would predict
(DINOv2 > CLIP > ResNet, and B/16 > B/32, RN50 > RN18), which is the strongest
evidence to date that these probes measure representation quality rather than
capacity.

**Schema v7 adds `pooling_requested`, and widening the corpus is what forced
it.** `pooling` is recorded *resolved* — right for reading one record, wrong for
comparing two, because `default` resolves to `cls` on a ViT and `mean` on a CNN.
Keyed on the resolution, **four of the twelve probes split along an
architectural line**: classification, retrieval, correspondence and similarity
each became two groups, and a CNN could never be ranked against a ViT. That
matters concretely — `resnet50` scores 0.9980 top-1 and 0.9357 mAP, which puts
it *first* on both boards ahead of DINOv2-B, and the split hid it.

- **The request is the protocol; the resolution is a property of the backbone.**
  `comparability_key` uses `pooling_requested`, falling back to `pooling` for v6
  and earlier. Two runs that both asked for `default` are comparable. Two that
  named `cls` and `mean` explicitly are **still not**, and a test pins that —
  without it the fix is indistinguishable from dropping pooling from the key,
  which would silently rank a CLS-pooled number against a mean-pooled one.
- **A v6 record groups with a v7 record that asked for `cls`, not with one that
  asked for `default`.** It cannot say which its `cls` was, so it takes the
  conservative side rather than joining a group defined by a request it may not
  have made.
- **`run()` must actually write the field, and there is a test for that alone.**
  A field declared and never populated is the QuickGELU failure in a new place.
  `FakeCNN` gained a real `preprocess` so the ViT-versus-CNN case is proved
  through `run()` rather than on hand-built records.
- **Do not read `resnet50` topping classification and retrieval as a finding.**
  `resnet50.a1_in1k` was trained on ImageNet-1k *with labels* and Imagenette is
  an ImageNet subset. The README already records this; the row belongs on the
  board with the caveat attached, not suppressed.

**The correspondence board looked wrong here, and the diagnosis in this section
was itself wrong — see 6f below.** 6e-2b recorded the problem as `num_matches`
being a per-backbone denominator (4,911 for ResNet-18 against 27,590 for
DINOv2-B). That is true and remains true, but it is not what inverted the board.
The threshold unit was.

**`slurm/corpus.sbatch` takes `VISBENCH_BACKBONES`** so the matrix widens
without editing it. The `--array` range **cannot** be derived from that list —
`#SBATCH` directives are read before the script runs — so the script refuses a
range that does not match `probes x backbones` unless `VISBENCH_PARTIAL=1` says
the gap is deliberate. Worth the guard because the failure is invisible: a short
array simply omits probes, and the corpus then looks complete, since every group
it *does* contain holds every backbone.

### Step 6e-3 — **done**. The renderer, and the two bugs it took to get right

`visbench/results/render.py` (28 fast tests), `scripts/render_tables.py`,
`LEADERBOARD.md`, and nine marker-delimited boards in `README.md`.
`tests/test_readme.py` runs `--check` in the **fast** suite, so a table that
drifts from the corpus fails a build instead of shipping to PyPI.

**The split from `leaderboard.py` is one-directional and must stay that way.**
That module decides what may be compared; this one only formats an answer.
Nothing in `render.py` may relax a rule from it — no backbone metadata table, no
"known good" number, no per-task special case beyond two listed dicts.

- **`HEADLINE_METRICS` is listed, and a task without an entry raises.** A board
  ordered by whichever metric sorted first asserts a ranking nobody chose. A
  test checks the dict against `list_probes()`, so adding a probe and forgetting
  the table fails immediately rather than at the next corpus run.
- **Bolding is per column, never per row**, and the disagreement note names the
  metrics that fight the ordering. `edge` orders its three metrics three ways;
  bolding a winning row would assert an outright winner that does not exist.
- **Diagnostics and ceilings are columns, not omissions.** `rank` refuses them,
  correctly — but a table that then drops `num_matches` presents a comparison
  whose terms differ as though they did not. `CAVEATS` carries the prose.
- **Narrowing for width cannot launder a board.** `metrics=` trims *rankable*
  columns only: diagnostics always survive, ceilings survive for the metrics
  kept, and the disagreement note is computed over every shared metric. All
  three are tested, and the denominator test was **vacuous when first written** —
  it asserted `` `num_matches` `` appeared in the rendered page, which the
  caveat prose satisfies whether or not the column exists. Assert against the
  header row. Found by mutation-testing, not by reading.
- **`COUNT_METRICS` is listed for the same reason directions are.** "Any
  diagnostic" renders `tie_rate` 0.0 as `0`; "any integral value" renders a
  saturated `ceiling_recall@4p` of 1.0 as `1`. Only `num_matches` and
  `classes_scored` are counts; `detections_per_image` is a mean.
- **A lazy `.*?` cannot delimit a possibly-empty marker body.** With `-->\n`
  consumed by the open group there is nothing left to match, so the regex ran on
  to the *next* pair's close marker and swallowed the open marker between them:
  nine empty markers produced four boards and silently deleted four. The body
  pattern is now "anything that is not another marker", and `rewrite()` asserts
  the marker count is unchanged before writing. **A generator that can delete
  its own inputs and report success is worse than no generator.**

**`render_leaderboard` has a caller on purpose.** It writes `LEADERBOARD.md`,
all twelve groups unnarrowed. A declared-but-uncalled mechanism is the QuickGELU
failure, and this module is exactly where one would hide.

### Step 6e-4 — **done**. The artifact, and why it refuses more than it accepts

`visbench/hub/` (`save_probe`, `load_probe`, `probe_metadata`,
`IncompatibleProbe`, `ARTIFACT_VERSION`), `head_spec()` / `probe_state()` /
`load_probe_state()` on `BaseTask`, and `examples/save_probe.py`. 22 fast tests.
**No network dependency** — 6e-5 adds `huggingface_hub` behind a `[hub]` extra;
saving and loading works in a core install.

**A head is only meaningful against the exact features it was fitted on, and
almost every way of getting that wrong is shape-compatible.** Measured on real
DINOv2-S weights over Imagenette: a linear head fitted on CLS tokens, then fed
*mean-pooled* tokens from the same backbone, scores **0.9620 against 0.9820**.
It does not crash and it does not produce garbage — it produces a number nobody
would question. That two-point gap is the entire argument for the module.

Four fields are checked on load, and each is its own silent failure:

- **`backbone_key`** — a fine-tuned DINOv2-S and its parent share a name, width,
  pooling rule, feature mode and depth. This is the *only* thing that differs.
- **`pooling`**, resolved — the 0.9620 case above. Raises nothing on its own.
- **`feature_mode`** — `dense_cls_broadcast` doubles the width, so this usually
  raises anyway. "Usually" is doing a lot of work when the head is a 1x1 conv.
- **`layers`** — right shape, wrong depth.

Decisions settled while building it:

- **`weights_only=True` on load is not optional.** 6e-5 fetches these from a
  hub, and an unrestricted `torch.load` on a downloaded file is arbitrary code
  execution. Nothing may enter the payload that needs unpickling to
  reconstruct — a test asserts the artifact still loads under it, so the day
  someone puts an object in the metadata it fails immediately.
- **The head *recipe* is captured in `_build_head`, not reconstructed at save
  time.** `in_channels` and `output_size` are measured from the first batch of
  features, so nothing outside that call knows them. A save that re-derived
  them would be guessing, and a guess that happens to be right for one backbone
  is the worst outcome.
- **`probe_state()` exists because `ClassificationTask` standardises.** Its
  `_mean`/`_std` are fitted on the training split, live outside `self.head`, and
  decide the answer; a head saved without them loads cleanly and scores against
  raw features. This was found by reading `fit()`, not by a failing test — check
  any new trained probe for fitted state outside its head.
- **`head_spec()` has two kinds and they are not merged.** `registered` for
  anything built through `build_head`, `linear` for the bare `nn.Linear` the
  classification probe fits on pooled features. The registered heads all map
  `(B, C, H, W)`; forcing the pooled one through would reload it at the wrong
  rank.
- **Zero-shot probes are refused by name.** Retrieval, correspondence and
  similarity train nothing, so an artifact would hold no weights and the
  backbone alone reproduces them.
- **`strict=False` warns and loads.** Deliberately probing how far a head
  transfers is a legitimate experiment; doing it silently is not, and a number
  produced that way is comparable with nothing, because `run()` would record the
  backbone actually used with nothing saying the head came from elsewhere.
- **`ARTIFACT_VERSION` is separate from `SCHEMA_VERSION`.** They version
  unrelated things and move for unrelated reasons; tying them would force a
  migration on one side every time the other changed.

**Every one of the four identity guards was mutation-tested** by deleting it and
re-running. Three failed immediately; **dropping `backbone_key` left all 21
tests passing**, because the cross-backbone test was being caught by the pooling
check instead. `test_different_weights_alone_are_refused` exists because of
that. A guard with no test that isolates it is a guard you do not have.

### Step 6e-5 — **done**. The transport, which deliberately adds no rules

`visbench/hub/remote.py` (`push_probe`, `load_probe_from_hub`, `probe_card`),
the `[hub]` extra, `--show-card` on `examples/save_probe.py`. 41 fast tests
across the two hub modules, **none of which touch the network**.

**`load_probe_from_hub` is `load_probe` with a download in front of it, and that
is the design.** A separate remote loading path is how one of the two ends up
without `weights_only=True` or without the identity checks — and a downloaded
probe is precisely the one that needs both. `push_probe` likewise calls
`save_probe`; a test asserts the uploaded bytes and a locally saved artifact
carry identical `meta` and `head_spec`, so the two formats cannot drift.

- **`private=True` is the default.** A push is not reversible the way a local
  write is: once a repository is public it may already have been fetched, and
  deleting it does not unpublish what was taken. Public is a decision, not a
  default someone discovers afterwards.
- **`save_probe` runs before `create_repo`.** An unfitted or zero-shot probe is
  refused *before* anything is created, so a rejected push leaves no empty
  public repository behind. Mutation-tested by swapping the order: two tests
  fail.
- **The card is generated from `probe_metadata`**, the same source the artifact
  uses, so the page and the file cannot disagree — a test pins that. A bare
  `.pt` on a model page does not tell a visitor the one thing they must know,
  which is that the weights belong to exactly one backbone.
- **`revision=` is offered because a Hub repo is mutable.** `main` today is not
  promised to be `main` next month, so anything whose number is quoted should
  pin a commit.
- **Publishing is the maintainer's, like PyPI.** `examples/save_probe.py`
  *prints* the card under `--show-card` rather than pushing; a push under
  someone's account is not something an example does as a side effect.

**`huggingface_hub` is imported inside the functions that need it**, so
`import visbench.hub`, `save_probe` and `load_probe` all work in a core install.
The test for this was **weak when first written**: it reloaded `visbench.hub`
but not `visbench.hub.remote`, which was already in `sys.modules`, so moving the
import to module scope passed. Reload the module that *holds* the import.

**A dependency change means `uv lock` in the same commit** — see the release
notes above. The diff here is seven lines and no version moved;
`huggingface_hub` was already in the lock transitively via timm.

**And that last fact is what broke CI.** Three pull tests used
`monkeypatch.setattr("huggingface_hub.hf_hub_download", ...)`, which needs the
real module to import. It does here — transitively via timm — and **CI installs
`.[dev]` only**, so it failed on both 3.10 and 3.12 with `ModuleNotFoundError`
at monkeypatch time. This is the "a local env with extra packages installed will
pass checks that CI fails" rule, hit exactly as written, by the person who wrote
the tests for the optional extra.

The fix is a stub module injected with `monkeypatch.setitem(sys.modules, ...)`,
not a `skipif`. Skipping would leave the **pull** path — the half that loads
someone else's file — untested in precisely the install where most users will
run it.

**To check an optional extra locally, block the import rather than trusting the
venv.** A pytest plugin inserting a `find_spec` that raises for the package
reproduces CI's environment in one command:

```python
# /tmp/blockhub.py, then: PYTHONPATH=/tmp pytest -p blockhub
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "huggingface_hub":
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None
sys.meta_path.insert(0, _Block())
```

Run this whenever a test touches `clip`, `timm` or `hub` — the five verification
commands cannot catch it, because they run in the environment that has
everything.

### Step 6f — **done**. The unit that inverted a published board

`threshold_units` defaults to `"pixel"`, the CLI's `--units` with it, the
headline metric is `recall@5px`, and the six correspondence records in the
corpus were re-run. **This corrects a board that shipped in v0.6.0 ranked
upside down.**

**A patch width is a property of the backbone, not of the protocol.** At 224px
it is 14px on DINOv2/14, 16px on CLIP ViT-B/16 and 32px on ViT-B/32 or a
ResNet's last stage. `_scale` divides pixel error by `patch_spacing`, so
`recall@1p` asks a coarse-grid backbone to land within 32px and a fine-grid one
within 14px — a 2.3x more permissive target — and prints both under one name.

Measured on the same 200 Imagenette pairs, only the unit changed:

| backbone | `recall@1p` (v0.6.0) | `recall@5px` (now) |
| --- | --- | --- |
| resnet18 | **0.8927** | 0.0973 |
| resnet50 | 0.8601 | 0.0887 |
| clip_vitb32 | 0.7992 | 0.0897 |
| dinov2_vits14 | 0.7834 | **0.3049** |
| dinov2_vitb14 | 0.7594 | 0.2816 |
| clip_vitb16 | 0.7179 | 0.2689 |

**First and last place swap**, and the pixel ordering is the one the taxonomy
predicts — DINOv2 > CLIP-16 > the 7x7 grids, matching all eight dense probes.

- **6e-2b's diagnosis was wrong and is corrected in place.** It blamed
  `num_matches`, the per-backbone denominator. That difference is real (4,911
  against 27,590) and still noted in `CAVEATS`, but it is a consequence of grid
  resolution, not the cause of the inversion. Normalising by the ceiling does
  not fix the unit problem either — it was tried, and RN18 still led. **When a
  board looks wrong, check what the threshold *means* on each row before
  reaching for the denominator.**
- **The old docstring argued the exact opposite and was persuasive.** It said
  patch widths made numbers "comparable across resolutions and architectures,
  which is the point of a benchmark", citing the real fact that `recall@1px` has
  a ceiling of 0.015 on DINOv2-S. The fact is true; the conclusion inverted a
  board. The 1px ceiling is an argument for choosing a sensible *pixel*
  threshold, not for a backbone-dependent unit.
- **The quantisation floor is stated, not divided out.** `ceiling_recall@5px` is
  ~0.10 on a 7x7 grid and ~0.41 on a 16x16 one. That is the honest way to carry
  a floor, and the mechanism already existed — `context_metrics` has done this
  since v0.1.
- **`patch` is kept, not removed.** Within one backbone it answers a real
  question, and the README's `max_warp` sweep is exactly that use. Removing it
  would delete a legitimate measurement to prevent a misuse the default now
  prevents.
- **A unit change is a protocol change, and the records say so.**
  `threshold_units` and `thresholds` live in `task_params`, so a pixel-unit
  record and a patch-unit one land in different comparability groups
  automatically. No v0.6.0 correspondence number can be silently ranked against
  a v0.6.1 one — the group digest moved from `1ac52b90` to `7db23175`.
### Still open in v0.3, beyond the numbered steps

- ~~Low-level tasks get their first real entry~~ — **done in 6d-1**, shipped in
  v0.4.0. Edge detection is in; optical flow, texture/reflectance and IQA are
  still scope only, and `visbench/tasks/low_level/README.md` says what each
  would cost.
- HF Hub integration for sharing pretrained probe heads and a public
  leaderboard, once there's enough task/backbone coverage for a leaderboard
  to be meaningful.

---

## Release history — what each upload verified

Moved out of `CLAUDE.md` on 2026-08-20 with the v0.3 step write-ups, and for
the same reason: every session was carrying five releases of upload detail.
The rules these releases *established* are in `CLAUDE.md` under "Releasing";
what follows is the record of each one, newest first. The current release is
still described in `CLAUDE.md` — only the superseded ones are here.

**Publishing needs the maintainer's credentials and is theirs to run.** Never
attempt it, and do not assume a tag means a release went out; check
[PyPI](https://pypi.org/project/visbench/) if it matters.

**v0.9.0**, the release before v0.10.0:

Package version was `0.9.0`, and it was **on PyPI: uploaded 2026-08-14 at
14:31 UTC**, wheel and sdist both (316 KB and 803 KB), tagged `v0.9.0` on merge
commit `7816517`, with a GitHub release created from that tag. **Verified the
standing way**: the wheel downloaded, its SHA256 checked against PyPI's digest
(`7334297d…`), extracted, put *first* on `sys.path` and **imported**, with an
assert on `visbench.__file__` so the editable checkout could not answer in its
place. It reports `__version__ = "0.9.0"`, `SCHEMA_VERSION = 7`,
`ARTIFACT_VERSION = 1`, thirteen probes, six backbones, and
`show_probes() == list_probes()` read back *through the import* — which is the
only way to check 0.9.0's actual content, since its release is a command and a
package rather than a number. Also confirmed through it: `get_probe(
"correspondence")` still reports `threshold_units="pixel"`, so v0.6.1's fix
survives a fourth release, and `DetectionTask.probe_state()` carries `grid_hw`,
which is 9a's fix.


**v0.8.0**, the release before it:

Package version was `0.8.0`, and it was **on PyPI: uploaded 2026-08-07 at
09:42 UTC**, wheel and sdist both (284 KB and 703 KB), tagged `v0.8.0` on
`574e792`. **Verified the standing way on 2026-08-07** — the published wheel
downloaded, its SHA256 checked against PyPI's digest, extracted, put *first* on
`sys.path` and **imported**, with an assert on `visbench.__file__` so the
editable checkout cannot answer in its place. That last step is the one worth
copying: without it the check passes on a machine where the package is already
installed, whatever the wheel contains. It reports `__version__ = "0.8.0"`,
`SCHEMA_VERSION = 7`, `ARTIFACT_VERSION = 1`, **thirteen** probes with `corner`
among them, six backbones, and `visbench.data.derived` exporting
`ShiTomasiResponse`/`DerivedTargetDataset`. Two further reads *through the
import*, since neither is visible in a version number: `get_probe(
"correspondence")` still reports `threshold_units="pixel"` and thresholds
`(1, 2, 5, 10)`, so v0.6.1's fix survived two releases; and METADATA puts
`huggingface-hub` only under `hub` and `all`, never in the core requirements.
The published README carries the concept DOI `10.5281/zenodo.21822684` and no
other.


**The tag-versus-artifact gap recurred, in the other direction this time.**
`main` is one commit ahead of `v0.8.0` — `48571bd`, the DOI badge fix — and that
commit landed at 09:50 UTC, **eight minutes after the 09:42 upload**, so unlike
v0.7.0 the extra commit is *not* in the wheel. It is one line of README, no
code, and it reaches PyPI with the next release; its own commit message says so.
Two consecutive releases have now had `main`, the tag and the wheel disagree
benignly, which is precisely the direction that trains you to stop checking.
**Do not fix either by moving a tag**: a PyPI version can never be re-uploaded
and the Zenodo archive is permanent, so a moved tag would disagree with both.

The upload before it was **v0.7.0 on 2026-08-06 at 15:57 UTC**, wheel and sdist
(266 KB and 665 KB). Verified the way v0.6.1 was — the wheel downloaded, put on
`sys.path` and *imported* — because v0.7.0's content is a command and a docs
extra, neither of which a version number shows: `__version__ = "0.7.0"`, twelve
probes, `demo` among the CLI's four commands, and `get_probe("correspondence")`
still reporting `threshold_units="pixel"` so v0.6.1's fix survived the release.
METADATA confirmed the `docs` extra (sphinx, furo, myst-parser,
sphinx-copybutton) and `huggingface-hub` still only under `hub` and `all`. Its
uploaded artifact was one commit *ahead* of the `v0.7.0` tag (`39e0495`), built
from `main` after `7ee0d07`, so `git show v0.7.0:README.md` has no DOI and the
published README does — harmless, and the first half of the pattern above.

The upload before it was **v0.6.0 on 2026-08-02**
([PyPI](https://pypi.org/project/visbench/)) — wheel and sdist both (276 KB and
645 KB), tagged `v0.6.0` on merge commit `77986e9`. Verified by downloading the published wheel and reading
`__version__ = "0.6.0"`, `SCHEMA_VERSION = 7`, `ARTIFACT_VERSION = 1` and the
five modules v0.6.0 added (`results/render.py`, `results/leaderboard.py`,
`hub/{__init__,artifact,remote}.py`) *out of it*, plus the METADATA confirming
`huggingface-hub` appears only under the `hub` and `all` extras and never in the
core requirements. Not by trusting the version number, which is the whole point
of the exercise.

**v0.6.1 followed the same day** — wheel and sdist both, tagged `v0.6.1` on
merge commit `dc5bc40`, verified the same way *and one step further*. A version
number cannot show what that release changed, because its entire content is a
changed default: so the published wheel was put on `sys.path` and imported, and
`get_probe("correspondence")` was constructed from it. It reports
`threshold_units="pixel"`, `thresholds=(1, 2, 5, 10)` and a headline of
`recall@5px`, with `"patch"` still accepted. **When a release's content is a
default value, read it back through an import, not out of the source text** —
source inspection cannot rule out a runtime override.
