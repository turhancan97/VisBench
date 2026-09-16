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
what follows is the record of each one, newest first — **including the current
release**, since 2026-09-03. `CLAUDE.md` names the version and the concept DOI
and points here for everything else.

**Publishing needs the maintainer's credentials and is theirs to run.** Never
attempt it, and do not assume a tag means a release went out; check
[PyPI](https://pypi.org/project/visbench/) if it matters.

The four most recent entries were lifted out of `CLAUDE.md` on 2026-09-03,
when that file passed the 150k-character limit it is loaded under for the second
time. Nothing was rewritten; each paragraph is as its release recorded it.

**`0.18.0` is fully released** (2026-09-12). On PyPI — wheel and sdist both
(408,877 and 1,317,230 bytes, wheel sha256 `54b85a27…`, sdist `d0c6deea…`) —
tagged `v0.18.0`, **annotated** (tag object `73015e4`), on merge commit
`2aee8c8`, with a GitHub release cut from that tag (published
2026-09-12T08:33:01Z) and archived by Zenodo as version DOI
**`10.5281/zenodo.22722545`**, the **thirteenth**, minted four seconds after the
release. **Verified out of the published wheel by import**: `__version__`
0.18.0, `SCHEMA_VERSION` **9**, the `hardware` field present on `ResultRecord`,
seventeen probes, thirteen backbones. Both artifacts match the locally built
ones byte for byte, checked against PyPI's own digests rather than assumed.

**Tag, wheel, release and `main` all agree at `2aee8c8`** — the sixth release
running with no gap. The concept DOI `10.5281/zenodo.21822684` resolves to
record 22722545 with `metadata.version` `v0.18.0`, read off Zenodo's API.

**It is a minor release because the schema moves, v8 → v9.** A v0.17.0 install
raises `Unsupported schema_version` on a record written by this one; records
written *before* it read unchanged here, so the incompatibility is forward-only
— which is more than a patch and less than a break. No measurement moves: the
corpus goes 264 → 360 records and stays at 204 board cells over seventeen
boards, because the 96 re-run cells reproduce what they superseded.

**What the release is, in one line:** the records learned to describe
themselves. Every trained board now carries how its *fit* went (v8) and every
run carries what *machine* it ran on (v9) — two fields added because the corpus
was twice asked a question it could not answer, and both added additively.

**The version lives in two places, and the lockfile is what said so.** Bumping
`visbench/__init__.py` and running `uv lock` produced **no diff**, which
contradicts the standing rule that a version bump always moves the lockfile.
That contradiction is what exposed `pyproject.toml` still reading 0.17.0 — so
the wheel would have built as **0.17.0 while `__version__` reported 0.18.0**,
the tag/artifact mismatch these rules exist to prevent, invisible until someone
imported the published package. **An empty `uv lock` diff after a bump is a
symptom, not a pass.** Both literals are bumped now and a check asserts
`pyproject` == `__version__` == `CITATION.cff`.

**The Zenodo API 302s, and without `-L` that reads as a broken archive.**
`https://zenodo.org/api/records/<conceptrecid>` redirects to the newest version
record, so a plain `curl` returns a 229-byte HTML redirect page and the JSON
parse fails with a traceback that looks like Zenodo serving garbage. It is
`curl -sL`. This sits directly on top of the existing rule to read the API
before concluding an archive is wrong — the check itself has a way of lying to
you.

**The abstracts were hand-checked before the tag, per the standing rule**, and
the archived text was read back after. Record 22722545's description carries
the new clause — a record "also carries how its fit went … and what hardware
produced it" — alongside "Seventeen probes"; the "Fifteen" that three earlier
archives carry permanently is absent. No probe shipped in this release, so the
count needed no change; the clause did, because the release is about what a
record carries.

**`0.17.0` is fully released** (2026-09-09). On PyPI — wheel and sdist both
(407,088 and 1,281,464 bytes, wheel sha256 `3cd28f6f…`, sdist `2d41ae4e…`) —
tagged `v0.17.0`, **annotated**, on merge commit `6f69872`, with a GitHub
release cut from that tag (published 2026-09-09T18:23:30Z) and archived by
Zenodo as version DOI **`10.5281/zenodo.22681020`**, the twelfth. **Verified
out of the published wheel by import**: `__version__` 0.17.0, `SCHEMA_VERSION`
8, seventeen probes with `instance_segmentation` among them, four heads
(`linear`, `dpt`, `detection`, `instance`). The wheel's PyPI digest matches the
locally built artifact byte for byte, so what was checked before the tag was
pushed is what shipped.

**Tag, wheel, release and `main` all agree at `6f69872`** — the fifth release
running with no gap. The concept DOI `10.5281/zenodo.21822684` now resolves to
record 22681020 with `metadata.version` `v0.17.0`, read off Zenodo's API rather
than assumed.

**The order was tag locally, build, verify, then push the tag.** A pushed tag
cannot be moved under this project's own rules, so anything wrong in the
artifact had to surface while the tag was still local and deletable. Checked in
that window: the import from the wheel with an assert on `visbench.__file__`;
all seven extras carrying dependencies, matched against **either** quote style
because Metadata 2.5 writes `extra == 'hub'` and a double-quoted check reports
every extra as empty; and the gallery absent from the sdist, which it would
otherwise near-triple.

**It is a minor release because it adds a probe.** `instance_segmentation`
(14a-1 to 14a-4) takes the corpus to 264 records / 204 board cells over
seventeen boards; schema stays v8 and no existing measurement moves. The
release also carries the split control, which showed a board's cluster
membership is partly a property of its split — `detection` on the instance
probe's images changes cluster — so no published number moved but one published
*reading* did.

**The abstract fix reached the archive, and that was checked.** Zenodo record
22681020's description reads "Seventeen probes span" and names instance
segmentation. That is the field 0.16.1 was spent discovering: `CITATION.cff`'s
abstract said "Fifteen probes" from v0.13.0 onward and three archives still say
so permanently. `tests/test_citation.py` compares the two metadata files'
*titles* and not their abstracts, so both remain hand-checked per probe.

**Two operational notes, each of which cost an attempt.** `twine upload` failed
with `ImportError: cannot import name 'errors' from 'packaging'` — and the
version that matters is **`packaging`**, not twine: both environments had twine
7.0.0, but conda's sat beside `packaging` 23.2 (no `packaging.errors`) and the
throwaway venv's beside 26.3. "Conda's twine is broken" had been the recorded
diagnosis and was wrong twice; it is the pin beside it. Separately, reading the
release back with `gh release view --json createdAt` failed twice before the
real field list was read — it is `publishedAt`. That is the standing
guessed-import rule arriving through a CLI rather than a Python import.

**`0.16.1` is fully released** (2026-09-07). On PyPI — wheel and sdist both
(375,412 and 1,203,926 bytes, wheel sha256 `21e01a55…`, sdist `a99afeaa…`,
uploaded 2026-09-07T16:47:11Z) — tagged `v0.16.1`, **annotated**, on merge
commit `1079bfc`, with a GitHub release cut from that tag (published
2026-09-07T16:52:53Z) and archived by Zenodo as version DOI
**`10.5281/zenodo.22647634`**, the eleventh. **Verified out of the published
wheel by import**: `__version__` 0.16.1, `SCHEMA_VERSION` 8, `ARTIFACT_VERSION`
1, sixteen probes, thirteen backbones, three heads (`linear`, `dpt`,
`detection`), `show_probes() == list_probes()`, and `correspondence` still
`threshold_units="pixel"` with `(1, 2, 5, 10)`.

**Tag, wheel, release and `main` all agree at `1079bfc`** — the fourth release
running with no gap. `gh api .../git/tags/{sha}` resolves, which is the call
that 404s on v0.12.0's lightweight tag.

**This release's content is what the gallery figures SAY, so that is what was
read back**, through the import rather than out of the source text:
`DisplayRange(1.632, 7.014).caption("m")` returns `'1.632 to 7.014 m'` and the
negative case `'-0.3318 to 1.956'`, which is the ambiguity the change exists to
remove; `frame_stem` and `frame_label` are present in `visbench.viz.panels`,
which is what stops the two gallery pages drifting again; `_DEPTH_ANCHORS` is
five anchors from `(13, 24, 61)` to `(250, 235, 110)`; `style_for("depth").kind`
is `"depth"` while `style_for("edge").kind` is still `"magnitude"`, so
magnitude stayed grey.

**Two claims this release published were checkable and had been asserted
instead. Both were caught by verifying the artifact, which is the argument for
doing it at all.**

**The `CITATION.cff` drift reached GitHub's cite button and no archive.** The
changelog entry as first written said three Zenodo archives described a
VisBench one probe smaller than the one they contain. They do not.
`.zenodo.json` carries its own `description`, Zenodo prefers it over
`CITATION.cff` — the rule this project had already written down — and it said
"Sixteen probes" throughout. Confirmed against Zenodo's API for every version
from v0.13.0 to v0.16.1. The bug was real and worth fixing, because GitHub
renders `CITATION.cff` for the cite button, but the permanent-damage claim was
false. **A divergence between those two files is half wrong and half fine, and
which half depends on which file the consumer prefers**; the two are read by
different consumers, which is what makes the pair confusing and is exactly why
`tests/test_citation.py` pins their titles together. It does not compare their
abstracts, which is how the count drifted for three releases.

**The depth ramp's luminance is non-decreasing, not "strictly monotonic".**
Measured on the published wheel over 256 samples: it never reverses and rises
24.3 → 229.2, with **5 ties in 255 steps**. Those are uint8 quantisation — the
ramp spans about 205 levels across 256 samples, so ties are forced — and strict
monotonicity is therefore unreachable, not merely absent. The property the
argument needs is the weaker one, and it holds: the grey panel stays
recoverable as the luminance channel, and the ramp cannot introduce a boundary
grey does not already have. `tests/viz/test_colour.py` asserts precisely that
(`np.diff(...) >= 0` plus a rising endpoint), so **the test was right and the
prose around it overstated**. `colour.py`'s docstring shipped in this wheel
with the stronger word, and since the archive is not editable it was corrected
on `main` afterwards (2026-09-08): the docstring, `docs/guides/visualising.md`,
and the test's own *name* — `test_luminance_increases_all_the_way_along_the_ramp`
became `test_luminance_never_reverses_along_the_ramp`, and its docstring now
records the tie count and says that tightening `>= 0` to `> 0` would fail,
because that tidy-up is how the stronger claim got in. No pixel moved. The
general form: when a docstring states a property as absolute, check whether the
test states it that way too, and prefer the test's wording.

**The METADATA check, now routine**: no relative links, all three `docs/*.md`
targets (`api/index.md`, `probes/overview.md`, `roadmap.md`) resolve, the
concept DOI present with **no** other Zenodo DOI over it, dependencies under
all seven extras (`all`, `clip`, `datasets`, `dev`, `docs`, `hub`, `timm`) read
with the either-quote match Metadata 2.5 requires, and the README's status line
reading v0.16.1.

**The upload failed on an environment, not an artifact, and the error names
neither.** `twine upload` on the conda interpreter died with `ImportError:
cannot import name 'errors' from 'packaging'`: that env has `packaging` 23.2,
and twine's `package.py` imports `packaging.errors`, added in 24.2. The
traceback is eight frames of `importlib` and mentions neither twine's version
nor `packaging`'s, so it reads as a broken twine. The standing rule already had
the fix — **build and upload from a throwaway venv**, never the project's
`.venv/` (no `pip`) and never a shared conda env, where upgrading `packaging`
to get one release out would move dependency resolution for everything else
installed in it.

**`0.16.0` is fully released** (2026-09-05). On PyPI — wheel and sdist both
(373,824 and 1,195,351 bytes, wheel sha256 `37a8f115…`, sdist `071cb0fd…`,
uploaded 2026-09-05T11:03:35Z) — tagged `v0.16.0`, **annotated**, on merge
commit `481fdae`, with a GitHub release cut from that tag (published
2026-09-05T11:03:44Z) and archived by Zenodo as version DOI
**`10.5281/zenodo.22340899`**, the tenth. **Verified out of the published wheel
by import**: `__version__` 0.16.0, `SCHEMA_VERSION` 8, `ARTIFACT_VERSION` 1,
sixteen probes, thirteen backbones, three heads (`linear`, `dpt`, `detection`),
`show_probes() == list_probes()`, and `correspondence` still
`threshold_units="pixel"` with `(1, 2, 5, 10)` — read back through the import
rather than the source text, as every release since 0.6.1 has been.

**Tag, wheel, release and `main` all agree at `481fdae`** — the third release
running with no gap. Tagging before building is what buys it.

**This release's own content is a README and a docs tree, so the check is the
METADATA, and it is the reason the release happened.** The README shipped with
0.15.0 linked to `docs/tasks.md` and `docs/show.md`, both deleted by the docs
restructure — two dead links on the package's front page that no later edit to
`main` could fix, because a PyPI version can never be re-uploaded. Confirmed
gone from the served METADATA, with `docs/roadmap.md` the only `docs/*.md` link
left and still resolving, **every link absolute** (the `tests/test_readme.py`
rule, checked against the artifact rather than the repo), the concept DOI
present and **no other Zenodo DOI** pasted over it.

**The extras were read back too, and they carry 13a's correction**: `docs` has
`sphinx-design` beside furo/myst/copybutton, and `sphinx` is in **both** `docs`
and `dev` — the optional-extra trap's fix, verified in the artifact rather than
in `pyproject.toml`. Metadata 2.5 writes those markers with **single** quotes
(`extra == 'hub'`), so the check matched either quote, per the standing rule.

**`0.15.0` is fully released** (2026-09-03). On PyPI — wheel and sdist both
(369,030 and 1,119,784 bytes, wheel sha256 `74e56ac9…`, sdist `021872a7…`) —
tagged `v0.15.0`, **annotated**, on merge commit `f8e2ae5`, with a GitHub
release cut from that tag (published 2026-09-03T15:40:56Z) and archived by
Zenodo as version DOI **`10.5281/zenodo.22283870`**, the ninth. **Verified out
of the published wheel by import**, the standing way: `__version__` 0.15.0,
`SCHEMA_VERSION` 8, `ARTIFACT_VERSION` 1, sixteen probes, thirteen backbones
(twelve corpus columns plus the `dinov2_vitb14_196` control), three heads,
`show_probes() == list_probes()`, `correspondence` still
`threshold_units="pixel"` with `(1, 2, 5, 10)`, and METADATA putting `datasets`
under `datasets` and `huggingface-hub` under `hub`.

This release's *own* content was read back through that import too, not out of
the source text: `DenseTrainingTask.evaluate_oracle` is present and its
docstring now says **bar** and names **DPT**, which is the correction 0.15.0
exists to publish. `visbench.metrics.boundary` and `BSDS500Dataset` import.

**Tag, wheel, release and `main` all agree at `f8e2ae5`** — the second release
running with no gap at all, and the digest PyPI serves is the one the tagged
commit builds. Tagging before building is what buys this; keep the order.

**The tag is annotated**, deliberately: `v0.12.0` is the only lightweight tag
in this project's history, which is why `gh api .../git/tags/{sha}` 404s on it.
`git tag -a`, always. The release page reports `targetCommitish: main`, which
looks wrong and is not — check the tag ref, which resolves through the tag
object to `f8e2ae5`.

**The concept DOI `10.5281/zenodo.21822684` is unchanged and now resolves to
v0.15.0** (`https://zenodo.org/records/22283870`), so `README.md`,
`docs/index.md` and `CITATION.cff` needed no edit and `22283870` appears in
none of them — asserted, not assumed, by grep and by
`tests/test_citation.py`.

**One thing this verification got wrong on the first attempt, worth keeping**:
the import check reached for `show_probes` in `visbench.cli.main`, where it
does not live — it is in `visbench.viz.styles`. The check *failed loudly*,
which is the right outcome, but a verification script that imports from a
guessed path can only ever fail; **read the symbol's home before writing the
check**, or a real regression and a wrong import look identical.

**`0.14.0` is fully released** (2026-09-01). On PyPI — wheel and sdist both
(368,805 and 1,106,178 bytes, wheel sha256 `6b1d7258…`) — tagged `v0.14.0` on
merge commit `aea8f1e`, with a GitHub release cut from that tag (published
2026-09-01T18:58:26Z) and archived by Zenodo as version DOI
**`10.5281/zenodo.22237026`**, the eighth. **Verified out of the published wheel
by import**, the standing way: `__version__` 0.14.0, `SCHEMA_VERSION` 8,
`ARTIFACT_VERSION` 1, sixteen probes, thirteen backbones (twelve corpus columns
plus the `dinov2_vitb14_196` control), `show_probes() == list_probes()`,
`correspondence` still `threshold_units="pixel"` with `(1, 2, 5, 10)`. This
release's *own* content was read back through that import too — the boundary
metric, `BSDS500Dataset`, `evaluate_oracle`, and the oracle's opt-in refusal
raising for `DepthTask`.

**Tag, wheel, release and `main` all agree at `aea8f1e`, and the digest was
checked on both sides of the upload.** The artifact was built and
`twine check`ed locally before publishing, and PyPI now serves a wheel with the
identical sha256 and byte count — so nothing was rebuilt or substituted in
between. `main` is **0 commits ahead of the tag**, the first release with no gap
at all. Tagging before building is what buys this; keep the order.

The release page reports `targetCommitish: main`, which looks wrong and is not —
that field records the default branch for a pre-existing tag, and the tag ref
itself resolves to `aea8f1e`. Check the ref, not that field.

**The concept DOI `10.5281/zenodo.21822684` is unchanged and now resolves to
v0.14.0** — no edit needed in `README.md`, `docs/index.md` or `CITATION.cff`,
and `22237026` appears in none of them. That is the point of quoting the concept
DOI rather than a version DOI; `tests/test_citation.py` rejects any other Zenodo
DOI in those files because pasting one over it is the realistic mistake and it
freezes every citation at one release.

**What 0.14.0 is**: the release that ships no probe. The corpus is unchanged at
16 probes x 12 backbones, 192 records, and schema stays at v8. What it adds is
the **oracle gate**, the **ceiling beside every dense score**, and **BSDS500's
dataset plus a validated ODS/OIS/AP metric** reproducing the published human
agreement at 0.8030 against 0.80 — and it is the release in which the gate
*refused* a probe, BSDS's own, at a 0.4193 ODS ceiling. `scipy` became a
declared core dependency.

**`0.13.0` is fully released** (2026-08-28, confirmed 2026-09-01). On PyPI —
wheel and sdist both (343,687 and 1,046,538 bytes, wheel sha256
`049f7f77…`) — tagged `v0.13.0` on merge commit `205cb0e`, with a GitHub
release cut from that tag (published 2026-08-28T15:37:48Z) and archived by
Zenodo as version DOI **`10.5281/zenodo.22147201`**, the seventh. **Verified
out of the published wheel by import**, the standing way: `__version__`
0.13.0, `SCHEMA_VERSION` 8, sixteen probes, twelve corpus backbones (plus the
`dinov2_vitb14_196` resolution control), `show_probes() == list_probes()`,
`correspondence` still `threshold_units="pixel"` with `(1, 2, 5, 10)`.

The release-prep also fixed something 0.12.0's prep missed: `CHANGELOG.md`'s
link refs at the bottom had **no `[0.12.0]` entry at all** and `[Unreleased]`
still compared from `v0.11.0`. Every section now has a matching ref, checked
both ways.

**`main` had already moved one merge past the tag when this was checked** —
the `superpixel-probe` PR (`d2075ca`, PR #78, the rejected photometric
superpixel probe) landed after `v0.13.0` was tagged and released, adding no
new probe or corpus record. The tag, wheel and release still agree with each
other exactly; only `main` is ahead, which is the benign gap the standing
rules below warn drifts silently into the *next* release if not watched.
**The concept DOI `10.5281/zenodo.21822684` is unchanged and now resolves to
v0.13.0** — no edit needed in `README.md`, `docs/index.md` or
`CITATION.cff`.

**`0.12.0` is fully released** (2026-08-28). On PyPI — wheel and sdist both
(338,079 and 1,023,738 bytes, wheel sha256 `c55a63b7…`) — tagged `v0.12.0` on
merge commit `f5afcd3`, with a GitHub release cut from that tag and archived by
Zenodo as version DOI **`10.5281/zenodo.22146664`**, the sixth. **Verified out
of the published wheel by import**, the standing way: `__version__` 0.12.0,
`SCHEMA_VERSION` 7 (as shipped; `main` is at v8 since), fifteen probes, twelve
corpus backbones, `show_probes() == list_probes()`, `correspondence` still
`threshold_units="pixel"` with `(1, 2, 5, 10)`, and METADATA putting `datasets`
under `datasets`/`dev`/`all`.

**Tag, wheel and release all agree at `f5afcd3`** — the third release running,
and `main` had already moved two merges past it when the release was cut, which
is exactly the case creating a release *from the tag* exists to handle. The
release page reports `targetCommitish: main`, which looks wrong and is not: that
field records the default branch for a pre-existing tag, and the tag ref itself
resolves to `f5afcd3`. Check the ref, not that field.

**The concept DOI `10.5281/zenodo.21822684` is unchanged and now resolves to
v0.12.0**, which is the whole point of quoting it rather than a version DOI —
`README.md`, `docs/index.md` and `CITATION.cff` needed no edit. Do not paste
`22146664` over it; `tests/test_citation.py` rejects any other Zenodo DOI in
those files because that paste is the realistic mistake and it freezes every
citation at one release.

Package version was `0.11.0` through the last release, and it is **on PyPI,
uploaded 2026-08-20**, wheel
and sdist both (321,900 and 860,881 bytes), tagged `v0.11.0` on merge commit
`1f0908e`, with a GitHub release created from that tag. **Verified the standing
way**: the wheel downloaded from the JSON API, its SHA256 checked against
PyPI's digest (`0c7f3ec1…`), extracted, put *first* on `sys.path` and
**imported**, with an assert on `visbench.__file__` so the editable checkout
could not answer in its place. It reports `__version__ = "0.11.0"`,
`SCHEMA_VERSION = 7`, `ARTIFACT_VERSION = 1`, thirteen probes, **twelve**
backbones with `dino_vitb16` and `sam_vitb16` among them, and
`show_probes() == list_probes()` read back through the import. Also confirmed
through it: `get_probe("correspondence")` reports `threshold_units="pixel"` and
`(1, 2, 5, 10)`, so v0.6.1's fix survives a sixth release, and METADATA puts
`huggingface-hub` only under `hub` and `all`.

**`main`, the tag and the wheel all agree exactly** — all three at `1f0908e`,
the second release running. The artifact was also verified *before* upload and
its digest is identical to what PyPI now serves, so nothing was rebuilt or
substituted in between. Tagging before building is what buys this; keep the
order.

**Zenodo archived it as version DOI `10.5281/zenodo.22027316`**, the fifth
under the concept DOI, which is unchanged and remains the only one quoted
anywhere.

**v0.10.0**, the release before v0.11.0:

Package version is `0.10.0`, and it is **on PyPI: uploaded 2026-08-19 at
17:26 UTC**, wheel and sdist both (321 KB and 840 KB), tagged `v0.10.0` on merge
commit `9f0b91e`, with a GitHub release created from that tag. **Verified the
standing way**: the wheel downloaded from the JSON API, its SHA256 checked
against PyPI's digest (`ac03455b…`), extracted, put *first* on `sys.path` and
**imported**, with an assert on `visbench.__file__` so the editable checkout
could not answer in its place. It reports `__version__ = "0.10.0"`,
`SCHEMA_VERSION = 7`, `ARTIFACT_VERSION = 1`, thirteen probes, **ten**
backbones with `supervised_vitb16` among them, and
`show_probes() == list_probes()` read back through the import. Also confirmed
through it: `get_probe("correspondence")` reports `threshold_units="pixel"` and
`(1, 2, 5, 10)`, so v0.6.1's fix survives a fifth release, and METADATA puts
`huggingface-hub` only under `hub` and `all`.

**The tag and the wheel agree exactly, which the two previous releases could
not say.** `v0.10.0` is on `9f0b91e` and the artifact was built from that
commit, so what is installable is what is tagged. Tagging *before* building is
what bought it, and it is the order to keep.

**`main` is one commit ahead of the tag** — the one recording this paragraph,
docs only, no code, landed after the upload. That is the same benign gap the
last two releases had, and the third in a row, which is precisely the pattern
that trains you to stop checking; it reaches PyPI with the next release. **Do
not fix it by moving the tag**: a PyPI version can never be re-uploaded and the
Zenodo archive is permanent, so a moved tag would disagree with both.

**Zenodo archived it as version DOI `10.5281/zenodo.22016457`**, the fourth
under the concept DOI, which is unchanged and remains the only one quoted
anywhere. The release commit also stopped `CITATION.cff` and `.zenodo.json`
saying "Twelve probes" — that text is what Zenodo archives permanently, no test
reads it, and it had been one probe short since v0.8.0 shipped `corner`.

**v0.9.0**, the release before it:

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

## 15a — `dino_vitb8`: crossing the grid axis from the other side

**2026-09-16.** A thirteenth backbone and its seventeen-cell board, added for
one reason: `CORPUS_FINDINGS.md`'s resolution control could only lower a grid,
never raise one, and said so — "it is one-sided because nothing else here can
be raised". `dinov2_vitb14_196` cut DINOv2 from 256 tokens to 196 at fixed
weights; nothing could take a non-DINOv2 the other way, because open_clip does
not interpolate position embeddings and timm needs `dynamic_img_size`.

`vit_base_patch8_224.dino` needs neither. It is the same objective, the same
pretraining set, the same width and the same depth as `dino_vitb16`, at a patch
of 8 — **784 tokens against 196**, and the only fine grid in the corpus that is
not a DINOv2. Registered in one step (PR #117), boarded in this one.

### What the run cost, and the two cells that skipped

Seventeen cells, `--array=0-16%4` on `dgx2`/`dgxa100`, V100s throughout.
Per-cell elapsed ran from **18 s** to **37 m 33 s**, the long one being
`scene_classification` on Places365 — which would have overrun the script's
45-minute `#SBATCH --time` default had a two-cell smoke test not been run first
to size it. 784 tokens is 4x a B/16's, so dense features are 4x the size; that
showed up in wall clock far less than expected, because extraction dominates
and it is decode-bound.

**`corner` and `orientation` failed in under a second**, which is the signature
of `build_corpus.sh`'s missing-data guard rather than a crash:
`data/corner_frames/` is gitignored and absent from a fresh checkout.
`scripts/stage_corner_frames.py` fixed it, and the staging was **verified
set-equal to the frames `edge` reads** on both splits, 600 each, rather than
assumed equal — the claim earning `corner` its place is that it ranks
differently from `edge` despite a 0.52 target correlation, which is only exact
on identical pixels.

Two process notes from that, both worth more than the cells:

- **A failure filter that greps for `Traceback|CANCELLED|error:` reports
  "none" when a script exits cleanly with `!!! SKIPPED`.** Silence looked like
  success. Widen a monitor's alternation to the *terminal states* rather than
  the crash shapes you predicted.
- **A verification check that guesses a format can only ever fail.** The first
  set-equality check reconstructed stems as `{building}__{point}` from the row
  tuple and reported `False` with near-zero overlap. The rows are
  `(building, point, view)` and the link is
  `{building}__point_{point}_view_{view}.png`. Near-zero overlap was the tell
  that the *check* was wrong; a real mismatch would have overlapped partially.

### `merge_corpus.sh` re-imports records deliberately held out

**The most expensive thing this step found, and it was one command from
landing.** Run as documented, the merge added **58** records where 19 were
expected. The other 39 were 2026-09-10 files still sitting in
`results/corpus/parts/` — including **all three A100
`fine_grained_classification` cells** that the v8 `training` re-run held out of
the corpus on purpose and parked in `results/controls/hardware_a100.jsonl`.

The script deduplicates by exact JSON line. That stops it re-adding what is
*already* in the corpus and gives no protection at all against records
deliberately kept *out* of it — and merging these three is precisely the
ranking change holding them out prevented: `convnext_base` below `resnet50` on
the CUB board, caused by a variable no record carried.

It was caught only by diffing the merge against the corpus before trusting it,
which is the standing rule. The corpus was restored and re-merged from a
staging directory holding the 17 new files alone: **+19, all `dino_vitb8`,
zero pre-existing records lost.** `parts/` is not a queue of pending work; it
is an archive with excluded records in it, and the merge cannot tell the
difference.

### Free reproducibility check

`classification` and `edge` were each run twice — once in the smoke test, once
in the array — so their parts files carry two records. `classification` is
**bit-identical**. `edge` reproduces to **~1e-7 relative** (`edge_correlation`
0.48245614 against 0.48245651). That matches the documented behaviour of the
linear dense boards exactly, on a backbone that had never been run before.

### The measurement

Corpus **360 -> 379 records**, `LEADERBOARD.md` **204 -> 221 cells**, 17 boards
at 13 backbones each. The reading is in `CORPUS_FINDINGS.md`; the short version
is that the resolution *correlation* survives at n=13 — tokens is still the
strongest structural correlate on **8 of 11** grid-reading boards against 9 of
11 — while the controlled pair says it is largely **not causal**. A 4x finer
grid at fixed objective and data gains 0.2861 on `correspondence`, where the
grid genuinely is the floor, and moves every other dense board by a rounding
error or the wrong way.

The mechanism is in the ceilings: they rise every time, as they must, and the
share a linear head recovers **falls every time, five of five**.

---

## Steps 7a-14a — write-ups lifted from `CLAUDE.md`

Moved on 2026-09-13, when `CLAUDE.md` passed the 150k-character limit it is
loaded under for the **third** time — and for the reason that file records
after the first two: the growth is retrospective narrative, not rules. Each
write-up below is as `CLAUDE.md` carried it, unrewritten; what stayed there is
the rule, pointing here for the derivation — so a "this file" inside one of
these blocks means `CLAUDE.md`, and "below"/"above" mean where it used to sit.
Read this before touching the code one of these steps built.


### The instance probe's head and board (14a-3, 14a-4)

- **The instance probe is `DetectionTask` plus a mask branch, and every
  decision in it is about staying attributable** (14a-3). `InstanceHead` is a
  `DetectionHead` plus **one 1x1 convolution** over RoI-aligned features, where
  Mask R-CNN's branch is four 3x3 convolutions and a deconvolution on an FPN.
  RoIAlign carries no parameters, so the only learned thing between features
  and mask is that convolution — which is what lets a difference between two
  backbones be a difference between two representations, the same argument
  behind `LinearHead` and `hidden_dim=0`. Class-**agnostic**, one channel: the
  class is already decided by the detection branch.

  **Registered at 14a-4, and the board ranks**: spread **0.2148** on
  `mask_map_50` over twelve backbones, `dinov2_vitb14` 0.2861 to
  `convnext_base` 0.0713, reproducing no other board's ordering (0 of 136
  pairs). Only `siglip_vitb16`/`supervised_vitb16` are inseparable (0.0005), so
  **quote it to three decimals** like `detection`. 14a-3's proof run reported
  0.2641 for DINOv2-S against the board's **0.2696** — the example constructs
  the backbone itself and `run()` seeds before constructing from a name, the
  documented RNG path difference. The board is the number to quote.

  Four things not to re-derive:

  **Both branches live in one module.** A mask convolution held beside the head
  would be outside `head.state_dict()`, so a saved probe would load its boxes
  and predict blank masks — 9a's `grid_hw` bug, one artifact round-trip later.

  **The mask branch trains on ground-truth boxes and predicts on detected
  ones**, recorded as `mask_train_boxes: "ground_truth"`. The boxes come from
  the same head being trained, so early epochs would hand the mask branch RoIs
  containing no object. `box_map_50` is reported beside `mask_map_50` so a low
  score is attributable to outlines or to localisation.

  **Target and prediction go through the same RoIAlign.** Cropping the ground
  truth by hand would put them on two sampling grids that agree almost
  everywhere — the `recall@1px = 0.003` failure. And the mask bias starts at
  **zero, not the focal prior**: a RoI is a detected object's box, so half its
  pixels are foreground, and copying the dense branch's prior starts every mask
  empty.

  **Collecting a split's mask predictions costs 5.4 GB** at the 74.4
  detections/image DINOv2-S actually decodes — `bool` rather than `float32` is
  what makes it feasible (21.6 GB otherwise). A first draft of that docstring
  said "~5 MB", which was per-image arithmetic labelled as a total; check a
  memory claim by multiplying it out.

### The instance board's cluster, and the box half that refuted the obvious reading (14a-4)

- **`instance_segmentation` is a high-level board whose four strongest partners
  are all mid-level — and the mask branch is not why** (14a-4). Mean rho
  +0.821 against mid-level, **+0.238 against its own tier**: `occlusion_edge`
  +0.958, `surface_normal` +0.930, `generic_segmentation` +0.909, `depth`
  +0.902, against `semantic_segmentation` +0.378 and `retrieval` −0.217. The
  sharpest case yet of `high_level` being a folder rather than a quantity.

  **The obvious explanation was checked and is wrong.** "Mask AP measures
  outlines, so it ranks with geometry" predicts the box half ranking elsewhere;
  the record carries `box_map_50` from the same runs and the two halves agree at
  **+0.986**, both topped by `occlusion_edge`. The box half alone ranks with the
  geometry cluster while inheriting every line from `detection`, whose board
  sits at +0.804 with `semantic_segmentation`. **The split control settled the
  rest** (2026-09-09) — see the next bullet.

### The split control, in full (2026-09-09/2026-09-10)

- **A board's cluster membership is partly a property of its *split*, and
  `detection` is the proof** (the split control,
  `results/controls/detection_split.jsonl`, 24 records). Run `detection` on the
  instance probe's 1464/1449 `ImageSets/Segmentation` images instead of its own
  600 `Main` frames — same probe, same head, same losses, same matcher, same
  metric — and it **changes cluster**: `occlusion_edge` **+0.965**, mean
  **+0.784** against mid-level and **+0.018** against high, where the published
  board reads +0.804 with `semantic_segmentation` and +0.483 with
  `occlusion_edge`.

  **Once images and size match, `detection` and `instance_segmentation` rank the
  same board (+0.958)**, so mask-derived boxes against VOC's XML plus the whole
  mask branch are worth ~0.04 of rho. Neither half of the data explains it
  alone (+0.818 for size, +0.818 for images) and the two compound (+0.510).
  `mae_vitb16` shows it plainly: **0.1296 on the published board (tenth of
  twelve) against 0.3371 on the segmentation split (first)**.

  So 14a-4's negative claim is now positive: **it is the split, not the probe.**
  Two consequences. **Never quote a cluster as a property of a *task*** — it is
  a property of a board as configured; the two-cluster structure and mid/low
  coherence are untouched, but which side a board falls on is contingent. And
  **no published number moves** — what was contingent was always the reading.

  **The one thing the corpus could not answer is now answered** (2026-09-10):
  the published board *does* underfit relative to the full split, on **12/12**
  backbones — mean `train_loss` 1.3500 against 1.3071. Its records used to
  carry `training: null`; the v8 re-run gave them the field. Size is the larger
  half (1.3071 against 1.4012 at equal images, 12/12) and is partly offset
  because the `Main` frames are *easier to fit* than the segmentation ones at
  equal size (12/12, +0.0512). `scripts/analyse_split_control.py` prints it
  under "THE FIT".

  **The control as first written down was impossible, and checking beat
  assuming.** `CORPUS_FINDINGS.md` had called for the instance head on
  `ImageSets/Main --limit 600`; `SegmentationObject` covers 2913 images, so 141
  of those 600 stems have a mask and 459 have no target. Inverting it — the
  published probe onto the new probe's images — is both runnable and the better
  experiment, since that baseline is the one already published.

  **It also retired an already-published finding's argument.** "The board
  clustering is not an artefact of shared datasets" rested on there being two
  boards on the identical 1449 VOC images, whose pair was the weakest of three;
  this probe is a third, and it is `generic_segmentation`'s **nearest neighbour
  of all** at +0.909. The conclusion survives on different evidence — the three
  same-image pairs span +0.378 to +0.909 and the weakest of all six VOC pairs is
  a same-image one, so identical pixels are neither sufficient nor necessary —
  and the test that pinned the old form failed exactly as its message predicted.
  See `CORPUS_FINDINGS.md`; do not quote the old ordering.

### Mask AP, and the guard the shape table exists for (14a-2)

- **Mask AP is the detection protocol with the overlap swapped, and the
  sharing is enforced by a listed table** (14a-2). `average_precision` takes
  `shapes="boxes"|"masks"`, keys of `SHAPE_KINDS`, and reads the annotation key,
  the coercion and the overlap from that row — so `VOCevaldet.m`'s matching has
  one implementation rather than two, and mask AP is comparable with this
  codebase's own box AP. **The guard is the point of the table**: annotations
  carrying the *other* geometry are refused by name, because masks scored as
  boxes read an absent key, coerce to empty and report **0.0** — a silent wrong
  number that looks like a detector finding nothing.

  Three things not to re-derive. **Box AP is bit-identical after the refactor**,
  checked over 4800 values on 400 random splits, twice — `detection` is a
  published board and "the tests still pass" is not that claim. **The sweep is
  an optimisation, not an approximation**: the best-matching shape and its
  overlap do not depend on the threshold, so `sweep_average_precision` overlaps
  once per class and re-tallies per threshold, which took mask mAP over VOC val
  from unusable to 7 seconds; a test pins the swept and naive paths equal on
  both geometries. And **rectangle masks score exactly as their boxes** — a
  rectangle's pixel IoU *is* its half-open box IoU — so any divergence between
  the two paths fails on a number. Calibrated at **1.0000** on perfect
  predictions; the 16x16 oracle is mask mAP@50 **0.6666**. Keys are prefixed
  `mask_` so they cannot sit beside detection's `map_50` meaning something else.

### The oracle gate, the rejection that produced it, and the DPT control that widened its claim

- **The gauntlet asks whether a target is distinctive; it never asked whether
  it is *recoverable*. Photometric superpixels is what that cost** (built and
  rejected 2026-08-28). SLIC boundary regression passed every gate — tail 0.055
  against `edge_occlusion`'s 0.46, overlap with `edge_texture` 0.267 against the
  0.52 `corner` shipped with, cross-image `|r|` 0.044 — and then scored
  **0.0434 / 0.0209 / 0.0238** on DINOv2-S, CLIP-B/16 and ResNet-50, where the
  weakest shipped low-level probe scores 0.179-0.236 and `corner` scores
  0.492-0.651. Spread 0.023, ResNet-50 "beating" CLIP by 0.003, and
  `train_loss` **lowest** for the worst scorers — the heads learned the mean
  boundary density and nothing about location.

  **The missing check was an oracle, and it now ships** (2026-09-01).
  `DenseTrainingTask.evaluate_oracle` pools the target to the feature grid,
  upsamples it back and scores it with the probe's own metric — what a perfect
  backbone would make available, since a dense probe sees one feature vector per
  patch and signal finer than a patch is *absent from its input* rather than
  merely hard to predict. No backbone, no features, no fitted head, so it costs
  one pass over a split rather than a board.
  `CorrespondenceTask.evaluate_ceiling` is the same idea, arrived at the same
  way. **Run `scripts/oracle_ceiling.py` before writing the next derived
  task**, and see the "oracle gate" section of
  `visbench/tasks/low_level/README.md` for the numbers.

  **The bar, calibrated against this rejection**, over the pinned 600 val frames
  at a 16x16 grid: the four shipped magnitude targets score 0.53–0.83 and
  photometric superpixels scores **0.25**. At a ResNet's 7x7 grid, 0.43–0.67
  against 0.11. Three things about it that are not obvious:

  - **A probe opts in**, `TARGET_STYLES`-style, and every other dense probe
    raises. Pooling is the right bottleneck only for a target that averages —
    the mean of classes 1 and 15 is class 8 — and a silently defaulting oracle
    would return a confident number about nothing, which is worse than none for
    a gate whose job is to stop work.
  - **The upsample is bilinear because `LinearHead`'s is**, so the gate is never
    more permissive than the heads it protects. Even a target built from hard
    grid cells scores ~0.88 rather than 1.0.
  - **It is a bar, never a denominator.** Unlike `evaluate_ceiling` it is an
    achievable score rather than a proven bound, and the ratio does not
    discriminate anyway: `corner` reaches 80% of its oracle and `keypoints2d`
    41%, and both rank backbones fine.
  - **It measures a candidate's ceiling and nothing measured its floor, which
    is the gap relative depth ordering cost** (2026-09-04). A probe needs room
    between the cheapest shortcut a head could learn and what a perfect
    backbone could reach, and the gate only ever checked the top. Relative
    depth **cleared the gate at a 94.0% oracle and was rejected anyway**:
    "the lower point in the image is nearer" scores **65.2%** with no features
    at all, so the usable band was 0.157 wide, the five backbones' own ceilings
    differed by 0.055 of it, and the three strongest landed **0.0007** apart —
    Spearman **+1.000** with the `depth` board it subclassed, at 38% of its
    spread. `corner` ranks fine at a comparable 0.83 ceiling *because its
    trivial floor is near zero*. **A ceiling of 0.9 above a floor of 0.7 is a
    worse probe than a ceiling of 0.6 above a floor of 0.** Name the cheapest
    shortcut — an image coordinate, a per-image constant, the dataset mean —
    and measure it on the samples the metric will use;
    `scripts/premeasure_ordering.py` is the worked example, and it costs one
    pass over a split.
  - **It models a *linear* head exactly, and exactly one backbone's DPT head
    beats it** (measured on two backbones 2026-09-01, widened to the whole
    corpus 2026-09-04; full write-up in `results/controls/README.md`).
    `LinearHead` is a 1x1 convolution per patch plus a bilinear upsample, which
    is literally what the oracle computes. Across the five probes and the nine
    twelve-block ViTs a DPT head reaches **54-104%** of the oracle (median 83%)
    and exceeds it in **2 of 45 cells** — both `mae_vitb16`. So it is a bar for
    the head VisBench reports, **not a bound on what is achievable**; but **do
    not read the 104% as a property of decoders** either, since it is one row
    and MAE is the only backbone trained by masked *pixel* reconstruction. The
    two-backbone version read as the general claim, which is the mistake
    widening it caught. It does not reopen BSDS500: scaling that 0.4193 linear
    ceiling by the best ratio seen anywhere (1.038) gives ~0.435 ODS, still
    below Canny's 0.60.

    **A CNN's DPT run is a different experiment and has its own file.**
    `_grid_of` takes the *finest* requested map, so a ResNet reading stages 1-4
    gets a 56x56 oracle where its linear run reading `layer4` got 7x7. The head
    and the bottleneck both moved, so only the DPT/linear *gain* is comparable.
    A ViT's blocks share one grid, which is what makes the ViT group the clean
    control and the one the gate's claim is stated over.

    **And a head is not a neutral magnifying glass**, counted rather than
    anecdotal: two of five ViT boards change leader and **24 of 174 separable
    pairs reorder**; on the three CNNs, three of five boards change leader and
    two invert outright (`convnext_base` first to last). That is the
    demonstration behind reporting the linear number when comparing
    representations. **A DPT number is good to three decimals** — re-running ten
    cells three days later moved them 2e-4 to 3.3e-3 relative, where the linear
    boards reproduce at ~1e-7 — so count a reordering only over pairs both
    boards separate by more than their own drift.

  **It has now refused something** (2026-09-01). The BSDS500 probe was not built
  because the gate put a linear probe's ceiling at **0.4193 ODS** on the 16x16
  grid every corpus backbone produces, against published detectors at 0.60-0.79
  and human agreement at 0.80. That cost one 60-second run instead of a
  12-backbone board. **Do not read that 0.42 against the 0.25 that rejected
  superpixels** — one is ODS and the other Pearson correlation, they are not
  comparable, and an earlier draft made exactly that mistake.

  **A pooled-resolution overlap check nearly became a false veto**: the
  boundary map reads 0.267 against `edge` at full resolution and 0.684 pooled to
  a 16x16 grid, which looked decisive until the shipped `corner` target read
  **0.781** there and its board ranks differently from `edge` anyway.
  **Calibrate a new rejection criterion against something that already passed
  before letting it reject anything.**

  What survived: `DerivedTargetDataset` memoises computed targets
  (`MEMO_LIMIT`), because `CachedFeatures.__getitem__` calls
  `dataset.target(index)` on every access — a ten-epoch streaming run was
  recomputing every target ten times, which `corner` and `orientation` both
  paid.

### The panel viewer's rules, and the depth ramp that replaced greyscale (9a)

- **A viewer that applies its own geometry is worse than no viewer** (9a). This
  is the single rule `visbench/viz/` exists to keep, and it inverts the usual
  cost/benefit: a panel's entire evidential content is whether the image and the
  target line up, so a viewer that resizes for layout, re-reads the source file
  or re-crops can make a *misaligned pipeline look fine and a correct one look
  broken*. It is guaranteed by pasting `np.asarray(dataset[i][0])` unchanged,
  which is cheap only because dense datasets already yield a PIL image at the
  working resolution rather than a normalised tensor — there is nothing to
  invert. A fast test pins the image panel byte-for-byte.

  **Four validity conventions, one listed table, no fallback.** The four
  conventions in the bullet above are invisible in a tensor's shape or dtype, so
  `TARGET_STYLES` is keyed per probe and `style_for` raises on an unlisted one —
  the posture `METRIC_DIRECTIONS` takes, for the same reason. A "scalar map,
  mask the zeros" default is right for depth and silently wrong for the four
  probes where 0 is a real reading, and it *renders*: the panel comes out
  looking like a target full of holes. There is a test per convention.

  **A prediction is drawn against the target's range, not its own.** Scaling
  each panel to its own extremes is the obvious implementation and it hides the
  most common way a regression head is wrong: a prediction uniformly half the
  target's magnitude renders identically to a correct one. The test asserts both
  halves — that the shared range separates them, *and* that independent ranges
  do not — because only the second one fails if someone "simplifies" it back.

  **Magenta for invalid, chosen because no colouriser here can produce it**:
  greyscale has no hue, `(n + 1) / 2` cannot reach it for a unit vector, and
  VOC's palette does not contain it. A test asserts that, so a future colouriser
  cannot quietly make the marker ambiguous.

  **Greyscale was argued for on two grounds and only one of them generalised**
  (2026-09-06). `colour.py` documented greyscale as deliberate: no lookup table
  means no dependency, and a ramp's transitions read as edges on a noisy
  magnitude map. The first holds everywhere — `_DEPTH_ANCHORS` is five anchors
  interpolated inline, as `voc_palette` and `_orientation` already are. The
  second is a fact about a *magnitude*, and **`depth` is not one**: mid-grey is
  an ordinary reading for "how much is here" and means nothing to the eye for
  "how far", so a grey depth panel reads as texture. It is a ramp now, dark blue
  near to pale yellow far, and `magnitude` stays grey.

  **The old argument is answered by a number, not by preference: the ramp's
  luminance never reverses**, so the grey panel is recoverable as its
  luminance channel and it cannot introduce a boundary grey does not already
  have. **Non-decreasing, not strictly increasing** — 5 ties in 255 steps,
  because the ramp spans ~205 of 255 uint8 levels, so strict monotonicity is
  unreachable at that sample count rather than absent. Assert `>= 0` plus a
  rising endpoint when adding a colouriser, never `> 0`. The anchors are the viridis family
  **with its purple end dropped** — viridis begins at `(68, 1, 84)`, hue 296°,
  four degrees from magenta — and the magenta test is a *distance* now, because
  exact inequality against `INVALID_RGB` would have passed on that purple. And
  the fixtures start at 0.1 rather than 0: depth's convention makes 0 invalid,
  so a ramp row starting there is painted magenta and every property is then
  measured against the marker, which is how these tests first failed.

  **A range caption reads `"1.632 to 7.014 m"`.** A magnitude probe's
  `_activate` is the identity, so a head may predict below zero and often does;
  `-0.3318--1.956` reads as a subtraction. `to` rather than an en dash, per the
  ASCII rule the bitmap font imposes.

  One thing it is deliberately **not**: it does not train. That is
  `run --save-probe`, added alongside, because `--push-to` needed a Hub account
  and the prediction column otherwise had no CLI-producible input.
  `correspondence` was out of scope for 9a and is covered by 9b, below.

### The gallery's licensing and its four prediction-only figures (9d/11a)

- **The docs gallery is real photographs, and the licence rule that made it
  generated was satisfied by better sourcing rather than waived** (9d, replaced
  2026-08-19). **VOC, ImageNet, NYUv2, Taskonomy and NIGHTS all restrict
  redistribution and appear nowhere in this repository.** Open Images'
  validation split is CC BY 2.0, so `scripts/fetch_gallery_frames.py` reads from
  there, the frames are committed (`assets/gallery_frames/`, 1.5 MB) and **the
  licence is verified per frame rather than inherited** — with a refusal for any
  frame lacking an author or landing page, since an unattributable CC BY image
  is one this repo may not redistribute. `CREDITS.md` is generated beside them
  and a test fails on an uncredited photograph, because CC BY compliance rots
  silently: the page renders correctly either way.

  **Four probes cannot have a target column and must not be given one.**
  `depth`, `surface_normal`, `keypoints2d` and `occlusion_edge` need sensor or
  reconstruction geometry no redistributable photograph carries, so they render
  `image | prediction` from a *published* Hub head with a footer saying so. An
  invented middle column would teach the wrong convention to exactly the reader
  who came to learn it. Two details cost an attempt each: a trained head's
  `output_size` is **fitted state**, so these emit 224x224 whatever they are fed
  and the figure must be rendered at 224; and they are drawn on **interiors**,
  since the heads were fitted on NYUv2 rooms.

  **The figures live under `docs/_static/`, not `assets/`** — Sphinx cannot
  follow a relative path escaping its source tree and MyST does not warn, so
  `-W` would not catch `../assets/...`; the site would simply have holes. The
  README points at the same files through `raw.githubusercontent.com`. They are
  excluded from the sdist, which they would otherwise nearly triple.

  **Every gallery bug so far was found by looking at the output, never by a
  test** — five of them now, the newest being an instance colour that blended
  into the magenta invalid marker (14a-4). The earlier four: a `(H, W)` mask
  against a `(3, H, W)` target; a ragged final row; a footer that truncated the
  *legend*; and four prediction-only figures captioning each row `str(index)`
  while computing a `DisplayRange` one line above, so a depth page stated no
  range and never named the **feature grid**. **When a page cannot be rendered
  in the fast suite, make what it *says* a pure function** — `frame_stem`/
  `frame_label` in `panels.py`, shared by both pages so they cannot drift again.

### The docstrings that had never been rendered (13a)

- **Docstrings had been written for an API reference for six steps and none of
  them had ever been rendered** (13a). ~5,198 lines of numpydoc went through
  docutils for the first time at 13a and nine source files had real defects —
  malformed simple tables, a `#:` block whose `History/-----` reached docutils
  as a section title (fatal under `-W`), `Returns`/`Raises` sections whose free
  prose had no type line so napoleon read *the prose* as the type, and dead
  cross-references. **A docstring convention nothing renders is not a
  convention, it is a guess** — `scripts/check_docstrings.py` runs each one
  through napoleon and docutils in the fast suite, in ~1s with no Sphinx
  *build*. The trick that makes it work is indenting the result three spaces
  under a dummy directive: un-nested, a section title is legal and docutils
  says nothing. It documents what it **cannot** reach and a test asserts that
  limit. Its `sphinx` import was the optional-extra trap for the third time —
  see that bullet below.

### The intersphinx warning filter, and why -W made it a deploy failure (7d)

- **A `-W` docs build must tolerate an unreachable intersphinx inventory, and
  the filter has two details that each cost an attempt** (7d). intersphinx
  fetches five `objects.inv` per cold build; a `ConnectionResetError` is logged
  as a warning, which `-W` turns into a failed deploy — it did, on the first
  push to `main`, minutes after the same commit passed on its PR. Losing
  intersphinx degrades gracefully (nitpicky is off), so the *warning* is the
  only real problem, and it carries no `type=`, so `suppress_warnings` cannot
  target it. The filter in `docs/conf.py` matches that one message, and: it goes
  on the **handlers, not the logger** (Sphinx emits from child loggers, and a
  parent's filters never see a propagated record), and it is inserted at
  **position 0, not appended** (`-W` is itself a filter on the same handler, so
  anything appended after it never runs — which looks correct and does nothing).
  Verified by checking a broken toctree still fails, so the filter did not
  disable the guard.

### The library-surface backlog, as it closed (2026-08-28)

Added 2026-08-14, after a read of what a new user would reach for and not find.
All three shipped: `visbench show` (9a-9d), `examples/custom_backbone.py`
(2026-08-19), and the **dataset bridges** (2026-08-28, below). `docs/roadmap.md`
has the public version. **None of these was a defect** — each was already
reachable by writing Python; what was missing was the shortest path. v0.7 is the
precedent for shipping a release that changes no number.

**The dataset bridges, as shipped.** `TorchvisionDataset` and
`HuggingFaceDataset` in `visbench/data/bridges.py` — thin `BaseDataset`
adapters over a `torch.utils.data` dataset / a `datasets.Dataset`. `torchvision`
is a core dep so its bridge imports at module scope; `datasets` is a `[datasets]`
extra, imported lazily inside `HuggingFaceDataset.__init__` and `_build_hf`, so
`import visbench` never needs it (and it is in `dev` too, or the bridge tests
skip in CI — the optional-extra trap, pre-empted this time). On the CLI,
`classification` / `retrieval` / `scene_classification` take
`--dataset torchvision:CIFAR10` / `--dataset hf:cifar100:name=cifar100` in place
of `--data` (a mutually-exclusive group; `resolve_named_dataset` in
`cli/datasets.py` parses `scheme:name:key=value…`). **Image-level probes only** —
a dense/pair/triplet probe with `--dataset` raises with a message, because an HF
dataset carrying a dense target is a much larger surface (per-probe
target-column plumbing, loader/dtype selection, the four validity conventions).

**`cache_identity` is the method a bridge must not skip, and both get it right by
leaning on index-order immutability.** Return `None` there and every run
re-decodes every image forever while appearing to work — the `view_identity`
failure. A `datasets.Dataset` carries a `_fingerprint` that changes on any
transform, so `f"{fingerprint}|{row}"` names a row's content exactly. A
`torchvision` dataset has no such hash: the `ImageFolder` family
(`.samples`/`.imgs`) uses the file path + size + mtime like
`ImageFolderDataset`, everything else a sha256 of `repr(dataset)` (which states
root, split, download flags) + length + index. The `repr` digest is weaker —
two different downloads with matching reprs would collide — and that is
documented on the class, not hidden. `describe()` adds `dataset_source`
(`"torchvision:CIFAR10"` / `"hf:<name>"`) so a bridge record lands in its own
comparability group rather than merging with a folder board.

**`balanced_subset` moved to `BaseDataset`.** It only needs `labels()` and
`subset()`, both of which the bridges have, so the CLI's per-class `--limit`
works on them for free. `ImageFolderDataset` lost its copy; the method is
otherwise unchanged.

**All three shipped in that cost order**: `examples/custom_backbone.py`
(hours), `visbench show` (the only one that guarded a silently wrong number),
the dataset bridges (largest). The pre-bridge reasoning for each — why a viewer
was the one that guarded a wrong number, what the two tiers of custom-dataset
support already covered, and why `CustomBackbone` needed showing rather than
building — is in `docs/roadmap.md` under "Library surface". What remains is the
candidate-task backlog.

### The candidate and library-surface backlogs, as `CLAUDE.md` summarised them before the third trim

**There is no `next` step.** The remaining work is the candidate task backlog
further down this file — and the cheapest items there need no new dataset at
all. **Two of those are done, and a third was built and rejected.**

**Relative depth ordering was the last cheap candidate and it did not earn a
board** (2026-09-04). The third rejection and **the first for failing to *rank*
rather than for failing to be recoverable**: it cleared the oracle gate at 94.0%
and then reproduced the `depth` board's ordering at Spearman **+1.000**, at 38%
of its spread, with two backbones 0.0007 apart that `depth` separates by 0.0707.
`RelativeDepthTask` is kept **unregistered** and its five records are a control.
The transferable lesson is the gauntlet's floor rule in "decisions already paid
for": **the oracle gate measures a ceiling and nothing measured the floor.**

**Three probes shipped after v0.11.0 and each has a 12-backbone board.** Their
board readings are in [`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md) and their
reference on their own page under `docs/probes/`; what matters here is what
each one *is*:

- **`scene_classification`** (14th, probe 2026-08-27) — scene category on the
  same linear-probe path as object `classification`, on `places365_standard`,
  read with no loader code.
- **`orientation`** (15th) — local gradient orientation, the fourth low-level
  task and the second computed from the frame, but the first whose target is a
  *direction*, so the first that could not reuse `DenseMagnitudeTask`. Target is
  `(cos 2θ, sin 2θ)` with its length set to the coherence; `orientation_error`
  is degrees of coherence-weighted angular error, halved so 45 is chance. It
  reuses `corner`'s pinned `data/corner_frames/` set. **DoG-blob was the first
  candidate for this slot and was rejected** at 0.51 overlap with `corner`.
- **`fine_grained_classification`** (16th) — subordinate category on the same
  path again, on **CUB-200-2011**, the official 5994/5794 split read from
  `vision/CUB-200/images_train_test/`. `probe_fine_grained_classification` runs
  the whole official split with no `--limit`, which is what makes the board
  comparable to the published CUB literature.

Three probes now share one implementation and ask three questions — basic-level,
place, and subordinate — and each is a distinct probe *name* for the reason in
"decisions already paid for", which the second and third instances confirmed
edit for edit.

**Two findings from those boards are load-bearing enough to state here**, both
expanded in `CORPUS_FINDINGS.md`: the two image-level classification probes
rank with the *localised* cluster (`detection`, `semantic_segmentation`) rather
than with the object board they subclass — `fine_grained_classification`
correlates **+0.832 with `detection`** against +0.322 with `classification` —
and `orientation`'s board is **not** independent even though its target is,
ranking like `keypoints2d` (rho +0.95), `corner` (+0.82) and `edge` (+0.79).


**The library-surface backlog is closed** (2026-08-28) — `visbench show`
(9a-9d), `examples/custom_backbone.py`, and the **dataset bridges**
(`TorchvisionDataset` / `HuggingFaceDataset`, plus
`--dataset torchvision:… | hf:…` on the three image-level probes). It shipped
no new number, the way v0.7 did; `docs/roadmap.md` has the public version and
the rules it established are in "decisions already paid for".

**The candidate-task backlog is what remains, and its cheap end is
exhausted.** `fine_grained_classification` came off it (CUB-200-2011);
**photometric superpixels was built and rejected** at 0.021-0.043;
**the gauntlet gained the oracle gate** that rejection was missing
(`scripts/oracle_ceiling.py`, calibrated so the four shipped magnitude targets
pass at 0.53-0.83 and the rejected one fails at 0.25); and **the BSDS500 line
is closed at two steps** (12a-1/12a-2) — the dataset and a validated ODS/OIS/AP
metric ship, reproducing the published human ODS of 0.80 at **0.8030**, and the
probe was **refused by the oracle gate** at a 0.4193 ODS ceiling against
Canny's published 0.60, which removed the only reason to add BSDS rather than
reuse `edge`. Its write-up and the two routes that could reopen it are in
`visbench/tasks/low_level/README.md`. **Instance segmentation on VOC shipped**
(14a-1 to 14a-4, 2026-09-08) as the seventeenth probe and board; it was the
exception to "nothing cheap remains" and is now closed, leaving **no open
candidate line**. Re-confirm what is wanted before starting anything; do not assume
this order is a plan. **The one thread that was open — why `detection`
alone fails to reproduce — is closed**: it is GPU non-determinism made visible
by a discrete metric, it was never a bug, and detection reproduces to *three*
decimals rather than four. **Settled 2026-08-14 on all six backbones**: only
the two 16x16-grid rows (DINOv2) drift; CLIP-B/16, CLIP-B/32 and both ResNets
are bit-identical and match their corpus records to every digit, so it tracks
the feature grid rather than the width, the architecture or the probe. See
[`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md) for the table. **There is no open lead
here any more.**

### Why the fit diagnostics took until schema v8 (2026-08-28)

- **A trained probe records how its fit went, and the reason it took until
  schema v8 is worth keeping** (2026-08-28). Every trained probe computed
  `train_loss` — and the classification family `train_top1` — printed it to a
  log line, and dropped it before the record. So the corpus could not answer
  the one question this file says matters most about a low score: whether the
  probe **underfitted**, which *understates* a backbone, or whether the
  representation genuinely does not carry the answer. Those are opposite
  conclusions from the same number, and the binary-segmentation bullet above is
  the proof — 0.16 IoU at the defaults against 0.87 at `epochs=40`, identical
  features.

  Found by friction, not by audit: the CUB write-up could claim "does not
  underfit" for the six backbones run by hand and could not check it for the
  six run on the cluster, because their `train_top1` existed only in a Slurm
  log. 156 trained records, none of them able to answer it.

  **`training` is a separate field, not entries in `metrics`.** `metrics` is
  what `evaluate()` returned about the *evaluation* split, and every leaderboard
  path reads it; a training number there is one the ranking code can only refuse.
  `DIAGNOSTIC_METRICS` does exist and would have worked, which is why this was a
  real choice rather than an obvious one — it was rejected because it blurs what
  `metrics` means and widens the key-collision surface the `ceiling_` guard
  exists for. **An open dict**, for `task_params`' stated reason: a future
  probe's own diagnostic must not force another bump. **`None`, not `{}`,** for
  the three zero-shot probes — no fit happened, which is a different statement
  from "trained and reported nothing", and it is what every pre-v8 record
  carries by absence, exactly as `finetune` does.

  **Never rank on it.** A probe that fits its training data perfectly has said
  nothing yet about a backbone; on CUB every backbone reaches `train_top1`
  1.0000, including the one that comes last.

### The v8 re-run, and the three cells that did not reproduce (2026-09-10/11)

- **A re-run replaces a published record only where it *reproduces* it**
  (the schema-v8 `training` re-run, 2026-09-10). The corpus is append-only and
  `latest_per_backbone` takes the newest, so re-running a board silently
  republishes whatever the re-run produced. Ninety-three of ninety-six cells
  reproduced; the three that did not are in
  `results/controls/hardware_a100.jsonl`, because merging them would have
  dropped `convnext_base` below `resnet50` on the CUB board — **a ranking
  change caused by a variable no record carries**, since the schema has never
  recorded which GPU produced a number. That is not picking the convenient
  number; it is refusing to let an unrecorded variable move a board.

  **Closed 2026-09-11**: `dgx2` came out of `DRAIN`, the three were re-run on a
  V100, and **all three reproduce their published value exactly** — so they
  merged and every board now carries `training`. The A100 records stay in the
  control as the evidence, because the three-way comparison is the finding:
  same code, same data, same seed, two silicons, and `convnext_base` reaches
  `train_top1` **1.000000** on a V100 against **0.989156** twice on an A100.
  **The disagreement is a fit that does not interpolate, not a metric that
  wobbles** — which is why the fit diagnostics, not the score, are what tell
  the two apart.

### The degraded node, cell by cell (2026-09-10)

- **A degraded node returns plausible wrong numbers, and only the fit
  diagnostics catch it** (same re-run). `dgx2` produced twenty cells before
  entering `DRAIN`; seven of its eight `scene_classification` cells were wrong
  by up to −0.0102, with the same seed, fingerprint and `task_params` as the
  records they disagreed with. Every one of them carried a visibly *worse fit*
  beside its worse score, which is what identified them — and re-running on
  healthy hardware reproduced all twelve exactly. **A saturated board cannot
  reveal this**: `classification` came off the same faulty node bit-identical,
  because top-1 ~0.99 with `train_top1` 1.0 has no margin left to flip. Check
  `training` before believing a re-run that disagrees, and prefer a board that
  is *not* saturated when you want a canary.

### The cluster's nodes, as `CLAUDE.md` recorded them (2026-08-19 to 2026-09-10)

- **`dgx1` was degraded on 2026-08-19 and it does not fail like a broken node**
  (found while running 10c). It accepts work and starves it: `import torch`
  spends 1-1.6 s *per submodule* there and never finishes inside 300 s, while
  dgx2 imports the same venv in 2.4 s. Nothing in Slurm reports it unhealthy, so
  a job hangs with an empty log and the first read is that your new code is
  broken. **Submit with `--exclude=dgx1`**, and when a job on this cluster hangs
  with no output, time an `import torch` on the node before suspecting the code.

  **"Never finishes" was a floor, not a ceiling** (measured 2026-09-10 by giving
  it three hours instead of five minutes): `import torch` returns in **1376.7 s**
  there — 23 minutes, against ~2 s on dgx2 — so the node is not hung, it is
  uniformly ~600x slow. That is worse news than a hang, because the *work* is
  slow too: the same job then spent **4.5 hours on one CUB cell** that takes 80 s
  on `dgxa100`, and timed out having written no record. So dgx1 is unusable for
  anything real, the exclusion stands, and the reason to state the number is
  that "it hangs" invites someone to retry with a longer walltime, which is
  what this was and it still did not finish.

  **The venv is NOT limited to `dgx1`/`dgx2`, which this file claimed until
  2026-09-10.** `dgxa100` runs Ubuntu 24.04 *and* ships `/usr/bin/python3.10`
  beside 3.12, so `.venv` resolves and runs there unchanged — checked by
  importing torch and visbench on the node, not inferred from the OS version.
  **`dgxh100` is the opposite case and cannot run it**: it ships
  `/usr/bin/python3.12` and *no* 3.10, so `.venv/bin/python` is a dead symlink
  there and a job dies in one second with "cannot execute: required file not
  found". Its `--qos=quick` refusal (a misleading `QOSMaxGRESPerJob`, fixed by
  `--qos=normal --gres=gpu:1`) is a *scheduling* obstacle in front of that, and
  clearing it only buys the right to fail on the node — which is why the QoS
  fix is not evidence the node is usable, and why this bullet said so for a day
  before the probe actually landed. **So the usable set is `dgx1`, `dgx2` and
  `dgxa100`**, of which dgx1 is degraded. That matters because **`dgx2` can go
  `DRAIN` mid-run** (it did, "Kill task failed"), leaving `dgxa100` as the only
  healthy node — and it is the one whose silicon differs. An A100 has
  TF32 where a V100 has none, so see the reproducibility entry in
  `CORPUS_FINDINGS.md` before putting corpus records on one: measured, TF32
  moves a dense board by ~1e-6, but three `fine_grained_classification` cells
  do not reproduce across the two.

  **Do not read `torch.backends.cudnn.allow_tf32` as evidence of the
  hardware.** It is `True` by default and reads `True` on a **V100** too, where
  there is no TF32 unit for it to enable — checked on dgx2. The flag says what
  PyTorch would permit, not what the silicon can do, so the A100/V100 question
  is settled by `get_device_name` or by measuring the effect, which is what the
  reproducibility entry does.

### The corner probe's three pre-measurements (8a)

- **A derived target is the cheapest kind to add and the easiest to fool
  yourself with** (8a). Corner detection computes its target from the RGB frame,
  so it needs no dataset — and three things had to be measured before it was
  worth shipping, none of which a probe run would have revealed. The numbers are
  in `visbench/tasks/low_level/README.md`; the rules are:

  **Check the tail, before writing the task.** Every raw corner response was
  more concentrated than `edge_occlusion`'s 0.46 — the case that scored 0.088
  and ranked nothing. `log1p(1e4·λ_min)` brings it to 0.089, and **Shi-Tomasi
  rather than Harris** because λ_min is non-negative by construction and has no
  `k`.

  **Check the overlap with what already ships**, which nothing previously asked
  for. The corner target correlates **0.52** with `edge_texture` where that and
  `keypoints2d` correlate 0.147 with each other — so the new target is more
  redundant with an existing one than the two existing ones are. The overlap is
  *intrinsic*, holding across eight transforms: a corner is a pixel whose
  gradient is large in two directions and an edge map is gradient magnitude. A
  first pass blamed the `log1p` and was wrong.

  **A correlated target can still rank differently, and that is the criterion.**
  Spread over six backbones 0.1603 against edge's 0.1136, with CLIP-B/16 first
  on edges and third on corners. Had the ordering matched, it should not have
  shipped. **Do not read one pair as a failure to rank** — DINOv2-S and B differ
  by 0.0014 here, which looks like the occlusion-edge failure and is not; ask
  about the spread over the full set.

  **Computing the target after the crop deletes the alignment hazard rather
  than testing for it.** No second geometry, no resampling of the response —
  the strongest property of this class of target, and why
  `DerivedTargetDataset` does not subclass `DenseFolderDataset`.

### The DoG-blob veto and the orientation target that replaced it (2026-08-28)

- **The overlap check is a veto, and `orientation` is the probe that proves it
  earns its keep** (2026-08-28). DoG-blob detection was the obvious next derived
  probe — the scale-space counterpart to `corner`. Its pre-measurement (the same
  afternoon of correlations `corner` established): tail@1% ≈ 0.084 raw, so *no
  compression needed*, which passed. But per-image `|r|` with `edge_texture` was
  0.50 and **with `corner` 0.51** — as redundant with an existing probe as
  `corner` is with `edge`. Rejected without a probe run: the check exists so you
  do not spend a per-backbone board to discover redundancy.

  **Structure-tensor orientation was the alternative and it pre-measures
  clean.** `|r|` under 0.09 with both `corner` and `edge`, because it measures
  *phase* and no other probe does. Its target is a *direction* — the unit vector
  `(cos 2θ, sin 2θ)`, the angle taken mod π so the double angle handles the wrap
  — with the coherence `(λ_max−λ_min)/(λ_max+λ_min)` folded into its length. So
  it is the first derived probe that could **not** reuse `DenseMagnitudeTask`:
  it needs a 2-channel L2-normalising `_activate`, a coherence-weighted angular
  `_loss`, and `orientation_metrics` (degrees, `orientation_error` halved so 45
  is chance). Coherence is a **weight, not a mask** — only 1.4% of Taskonomy
  tiny val pixels fall below 0.1 — folded into the target length exactly as a
  zero-length normal marks an invalid pixel, so the loss and metric both weight
  by `target.norm(dim=1)`. An angle has **no tail**, so the compression `corner`
  needed is absent here; the pre-measurement confirmed that before the task was
  written. Proved end to end on DINOv2-S: `orientation_error` 35° against the
  45° floor on 40 training frames. The 12-backbone board is the next step and
  reuses `corner`'s pinned `data/corner_frames/` set.

  Viz: `orientation` is drawn in **colour**, not greyscale — a new `"orientation"`
  `Kind` whose `_orientation` colouriser maps `2θ` to hue and coherence to
  brightness (inline HSV→RGB, no new dependency), so a flat patch reads as black
  rather than a confident wrong colour.

### The corpus claims, as `CLAUDE.md` stated them before the third trim

- **What the corpus says is in
  [`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md), and you must read it before
  quoting any board.** Eight findings moved there on 2026-08-20 when this file
  passed the context limit it is loaded under. The claims are below; the
  evidence, the numbers and the readings each one corrected are in that file.
  `scripts/analyse_board_correlates.py` reproduces the correlational ones.

  - **"Which backbone is best" is not a well-formed question against this
    corpus.** `mae_vitb16` is first on six of the sixteen boards and last on
    four. A summary that picks a winner is discarding the result.
  - **A count over a corpus is a fact about that corpus, not about a
    backbone.** Three of MAE's counts have now moved without its features
    changing — twice because a column was added, once because a *board* was.
    Re-read counts off `LEADERBOARD.md`.
  - **Quote an objective gap against the *recipe* gap on the same board, never
    against zero.** `sam_vitb16` and `supervised_vitb16` share architecture,
    data, labels and normalisation and differ only in training recipe; on seven
    of the thirteen boards that gap is more than a third of the whole objective
    spread. Nothing in a record says which board you are on.
  - **The semantic segmentation board separates neither training objectives nor
    feature resolution**, which every other dense board ranks by. Do not
    present it as evidence about an objective.
  - **The high-level tier is two clusters, not one** — `classification`/
    `retrieval` (image-level categorisation) and `detection`/
    `semantic_segmentation`/`scene_classification`/`fine_grained_classification`
    (localised / spatial-context prediction) — that barely correlate with each
    other. **Two probes that are mechanically object classification with a
    different folder both land in the *localised* cluster**, which is the
    replication that makes this a property of the cluster rather than a fact
    about Places365: `fine_grained_classification`'s strongest partner in the
    whole corpus is `detection` at **+0.832**, against +0.322 with the object
    board it subclasses. The tier-mean-vs-
    cross-tier sign has flipped both ways with corpus composition (below the
    line at 13 boards, marginally above at 14) and is noise; the two clusters
    are the stable finding. `scene_classification` is image-level classification
    yet lands with the localised cluster (+0.72 with `detection`, −0.22 with
    `retrieval`). Treat `high_level` as a folder, not a quantity to average
    over. Mid- and low-level cohere. This is not the taxonomy being wrong.
  - **That clustering is not a shared-dataset artefact**, checked: the two
    boards reading the *same 1449 images* agree least of the three VOC pairs,
    and Imagenette's three probes average +0.128.
  - **Quote `detection` to three decimals, not four**, and treat
    `clip_vitb16`/`clip_vitb32` as **tied**. It is GPU non-determinism a
    discrete metric can see, and there is nothing to fix. Two corrections from
    the v8 re-run: the drift is not confined to the two 16x16-grid backbones —
    every 14x14-token ViT-B/16 moved too — and the claim that the two CLIP rows
    were "verified rather than lucky" is refuted, because they swapped. Their
    gap is 0.0001 on a board that drifts by more than that.
  - **Two backbones' high-level scores are close to in-distribution recall**,
    not transfer: `convnext_base` and `supervised_vitb16` are ImageNet-1k
    supervised and Imagenette's classes are ImageNet-1k wnids.
  - **Feature resolution is the strongest correlate of nearly every dense
    board, and it is not what DINOv2's lead is made of.** Nine of the eleven
    grid-reading boards, after the 2026-09-13 tie fix corrected three published
    coefficients; `semantic_segmentation` and `detection` are the exceptions,
    and a board's *fit* tracks the grid too (11 of 11, mean rho −0.681).
    **Every published board-*pair* number survived that fix unchanged.** Holding weights fixed and
    cutting DINOv2-B from 256 to 196 tokens costs under 3% on all five dense
    boards, and it keeps its lead over the whole ViT-B/16 pack on both boards
    it led — 21% of the `generic_segmentation` gap and 7% of the `depth` one.
    On the other three boards DINOv2-B never led, so there was nothing to
    explain. **Check who leads a board before explaining their lead.**
  - **The `depth` board is not ranking by metric accuracy.** A readout that
    discards scale and shift entirely — never supervising or scoring them —
    reproduces its ranking at Spearman **+1.000** over five backbones, so the
    board ranks *ordering plus feature resolution* and reports it in metres.
    Not a defect: it reproduces probe3d's protocol, which is why its numbers
    compare to anything. See the relative-depth control.
  - **A control is rankable and still must not be listed beside the corpus.**
    `results/controls/` holds records that pass `comparability_key` against
    their board and answer a different question from it — the corpus says what
    a backbone scores, a control says what changes when one thing about one
    backbone moves. Nothing there feeds a generated table.
  - **n=12.** Every correlation above has wide error bars.

### The build table's v0.1-v0.3 rows, one line each

| Step | What | Status |
| --- | --- | --- |
| 1 | Scaffold every folder and module, docstrings + stubs, no logic | done |
| 2 | `BaseBackbone` + feature cache + DINOv2, with tests | done |
| 3 | `BaseTask` + one task (retrieval) end to end on a local folder | done |
| 4 | All three v0.1 tasks, both v0.1 backbones, `uv.lock`, `run()` | done |
| 5a | ResNet/timm backbone — first non-ViT, validates the CNN half | done |
| 5b | Custom `nn.Module` backbones, and pluggable heads (linear + DPT) | done |
| 5c | Multi-layer extraction through every backbone and the cache | done |
| 5d | Depth estimation — first dense task, full probe3d protocol | done |
| 5e | Streaming features from disk, for splits larger than memory | done |
| 5f | Surface normals + the shared `DenseTrainingTask` | done |
| 5g | Generic (binary) segmentation | done |
| 5h | High-level semantic (multi-class) segmentation | done |
| 5i | Mid-level image similarity | done |
| 5j | The CLI — last, once the dense-task Python API has settled | done |
| 6a | Fine-tuning: unfreeze last N blocks, cache out of the path, DINOv2 only, proved on VOC segmentation | done |
| 6b | Cache the frozen prefix — works, saves 21%, and found the real bottleneck | done |
| 6c-1 | Detection: the box dataset and VOC loader | done |
| 6c-2 | Detection: `average_precision`, mAP@50, mAP@50:95 | done |
| 6c-3 | Detection: the head, against a metric already trusted | done |
| 6d-0 | Dataset listing: `scandir`, not a stat per file | done |
| 6d-1 | Edge detection — the first low-level task, on Taskonomy | done |
| 6d-2 | `mask_valid`, keypoints2d + occlusion_edge, `DenseMagnitudeTask` | done |
| 6e-1 | Leaderboard: the comparability rules, as pure functions | done |
| 6e-2 | Leaderboard: regenerate a record corpus for all twelve probes | done |
| 6e-3 | Leaderboard: render it, and generate the README tables from records | done |
| 6e-4 | Hub: serialise a trained head, with the backbone identity beside it | done |
| 6e-5 | Hub: push/pull through `huggingface_hub`, behind a `[hub]` extra | done |
| 6f | Correspondence: score in pixels — the unit that inverted the board | done |

### The candidate-task backlog's dataset survey, in full (checked 2026-08-01)

`docs/roadmap.md` has the public version of this list, grouped by cost — it was
in the README until 7b moved it. What follows
is the part a contributor cannot see: **which of these have data on this
machine**, checked on 2026-08-01 rather than assumed. A candidate whose dataset
is absent is not cheap, however simple its protocol.

**`/shared/sets/datasets/` has a `vision/` subdirectory, and a top-level listing
does not see into it.** 96 more datasets live there, including ones a first pass
recorded as absent. Check both levels before concluding anything is missing —
this note exists because the first version of this section did not, and said
Places365 and NIGHTS were absent when both are on disk.

**Verified present at the top level:** `ADE20K` (`ADEChallengeData2016`), `COCO`
(`annotations/` has `instances_*`, `captions_*`, `person_keypoints_*` — **no
panoptic and no stuff**), `cub_200_2011`, `stanford_cars`, `stanford_dogs`, many
ImageNet variants, `Imagenette`.

**Verified present under `vision/`:** `nights` (`data.csv`, `ref/`, `distort/` —
this is what the `similarity` probe reads), `places365_standard` (`train/`,
`val/`, `categories_places365.txt`), `SUN397`, `mit67_indoor_scenes`,
`caltech101`, `country211`, `CUB-200`, `oxford_flowers102`. **Scene
classification was a dataset-swap on the existing linear-probe path** and
shipped 2026-08-27 as the `scene_classification` probe on `places365_standard`
(a new probe *name* rather than a flag — see the "decisions already paid for"
bullet). Its 12-backbone corpus board landed 2026-08-28. **Fine-grained
recognition shipped the same way on 2026-08-28** as
`fine_grained_classification`, on **CUB-200-2011** — and the copy to use is
`vision/CUB-200/images_train_test/`, which already holds the official
5994/5794 split as `train/<class>/` + `val/<class>/`. Two traps in that
directory: the top-level `cub_200_2011/CUB_200_2011` is **permission-denied**,
and `test/` is a **symlink to `val/`**, so naming `val` is naming the official
test set and `--split test` would index the same files under a different path
and so a different fingerprint. Stanford Cars (`train_cars`/`test_cars`, 196
numeric class dirs) is the same folder shape and still open; Stanford Dogs and
Flowers102 are **not** — both keep their splits in `.mat` files and so need
loader code, which is a different cost class from a folder swap.

**Verified absent, both levels:** any optical-flow set (Sintel, KITTI,
FlyingChairs), NYUv2, any intrinsic-image set (IIW, SAW, MIT intrinsic).
`bsds300` is still the MAF density-estimation benchmark, not BSDS500 (its
`bsds300.hdf5` sits beside `gas` and `hepmass`) — see 6d-1. **BSDS500 itself is
no longer absent**: Berkeley is unreachable from this machine (`www2.eecs`
times out, the old host 403s) while the network is otherwise fine, so
`scripts/fetch_bsds500.py` reads the `BIDS/BSDS500` GitHub mirror at a pinned
commit into gitignored `data/bsds500/`.
`davis` exists but holds two sequences of derived output (`dpt/`,
`epipolar_error*`), not the DAVIS annotations, so it is not a video-segmentation
benchmark.

**The Taskonomy copy on disk carries eight domains only**: `depth_zbuffer`,
`edge_occlusion`, `edge_texture`, `keypoints2d`, `keypoints3d`, `normal`,
`principal_curvature`, `reshading`, plus `rgb` and `mask_valid`. Taskonomy
*publishes* `vanishing_point`, `room_layout`, `segment_unsup2d/25d` and
`point_matching`, and none of them are here. So the roadmap items that look like
free Taskonomy wins — vanishing points, room layout, superpixel segmentation —
each need a download first, and are not in the same cost class as 6d-1 and 6d-2
were.

**The cheapest items on the list need no dataset at all, and that is the useful
observation.** `edge_texture` is a target Taskonomy *computed from the RGB
frame*, and so are these. **`corner` (8a) and `orientation` (2026-08-28) are
done**; **DoG blobs was rejected** for overlapping 0.51 with `corner`;
**photometric superpixels** is the one that remains derivable from any image
folder already here. A magnitude target is a generator plus a
`DenseMagnitudeTask` subclass; a vector one (`orientation`) needs its own small
task base, which `visbench/tasks/low_level/orientation.py` now provides as the
second worked example.

Three hazards to carry into any of them, all paid for:

- **Check the tail before assuming the magnitude protocol transfers.** A corner
  response is spikier than an edge response, and `edge_occlusion` at 46% mass in
  its strongest 1% of pixels is the case where L1 and Pearson pull apart and the
  probe stops ranking backbones. (An *angle* has no tail — `orientation` needed
  no compression, confirmed by the pre-measurement.)
- **Check the overlap with what already ships, before building.** DoG blob was
  vetoed on this: 0.51 with `corner`. `orientation` passed it: under 0.09 with
  both `corner` and `edge`, because it measures phase. One afternoon of
  correlations, not a probe run per backbone.
- **A derived target is only as honest as its generator, and `protocol` must say
  which generator.** "Harris corners" is a family, not a definition — the
  k parameter, the window, the smoothing and the non-maximum suppression all
  move the target. A record claiming a bare `"harris"` says less than it looks.

### The Hub push paragraphs, with the collection script's two limits

**`visbench run --push-to REPO_ID` publishes the head it just trained**
(`--public` overrides the private default), and `scripts/build_corpus.sh` takes
`PUSH_TO` / `PUSH_PUBLIC` so a whole board publishes from the file that already
holds every probe's flags. **Publishing from the run, not a second script, is
the design**: a head is only meaningful against the features it was fitted on
and the run's flags are what fitted them, so a separate publish step is a second
copy of every dataset flag, free to drift — and a head trained under drifted
flags uploads, loads and scores without complaint. The CLI refuses a zero-shot
probe *before* the run rather than after spending it.
`scripts/publish_collection.py` groups the pushed repositories into one
collection, dry-run unless `--create`; its two Hub limits (a 150-character
description cap, and needing `collection.write` rather than `repo.write`) each
cost an attempt and are asserted or recorded in the log.

**Twenty trained heads are published and public**, as of 2026-08-07: ten
probes against DINOv2-S/14 and DINOv2-B/14, one repository per pair at
`turhancan97/visbench-<probe>-<backbone>`, in a collection whose URL is quoted
in `README.md` and `docs/guides/sharing.md` — **read it from one of those two
files rather than reconstructing it**, since a Hub collection slug carries a
generated hash suffix. **That is ten of the fourteen probes that train a head,
not all of them**: `scene_classification`, `fine_grained_classification`,
`orientation` and `instance_segmentation` all shipped after the push and have
no published head. The three zero-shot probes are deliberately absent, which is
a different reason.

Republishing the board is `PUSH_TO=... PUSH_PUBLIC=1 scripts/build_corpus.sh`.
**Point `RESULTS=` at a scratch file, never `results/corpus/visbench.jsonl`**,
so the run can be diffed against the corpus instead of replacing the reference
it would be checked against — that diff is the only reason the seeding bug below
was ever found. And **do not pipe a long publishing run through `tail`**: it
buffers, so a run killed part-way leaves no log and the Hub has to be queried to
find out what shipped, which happened and was recoverable only because each
record names its own pair.

### The ceilings, and the re-run that put them in the corpus (2026-09-01)

- **A ceiling travels with its score, through `BaseTask.context_metrics` — for
  correspondence and, since 2026-09-01, for every dense probe that declares an
  oracle.** A match can only land on a patch centre, so a
  coarse grid has a hard floor on achievable precision: `ceiling_recall@5px` is
  ~0.10 on a 7x7 grid against ~0.41 on a 16x16 one. The score alone therefore
  says the wrong thing. Measured on 200
  Imagenette pairs, DINOv2-S: `recall@5px` 0.3049 against a ceiling of 0.4123.

  **The dense probes have the same problem and now say so.** The head reads one
  feature vector per patch, so part of every dense target is out of reach before
  the backbone is chosen, and how much varies by *backbone*: `corner`'s ceiling
  is 0.8316 on a 16x16 grid and 0.6685 on a ResNet's 7x7. Ranking those two
  against each other silently invites a reader to attribute a grid difference to
  a representation. `edge`, `keypoints2d`, `occlusion_edge`, `corner` and
  `orientation` emit `ceiling_*`; every other dense probe still returns `{}`,
  because pooling a class-index or bin-expectation target is meaningless.

  Everything else is unchanged: `run()` refuses a context key that
  collides with a score, since they share one flat dict, and both prefix
  `ceiling_`. **Never rank or average on a ceiling** — it says what was
  available, not what was recovered, and since it falls with the grid, ranking
  on it would rank feature resolution directly.

  **The corpus carries them since 2026-09-01.** The five boards were re-run —
  60 records, every value produced by a run rather than backfilled, which is the
  distinction that matters: a number in a record no run produced would be a
  fabrication however easy it is to compute. The old lines stay (the corpus is
  append-only) and `latest_per_backbone` picks the new ones, so the file is 252
  lines for 16 boards x 12 backbones. Those records also gained the schema-v8
  `training` block, because they predated it. Schema is untouched by the
  ceilings themselves: they are keys inside `metrics`, not a new field.

  **Four of the five boards reproduced to ~1e-7 relative; `orientation` did
  not.** See its entry in [`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md) — its
  metric is ill-conditioned and two of its rows are not separable.

### The pinned corner frame set, and the staging that had no records (8a/8b)

- **A probe that runs on any folder cannot have a leaderboard without a chosen
  folder** (8a). This is the cost of a derived target and it is not obvious from
  the API: two people's corner numbers are comparable only if they ran the same
  images, and nothing in the probe pins which. **The set chosen is Taskonomy
  tiny, the first 600 rows of each split list — the same frames `probe_edge`
  reads** — and `scripts/stage_corner_frames.py` is what makes them readable,
  symlinking the building-nested RGB frames into the flat `<split>/images/`
  layout `DerivedTargetDataset` expects. `build_corpus.sh` skips the probe with
  an actionable message if that folder is absent, as it already did for
  `generic_segmentation`'s binarised masks.

  **Shared frames are the point, not a convenience.** The corner target
  correlates 0.52 with `edge_texture`; the claim that earns the probe its place
  is that the two nonetheless rank backbones differently. That is exact only on
  identical pixels, so the staging is verified **set-equal** to the edge
  probe's 600 rather than assumed equal.

  **Symlinks, not copies**: `cache_identity` keys on path, size and mtime, and a
  symlink reports its target's, so a staged frame and the original share one
  feature-cache entry instead of doubling the cache.

  The cost of getting this wrong was demonstrated rather than argued: 8a's
  numbers were produced on an ad-hoc staging that was never committed, so the
  six published figures had **no surviving records** — 6e-2's exact failure,
  recurring on the newest probe two steps after that step ended it. The
  regenerated corpus reproduces all six to four decimals, which is what
  retired the hand-written table.

### TimmBackbone's three silently-wrong-number decisions (10a)

- **`TimmBackbone` reads a model's own structure; it used to assume a CNN's**
  (10a). `has_cls_token` and `patch_size` were *class* attributes declaring
  "CNN" for everything, so timm ViTs were refused outright — and a false
  `has_cls_token` discards the CLS token while the record claims there was none
  to keep. Read per instance from `num_prefix_tokens` and `patch_embed`, any
  timm ViT becomes usable *and honest*, which added ConvNeXt-B, MAE ViT-B/16
  and SigLIP-GAP ViT-B/16 in one change rather than three.

  **`default` pooling is read from timm's `global_pool`, not inferred from
  whether a CLS token exists.** "CLS if there is one, mean otherwise" is only a
  proxy: MAE reports `token` and SigLIP-GAP reports `avg`, so `default` means
  different things for two models of identical shape — each matching what the
  model hands its own classifier.

  **SigLIP is the `_gap_` variant deliberately.** Canonical SigLIP pools with an
  `AttentionPoolLatent` (`global_pool='map'`) — a *trained module*, not a
  reduction over tokens, so it cannot be a pooling mode over cached features.
  `describe_transformer` refuses `map` by name. Do not "add a map mode" without
  first deciding a pooling mode may carry weights.

  **ConvNeXt breaks the "pooled is what the model hands its classifier" rule,
  and the exception is documented rather than smoothed over.** Its head is
  `avg -> LayerNorm2d`, so the model's vector is `norm(mean(x))` where this
  class returns `mean(x)` — max absolute difference 27.5 on one frame. Both
  invariants cannot hold, and the one kept is structural: **`pooled` is always a
  reduction of `dense`**, because the cache stores dense features and every
  pooling task reduces them. A test pins which four backbones match their own
  head and that ConvNeXt does not, in both directions.

  **The guards have fast tests, which is why `describe_transformer` is a
  module-level function.** Every timm backbone test needs real weights and is
  `slow`, which CI does not run; these three decisions each produce a silently
  wrong number rather than an error, so the logic takes a stub.

### The optional-extra trap, all four instances

- **The optional-extra trap has now been hit three times, by the same person.**
  v0.6.0's hub tests needed `huggingface_hub` at monkeypatch time and CI
  installs `.[dev]` only; 7c's issue-template test needed PyYAML, present
  locally via timm and absent from `.[dev]`; **13a's docstring guard needed
  `sphinx` and `docutils`**, which are in the `docs` extra, and it reached a
  pull request — seven red tests on both Pythons after all six local checks
  were green. The second was caught *before* pushing by blocking the import the
  way `CONTRIBUTING.md` documents — the `find_spec` recipe there, and in 6e-5's
  section of the engineering log. **The third was not, and the reason is worth
  keeping: the docs build had been run and passed, which felt like it covered
  Sphinx.** It does not — that build installs `.[all,docs]`, and the *test*
  suite does not. Run the blocker whenever a fast test touches `clip`, `timm`,
  `hub`, `yaml`, `datasets` or `sphinx`; the six verification commands cannot
  catch this, because they run in the environment that has everything. PyYAML,
  `datasets` and `sphinx` are all declared `dev` dependencies now.

  **Declare it; do not skip it.** A `pytest.importorskip` would have made the
  suite green and left the guard uninstalled in exactly the environment that
  gates every pull request — the `slow`-only failure with a different label.

  **A fourth instance, and it is not a package: `.venv/` itself** (2026-09-10).
  Four tests running `slurm/corpus.sbatch` pointed `VISBENCH_REPO` at the real
  repository, so the script's "Not a VisBench checkout with a .venv" guard
  fired on CI — which installs into the runner's own environment and has no
  `.venv/` — while passing locally. The family is wider than optional extras:
  **a test must not depend on anything that exists because of how this machine
  is set up.** The fix is a stub the test builds itself (a directory holding
  `pyproject.toml` and an empty `.venv/`), which is also what makes it a test
  of the script rather than of the checkout.

## v0.1 and v0.2 — the completed scope, as those releases recorded it

Lifted out of `CLAUDE.md` on 2026-09-03, when that file passed the
150k-character limit for the second time. Both releases are complete and their
boundaries no longer constrain new work; the rules they established that still
do are in `CLAUDE.md` under "decisions already paid for".

### v0.1 — prove the abstraction — **COMPLETE**

Kept for the record; the boundaries below applied to v0.1 only and no longer
constrain new work. Dense-prediction training loops are v0.2 scope and exist.

**Hard boundary (v0.1 only): no fine-tuning, no dense-prediction training
loops. Every v0.1 task either needs no training (zero-shot) or trains a linear
layer on cached features.**

- **Backbones**: DINOv2 (ViT-S/B) and CLIP (OpenCLIP ViT-B) only. No
  ResNet/timm, no custom-backbone support yet — that's v0.2.
- **Tasks**:
  - High-level image classification — linear probe on cached pooled features.
  - High-level image retrieval — zero-shot, cosine similarity over cached
    pooled features (CLS default for ViT backbones).
  - Mid-level geometric correspondence — zero-shot, dense feature matching
    (conceptually reusing matching logic familiar from vismatch (https://github.com/gmberton/vismatch), applied to
    raw backbone features instead of dedicated matcher networks).
- **Required infrastructure before any task code**: reviewed folder skeleton
  → `BaseBackbone` with dual pooled+dense output → feature cache → `BaseTask`
  abstraction → structured JSON result logging (see below), from the very
  first task, not retrofitted later.
- **Explicitly deferred**: CLI, custom backbones, ResNet/timm, multi-layer
  extraction, any dense-prediction task, fine-tuning.

---

### v0.2 — dense mid-level tasks + broader backbone support — **COMPLETE**

- [x] ResNet/timm backbones and user-supplied custom-backbone support
      (arbitrary `nn.Module` + preprocessing function).
- [x] **Pluggable task heads**, never hardcoded to one architecture: `linear`
      and `dpt` ship, `register_head` is the extension point. A head declares
      which feature modes it consumes and rejects a mismatch at construction.
      `DPTHead` refuses a single feature map rather than duplicating it.
- [x] Multi-layer feature extraction, now that the single-layer path is proven.
- [x] Depth estimation, surface normal estimation — probe3d's protocols used
      directly rather than re-deriving metrics.
- [x] **Generic (binary) object segmentation** — `GenericSegmentationTask`,
      sigmoid + masked BCE + `binary_iou` (foreground IoU, Dice, pixel
      accuracy). `load_mask` reads 0/1 or 0/255 as "non-zero is foreground" and
      never rescales; `ignore_index=` maps a dataset's ignore value to -1.
      `DenseFolderDataset` needed no change, but **do not pass `max_target` for
      a mask** — it would erase the foreground class.
- [x] **High-level semantic (multi-class) segmentation** —
      `SemanticSegmentationTask`, cross-entropy over class indices with a
      logit-passthrough `_activate`, reporting mIoU both ways plus pixel and
      mean class accuracy. `num_classes` is required, since a wrong one does
      not raise. The base gained `target_dtype` (the class-index target is the
      one that is not a float measurement) and `DenseFolderDataset` gained
      `stems=` for official split lists. `load_label_map` reads palette PNGs
      without conversion. Proved on Pascal VOC 2012 val at 224px with a linear
      head and the default schedule: DINOv2-S/14 **0.732 mIoU**, DINOv2-B/14
      **0.753** — the ordering you would hope for, which is itself a check that
      the probe measures something.
- [x] **Mid-level image similarity** — `MidLevelSimilarityTask`, zero-shot 2AFC
      over pooled features, kept separate from high-level retrieval as the
      task-categorization note requires. Proved on NIGHTS (1,824-triplet test
      split): DINOv2-S/14 **0.870**, DINOv2-B/14 0.858, CLIP-B/16 0.828,
      ResNet50 0.827.
- [x] **The CLI** — `visbench list`, `visbench run <probe>`,
      `visbench cache stats|clear`, a thin wrapper over `visbench.run()` with a
      `ProbeSpec` table supplying the dataset construction `run()` cannot know.
      Every probe is a subcommand with only its own flags. Proved on real data
      against the numbers the Python API already produced: NIGHTS similarity
      **0.8701** (identical to 5i's) and VOC val semantic segmentation
      **0.733 mIoU** against 5h's 0.732.
- [x] **`run()` covers correspondence.** The open question from v0.1 —
      how pairwise extraction is expressed — is answered by `uses_pairs` plus
      `PairViewDataset`: flatten to `2N` single images, regroup by index, leave
      the cache alone. `--stems` was added alongside so a dense probe can take
      an official split list, without which the CLI could not run VOC at all.
