# Changelog

All notable changes to VisBench are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

Each released section is written to be pasted straight into a GitHub release,
so it stands on its own rather than assuming you have read the ones above it.

## [Unreleased]

### Added

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

Still to come in v0.2: surface normals, generic and semantic segmentation,
mid-level similarity, CLI.

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
[Unreleased]: https://github.com/turhancan97/VisBench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/turhancan97/VisBench/releases/tag/v0.1.0
