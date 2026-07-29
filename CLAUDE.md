# VisBench

Unified library for probing vision backbones (DINOv2, CLIP, custom) across
high-level, mid-level, and eventually low-level computer vision tasks, through
a `get_backbone()` / `get_probe()` API. Sibling project to vismatch (image
matching - https://github.com/gmberton/vismatch), same ergonomic philosophy, applied to representation probing
instead of matching.

**Distribution**: this ships as a pip-installable Python package on
[PyPI](https://pypi.org/) (`pip install visbench`), not just a research repo.
Packaging conventions (pyproject.toml, semantic versioning, a lockfile) apply
from v0.1 onward, not bolted on later.

**Prior art to credit explicitly, not re-derive**:
- [probe3d](https://arxiv.org/abs/2404.08476) (El Banani et al., CVPR 2024) —
  reuse its evaluation protocols for depth, surface normal, and correspondence.
- Chen, Marks & Cheng, ["Probing the Mid-level Vision Capabilities of
  Self-Supervised Learning"](https://arxiv.org/abs/2411.17474) — the task
  categorization below follows this paper directly.

---

## Critical: build order — read this before writing any code

Do not implement all tasks, backbones, and versions in one pass. This is a
multi-month roadmap; **each session completes one step and stops for review**,
not racing ahead. If asked to "build VisBench" or "continue", re-confirm which
step is next rather than attempting the whole roadmap in one session.

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
| **6c** | **Detection groundwork — dataset and metric first, head second** | **next** |
| 6d+ | Low-level tasks if there is bandwidth | later |

---

## Current state

**v0.1 and v0.2 are both complete** — every task, both backbone families, and
the CLI. Everything below exists, is tested, and is on `main`. **v0.2.0 is
released on PyPI** — `pip install visbench` works.

**v0.3 is in progress: steps 6a (fine-tuning) and 6b (prefix caching) are both
done and unreleased.** Dense probes take `finetune_blocks=N` /
`--finetune-blocks N`, DINOv2 only, recorded under schema v6's `finetune` field.
Proved on VOC at two scales: 0.7758 against the frozen 0.7328 on DINOv2-S, and
0.7992 against 0.7533 on DINOv2-B. 6b caches the frozen blocks below the cut in
a separate `PrefixCache`, cutting a fine-tuned ViT-B run from ~345 s to ~272 s
with the mIoU unchanged to four decimals. The next build step is **6c**.

Registered names — `visbench.list_backbones()`, `list_probes()`,
`visbench.heads.list_heads()`:

```text
backbones  dinov2_vits14, dinov2_vitb14, clip_vitb16, clip_vitb32,
           resnet18, resnet50            (+ CustomBackbone, unregistered)
probes     classification, retrieval, correspondence, depth, surface_normal,
           generic_segmentation, semantic_segmentation, similarity
heads      linear, dpt
```

The CLI exposes all eight probes: `visbench list`, `visbench run <probe>`,
`visbench cache stats|clear`. A test asserts the CLI's table and
`list_probes()` are the same set, so a probe cannot ship unreachable from a
shell by accident.

Package version is `0.2.0`, tagged `v0.2.0`, live on
[PyPI](https://pypi.org/project/visbench/). **Publishing needs the maintainer's
credentials and is theirs to run** — never attempt it. A version number on PyPI
can never be reused, so anything that renders wrong ships until the next
release: anything wrong in the README ships with it. Two separate checks cover
that, and neither replaces the other. **Rendering** is CI's `build` job —
`twine check dist/*` runs `readme_renderer`, which is what PyPI itself uses, so
a README that fails to render cannot reach a tag. **Relative paths** are
invisible to that check: they are valid markdown, render without complaint, and
merely point nowhere once the page is served from `pypi.org` rather than
GitHub. `tests/test_readme.py` is the guard for those, in the fast suite —
every link and image must be absolute. Do not "tidy" one back to relative;
point it at `.../blob/main/...`, or `raw.githubusercontent.com` for an image.
Result schema is at **v6** (`finetune` added in 6a; `dataset_params` was 5j) and
is **additive only**: never remove or repurpose a field, or old records stop
being readable.

### Layout worth knowing before editing

```text
visbench/
  backbones/     base.py (resolve_layers, _assemble), dinov2, clip,
                 timm_backbone, custom, pooling.py (feature modes)
  cache/         feature_cache.py (_Plan/_walk, extract_dataset, materialise)
                 streaming.py (CachedFeatures — a torch Dataset over the cache)
                 prefix_cache.py (PrefixCache — frozen prefixes, 6b; nests in
                   _prefix/ under the same root and is never counted as features)
  cli/           main.py (build_parser + the three commands),
                 datasets.py (ProbeSpec table: flags -> datasets, per probe)
  data/          image_folder (+ balanced_subset), pair_dataset
                 (PairDataset, HomographyPairDataset, PairViewDataset),
                 triplet.py (TwoAFCDataset — NIGHTS-style 2AFC), dense.py
                 (DenseFolderDataset + stems= for official splits,
                  load_depth_map, load_normal_map, load_mask, load_label_map)
  heads/         base.py (register_head/build_head), linear.py, dpt.py
  metrics/       classification, retrieval, correspondence, similarity, dense.py
  tasks/         base.py (BaseTask)
                 dense_base.py (DenseTrainingTask — shared by every dense probe)
                 high_level/  classification, retrieval, semantic_segmentation
                              (+ stubs)
                 mid_level/   correspondence, depth, surface_normal,
                              generic_segmentation, similarity
  results/       schema.py (ResultRecord, SCHEMA_VERSION), writer.py
  runner.py      visbench.run() — the one call the CLI wraps
examples/        classify, retrieve, correspond, depth, normals, segment,
                 segment_semantic, similarity
```

### The CLI — add a probe by adding a row

`visbench/cli/datasets.py` holds one `ProbeSpec` per probe: its summary, the
folder layout it expects, the flags it adds, how those become `Splits`, and the
kwargs its constructor takes. That is a **table, not a hierarchy** — the eight
probes share flag *groups* (`_dense_flags`, `_split_flags`) but not a class
tree, because what they have in common is a set of options, not behaviour.

Two things the CLI must keep doing, both already tested:

- **Build the probe as an object, never by name with kwargs.** `run()` owns
  `batch_size` (extraction) and `device` (the backbone's), and every dense probe
  takes constructor arguments of those names meaning something else. Passing
  them through `run(**task_kwargs)` is a `TypeError`. The CLI keeps them apart
  as `--batch-size` and `--train-batch-size`.
- **`--limit` is per class on a labelled folder** (`balanced_subset`), by
  triplet on `TwoAFCDataset` (`max_triplets=`), by stem on a dense split, and a
  prefix on pairs. A plain prefix of an Imagenette split is entirely class 0 and
  scores 1.0 while measuring nothing.

### `DenseTrainingTask` — subclass this for a new dense task

`visbench/tasks/dense_base.py` holds everything a trained dense probe needs:
feature sources (in-memory dict *or* streaming `CachedFeatures`, normalised to
one indexable source), batching, head construction, the AdamW + warmup/cosine
schedule, the training loop, batch-wise `predict`/`evaluate`, and per-image
metric averaging. A subclass supplies only:

- `out_channels` — how many channels the head emits
- `_activate(raw)` — raw head output → prediction (applied in loss, metrics
  *and* `predict`, so those three can never disagree)
- `_loss(pred, target)` — both `(B, C, H, W)`
- `_batch_metrics(pred, target)` — must return **per-image averages**, which is
  what lets `evaluate` weight each batch by size and recover the split number
- `target_channels`, `display_name`, `target_noun`, `level`, `name`
- `target_dtype` if the target is not a float measurement — `long` for class
  indices, which is the one place a classification target leaves the path the
  other three share
- optionally `_task_params()` (extra `task_params` for the record) and
  `_on_epoch_start()` (per-epoch diagnostic hook)

`DepthTask` is 224 lines, `SurfaceNormalTask` 299,
`GenericSegmentationTask` 173 and `SemanticSegmentationTask` 186 because of
this — read them before writing a fifth. Between them they show a scalar target and a vector one; a
bin-expectation activation, a normalising one and a sigmoid; a protocol borrowed
wholesale from probe3d and one that only borrows its schedule. The base was
lifted out of a *working* `DepthTask` when the second task arrived, not
designed up front; extend it the same way, from a case that already runs.

### Decisions already paid for — do not re-litigate or re-derive

- **Fetch probe3d's real source before implementing any of its protocols.**
  Reconstructing depth from memory would have produced scalar regression
  instead of the 256-bin expectation, which is a materially different probe.
- **Not all of probe3d is MIT.** `evals/utils/metrics.py`, `losses.py` and
  `probes.py` are safe to follow. `evals/utils/correspondence.py` and
  `evals/models/croco_models/` are **CC BY-NC** and must never be copied — see
  `NOTICE`, which is the consolidated record.
- **Dense geometry**: image and target must survive the *same* resize and crop,
  applied by the dataset, and targets resample **nearest-neighbour**. Bilinear
  averages across depth discontinuities and turns a hole's zeros into a halo of
  plausible wrong values the valid mask no longer excludes. The correspondence
  task already paid for a misalignment bug once (recall@1px = 0.003).
- **Validity convention**: a pixel is invalid where the target is 0 (depth) or
  zero-length (normals). Cap out-of-range values by *marking them invalid*, not
  clamping — clamping trains and scores against a wall of fabricated values.
  **Label maps are the exception and shift by one**: for segmentation 0 is a
  real class (background) and an unlabelled pixel is *negative*. Reusing the
  depth convention there would discard every background pixel and train the
  probe to answer foreground everywhere. `SemanticSegmentationTask` inherits
  this: `IGNORE_INDEX = -1`, and it is what `cross_entropy(ignore_index=)` and
  the confusion matrix both mask on, so loss and metric drop the same pixels.
- **A label map must be read without mode conversion, and this is silent when
  wrong.** VOC's `SegmentationClass` PNGs are palette images (mode `P`) whose
  raw bytes *are* the class indices; `convert("L")` resolves the palette and
  turns classes `[0, 1, 15, 255]` into `[0, 38, 147, 220]`, which loads, trains
  and scores against labels that mean nothing. `load_label_map` therefore never
  converts, while `load_mask` must (it only asks "non-zero?"). The two cannot
  share that step, and **`load_mask` is wrong on a palette file** — VOC's void
  255 resolves to a light grey, i.e. foreground, and `ignore_index=255` never
  matches because it compares against the resolved value. Binarise
  `load_label_map` instead.
- **Two mIoUs, and they disagree by about 5 points.** Dataset-level (one
  confusion matrix over the split, ratios taken once) is what VOC, ADE20K and
  Cityscapes define and the only one comparable to published numbers;
  per-image-then-averaged is this codebase's rule everywhere else. Measured on
  VOC val with DINOv2-S: 0.732 against 0.683; with DINOv2-B, 0.753 against
  0.712. `SemanticSegmentationTask`
  reports both under distinct names and overrides `evaluate` to do it, because
  no weighted mean of per-batch ratios equals the ratio of the sums. Do not
  collapse them to one number.
- **Not every dense task gets to borrow probe3d.** It has no binary or semantic
  segmentation task, so `GenericSegmentationTask` and `SemanticSegmentationTask`
  keep only its *optimiser* schedule and record `protocol:
  "visbench_binary_seg"` / `"visbench_semantic_seg"`. Do not let a record
  claim `"probe3d"` for a loss and metric that paper never defined; the whole
  value of the field is that it says what a number is comparable to.
- **The ten-epoch schedule assumes NYUv2-sized data.** Measured on 80 training
  images: 0.16 IoU at the defaults, 0.87 at `epochs=40, lr=5e-3`, identical
  features. That is underfitting, not a weak representation, and `train_loss`
  is what separates the two. Do not tune the defaults away from probe3d's —
  say so in the example instead, which `examples/segment.py` does. **At real
  scale the defaults are fine**: 1464 VOC training images reach 0.73 mIoU at
  ten epochs with `train_loss` 0.19, so the schedule is not the problem, small
  splits are.
- **Bigger is not better on every task, and this is the point of the library.**
  DINOv2-S beats DINOv2-B on mid-level similarity (0.870 vs 0.858) while losing
  to it on semantic segmentation (0.732 vs 0.753). Do not "sanity check" a new
  task by asking whether the larger model won.
- **The NIGHTS ImageNet split is a contamination check, not a subset.**
  `test_imagenet` and `test_no_imagenet` partition the test set by whether the
  reference image came from ImageNet. DINOv2-S scores 0.882 against 0.854 across
  them; quoting the combined 0.870 without that gap overstates how much of it is
  perceptual alignment.
- **A triplet task is a flat image dataset plus indices, not a widened cache.**
  `TwoAFCDataset` presents unique images and puts the triplet structure in
  `labels()` as `(ref, left, right, vote)` indices into itself. That keeps the
  cache, the fingerprint and `run()` unchanged, extracts a shared image once,
  and makes the pairing travel by index. It is also why `subset()` is refused
  there — slicing images would silently repoint every triplet — and why
  `max_triplets=` exists on the constructor instead.
- **The mid-level similarity protocol is zero-shot, despite what its own paper's
  README says.** `evaluate_model_percepture.py` builds a test loader, freezes
  the backbone and compares two cosine similarities; nothing is trained. The
  README says otherwise. Follow the code, and do not add a head "to match the
  description".
- **Read a CSV by column name.** The reference reads the vote as `iloc[idx, 2]`
  and the paths as `4`/`5`/`6`. Reordering the file would silently score
  against the wrong column, and the failure looks like a mediocre number rather
  than an error.
- **A pair task is a flat image dataset plus interleaving, not a widened cache.**
  Same resolution as the triplet one. `PairViewDataset` presents a
  `PairDataset`'s two views as `2N` single images — item `2i` and `2i+1` are
  pair `i` — so extraction, batching, streaming and the identity memo need no
  change, and `regroup` restores the pairing. `run()` forks on the task's
  declared `uses_pairs`, one explicit branch rather than a general "dataset
  adapter" mechanism built to fit a single case. **Both directions stay lazy**:
  materialising the pairs would pull a whole split of dense features back into
  memory, undoing the streaming that had just written them to disk.
- **`view_identity` is the reason a cached correspondence run is fast, and it
  had no caller for a year.** The two views of a pair come from one file and
  would otherwise share a `cache_identity`, so the memo would serve view 1 the
  features of view 0 — trivially perfect matches, no error. It existed and was
  tested from v0.1; `examples/correspond.py` passed the cache a bare list of PIL
  images, which has no identity, so a fully cached run still decoded, cropped and
  warped everything. 16.4 s cold against 8.2 s warm on 200 pairs, once `run()`
  used it. **A declared-but-uncalled mechanism is the same failure as the
  QuickGELU guard**: it passes its own tests forever while doing nothing.
- **Correspondence's ceiling travels with its score, through
  `BaseTask.context_metrics`.** `recall@1px` has a ceiling of 0.015 on DINOv2
  ViT-S/14 at 224px, so the score alone says the wrong thing. The hook returns
  `{}` for every other task; correspondence prefixes `ceiling_`. `run()` refuses
  a context key that collides with a score, since they share one flat dict.
  Measured on 200 Imagenette pairs: 0.783 against a ceiling of 0.951.
- **The dataset half of a run gets `dataset_params`, like the task half gets
  `task_params`** (schema v5). Filled from whatever `describe()` returns beyond
  the record's own fields, so `max_warp`, `image_size` and `num_triplets` land
  there without a per-setting column. Before this they changed the fingerprint
  and nothing else — two runs were distinguishable only as "not the same data".
- **Shorten a labelled folder with `balanced_subset(n)`, not `subset(n)`.**
  The file list is grouped by class, so a prefix is entirely class 0 and a
  single-class retrieval scores 1.0 while measuring nothing. Two examples
  carried their own copy of this before it became a method.
- **Shorten a split with `dataset.subset()`, never by slicing its attributes.**
  A dense dataset carries three index-parallel lists and slicing one alone pairs
  a target with the wrong image, silently, since every later step still sees
  equal lengths. Subclasses declare `_parallel_attrs`; the base reindexes them
  together and the fingerprint follows, so a limited run cannot be mistaken for
  a full one. The CLI (5j) needs exactly this — do not reinvent it there.
- **Per image, then averaged.** Never pool every pixel of the split; that lets
  uneven hole coverage silently reweight the dataset.
- **Dense features stream.** ~250x the size of pooled ones (24k NYUv2 images at
  DINOv2-B is ~19 GB). `run()` streams automatically for `uses_dense` tasks.
  Measured: 10.8 GB peak RSS in memory vs 1.7 GB streaming for 0.63 GB of
  features. `CachedFeatures` is random-access, not a generator — training
  reshuffles every epoch and a generator can only shuffle *within* a batch.
- **Targets travel by index, never by iteration order**, or they drift from
  their features the moment a loader shuffles. Silent failure: it still trains.
- **probe3d's uncertainty-aware angular loss can switch itself off** near chance
  accuracy. `SurfaceNormalTask.fit` documents the measured dynamics, detects it
  and warns. The loss is deliberately left as probe3d wrote it — silently
  substituting the plain one would make VisBench's numbers incomparable with
  the published ones, which is the only reason to borrow a protocol at all.
- **mypy's `python_version` tracks the newest syntax any *dependency stub* uses,
  not the package's floor.** It is pinned to 3.12 in `pyproject.toml` because
  numpy 2.x uses PEP 695 `type` statements. Do not "fix" it down to the floor.
- **A fine-tuned number and a frozen one are different measurements, and the
  record is what keeps them apart.** Frozen asks what a representation already
  carries; fine-tuned asks what it can be adapted into. Every published VisBench
  number is frozen. Schema v6's `finetune` field is `None` for those and a dict
  otherwise — never rank or average across it. The trainable forward pass is a
  **separate entry point** (`extract_features_trainable`), not a flag on
  `extract_features`, because the cache depends on getting detached tensors and
  a keyword defaulting to the safe value puts the expensive mistake one typo
  away. The unfrozen backbone **stays in `eval()`**: train mode would start
  BatchNorm updating and dropout firing, moving a fine-tuned number for two
  reasons at once with one of them unrecorded.
- **A wall clock is not a metric — repeat it before concluding anything from
  it.** Every score in this codebase is deterministic and reproduces to four
  decimals across runs, which makes it tempting to treat a `duration_seconds`
  from the same record as equally solid. It is not: 6a timed one frozen/
  fine-tuned pair, got 252 s against 238 s, and recorded "fine-tuning is not
  slower" in three files and a merged PR. Re-running the identical commands
  gave 156 s and 126 s frozen against 200 s fine-tuned — same metrics to the
  digit, opposite conclusion. The machine is shared. Run a timing at least
  twice, and prefer the *repeat* to the first, since the first also pays for
  whatever the page cache had evicted.
- **Verify with the exact commands CI runs** (below). A local env with extra
  packages installed will pass checks that CI fails.
- **A guard whose only test is `slow` is a guard CI never runs.** `addopts`
  deselects `slow`, and CI runs a plain `pytest`, so the entire
  weight-downloading suite is invisible to it. The CLIP QuickGELU check filtered
  on a phrase open_clip has never emitted and was dead code for its whole life;
  its test existed, failed correctly, and never ran. When a check exists to stop
  a *silently wrong number*, give it a test in the fast suite — extract the
  logic to a pure helper if that is what it takes.

- **The Python floor is 3.10 because DINOv2 requires it, and that was the
  cheaper of two bad options.** The pinned `HUB_REF` uses `float | None` at
  class-body scope, which 3.9 evaluates at import and rejects — so DINOv2, six
  of seven `examples/`, and every slow test were broken on the declared floor
  (#1). The alternative, repinning `HUB_REF` to a 3.9-compatible commit, would
  have **invalidated every cached DINOv2 feature on every machine**, since
  `HUB_REF` feeds `cache_key()`. Raising the floor keeps the ref and therefore
  the caches: verified identical keys before and after. Do not lower it back
  without checking DINOv2 still imports.

### Open issues — read before assuming a red suite is your fault

**Every issue below is closed; the tracker is empty as of 2026-07-29.** All
five verification commands were re-run on that date and are green: 1024 fast
tests, 76 slow, and the three lint steps. If anything is red for you, that is
new — do not go looking for a known cause here.

The entries are kept because each one records a *class* of failure this
codebase has actually shipped, and the next one will rhyme with them.

- **[#2] CI never ran `-m slow`** — fixed. `.github/workflows/slow.yml` runs it
  on every push to `main`, nightly at 03:00 UTC, and on demand, with the
  downloaded weights cached against `HUB_REF`. It is **not** part of the gating
  CI workflow and does not run on pull requests, so a 1.7 GB download never
  blocks ordinary work. If you add a check that guards a *silently wrong
  number*, it still belongs in the fast suite — this catches the ones that can
  only be caught with real weights, a day later at worst, not instead.
- **[#4] `zip(strict=)`** — done. `B905` is enforced, not ignored: 12 sites take
  `strict=True`, and `zip(resolved, resolved[1:])` in `backbones/base.py` takes
  `strict=False` because pairing a list with its own tail is meant to be ragged.
  Most of the 12 are backstops for invariants already enforced a few lines
  above, but one was a real hole: `CorrespondenceTask.evaluate_ceiling` never
  length-checked its arguments, so nine geometries against ten pairs scored nine
  and reported the number as covering the split. `evaluate` had always checked.
  **When you add a `zip` over two things paired by index, `strict=True` is the
  default** — the cost is nothing and the failure it prevents still trains.
- **[#1] DINOv2 on 3.9** — fixed by raising the floor; see above.
- **[#3] CLIP QuickGELU guard** — fixed; see the bullet above.

`CHANGELOG.md` under `[Unreleased]` is the full record of what each step
added and why; `README.md` has the user-facing view and the measured Imagenette
numbers. Both are kept current per step — update them in the same commit as the
code, not afterwards.

**Update this file at the end of every step, in that same commit.** The build
table, the registered names, the layout block, the v0.2 checklist and the
"decisions already paid for" list are how the next session knows where the work
stands and what it must not re-derive; a step that ships code without updating
them has left the next session to rediscover its findings the expensive way.

---

## Architecture

### `BaseBackbone`

- One method, `.extract_features(image, pooling="default", layers=None,
  feature_mode="dense_only")`, returning a `FeatureDict`:
  `{"dense", "pooled", "grid_hw", "cls", "dense_layers", "layer_indices"}`.
- Returns **both** the dense spatial features and a pooled single vector from
  the same call — tasks pick whichever they need, backbones never expose
  separate methods per use case.
- Subclasses implement `_forward_features(image, layers) -> list[LayerOutput]`,
  one `(patch_tokens, cls_or_None, grid_hw)` per requested depth, from **one**
  forward pass. `resolve_layers()` on the base turns `None`/negatives into
  absolute indices and enforces strictly increasing order — order is meaningful,
  since DPT reads the first as coarsest.
- `dense`, `pooled` and `cls` always describe the **last** requested layer, so a
  multi-layer call is a strict superset of a single-layer one and a task reading
  only `dense` is unaffected. Multi-layer maps live under `dense_layers`, a
  separate key rather than `dense` sometimes being a list.
- Same method signature for every backbone type (ViT or CNN) even though the
  internals differ completely — see CNN vs ViT handling below.

### `BaseTask` (a.k.a. probe)

- `.fit(features, labels)` — no-op for zero-shot tasks (retrieval,
  correspondence).
- `.evaluate(features, labels) -> dict` — always returns a flat metrics dict,
  never prints results directly (see structured logging below).
- `.predict(features)`.
- **Pooling strategy is chosen here, not on the backbone.** A task passes
  `pooling="cls"` or `pooling="mean"` (etc.) into `extract_features()`; the
  backbone just executes whatever is asked. This keeps backbones dumb and
  interchangeable, and keeps the "what representation does this task need"
  decision in one place. Same for `feature_mode` and `layers`.
- A task **declares** `uses_dense`, rather than it being inferred from the
  task's level: it tells the cache which half of the extraction to keep and
  whether to stream, and dense features are ~250x larger, so guessing would be
  an expensive guess.
- `describe()` returns the metadata for the result record and always includes
  `task_params`, empty by default — a caller building a record should never
  have to ask whether a particular task has hyperparameters.
- `run()` records `pooling` **resolved** (`"default"` means CLS on a ViT and
  mean on a CNN, so the literal word does not say what produced the number).

### Feature cache

Mandatory in v0.1, not an optional speed-up added later. Disk-backed
key-value store, one file per (image, layer), keyed by
`backbone_key | layer | pooling | feature_mode | image_hash`. Every task reads
from the cache; the backbone forward pass runs at most once per image per
backbone. Two front doors:

- `extract_dataset(...)` stacks everything and returns one `FeatureDict`.
  Right for pooled features; impossible for dense ones.
- `materialise(...)` runs the same extraction, keeps nothing in memory, and
  returns a `CachedFeatures` — an ordinary `torch.utils.data.Dataset` over the
  files already on disk, so a `DataLoader` supplies batching, shuffling and
  workers. Pass `targets=dataset.target` to pair supervision by index.

---

## Feature extraction design — the most important decision in this codebase

Handle this consistently; don't improvise per-backbone.

### Default pooling rules
- ViT backbones with a CLS token → default single-vector representation is
  the **CLS token**.
- CNNs, and any backbone without a CLS token → default is **mean-pooling**
  over the dense feature map / patch tokens.
- Either default can be overridden per task call via the `pooling` argument.

### Dense-task feature modes (all three implemented and reachable through
`extract_features(feature_mode=...)`; mode 1 is the default)

1. **`dense_only`** (default) — just the spatial grid of patch/conv
   features, no CLS involved.
2. **`dense_cls_broadcast`** — the CLS token is broadcast spatially and
   concatenated onto every patch location, increasing channel dim uniformly
   across the grid.
3. **`dense_plus_cls`** — the dense grid and a single global CLS vector are
   kept **separate** and both handed to the task head, which decides how to
   fuse them (e.g. only at a bottleneck, or as a global conditioning vector),
   rather than broadcasting CLS into every spatial location.

Modes 2 and 3 are opt-in — a task must explicitly request them. The cache keys
on the mode, and `dense_plus_cls` returns the global vector under a separate
`cls` key. `DPTHead` is the consumer these were built for.

### CNN vs ViT handling
- **CNNs**: "dense features" = the last conv feature map before global
  pooling (e.g. `layer4` output of a ResNet).
- **ViTs**: "dense features" = the patch token grid, reshaped from
  `(num_patches, dim)` to `(H, W, dim)` using the model's known patch size and
  input resolution.
- Both are exposed through the **identical** `.extract_features()` signature
  and return shape, even though the internal extraction logic is completely
  different per architecture family.

### Multi-layer extraction
Declared in the interface from v0.1, **wired up in v0.2 (step 5c)** for every
backbone: `layers=[2, 5, 8, 11]` returns one map per depth from a single
forward pass. `visbench.run()` carries a task's declared `layers` into
extraction, and the result record stores them **resolved** against that
backbone's depth — `[-4, -1]` names different blocks on a 12- and a 24-block
ViT, so an unresolved record does not say what produced the number.

---

## Task categorization

Tasks are organized into three levels, following Chen, Marks & Cheng
(arXiv:2411.17474):

```text
tasks/
  high_level/   classification, semantic (multi-class) segmentation, detection
  mid_level/    generic (binary) object segmentation, depth estimation,
                surface normal estimation, geometric correspondence,
                mid-level image similarity
  low_level/    placeholder only until v0.3+ — edge detection, optical flow,
                texture/reflectance, image quality
```

- **High-level** = semantic/category understanding.
- **Mid-level** = geometry and generic structure prior to semantic labeling —
  this is the paper's core contribution area, and it's where VisBench should
  be strongest relative to existing tools.
- **Mid-level image similarity is a distinct task class from high-level
  (semantic) retrieval** — mid-level similarity judges perceptual/geometric
  resemblance between candidates and a reference (scene layout, geometry),
  not category membership. Do not merge these two into one task even though
  both are "similarity"-flavored.
- **Low-level** is a folder with a README describing future scope only —
  nothing implemented there before v0.3, and possibly not even then without
  contributor bandwidth.

---

## v0.1 — prove the abstraction — **COMPLETE**

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

## v0.2 — dense mid-level tasks + broader backbone support — **COMPLETE**

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

### Still open in v0.3, beyond the numbered steps

- Low-level tasks get their first real entries if there's contributor
  bandwidth (edge detection, optical flow); otherwise the folder stays a
  placeholder.
- HF Hub integration for sharing pretrained probe heads and a public
  leaderboard, once there's enough task/backbone coverage for a leaderboard
  to be meaningful.

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

Two conventions to settle **before** writing the dataset, since both are silent
when wrong: absolute pixels vs normalised `[0,1]`, and `xyxy` vs `xywh`. Pick
one internally, convert at the loader boundary, and assert the choice in a fast
test — a swapped pair loads, trains and scores.

Prove it on **VOC2012 Detection**, whose `Annotations/` XML sits beside the
`JPEGImages/` and `SegmentationClass/` this repo already runs on, so no new
download is needed. `DetectionTask` is currently a `NotImplementedError` stub
with `level`/`feature_mode`/`zero_shot` already declared — extend it, do not
rewrite it.

**Known deferred**: keying the prefix cache on dataset identity (path + mtime)
rather than image content, which 6b's profile identified as the 128.3 s
remaining cost. Explicitly not part of 6c — see 6b's closing note for why it
reaches into the data layer and what it would break.

---

## Engineering conventions

- PyTorch, Python 3.10+. Optional extras: `clip` (open_clip), `timm`, `dev`.
  **A backbone whose extra is missing is still registered and still listed** —
  both CLIP and timm import their dependency lazily inside `__init__`, so the
  registration module imports cleanly and `_REGISTRATION_MODULES`' skip logic
  never fires for them. Constructing one raises `ImportError: ... pip install
  visbench[clip]`, which is *better* than the registry raising "Unknown
  backbone", so do not "fix" it by moving the imports to module scope. Use
  `registry.missing_extra(name)` to ask without importing; the CLI's `list`
  marks them. This was documented backwards until the v0.2.0 wheel test, where
  a core-only install listed all six backbones under a footer promising it
  would not.
- Pin exact dependency versions via `uv.lock` — this is a reproducible
  benchmark library, not a moving-target research repo.
- Write tests alongside every new module; don't defer testing to "later."
- Every task run logs a structured JSON record — backbone, task, dataset,
  pooling mode, feature mode, layers, metrics, timestamp — under one
  **additive-only** schema, so leaderboard tooling never needs a retrofit.
  Bump `SCHEMA_VERSION` when adding a field; never remove or repurpose one.
- Package for PyPI from v0.1: `pyproject.toml`, semantic versioning,
  `pip install visbench` as the eventual target install path.
- Cite prior art in code comments and docs wherever an evaluation protocol is
  borrowed, not just in the README. `NOTICE` is the consolidated list.

### Verifying — use these exact commands

The project venv is `.venv/` — Python 3.10.12, the supported floor, with
`visbench` installed editable and all extras present. **Use it.** Another
interpreter on the machine will not have `visbench` importable (examples fail
with `ModuleNotFoundError`) and may have different dependency versions.

```bash
source .venv/bin/activate       # or call .venv/bin/<tool> directly

pytest                                              # 1024 fast tests
pytest -m slow                                      # 76, real DINOv2/CLIP weights
ruff check visbench/ tests/ conftest.py examples/
ruff format --check visbench/ tests/ conftest.py examples/
mypy visbench/ examples/ --ignore-missing-imports   # reads [tool.mypy], py 3.12
```

CI runs all five: the four fast ones gate every push and pull request, and
`-m slow` runs in a separate workflow on pushes to `main` and nightly. A local
environment with extra packages installed will pass checks that CI fails, so do
not substitute your own invocations — particularly for mypy, which reads
`python_version` from `pyproject.toml` and checks nothing useful if you
override it.

Both suites and all three lint steps must be clean before a commit. Prove a
new task end to end on a real backbone via its `examples/` script, not only
against the fake backbones in `tests/conftest.py`; the toy backbones cannot
show a training-dynamics problem, and one has already been found that way.
