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
| 6c-1 | Detection: the box dataset and VOC loader | done |
| 6c-2 | Detection: `average_precision`, mAP@50, mAP@50:95 | done |
| 6c-3 | Detection: the head, against a metric already trusted | done |
| 6d-0 | Dataset listing: `scandir`, not a stat per file | done |
| 6d-1 | Edge detection — the first low-level task, on Taskonomy | done |
| 6d-2 | `mask_valid`, keypoints2d + occlusion_edge, `DenseMagnitudeTask` | done |
| 6e-1 | Leaderboard: the comparability rules, as pure functions | done |
| 6e-2 | Leaderboard: regenerate a record corpus for all twelve probes | done |
| 6e-3 | Leaderboard: render it, and generate the README tables from records | next |
| 6e-4 | Hub: serialise a trained head, with the backbone identity beside it | |
| 6e-5 | Hub: push/pull through `huggingface_hub`, behind a `[hub]` extra | |

---

## Current state

**v0.1 through v0.5 are all complete** — every task, all three backbone
families, the CLI, fine-tuning, detection and two of the low-level probes.
Everything below exists, is tested, and is on `main`. v0.4.0 filled the
low-level tier; **v0.5.0 is the `mask_valid` release** (2026-08-01), which
unblocked four Taskonomy domains and added the eleventh and twelfth probes.

**v0.3's numbered steps are all done: 6a (fine-tuning), 6b
(prefix caching) and all of 6c (detection).** Dense probes take `finetune_blocks=N` /
`--finetune-blocks N`, DINOv2 only, recorded under schema v6's `finetune` field.
Proved on VOC at two scales: 0.7758 against the frozen 0.7328 on DINOv2-S, and
0.7992 against 0.7533 on DINOv2-B. 6b caches the frozen blocks below the cut in
a separate `PrefixCache`, cutting a fine-tuned ViT-B run from ~345 s to ~272 s
with the mIoU unchanged to four decimals. 6c added the box dataset, the VOC
metric and an anchor-free single-scale detection probe, in that order. **The
schema is still v6 — detection needed no bump**, because `task_params` and
`dataset_params` are both open dicts and the protocol, the decoding settings and
`include_difficult` all land in them.

**6d-1 added the first low-level task**, so all three levels of the taxonomy now
have entries and `visbench/tasks/low_level/` is no longer a placeholder. Edge
detection is dense magnitude regression on Taskonomy's `edge_texture`, scored by
per-image Pearson correlation and recorded as `visbench_edge_regression` — not
BSDS500's, which is a correspondence metric and a step of its own.

**6d-2 wired up `mask_valid/` and added two more probes**, released as v0.5.0.
Four of the six Taskonomy domains that were refused are now supported — `depth_zbuffer`, `normal`, `edge_occlusion`, `keypoints3d` —
each declaring how it marks an invalid pixel rather than exposing a mask.
`keypoints2d` (low-level) and `occlusion_edge` (mid-level) joined `edge` on a
lifted `DenseMagnitudeTask`. **Still schema v6**: twelve probes, and the new
`target_transform` / `invalid` / `masked` settings land in `dataset_params`,
which is what that field was added for. The next step is **6d-3+** — another
task, or the HF Hub / leaderboard work.

Registered names — `visbench.list_backbones()`, `list_probes()`,
`visbench.heads.list_heads()`:

```text
backbones  dinov2_vits14, dinov2_vitb14, clip_vitb16, clip_vitb32,
           resnet18, resnet50            (+ CustomBackbone, unregistered)
probes     classification, retrieval, correspondence, depth, surface_normal,
           generic_segmentation, semantic_segmentation, similarity, detection,
           edge, keypoints2d, occlusion_edge
heads      linear, dpt, detection
```

The CLI exposes all twelve probes: `visbench list`, `visbench run <probe>`,
`visbench cache stats|clear`. A test asserts the CLI's table and
`list_probes()` are the same set, so a probe cannot ship unreachable from a
shell by accident.

Package version is `0.5.0`. The last upload to
[PyPI](https://pypi.org/project/visbench/) this file witnessed was **v0.5.0 on
2026-08-01** — wheel and sdist both, verified by downloading the published wheel
and reading `__version__` and the three new modules out of it, rather than by
trusting the version number. Whether anything followed it is not something this
paragraph can know.
**Publishing needs the maintainer's credentials and is theirs to
run** — never attempt it, and do not assume a tag means a release went out, or
that `main` matches what is installable; check
[PyPI](https://pypi.org/project/visbench/) rather than this paragraph if it
matters. A version number on PyPI
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
Result schema is at **v7** (`pooling_requested` added in 6e-2b; `finetune` was
6a; `dataset_params` was 5j) and is **additive only**: never remove or repurpose
a field, or old records stop being readable.

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
  data/          detection.py (DetectionFolderDataset, load_voc_boxes,
                   VOC_CLASSES — boxes transform, they do not resample)
                 image_folder (+ balanced_subset), pair_dataset
                 (PairDataset, HomographyPairDataset, PairViewDataset),
                 triplet.py (TwoAFCDataset — NIGHTS-style 2AFC), dense.py
                 (DenseFolderDataset + stems= for official splits,
                  _init_geometry() — the crop, shared with taskonomy.py,
                  load_depth_map, load_normal_map, load_mask, load_label_map,
                  load_edge_map)
                 taskonomy.py (TaskonomyDataset — building-nested, indexed from
                   splits/*.csv; subclasses DenseFolderDataset for geometry only.
                   _DOMAIN_SPECS: per-domain loader, scale, invalid convention,
                   log_transform. load_valid_mask — mask_valid/, 6d-2)
                 base.py (BaseDataset, list_files — scandir, never a stat/entry)
  heads/         base.py (register_head/build_head), linear.py, dpt.py,
                 detection.py (DetectionHead — cls + box branches, focal prior)
  metrics/       classification, retrieval, correspondence, similarity, dense.py
                 (+ magnitude_metrics — per-image Pearson, masks NaN;
                    edge_metrics is it under the published key)
                 detection.py (box_iou, average_precision, detection_metrics —
                   VOC protocol, dataset-level, difficult ignored not dropped)
  tasks/         base.py (BaseTask)
                 dense_base.py (DenseTrainingTask — shared by every dense probe)
                 magnitude_base.py (DenseMagnitudeTask — edge/keypoints2d/
                   occlusion_edge; identity activation, masked L1, correlation)
                 schedule.py (warmup_cosine/check_schedule — probe3d's schedule,
                   shared by DenseTrainingTask and DetectionTask)
                 high_level/  classification, retrieval, semantic_segmentation,
                              detection (anchor-free, single-scale, 6c-3)
                 mid_level/   correspondence, depth, surface_normal,
                              generic_segmentation, similarity, occlusion_edge
                 low_level/   edge (6d-1), keypoints (Keypoint2DTask, 6d-2)
  results/       schema.py (ResultRecord, SCHEMA_VERSION), writer.py
  runner.py      visbench.run() — the one call the CLI wraps
examples/        classify, retrieve, correspond, depth, normals, segment,
                 segment_semantic, similarity, detect, edges, keypoints,
                 occlusion_edges
```

### The CLI — add a probe by adding a row

`visbench/cli/datasets.py` holds one `ProbeSpec` per probe: its summary, the
folder layout it expects, the flags it adds, how those become `Splits`, and the
kwargs its constructor takes. That is a **table, not a hierarchy** — the nine
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
  **Edge maps are the third case and have no invalid value at all** (6d-1): 0
  means "no edge", a real reading covering most of a frame, so `edge_metrics`
  masks nothing. **`NaN` is the fourth, and exists because the third has no
  spare value** (6d-2): a magnitude map derived from Taskonomy's 3D
  reconstruction *does* have holes, and they hold a plain 0 that is
  indistinguishable from a real reading, so validity has to travel out of band.
  `NaN` is also the loud choice — it makes an unmasked loss `NaN` on the first
  step, where a fabricated 0 trains quietly and merely scores badly. Four
  targets, four conventions — check which one a new dense task needs rather
  than inheriting the nearest, and note that `depth_zbuffer` and `normal` on
  Taskonomy needed *none* of the new machinery because their existing in-band
  sentinels turned out to match `mask_valid/` pixel for pixel.
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
  segmentation task, and no edge task, so `GenericSegmentationTask`,
  `SemanticSegmentationTask` and `EdgeTask` keep only its *optimiser* schedule
  and record `protocol: "visbench_binary_seg"` / `"visbench_semantic_seg"` /
  `"visbench_edge_regression"`. Do not let a record claim `"probe3d"` for a loss
  and metric that paper never defined; the whole value of the field is that it
  says what a number is comparable to. The same applies to *other* papers'
  protocols: the edge probe must not claim BSDS500's, which is a correspondence
  metric it does not implement.
- **The ten-epoch schedule assumes NYUv2-sized data.** Measured on 80 training
  images: 0.16 IoU at the defaults, 0.87 at `epochs=40, lr=5e-3`, identical
  features. That is underfitting, not a weak representation, and `train_loss`
  is what separates the two. Do not tune the defaults away from probe3d's —
  say so in the example instead, which `examples/segment.py` does. **At real
  scale the defaults are fine**: 1464 VOC training images reach 0.73 mIoU at
  ten epochs with `train_loss` 0.19, so the schedule is not the problem, small
  splits are.
- **Bigger is not better on every task, and this is the point of the library.**
  DINOv2-S beats DINOv2-B on mid-level similarity (0.870 vs 0.858), low-level
  edges (0.4558 vs 0.4481) and low-level keypoints (0.2356 vs 0.2248) while
  losing to it on semantic segmentation (0.732 vs 0.753), detection (0.213 vs
  0.262), occlusion edges (0.2924 vs 0.3167) and Taskonomy depth (d1 0.5832 vs
  0.5986). Do not "sanity check" a new task by asking whether the larger model
  won. **And a task can disagree with itself**: on Taskonomy normals DINOv2-S
  wins on mean angular error while DINOv2-B wins on the 11.25° threshold, so
  quoting one and dropping the other manufactures a result.
- **Ask whether a new probe *ranks*, not whether its number is high.** A low
  absolute score can be by design — detection's 0.21 mAP is, because a
  single-scale head has no pyramid. What is never acceptable is failing to
  separate two backbones, and that is the question a suspiciously low number
  should prompt. 6d-2 nearly shipped an occlusion-edge probe scoring 0.088 with
  an S-versus-B gap of 0.0035; the score alone looked like "a hard task", and
  the gap is what showed it was measuring nothing.
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

**Every issue below is closed; the tracker is empty as of 2026-07-31.** All
five verification commands were re-run on 2026-08-01 and are green: 1216 fast
tests, 76 slow, and the three lint steps — plus CI's two extra jobs, `lock` and
`build`. If anything is red for you, that is new — do not go looking for a
known cause here.

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
                mid-level image similarity, occlusion-edge detection (6d-2)
  low_level/    edge detection (v0.4), 2D keypoint detection (6d-2)
                — still scope only: optical flow, texture/reflectance,
                image quality
```

The occlusion-edge and texture-edge probes **share every line of their
implementation and sit one tier apart**, which is the cleanest statement of what
the tiers mean that this codebase has: recovering a depth discontinuity needs
scene geometry, recovering an intensity one does not. Taskonomy has no
reflectance domain, so the texture/reflectance row is *not* unblocked by 6d-2's
mask work — see `visbench/tasks/low_level/README.md`.

- **High-level** = semantic/category understanding.
- **Mid-level** = geometry and generic structure prior to semantic labeling —
  this is the paper's core contribution area, and it's where VisBench should
  be strongest relative to existing tools.
- **Mid-level image similarity is a distinct task class from high-level
  (semantic) retrieval** — mid-level similarity judges perceptual/geometric
  resemblance between candidates and a reference (scene layout, geometry),
  not category membership. Do not merge these two into one task even though
  both are "similarity"-flavored.
- **Low-level** = signal-level properties recoverable without naming an object.
  Was a README describing future scope only until step 6d-1 (v0.4), which added
  edge detection. The folder's own README now separates what is implemented from
  what is still scope, and is the place to look before starting a second one.

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

- ~~Low-level tasks get their first real entry~~ — **done in 6d-1**, shipped in
  v0.4.0. Edge detection is in; optical flow, texture/reflectance and IQA are
  still scope only, and `visbench/tasks/low_level/README.md` says what each
  would cost.
- HF Hub integration for sharing pretrained probe heads and a public
  leaderboard, once there's enough task/backbone coverage for a leaderboard
  to be meaningful.

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
  toward the published pair.
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

**Unresolved, and 6e-3 must not render the correspondence board until it is:
`recall@t` is an average over a denominator each backbone chooses for itself.**
It is `(errors <= t).mean()` over the matches `nn_match` + the ratio test
proposed, and `num_matches` is that denominator. Across the six:

| backbone | recall@1p | ceiling | num_matches |
| --- | --- | --- | --- |
| resnet18 | **0.8927** | 0.9762 | **4,911** |
| resnet50 | 0.8601 | 0.9785 | 4,373 |
| clip_vitb32 | 0.7992 | 0.9741 | 4,283 |
| dinov2_vits14 | 0.7834 | 0.9509 | 23,439 |
| dinov2_vitb14 | 0.7594 | 0.9471 | **27,590** |
| clip_vitb16 | 0.7179 | 0.9584 | 12,798 |

**ResNet-18 tops the board while proposing 5.6x fewer matches than DINOv2-B**,
and a coarse 7x7 grid proposes few, easy, well-separated candidates. This is
exactly the `classes_scored` case 6e-1 already refuses — two averages over
different denominators are averages of different quantities — except that here
*every* backbone differs, so making `num_matches` a guard metric would render
correspondence permanently unrankable rather than fixing it. Normalising by the
ceiling does not resolve it either (RN18 still leads, 0.9145 against 0.8238).

Do not ship this row as "ResNet-18 is the best correspondence backbone". The
options are a fixed candidate set across backbones, a match-count-matched
protocol, or rendering the board with `num_matches` beside every score and
saying plainly what it does to the comparison. That is a protocol decision, and
it is the first thing 6e-3 has to settle.

**`slurm/corpus.sbatch` takes `VISBENCH_BACKBONES`** so the matrix widens
without editing it. The `--array` range **cannot** be derived from that list —
`#SBATCH` directives are read before the script runs — so the script refuses a
range that does not match `probes x backbones` unless `VISBENCH_PARTIAL=1` says
the gap is deliberate. Worth the guard because the failure is invisible: a short
array simply omits probes, and the corpus then looks complete, since every group
it *does* contain holds every backbone.

### The candidate task backlog — and what is actually on this machine

`README.md` has the public version of this list, grouped by cost. What follows
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
classification is therefore a dataset-swap on the existing linear-probe path**,
not an acquisition step.

**Verified absent, both levels:** any optical-flow set (Sintel, KITTI,
FlyingChairs), NYUv2, any intrinsic-image set (IIW, SAW, MIT intrinsic).
`bsds300` is still the MAF density-estimation benchmark, not BSDS500 — see 6d-1.
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
frame*; Harris corners, DoG blobs, structure-tensor orientation and photometric
superpixels are all equally derivable, deterministically, from any image folder
already here. That makes them a target generator plus a `DenseMagnitudeTask`
subclass — the two pieces that already exist — rather than a data acquisition
step. If a low-level probe is wanted next, start there.

Two hazards to carry into any of them, both already paid for once:

- **Check the tail before assuming the magnitude protocol transfers.** A corner
  response is spikier than an edge response, and `edge_occlusion` at 46% mass in
  its strongest 1% of pixels is the case where L1 and Pearson pull apart and the
  probe stops ranking backbones. 6d-2's note has the numbers.
- **A derived target is only as honest as its generator, and `protocol` must say
  which generator.** "Harris corners" is a family, not a definition — the
  k parameter, the window, the smoothing and the non-maximum suppression all
  move the target. A record claiming a bare `"harris"` says less than it looks.

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

pytest                                              # 1216 fast tests
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

**CI gates two more jobs the five commands do not cover, and a release touches
both.** `lock` runs `uv lock --check`, and `build` runs `python -m build` +
`twine check dist/*`.

- **A version bump requires `uv lock`.** `uv.lock` pins visbench *itself*, so
  editing `version` in `pyproject.toml` desynchronises it and `lock` fails while
  all five local commands pass — which is exactly what happened on the v0.3.0
  PR. Re-lock in the same commit as the bump and confirm the diff is the one
  line: anything more means dependencies moved too, which is a separate
  decision and not part of a release.
- **`twine check` is the only local proxy for how PyPI will render the README.**
  Neither `build` nor `twine` is in `.venv/`; install them into a throwaway venv
  rather than the project one, so `.venv/` keeps matching what CI has.

Both suites and all three lint steps must be clean before a commit. Prove a
new task end to end on a real backbone via its `examples/` script, not only
against the fake backbones in `tests/conftest.py`; the toy backbones cannot
show a training-dynamics problem, and one has already been found that way.
