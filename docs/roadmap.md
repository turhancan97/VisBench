# Roadmap, build order and future directions

This is the project's own record of how it was built and where it might go. It
is here rather than in the README because it answers "what is the plan", not
"how do I use this" — see the [documentation home](index.md) for the latter.

## Build order

This is a multi-month roadmap, built one reviewed step at a time.

- [x] **1. Scaffold** — every folder and module, docstrings and stubs, no logic
- [x] **2.** `BaseBackbone` + feature cache + DINOv2, with tests
- [x] **3.** `BaseTask` + one task end-to-end on a local image folder
- [x] **4.** Next task, then next backbone — all three v0.1 tasks, both v0.1
      backbones, `uv.lock`, and the `run()` entry point
- [x] **5a.** ResNet/timm backbone — the first non-ViT, validating the CNN half
      of `BaseBackbone`
- [x] **5b.** custom `nn.Module` backbones, and pluggable heads (linear + DPT)
- [x] **5c.** multi-layer extraction — `layers=[...]` through every backbone
      and the cache, so the DPT head has something real to consume
- [x] **5d.** depth estimation — the first dense task, end to end on probe3d's
      protocol: dense dataset, metrics, loss, pluggable head
- [x] **5e.** streaming features from disk, so a dense task can run a dataset
      larger than memory
- [x] **5f.** surface normals — probe3d's angular protocol, reusing the dense
      dataset, the streaming path and the shared `DenseTrainingTask`
- [x] **5g.** generic (binary) segmentation — the first dense task whose
      protocol is not probe3d's, and the first target where 0 is a label rather
      than a hole
- [x] **5h.** semantic (multi-class) segmentation — the high-level counterpart
      to 5g, on the same base class, with a class-index target and mIoU under
      both reductions
- [x] **5i.** mid-level image similarity — zero-shot 2AFC against human
      judgement, deliberately distinct from high-level retrieval
- [x] **5j.** the CLI, a thin wrapper over `visbench.run()` — which also
      taught `run()` to cover correspondence, the one task it had never been
      able to express
- [x] **6a.** opt-in fine-tuning — unfreeze the last N backbone blocks, with the
      feature cache out of the path and the result record saying which a number
      came from
- [x] **6b.** cache the *frozen prefix*, so a fine-tuned run recomputes only the
      blocks it is training
- [x] **6c.** detection, in three parts and in this order: the box dataset, then
      the VOC metric, then the head — so the head is judged by a scorer that was
      already cross-checked against `VOCevaldet.m`
- [x] **6d.** the first low-level task — edge detection on Taskonomy, filling a
      folder that had been a documented placeholder since v0.1
- [x] **6d-2.** `mask_valid/`, so the reconstruction-derived Taskonomy domains
      can be read at all; 2D keypoints and occlusion edges on the lifted
      `DenseMagnitudeTask`
- [x] **6e.** the leaderboard and probe sharing, in five parts: the
      comparability rules as pure functions, a committed record corpus covering
      every probe against every backbone, the renderer that generates the
      published tables from it, a serialised probe artifact carrying the
      backbone identity beside the weights, and the Hub transport behind a
      `[hub]` extra
- [x] **6f.** correspondence scored in pixels rather than patch widths — a unit
      change that inverted the published board
- [x] **7a–7e.** the contributor-facing surface, shipping no new number:
      `visbench demo`, the README reorganised around a reader with `docs/` split
      out of it, `CONTRIBUTING.md` and the issue/PR templates, the Sphinx site
      and its workflow, and citation metadata with a DOI
- [x] **8a–8b.** corner detection — the first probe whose target is *computed*
      from the image rather than downloaded, then put into the record corpus on
      a frame set a script pins and reconstructs
- [x] **9a–9d.** `visbench show` — the panel viewer, then the correspondence
      pair renderer, then the three probes whose answer is a choice among images
      rather than a map, then the generated docs gallery. Every probe is
      drawable, and `show_probes() == list_probes()` is asserted
- [x] **10a.** three more backbones — `TimmBackbone` learns to read a ViT, which
      added ConvNeXt-B, MAE ViT-B/16 and SigLIP-GAP ViT-B/16 in one change
- [x] **10b.** the corpus at 13 probes x 9 backbones — 117 records, and the
      first time the three tiers visibly separate: at nine backbones MAE was
      first on six boards and last on four (five and three at twelve — a count
      over a corpus is a fact about that corpus)
- [x] **10c.** a supervised ViT-B/16 — the same architecture and the same
      pretraining set as MAE, differing only in objective, so the corpus gets
      its first controlled experiment: 130 records, and supervised takes every
      high-level board while MAE takes every low-level one. The tidy version of
      that claim — "the winner changes exactly at the tier boundary" — is wrong,
      because mid-level similarity crosses it; count tiers from `record.level`,
      not from which boards feel semantic
- [x] **11a.** the documentation gallery drawn on real photographs — Open
      Images frames under a per-image licence check, rather than generated
      scenes
- [x] **10d.** DINO ViT-B/16 — a third value of the objective variable, which
      answers what the supervised/MAE pair could not: high-level structure comes
      from a semantic training signal, not from labels
- [x] **10e.** a recipe control — the same objective trained two ways, which
      supplies the denominator every objective claim needs, and refutes 10d's
      semantic-segmentation evidence on arrival
- [x] **backlog: `scene_classification`** — scene category on the object
      `classification` linear-probe path (Places365), a new probe *name* rather
      than a dataset flag; ranks backbones almost independently of the object
      board (Spearman +0.16)
- [x] **backlog: `orientation`** — gradient orientation, the fourth low-level
      probe and the second derived from the frame, but the first whose target
      is a direction; DoG-blob was rejected first for overlapping 0.51 with
      `corner`
- [x] **backlog: `fine_grained_classification`** — subordinate category on the
      same linear-probe path (CUB-200-2011), the third distinct question on that
      one implementation; a new probe *name* rather than a dataset flag, for the
      reason `scene_classification` is. The twelve-backbone board is the next
      step

## Roadmap

**v0.1** — prove the abstraction. DINOv2 + CLIP. Zero-shot or linear-probe-on-cached-features only; no fine-tuning, no dense training loops. Deferred: CLI, custom backbones, ResNet/timm, multi-layer extraction.

**v0.2** — ResNet/timm + custom backbones *(done)*, pluggable heads (linear + DPT) *(done)*, multi-layer extraction *(done)*, depth estimation *(done)*, surface normals *(done)*, generic (binary) segmentation *(done)*, semantic segmentation *(done)*, mid-level similarity *(done)*, CLI *(done)*.

**v0.3** — opt-in fine-tuning of the last N blocks *(done)*, prefix caching *(done)*, detection *(done)*.

**v0.4** — edge detection, the first low-level task *(done)*.

**v0.5** — Taskonomy's `mask_valid/` and the four domains it unblocks *(done)*, 2D keypoint detection *(done)*, occlusion-edge detection *(done)*.

**v0.6** — the leaderboard: comparability rules, a committed record corpus, and
generated tables *(done)*; HF Hub probe sharing *(done)*. **v0.6.1** corrects
the correspondence board it shipped, which was ranked upside down by a
backbone-dependent threshold unit.

**v0.7** — the contributor-facing surface: `visbench demo`, the reorganised
README, `CONTRIBUTING.md`, the documentation site, and a citable DOI *(done)*.
Every measurement v0.6.1 reported, v0.7.0 reports identically.

**v0.8** — corner detection, the first probe whose target is computed from the
image rather than downloaded *(done)*.

**v0.9** — `visbench show`: every probe drawable, four renderers, and a
generated docs gallery *(done)*. It adds no probe and moves no number.

**v0.10** — four more backbones *(done)*: ConvNeXt-B, MAE ViT-B/16 and
SigLIP-GAP ViT-B/16 through a `TimmBackbone` that now reads a ViT's own
structure, then a supervised ViT-B/16 that turns the corpus into a controlled
experiment. Thirteen probes against ten backbones, 130 records. The gallery
moves to real photographs in the same release.

**v0.11** — two more backbones *(done)*, and with them the corpus gets its
controls: DINO ViT-B/16 completes an objective family three wide on one
architecture and one pretraining set, and a SAM-trained ViT-B/16 varies only
the *recipe*, supplying the denominator an objective gap has to be quoted
against. Thirteen probes against twelve backbones, 156 records. The control
refuted half of its own family's published claim on arrival. It also ships
`examples/custom_backbone.py`, the escape hatch that had been documented and
never demonstrated.

**Next** — there is no committed next step. What follows is a candidate pool.

## Future directions

A candidate pool, not a commitment. VisBench is built one reviewed step at a
time, so these are ordered by *what they would cost*, not by preference — and
several are cheap only because the machinery they need already exists.

Anything here is open to contribution. `visbench/tasks/dense_base.py` supplies
everything a trained dense probe needs bar four methods, and
`visbench/tasks/magnitude_base.py` supplies the rest when the target is a
magnitude map.

### Cheapest — targets derived from the image itself

No new dataset. Taskonomy's `edge_texture` is already a target *computed from
the RGB frame*, and the same is true of these, so a target generator plus a
task subclass is most of the work. `corner` (v0.8) is the worked example for a
magnitude target and `orientation` for a vector one — read
`visbench/data/derived.py` and `visbench/tasks/low_level/{corner,orientation}.py`
before starting one of these.

| Task | Level | Note |
|---|---|---|
| Local orientation / gradient fields | low | **Implemented** as `orientation` — a 2-channel `(cos 2θ, sin 2θ)` field with coherence-weighted angular error; the first derived target that could not reuse `DenseMagnitudeTask` |
| ~~Superpixel / texture segmentation~~ | low | **Rejected after building it** — the SLIC boundary target passed every pre-measurement and then scored 0.021–0.043 correlation on three backbones, against 0.18–0.65 for every shipped low-level probe. A 1px partition boundary is not recoverable from patch features |
| Blob detection (DoG, LoG) | low | **Rejected** — the pre-measurement found its target correlates 0.51 with `corner`, as redundant with an existing probe as `corner` is with `edge` |

Three things a derived target has to establish before it is worth shipping,
none of which a probe run reveals on its own — all three cost real time on the
`corner` and `orientation` probes:

1. **Check the response's tail** before assuming the magnitude protocol
   transfers. A target with too much mass in its strongest 1% of pixels scores
   badly and ranks nothing. (An angle has no tail, so `orientation` skipped the
   compression — confirmed by the pre-measurement.)
2. **Check the overlap with what already ships, *before building*.** Cornerness
   correlates 0.52 with `edge_texture`; DoG blob correlated 0.51 with `corner`
   and was dropped; `orientation`'s `|r|` with both is under 0.09, because it
   measures phase. This costs an afternoon of correlations, not a probe run per
   backbone.
3. **A correlated target still earns its place if it *ranks* differently**, and
   that — not the absolute score — is the criterion.

Compute the target *after* the crop. There is then no second geometry and no
resampling of the response, which deletes the alignment hazard every other
dense probe has to test for.

### Reachable with data already common

| Task | Level | Note |
|---|---|---|
| Instance segmentation | high | The category-labelled counterpart to the existing binary segmentation; COCO-style polygon annotations |
| ~~Fine-grained recognition (CUB-200-2011)~~ | high | **Done** — `fine_grained_classification`, a distinct probe on the linear-probe path. Subordinate categories where the object board asks a basic-level question, which is why the object board is saturated and this one is not. The corpus board is pending. |
| ~~Scene classification (Places365)~~ | high | **Done** — `scene_classification`, a distinct probe on the linear-probe path. Ships with a rank check; the full corpus board is pending. |
| Relative depth ordering | mid | A ranking-only weakening of the existing depth probe — different protocol, same targets |
| Intrinsic image decomposition (albedo vs shading) | mid | Classic Marr-style separation of appearance from geometry and lighting. Ground truth is scarce outside synthetic data |
| Room / scene layout estimation | mid | Floor–wall–ceiling boundaries |
| Vanishing point / line detection | low | Published as a Taskonomy domain |
| Color constancy / illuminant estimation | low | Needs measured illuminant ground truth |

### Harder — new machinery, not just a new dataset

| Task | Level | Note |
|---|---|---|
| Optical flow | low | Needs image pairs and a flow head. `PairViewDataset` already expresses the pairing; the head is the real cost |
| Relative camera pose (essential / fundamental matrix) | mid | Pairwise, regressing a geometric relation rather than a per-pixel map |
| Multi-view stereo / point-cloud / mesh recovery | mid | Multi-view input and a non-raster output; the largest departure from every probe here |
| Motion / video object segmentation | mid | Grouping by motion, not identity. Needs video input, which nothing here consumes yet |
| Action / activity recognition | high | Same video constraint |
| Amodal completion of occluded boundaries | mid | Targets extend beyond visible evidence |
| Symmetry / repeated-structure detection | mid | Annotation is scarce |
| Object counting without recognition | mid | Scalar-per-image target rather than dense or categorical |
| Attribute recognition (material, function) | high | Multi-label rather than multi-class |
| Panoptic segmentation | high | Instance segmentation plus stuff classes |
| Image captioning / VQA | high | Needs a language decoder — a different kind of probe from anything here |
| Zero-shot / open-vocabulary classification | high | Only meaningful for backbones with a text tower, so it does not rank the full set |
| Animal identification | high | Fine-grained recognition at individual rather than species level |

### Already partly answered

Worth naming so they are not re-scoped from scratch:

- **Edge / contour detection** is implemented (v0.4) as dense magnitude
  regression on Taskonomy. What is missing is **BSDS500's** ODS/OIS/AP boundary
  protocol, which is a bipartite matching after non-maximum suppression and a
  step of its own — not a dataset swap.
- **Occlusion-edge detection** (v0.5) already covers the depth-discontinuity
  half of contour detection, at mid level.
- **Corner detection** is implemented (v0.8) as Shi-Tomasi cornerness computed
  from the RGB frame — λ_min rather than Harris's `R`, because it is
  non-negative by construction and has no `k` to record.
- **Gradient orientation** is implemented as `orientation` — a coherence-weighted
  `(cos 2θ, sin 2θ)` field, also computed from the frame, the first derived
  target that is a direction rather than a magnitude. **Blob detection (DoG)
  was rejected** by its pre-measurement: the target correlates 0.51 with
  `corner`.
- **Texture / reflectance** overlaps with intrinsic decomposition above.
  Taskonomy ships no reflectance domain, so `mask_valid/` did not unblock it.

## Library surface — candidates that ship no new number

Everything above adds a probe. These do not: they are the parts of the library
someone reaches for *around* a measurement, and each one is a gap found by
asking what a new user would try and failing to find it. v0.7 (7a–7e) is the
precedent — a release that changed no number and was worth shipping anyway.

None of these is a defect. Every one is reachable today by writing Python; what
is missing is the shortest path to it.

### Looking at what a probe saw and what it predicted — **done**

Shipped as `visbench show`, the first visualisation anywhere in the package. It
writes a grid of image / target / prediction panels to a file for the nine
probes with a spatial target — the eight dense ones plus `detection` — and
measures nothing.

**The argument for it was the project's own bug history, not convenience.** Two
of the most expensive failures here were geometry misalignments that stayed
invisible because nothing ever rendered a target next to its image:

- the correspondence misalignment that scored `recall@1px = 0.003`
- VOC's palette PNGs read through `convert("L")`, turning classes
  `[0, 1, 15, 255]` into `[0, 38, 147, 220]` — which loads, trains and scores
  against labels that mean nothing

Both were found by reading code. Both are obvious in one frame of output.

The two open decisions were settled as: **PIL and numpy only**, no matplotlib
and no dependency change; and **a saved artifact rather than training on the
spot**, which is what `visbench run --save-probe` was added to produce. The
stated hazard — that a viewer applying its own resize or colour-map is a second
geometry — is guarded by a test pinning the image panel byte-for-byte against
`np.asarray(dataset[i][0])`.

See [looking at a probe](show.md) for the three rules it keeps and why invalid
pixels are magenta.

**The pair renderer for `correspondence` followed immediately**, and step 9c
covered the last three, so **every probe is drawable** and
`show_probes() == list_probes()` is asserted. Two frames with the matches between them is a
different layout from a panel grid, and it is the probe whose historical bug is
quoted above — which is why it reports **coherence**, the mean resultant length
of the error directions. Measured with ResNet-18 features: 0.29-0.40 when the
geometry is right, 0.98-1.00 when the homography is in the wrong pixel frame,
while the median error alone cannot separate "broken" from "hopeless".

`classification`, `retrieval` and `similarity` have no spatial target, so they
draw the *decision* instead — a contact sheet, a query with its neighbours, a
triplet with the human vote marked. Each states a diagnostic as a figure the way
coherence does: **class balance**, which catches a split collapsed to one class
scoring 1.0, and **vote balance**, which catches a vote read from the wrong CSV
column.

A rendered figure for every probe now ships in the README and on the docs site,
drawn by `scripts/render_gallery.py` on **real photographs** — Open Images
validation frames, CC BY 2.0, committed under `assets/gallery_frames/` so the
gallery still rebuilds from one command with no downloads. Fetching them is a
separate one-off step (`scripts/fetch_gallery_frames.py`).

**The licence rule that made the first gallery synthetic was satisfied by
better sourcing, not waived.** VOC, ImageNet, NYUv2, Taskonomy and NIGHTS each
restrict redistribution and appear nowhere in this repository; Open Images
grants it, and the grant is verified *per frame* rather than inherited from
that sentence — a frame whose metadata carries no author or landing page is
refused, because an unattributable CC BY image is one this repository may not
ship. `CREDITS.md` is generated beside the frames and a test fails if a
committed photograph has no credit.

Real annotation cost one property and improved another. Ground truth is now
what a human marked wherever it is annotated, rather than what a script
constructed — but **invalid pixels are no longer placed on purpose**, since
real annotation has holes only where it has them. Four probes cannot have a
target column at all: `depth`, `surface_normal`, `keypoints2d` and
`occlusion_edge` need sensor or reconstruction geometry no redistributable
photograph carries, so they render `image | prediction` from a published Hub
head and say so in the footer.

### Bringing a dataset VisBench has never heard of — **done**

Three tiers now:

- **Folder layouts, no code.** `ImageFolderDataset` (class subdirectories or
  flat), `DenseFolderDataset` (`images/` + `targets/`, with `stems=` for an
  official split list), `DetectionFolderDataset` (VOC-style XML). NYUv2 joined
  the corpus with *no code change* because its layout already matched.
- **A `torch.utils.data` dataset or a Hugging Face `datasets.Dataset`** wraps in
  `TorchvisionDataset` / `HuggingFaceDataset` (`visbench.data`). Both are thin
  adapters over `BaseDataset`. The image-level CLI probes take
  `--dataset torchvision:CIFAR10` / `--dataset hf:cifar100` in place of
  `--data <path>`; `datasets` is a `[datasets]` extra, `torchvision` is already
  a core dependency.
- **Anything else, by subclassing `BaseDataset`** — two abstract methods,
  `__len__` and `__getitem__`.

**The trap the bridges had to avoid.** `BaseDataset` has four optional methods
beyond the two abstract ones, and skipping them fails *silently*: `labels()`
(supervised probes have no targets), `cache_identity()` (every run re-decodes
every image — the memo cannot recognise the file), `fingerprint()` (records
cannot tell your dataset from another), `describe()` (`dataset_params` comes out
empty). `cache_identity` is the worst: return `None` and everything still works,
just slowly, forever — the `view_identity` failure, a mechanism tested and
correct for a year while a caller passed bare PIL images and paid a full decode
on every "cached" run.

Both bridges supply a real `cache_identity` by leaning on one property: **the
wrapped dataset is immutable in index order.** A `datasets.Dataset` carries a
`_fingerprint` that changes on any transform, so `(fingerprint, row index)`
names a row's content exactly. A `torchvision` dataset has no such hash, so the
`ImageFolder` family uses the file path and everything else a digest of the
`__repr__` (which states root, split and download flags) plus the index — weaker,
and documented on the class rather than hidden.

### Probing a model VisBench has never heard of

This is the best-supported of the three and mostly needs *showing*, not
building. `CustomBackbone` is the documented escape hatch and its docstring
already names the case — a fine-tuned checkpoint, an architecture VisBench has
never heard of, something from a paper's repo:

```python
backbone = CustomBackbone(my_model, preprocess=my_transform, name="mine")
visbench.run(backbone, "retrieval", dataset)
```

`hash_weights()` keys the cache on the parameters themselves, so a fine-tuned
checkpoint automatically gets a different cache entry from the model it was
fine-tuned from. `register_backbone` is a top-level export for anyone who would
rather have a registry name.

**`examples/custom_backbone.py` closes the first of these** — it wraps
torchvision's ResNet-18, shows the cache key moving when the weights change,
and registers a named subclass, all through ordinary `visbench.run()` calls on
generated data that needs no download.

What remains:

| gap | note |
| --- | --- |
| Not reachable from the CLI | `--backbone` is a registry name, and a string cannot carry an `nn.Module`. Registering a named `BaseBackbone` subclass is the workaround, and the example demonstrates it |
| Fine-tuning does not apply | `finetune_blocks` is DINOv2-only by design and raises elsewhere |

**One thing the example measured that was not previously written down**: a
wrapped model is constructed *before* `run()` seeds, while a registry name is
constructed *after*, so a trained probe's head is initialised from a different
RNG state on the two paths. Measured on bit-identical features (max absolute
difference 0.0), classification top-1 came out 0.9125 wrapped against 0.9062
named — and each path's own spread across five seeds is 0.0062, so this is RNG
jitter rather than a cost of wrapping. The wrapped path is *perfectly*
reproducible, which is the opposite of what the hazard sounds like. Zero-shot
probes are identical bit for bit, since no head is fitted.

### If these are picked up, this order

Ordered by cost against what they prevent, not by preference:

1. ~~**`examples/custom_backbone.py`**~~ — **done.** It closed a gap between
   what the docs promised and what was demonstrated, and measuring it turned up
   the RNG-position note above.
2. ~~**`visbench show`**~~ — **done** in v0.9 (steps 9a-9d).
3. ~~**The dataset bridges**~~ — **done.** `TorchvisionDataset` /
   `HuggingFaceDataset` plus `--dataset torchvision:… | hf:…` on the image-level
   probes. Dense/pair/triplet probes stay folder-only for now — an HF dataset
   carrying a dense target is a much larger surface (per-probe target-column
   plumbing, loader/dtype selection, the four validity conventions).

**The whole library-surface backlog is now closed.**
