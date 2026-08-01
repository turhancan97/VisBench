<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/turhancan97/VisBench/main/assets/visbench-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/turhancan97/VisBench/main/assets/visbench-logo-light.svg" alt="VisBench" width="420">
  </picture>
</p>

<p align="center">
  <em>Probe any vision backbone across high-, mid- and low-level computer vision tasks.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/visbench/"><img src="https://img.shields.io/pypi/v/visbench.svg" alt="PyPI"></a>
  <a href="https://github.com/turhancan97/VisBench/actions/workflows/ci.yml"><img src="https://github.com/turhancan97/VisBench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/turhancan97/VisBench/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10+">
</p>

---

> **Status: v0.5.0, on PyPI.** Three backbone families (DINOv2, CLIP, timm CNNs) and
> twelve tasks run end-to-end across all three levels — high, mid and low —
> including seven trained dense probes and an anchor-free detection probe, from
> Python or from the `visbench` command line. v0.5 adds 2D keypoint detection
> and occlusion-edge detection, and reads the Taskonomy domains that need
> `mask_valid/`; v0.4 filled the low-level tier with edge detection; v0.3 added
> opt-in fine-tuning of the last N blocks — a *different measurement* from a
> frozen probe, kept apart in the record rather than averaged with it. See
> [Build order](#build-order).

## What it is

VisBench answers one question with as little ceremony as possible: *what does
this vision backbone actually encode?*

Working today — folder to scored, logged metrics, on any image folder laid out
as `root/<class_name>/<image>`:

```python
import visbench
from visbench.data import ImageFolderDataset

result = visbench.run(
    "dinov2_vitb14",
    "retrieval",
    ImageFolderDataset("data/tiny", split="val"),
    results="results/visbench.jsonl",
)
result.metrics    # {"recall@1": 0.94, "recall@5": 0.99, "mAP": 0.87}
result.record     # the ResultRecord that says exactly how they were produced
```

`run()` resolves pooling, extracts through the cache, fits the probe if it
trains, evaluates, and appends the record. The pieces are public if you want
them separately:

```python
from visbench.cache import FeatureCache

backbone = visbench.get_backbone("dinov2_vitb14")      # frozen, eval mode
probe    = visbench.get_probe("retrieval")             # zero-shot
features = FeatureCache().extract_dataset(
    backbone, dataset, pooling=probe.pooling, keep="pooled"
)                                                      # one forward pass per image
probe.evaluate(features, dataset.labels())
```

Re-running is cheap. On Imagenette (13,394 images, DINOv2 ViT-S, one V100):

| | cold | cached |
|---|---|---|
| wall time | 208 s | **26 s** |
| on-disk cache | 107 MB | — |
| val top1 | 0.9939 | 0.9939 |

A cached image is resolved from its file identity and never decoded, and
`keep="pooled"` also stops dense features being written — storing them for a
task that never reads them cost 5 GB instead of 107 MB. Results go to JSONL
through `visbench.results.ResultWriter`, under one schema from the first
record.

Trained probes take the same call with a training split. A train/test split is
just two datasets, so each half carries its own fingerprint:

```python
result = visbench.run(
    "dinov2_vitb14", "classification", val_dataset, train_dataset=train_dataset
)
result.metrics             # {"top1": ..., "top5": ...}
result.probe.train_top1    # 0.99 — if this is low, the probe underfitted,
                           # not the backbone. Raise `lr` or `epochs`.
```

Passing `train_dataset` to a zero-shot task raises rather than being ignored:
silently dropping it would leave the caller's intent and the result
disagreeing.

The linear probe trains with AdamW on cached features, so its hyperparameters
are part of the reported number and travel with it in the record's
`task_params`.

### Your own model

Any `nn.Module` works, without adding anything to this package:

```python
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
backbone = visbench.CustomBackbone(
    convnext_tiny(weights=weights).features,
    preprocess=weights.transforms(),
    name="convnext_tiny",
)
visbench.run(backbone, "retrieval", dataset)
```

The grid comes from the module's output shape, `embed_dim` from the first
forward pass, and the cache key from a hash of the weights — so a fine-tuned
checkpoint never reuses its parent's cached features. Where the output shape is
genuinely ambiguous VisBench raises rather than guesses; pass `patch_size=`,
`has_cls_token=` or a `feature_fn=` to say what it cannot infer.

To give a custom backbone a registry name, subclass `BaseBackbone` and apply
`@visbench.register_backbone("my_model")` — the same path the built-ins use.

Sibling project to [vismatch](https://github.com/gmberton/vismatch) — same
ergonomic philosophy, applied to representation probing instead of image
matching.

## Design in three points

**One extraction method.** `backbone.extract_features(image, pooling=..., layers=...)`
returns `{"dense": (B, C, H, W), "pooled": (B, C), "grid_hw": (H, W)}` — both
representations from one forward pass. ViTs and CNNs share the exact same
signature and return shape despite completely different internals.

**Tasks choose pooling, backbones don't.** A task passes `pooling="cls"` or
`"mean"` down into extraction. Backbones stay dumb and interchangeable; the
"what representation does this task need" decision lives in one place.

**The cache is not optional.** Disk-backed, keyed by
`(image_hash, backbone_name, layer, pooling)`. Every task reads through it, so
the backbone forward pass runs at most once per image per backbone.

## Task levels

Following [Chen, Marks & Cheng (arXiv:2411.17474)](https://arxiv.org/abs/2411.17474):

| Level | Tasks | Status |
|---|---|---|
| **High-level** — semantic / category | classification, retrieval | v0.1 |
| | semantic (multi-class) segmentation | v0.2 |
| | detection (anchor-free, single-scale) | v0.3 |
| **Mid-level** — geometry & generic structure | geometric correspondence | v0.1 |
| | depth, surface normals, generic (binary) segmentation, mid-level similarity | v0.2 |
| | occlusion-edge detection | v0.5 |
| **Low-level** — signal properties | edge detection (dense magnitude regression) | v0.4 |
| | 2D keypoint detection | v0.5 |
| | optical flow, texture, IQA | [scope only](https://github.com/turhancan97/VisBench/blob/main/visbench/tasks/low_level/README.md) |

Mid-level is where VisBench aims to be strongest relative to existing tooling.
Note that **mid-level image similarity and high-level retrieval are separate
tasks** — one judges perceptual/geometric resemblance, the other category
membership.

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

## Roadmap

**v0.1** — prove the abstraction. DINOv2 + CLIP. Zero-shot or linear-probe-on-cached-features only; no fine-tuning, no dense training loops. Deferred: CLI, custom backbones, ResNet/timm, multi-layer extraction.

**v0.2** — ResNet/timm + custom backbones *(done)*, pluggable heads (linear + DPT) *(done)*, multi-layer extraction *(done)*, depth estimation *(done)*, surface normals *(done)*, generic (binary) segmentation *(done)*, semantic segmentation *(done)*, mid-level similarity *(done)*, CLI *(done)*.

**v0.3** — opt-in fine-tuning of the last N blocks *(done)*, prefix caching *(done)*, detection *(done)*.

**v0.4** — edge detection, the first low-level task *(done)*.

**v0.5** — Taskonomy's `mask_valid/` and the four domains it unblocks *(done)*, 2D keypoint detection *(done)*, occlusion-edge detection *(done)*.

**Next** — HF Hub probe sharing and a public leaderboard.

## Reproducibility

Every run logs a structured JSON record — backbone, weights key, task, dataset,
pooling, feature mode, metrics, seed, timestamp — under one schema from v0.1,
so leaderboard tooling never needs a retrofit. Dependencies are pinned in
[`uv.lock`](https://github.com/turhancan97/VisBench/blob/main/uv.lock) — exact versions and hashes for every platform, covering
the `clip` and `dev` extras too — and CI fails if it drifts from
`pyproject.toml`. The ranges in `pyproject.toml` carry upper bounds so that a
minor dependency release cannot quietly move reported numbers even when
installing without the lock:

```bash
uv sync --all-extras     # exact locked versions
pip install -e ".[dev,clip]"   # ranges, for day-to-day work
```

Backbone weights are pinned the same way. DINOv2 loads from a fixed upstream
commit rather than the default branch, and that ref is part of the cache key —
so bumping it invalidates every stale entry instead of silently serving
features from the old weights. Pass `checkpoint=` to load local weights; the
cache key then carries a hash of that file instead. CLIP's cache key carries
its pretrained tag, since `openai` and `laion2b` are different models behind
one name.

CLIP returns the **pre-projection** CLS token by default, not the 512-d
image-text embedding. The projection is trained to discard whatever does not
help match a caption, which is exactly what a mid-level probe measures, and
DINOv2 has no equivalent head to compare against. `use_projection=True` gets
the projected vector, under its own cache key.

## Changelog

Release notes live in [CHANGELOG.md](https://github.com/turhancan97/VisBench/blob/main/CHANGELOG.md); each released section is
written to stand alone, so it doubles as the GitHub release body.

## Prior art

VisBench reuses established protocols rather than re-deriving them, and cites
them at the point of use in the code:

- **[probe3d](https://arxiv.org/abs/2404.08476)** (El Banani et al., CVPR 2024)
  — evaluation protocols for depth, surface normal and correspondence.
- **[Probing the Mid-level Vision Capabilities of Self-Supervised Learning](https://arxiv.org/abs/2411.17474)**
  (Chen, Marks & Cheng) — the task categorization used throughout.
- **[vismatch](https://github.com/gmberton/vismatch)** (Berton) — API
  philosophy, and the matching logic mirrored in the correspondence task.

## Install

```bash
pip install visbench                 # core: DINOv2, every task, the CLI
pip install 'visbench[clip,timm]'    # + CLIP and timm CNN backbones
```

`clip` and `timm` are optional extras. A backbone whose extra is missing stays
listed — `visbench list backbones` marks it — and constructing one tells you
which extra to install rather than pretending the name does not exist.

Development:

```bash
git clone https://github.com/turhancan97/VisBench && cd VisBench
uv sync --all-extras            # exact locked versions — what the numbers below used
# or
pip install -e ".[dev,clip,timm]"
pytest              # fast tests, no weights downloaded
pytest -m slow      # also runs the real DINOv2 and CLIP checkpoints

# The three gating lint steps, exactly as CI runs them. Run them verbatim —
# mypy in particular reads [tool.mypy] from pyproject.toml, so invoking it
# with different flags checks something CI does not.
ruff check visbench/ tests/ conftest.py examples/
ruff format --check visbench/ tests/ conftest.py examples/
mypy visbench/ examples/ --ignore-missing-imports
```

## The command line

Installing the package puts a `visbench` command on your path. It is a thin
wrapper over `visbench.run()` — same cache, same result records, same numbers.

```bash
visbench list                       # backbones, probes and heads that exist
visbench run retrieval --data /path/to/imagenette2 --split val
visbench cache stats
```

Each probe is its own subcommand, because they do not take the same data.
`visbench run depth --help` shows the folder layout depth expects and only
depth's flags:

```bash
# mid-level geometry, zero-shot, no annotation needed
visbench run correspondence --data /path/to/images --split val --limit 200

# a dense probe: <data>/<split>/{images,masks}, paired by filename stem
visbench run generic_segmentation --data /path/to/data --epochs 40 --lr 5e-3

# an official split list instead of split directories — how real benchmarks
# express one. Passing --stems makes --data the dataset root itself.
visbench run semantic_segmentation --data VOCdevkit/VOC2012 \
    --image-dir JPEGImages --target-dir SegmentationClass \
    --stems ImageSets/Segmentation/val.txt \
    --train-stems ImageSets/Segmentation/train.txt \
    --num-classes 21 --backbone dinov2_vits14

# detection reads the same way, from ImageSets/Main
visbench run detection --data VOCdevkit/VOC2012 \
    --stems ImageSets/Main/val.txt \
    --train-stems ImageSets/Main/train.txt \
    --backbone dinov2_vits14
```

That last one reports `miou 0.733` on VOC val, against the 0.732 the Python API
records for the same backbone — which is the check that matters for a wrapper.

Two flags worth knowing. `--batch-size` is the *extraction* batch;
`--train-batch-size` is the head's, and they are separate because they are
different numbers with the same name. `--limit` shortens a split correctly for
whatever kind of split it is — per class on a labelled folder, by triplet for
similarity, by stem for a dense split — rather than taking a prefix, which on a
class-grouped folder would leave you evaluating one class and scoring 1.0.

## Try it on your own data

[`examples/classify.py`](https://github.com/turhancan97/VisBench/blob/main/examples/classify.py) runs the whole path on any
folder laid out as `<data>/train/<class>/…` and `<data>/val/<class>/…`:

```bash
pip install -e .                                   # required: the script imports visbench
python examples/classify.py --data /path/to/dataset
python examples/classify.py --data /path/to/dataset --limit 20   # 20 images per class, quick
```

The first run extracts features; every later run on the same data reads them
from disk and the backbone never executes, so sweeping probe settings costs
only the probe:

```bash
python examples/classify.py --data /path/to/dataset --epochs 500 --lr 0.05
```

It prints `train top1` next to the validation score. If the validation number
is low *and* `train top1` is low, the probe underfitted — raise `--lr` or
`--epochs`. If `train top1` is near 1.0, the backbone genuinely does not
separate those classes.

[`examples/retrieve.py`](https://github.com/turhancan97/VisBench/blob/main/examples/retrieve.py) does the zero-shot version —
no training at all, every image queries every other by cosine similarity:

```bash
python examples/retrieve.py --data /path/to/dataset --split val
python examples/retrieve.py --data /path/to/dataset --split val --pooling mean
```

Both examples share one cache, so running retrieval after classification on
the same split costs nothing but the ranking.

[`examples/correspond.py`](https://github.com/turhancan97/VisBench/blob/main/examples/correspond.py) runs the mid-level task —
also zero-shot, and needing no annotation at all, since each image is warped by
a known homography:

```bash
python examples/correspond.py --data /path/to/folder --limit 50
```

It reports a **ceiling** beside every score: matches can only land on patch
centres, so with 14px patches a target falling between them cannot be hit
exactly. A low `recall@1px` almost always means the grid is coarse, not that
the backbone failed.

### Mid-level image similarity

[`examples/similarity.py`](https://github.com/turhancan97/VisBench/blob/main/examples/similarity.py) asks whether the backbone
agrees with a human about which of two candidates looks more like a reference —
a two-alternative forced choice over [NIGHTS](https://dreamsim-nights.github.io)
(Fu et al., *DreamSim*). Also zero-shot: the probe is two cosine similarities
and a comparison, with no head and no training split.

```bash
python examples/similarity.py --data /path/to/nights
python examples/similarity.py --data ... --split test_no_imagenet
```

**This is not retrieval.** The ground truth is perceptual — layout, pose,
structure — not category membership, which is why the two are separate tasks. A
backbone can be strong at one and ordinary at the other.

Measured on the NIGHTS test split (1,824 triplets, `min_votes=6`), pooled
features at 224px. Humans chose "right" 49.1% of the time, so chance is ~51%:

| backbone | accuracy | f1 |
| --- | --- | --- |
| dinov2_vits14 | **0.870** | 0.869 |
| dinov2_vitb14 | 0.858 | 0.858 |
| clip_vitb16 | 0.828 | 0.827 |
| resnet50 | 0.827 | 0.828 |

**The small DINOv2 beats the base one here** — the reverse of semantic
segmentation, where B leads S (0.753 against 0.732). Two tasks, two orderings,
same four backbones: which is the entire reason for probing more than one level
rather than assuming a single ranking of representations.

Run `--split test_imagenet` and `test_no_imagenet` before quoting a number: they
partition the test set by whether the reference came from ImageNet, so a gap
between them is a contamination signal rather than a similarity result. For
`dinov2_vits14` that gap is **0.882 against 0.854** — worth knowing before
reading 0.870 as a clean measure of perceptual alignment.

### Detection

[`examples/detect.py`](https://github.com/turhancan97/VisBench/blob/main/examples/detect.py) trains an **anchor-free,
single-scale** box head on frozen dense features: two 1x1 convolutions over the
patch grid, FCOS-style centre-inside-box assignment, focal loss on the classes
and GIoU on the boxes. It reads the Pascal VOC devkit directly, using
`ImageSets/**Main**` — the detection split, roughly four times the segmentation
one:

```bash
python examples/detect.py --data /path/to/pascal_voc --voc
```

**Read these numbers against another backbone, never against published VOC
detectors.** A single-scale head has no feature pyramid, so small objects fall
between grid cells and are simply unrecoverable. That ceiling is the point: the
probe measures what a frozen representation carries, and every point an FPN
would add is a point about the FPN. Records say
`protocol: "visbench_anchor_free_det"` so the number cannot be mistaken for a
detector's.

Measured on VOC 2012, 600 train / 600 val images at 224px, ten epochs:

| metric | DINOv2-S/14 | DINOv2-B/14 |
| --- | --- | --- |
| `map_50` (VOC-comparable) | 0.213 | **0.262** |
| `map_50_95` (COCO-*style*) | 0.072 | **0.093** |
| `classes_scored` | 20 of 20 | 20 of 20 |
| `detections_per_image` | 84.6 | 88.5 |
| `train_loss` | 1.208 | 1.112 |

`map_50` follows VOC's protocol as `VOCevaldet.m` defines it, cross-checked
against a literal transcription of that MATLAB over 3,060 generated APs with
zero mismatches. `map_50_95` averages COCO's ten IoU thresholds but integrates
all recall points at each, where COCO quantises recall to 101 — so it is
COCO-*style*, not a COCO number.

DINOv2-B leads DINOv2-S here by 4.9 mAP@50, the same direction as semantic
segmentation and the opposite of mid-level similarity. That is a recorded
observation, not a check the probe passed — see the similarity numbers above
for why "did the bigger model win?" is not a way to validate a task.

Two things that are protocol rather than detail:

- **`difficult` objects are ignored, not dropped.** VOC removes a detection
  matching one from the tally entirely; dropping those boxes from the ground
  truth instead scores **4.3 mAP lower** on VOC val and reads as a weaker
  detector. So the scored split is built with `include_difficult=True` and the
  training split without — the example and the CLI both do this.
- **`classes_scored` is mAP's real denominator.** A class with no non-difficult
  objects in the split has undefined AP and is excluded rather than scored 0.
  Check it matches before comparing two runs.

### Edge detection

[`examples/edges.py`](https://github.com/turhancan97/VisBench/blob/main/examples/edges.py) is the first
**low-level** probe: dense edge-magnitude regression on
[Taskonomy](https://arxiv.org/abs/1804.08328)'s `edge_texture` maps.

```bash
python examples/edges.py --data /path/to/taskonomy --limit 600
visbench run edge --data /path/to/taskonomy --limit 600
```

Taskonomy's splits are **disjoint by building** — 25 rooms train, 4 validate,
5 test — so a val number is measured in rooms the probe has never seen.

600 train / 600 val frames at 224px, linear head, ten epochs:

| metric | DINOv2-S/14 | DINOv2-B/14 |
| --- | --- | --- |
| `edge_correlation` (quote this) | **0.4558** | 0.4481 |
| `rmse` | 0.9226 | 0.9265 |
| `mae` | 0.5028 | 0.4972 |
| `train_loss` | 0.5721 | 0.5631 |

DINOv2-S edges out DINOv2-B here, by 0.008 — the same ordering as mid-level
similarity and the opposite of segmentation and detection. The margin is small
enough to be worth stating as *consistent with* the level taxonomy rather than
as evidence for it.

Three things that are protocol rather than detail:

- **Quote `edge_correlation`, not `rmse`.** Edge magnitude is concentrated near
  zero, so a probe that ignores its input and predicts the split's mean
  everywhere gets a *small* RMSE while having learned nothing. Pearson
  correlation is invariant to scale and offset, so it asks only whether the
  representation knows **where** the edges are, and scores that probe 0. RMSE
  and MAE are reported alongside because correlation is blind to the opposite
  failure — right shape, wrong magnitude.
- **Nothing is masked.** Depth has holes and normals have zero-length vectors,
  both meaning "no ground truth". Here 0 means *no edge*, a real reading
  covering most of most frames. Masking it away would score the probe only
  where an edge already is.
- **This is not BSDS500's protocol and must not share a table with one.** BSDS's
  ODS/OIS/AP matches edge pixels by bipartite correspondence after non-maximum
  suppression, swept over thresholds, against several annotators. Records say
  `protocol: "visbench_edge_regression"`.

### Keypoints, and occlusion edges

Two more probes share the edge probe's implementation and differ only in what
they read — [`examples/keypoints.py`](https://github.com/turhancan97/VisBench/blob/main/examples/keypoints.py)
on Taskonomy's `keypoints2d` response maps, and
[`examples/occlusion_edges.py`](https://github.com/turhancan97/VisBench/blob/main/examples/occlusion_edges.py)
on its `edge_occlusion` maps.

```bash
visbench run keypoints2d     --data /path/to/taskonomy --limit 600
visbench run occlusion_edge  --data /path/to/taskonomy --limit 600
```

600 train / 600 val frames at 224px, linear head, ten epochs:

| probe | level | DINOv2-S/14 | DINOv2-B/14 |
| --- | --- | --- | --- |
| `keypoints2d` (`keypoint_correlation`) | low | **0.2356** | 0.2248 |
| `occlusion_edge` (`occlusion_edge_correlation`) | mid | 0.2924 | **0.3167** |

**The occlusion-edge probe is mid-level and the texture-edge probe is
low-level, and they are otherwise the same code.** An occlusion edge is a depth
discontinuity — a painted line on a wall is not one, and the silhouette of a
chair against a similarly-toned wall is one with almost no intensity gradient —
so recovering it needs scene geometry. Running both on the same frames is about
as direct a comparison of the two tiers as VisBench offers.

Two things worth knowing before adding a fourth probe of this shape:

- **`edge_occlusion` is loaded in log space, and nothing else is.** Its target
  holds 46% of its mass in the strongest 1% of pixels, against ~0.10 for the
  other two. At that tail the L1 loss (chosen so strong pixels cannot dominate)
  and the Pearson metric (dominated by exactly those pixels) pull apart: the
  linear-target probe scored 0.088 and stayed flat under four target scales, 4x
  the training budget and a 10x learning rate. `dataset_params` records
  `target_transform`, so a log-space correlation can never be pooled with a
  linear-space one.
- **Ask whether a probe ranks, not whether its number is high.** The linear
  occlusion probe's DINOv2-S-versus-B gap was 0.0035 — noise. A low score can be
  by design; failing to separate two backbones never is.

### Dense tasks

[`examples/depth.py`](https://github.com/turhancan97/VisBench/blob/main/examples/depth.py),
[`examples/normals.py`](https://github.com/turhancan97/VisBench/blob/main/examples/normals.py),
[`examples/segment.py`](https://github.com/turhancan97/VisBench/blob/main/examples/segment.py) and
[`examples/segment_semantic.py`](https://github.com/turhancan97/VisBench/blob/main/examples/segment_semantic.py) train a probe
head on frozen dense features. Depth and normals follow
[probe3d](https://arxiv.org/abs/2404.08476)'s protocols; both segmentation tasks
borrow only its optimiser schedule, since that paper has neither. They want
images and per-pixel targets paired by filename stem under `train/` and `val/`:

```bash
python examples/depth.py   --data /path/to/dataset --target-scale 1000
python examples/normals.py --data /path/to/dataset --normal-source geonet
python examples/segment.py --data /path/to/dataset
python examples/segment_semantic.py --data /path/to/dataset --num-classes 21
```

Semantic segmentation also reads the Pascal VOC devkit directly, using the
official split lists rather than whatever the folders contain:

```bash
python examples/segment_semantic.py --data /path/to/pascal_voc --voc
```

Measured on VOC 2012 val (1449 images), linear head, 224px, at the default
ten-epoch schedule:

| metric | DINOv2-S/14 | DINOv2-B/14 |
| --- | --- | --- |
| `miou` (dataset-level) | 0.732 | **0.753** |
| `miou_per_image` | 0.683 | 0.712 |
| `pixel_acc` | 0.926 | 0.931 |
| `mean_acc` | 0.831 | 0.838 |
| `train_loss` | 0.193 | 0.166 |

**Report the linear head.** It is the default and the only one under which a
difference between two backbones is a difference between two *feature maps*.
The DPT head is probe3d's own choice and scores higher for everyone, so run
both and say which:

```bash
python examples/normals.py --data ... --head dpt --layers 2 5 8 11
```

Features are shared between the three tasks when the images and `--image-size`
match, so probing all of them on one dataset costs one extraction. Splits larger
than memory are fine — dense features stream from the cache a batch at a time
rather than being stacked.

Things that will bite otherwise:

- **Say where surface normals came from.** NYU's are derived (GeoNet's
  extraction, or Ladicky's) rather than sensed, and the sources disagree enough
  to move every metric. `--normal-source` is recorded verbatim in the result.
- Surface normals default to probe3d's uncertainty-aware loss, which has a
  failure mode near chance accuracy where it all but switches its own
  supervision off. VisBench detects it and warns; `--no-uncertainty` is the way
  out. See `SurfaceNormalTask.fit` for the measured dynamics.
- **Quote IoU, not pixel accuracy, for segmentation.** Objects are a minority of
  most frames, so a probe predicting background everywhere already scores high
  accuracy and zero IoU. `examples/segment.py` prints the foreground fraction
  and that baseline before it trains, so the comparison is unavoidable.
- **Two mIoUs are reported and they differ.** `miou` accumulates one confusion
  matrix over the whole split, which is what VOC and the literature define;
  `miou_per_image` averages each image's own mIoU, this codebase's convention
  elsewhere. On VOC they sit five points apart. Quote `miou` against published
  numbers, and say which one you mean.
- **Label maps are read without mode conversion, and getting this wrong is
  silent.** VOC's PNGs are palette images whose raw bytes are the class indices;
  resolving the palette turns classes `[0, 1, 15]` into `[0, 38, 147]`, which
  trains and scores perfectly happily against labels that mean nothing. Use
  `load_label_map`, not `load_mask`, for anything multi-class — including
  binarising a VOC map, since `load_mask` would read its void border as
  foreground.
- **The ten-epoch schedule assumes a dataset the size of NYUv2.** On a small
  split it underfits badly — 80 training images gave 0.16 IoU at the defaults
  and 0.87 at `--epochs 40 --lr 5e-3`, on identical features. `train_loss` is
  printed for exactly this: a poor score with a high training loss means the
  probe did not converge, which is a different finding from a representation
  that does not carry the signal.

### Measured on Imagenette

3,925-image val split, one V100. Correspondence on 50 pairs at `max_warp=0.2`.

| task | metric | DINOv2 ViT-S/14 | CLIP ViT-B/16 | ResNet-50 |
|---|---|---|---|---|
| classification | top1 | 0.9939 | 0.9954 | 0.9980\* |
| retrieval | recall@1 | 0.9921 | 0.9893 | 0.9901\* |
| retrieval | mAP | 0.8893 | 0.9102 | 0.9357\* |
| correspondence | recall@1p | 0.7650 | 0.6993 | 0.8443 |
| correspondence | ceiling | 0.9408 | 0.9505 | 0.9709 |
| dense grid @224 | | 16x16 | 14x14 | 7x7 |

**\* Read the ResNet column with care.** Imagenette's ten classes are ImageNet-1k
wnids, and `resnet50.a1_in1k` was trained on ImageNet-1k with labels — it has
seen these exact categories, while DINOv2 is self-supervised and CLIP is
image-text. Its semantic scores are close to in-distribution recall, not a
transfer result. This says more about the dataset than the backbone; a
benchmark comparing supervised against self-supervised features needs data the
supervised model has not been trained on.

Correspondence is less exposed to that (no labels are used), but comes with its
own caveat: ResNet's 7x7 grid means matching among 49 candidates against
DINOv2's 256. Patch-width thresholds make the *error* comparable across grids;
they do not make the matching problem equally hard.

The DINOv2/CLIP split is the cleaner comparison, and it lands where the task
taxonomy predicts: CLIP ahead on the semantic tasks, behind on the geometric
one despite a higher ceiling.

Retrieval with `--pooling mean` instead of CLS costs DINOv2 about 1.8 points of
recall@1 (0.9740, mAP 0.8314).

**Correspondence thresholds are in patch widths (`p`), not pixels.** A match
can only land on a patch centre, so patch spacing is a hard floor on
achievable error — and in pixels that floor moves with every configuration. At
224px on DINOv2 ViT-S/14, `recall@1px` has a *ceiling* of 0.015: the metric
reports patch size, not feature quality. It also makes comparison invalid,
since DINOv2's 14px patches and CLIP ViT-B/16's 16px are different yardsticks
under the same name. Pass `threshold_units="pixel"` to compare against a
published pixel number.

Degradation with viewpoint is gradual — 50 pairs, `recall@1p` as
score/ceiling:

| `max_warp` | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|
| recall@1p | 0.872 | 0.834 | 0.765 | 0.744 | 0.732 |
| ceiling | 1.000 | 0.980 | 0.941 | 0.916 | 0.891 |
| matches kept / pair | 160 | 143 | 115 | 86 | 55 |

The ratio test rejects more as the warp grows (164 → 59 matches), which is the
behaviour it exists for: fewer matches, still mostly correct.

Chance recall@1 is 0.10. Retrieval reused the classification cache: 3,925
hits, 0 misses, 8 s end to end. Switching to `--pooling mean` is a genuine
re-extraction (3,925 misses, 56 s) because pooling is part of the cache key —
and it costs about 1.8 points of recall@1 here, which is the sort of question
these two lines of CLI exist to answer.

Every one of these examples has a `visbench run` equivalent — see
[The command line](#the-command-line). They stay because an example is readable
top to bottom and a subcommand is not: when you want to know *how* a probe is
wired up, the script is the answer.

## License

MIT — see [LICENSE](https://github.com/turhancan97/VisBench/blob/main/LICENSE).

VisBench borrows evaluation protocols from prior work, all permissively
licensed and MIT-compatible; [NOTICE](https://github.com/turhancan97/VisBench/blob/main/NOTICE) records what came from where.
Backbone weights are downloaded at runtime, never redistributed here, and
carry their own upstream terms.
