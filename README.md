<p align="center">
  <img src="assets/visbench-logo-light.svg#gh-light-mode-only" alt="VisBench" width="420">
  <img src="assets/visbench-logo-dark.svg#gh-dark-mode-only" alt="VisBench" width="420">
</p>

<p align="center">
  <em>Probe any vision backbone across high-, mid- and low-level computer vision tasks.</em>
</p>

<p align="center">
  <a href="https://github.com/turhancan97/VisBench/actions/workflows/ci.yml"><img src="https://github.com/turhancan97/VisBench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10+">
</p>

---

> **Status: v0.2.0.** Three backbone families (DINOv2, CLIP, timm CNNs) and
> eight tasks run end-to-end, including four trained dense probes, from Python
> or from the `visbench` command line. Not yet on PyPI — install from source.
> See [Build order](#build-order).

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
| | detection | v0.3 |
| **Mid-level** — geometry & generic structure | geometric correspondence | v0.1 |
| | depth, surface normals, generic (binary) segmentation, mid-level similarity | v0.2 |
| **Low-level** — signal properties | edge detection, optical flow, texture, IQA | v0.3+, [scope only](visbench/tasks/low_level/README.md) |

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

## Roadmap

**v0.1** — prove the abstraction. DINOv2 + CLIP. Zero-shot or linear-probe-on-cached-features only; no fine-tuning, no dense training loops. Deferred: CLI, custom backbones, ResNet/timm, multi-layer extraction.

**v0.2** — ResNet/timm + custom backbones *(done)*, pluggable heads (linear + DPT) *(done)*, multi-layer extraction *(done)*, depth estimation *(done)*, surface normals *(done)*, generic (binary) segmentation *(done)*, semantic segmentation *(done)*, mid-level similarity *(done)*, CLI *(done)*.

**v0.3** — opt-in fine-tuning of the last N blocks, detection groundwork, HF Hub probe sharing and a public leaderboard.

## Reproducibility

Every run logs a structured JSON record — backbone, weights key, task, dataset,
pooling, feature mode, metrics, seed, timestamp — under one schema from v0.1,
so leaderboard tooling never needs a retrofit. Dependencies are pinned in
[`uv.lock`](uv.lock) — exact versions and hashes for every platform, covering
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

Release notes live in [CHANGELOG.md](CHANGELOG.md); each released section is
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
pip install visbench          # not yet published — v0.2.0 is tagged, not uploaded
```

Development:

```bash
git clone https://github.com/turhancan97/VisBench && cd VisBench
pip install -e ".[dev,clip]"
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

[`examples/classify.py`](examples/classify.py) runs the whole path on any
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

[`examples/retrieve.py`](examples/retrieve.py) does the zero-shot version —
no training at all, every image queries every other by cosine similarity:

```bash
python examples/retrieve.py --data /path/to/dataset --split val
python examples/retrieve.py --data /path/to/dataset --split val --pooling mean
```

Both examples share one cache, so running retrieval after classification on
the same split costs nothing but the ranking.

[`examples/correspond.py`](examples/correspond.py) runs the mid-level task —
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

[`examples/similarity.py`](examples/similarity.py) asks whether the backbone
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

### Dense tasks

[`examples/depth.py`](examples/depth.py),
[`examples/normals.py`](examples/normals.py),
[`examples/segment.py`](examples/segment.py) and
[`examples/segment_semantic.py`](examples/segment_semantic.py) train a probe
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

MIT — see [LICENSE](LICENSE).

VisBench borrows evaluation protocols from prior work, all permissively
licensed and MIT-compatible; [NOTICE](NOTICE) records what came from where.
Backbone weights are downloaded at runtime, never redistributed here, and
carry their own upstream terms.
