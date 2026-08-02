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

> **Status: v0.6.1, on PyPI.** Three backbone families (DINOv2, CLIP, timm CNNs) and
> twelve tasks run end-to-end across all three levels — high, mid and low —
> including seven trained dense probes and an anchor-free detection probe, from
> Python or from the `visbench` command line. **v0.6 is the leaderboard
> release**: a committed corpus of twelve probes against six backbones, the
> comparability rules that decide what may be ranked together, README tables
> generated from those records, and probe heads you can save, publish and
> reload. v0.5 added 2D keypoint detection and occlusion-edge detection; v0.4
> filled the low-level tier with edge detection; v0.3 added opt-in fine-tuning
> of the last N blocks — a *different measurement* from a frozen probe, kept
> apart in the record rather than averaged with it. See
> [LEADERBOARD.md](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md)
> and [Build order](#build-order).

## Try it in thirty seconds

No dataset, no configuration, no large download:

```bash
pip install visbench
visbench demo
```

```text
drawing 20 images per class for 4 shapes...
loading resnet18 (torchvision, ~45 MB on first run)...
running the classification probe...

  top1         0.8125

  chance is 0.25 — the shapes differ in outline only.
```

That is a real probe, on a real pretrained backbone, through the same code path
every other run uses. The images are generated: four shapes with **colour,
size, position and rotation randomised**, so only geometry identifies a class
and a backbone that has not learned shape scores about chance.

The number is deliberately not 1.0. Turn the difficulty up and watch it fall:

```bash
visbench demo --noise 90      # top1 ~0.31, against a chance of 0.25
```

| `--noise` | 28 | **45** (default) | 60 | 75 | 90 |
|---|---|---|---|---|---|
| top1 | 0.975 | **0.812** | 0.550 | 0.438 | 0.312 |

A probe whose score does not move when you destroy the signal is not measuring
the signal. That slide into chance is the demo's actual point.

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
`DenseMagnitudeTask` subclass is most of the work.

| Task | Level | Note |
|---|---|---|
| Corner / blob detection (Harris, DoG) | low | Classical-target counterpart to the learned `keypoints2d` response maps |
| Local orientation / gradient fields (structure tensor, HOG-style) | low | Vector-valued rather than magnitude; closer to surface normals in shape |
| Superpixel / texture segmentation | low | Grouping by local photometric similarity alone, no figure-ground reasoning |

### Reachable with data already common

| Task | Level | Note |
|---|---|---|
| Instance segmentation | high | The category-labelled counterpart to the existing binary segmentation; COCO-style polygon annotations |
| Fine-grained recognition | high | Reuses the existing linear-probe classification path; only the dataset changes |
| Scene classification (Places365) | high | Same, at scene rather than object granularity |
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
- **Texture / reflectance** overlaps with intrinsic decomposition above.
  Taskonomy ships no reflectance domain, so `mask_valid/` did not unblock it.

## Sharing a trained probe

A probe head is small — 17 KB for a linear classifier — so the cheapest way to
let someone check your number is to hand them the probe rather than the recipe.

```python
from visbench.hub import save_probe, load_probe

save_probe(probe, "checkpoints/voc.pt", backbone=backbone)
probe = load_probe("checkpoints/voc.pt", backbone=backbone)
```

With `pip install 'visbench[hub]'`, the same thing over the network. Pushing
creates a **private** repository unless you ask otherwise, and writes a model
card alongside the weights:

```python
from visbench.hub import push_probe, load_probe_from_hub

push_probe(probe, "you/dinov2-vits14-voc", backbone=backbone, metrics=scores)
probe = load_probe_from_hub("you/dinov2-vits14-voc", backbone=backbone)
```

**A head only works with the backbone it was fitted on, and getting that wrong
is silent.** Loading a head trained on DINOv2-S CLS tokens against *mean-pooled*
tokens from the same backbone gives the right shapes and a plausible number —
measured on Imagenette, **0.9540 against 0.9830**. Nothing about the tensors
says anything is wrong, so `load_probe` checks the backbone weights, the
pooling, the feature mode and the layers, and refuses a mismatch. Pass
`strict=False` if you are deliberately testing transfer; it warns rather than
raising, and the number is then comparable with nothing.

Downloaded probes are read with `torch.load(weights_only=True)`, so fetching one
from a stranger's repository cannot execute code. See
[`examples/save_probe.py`](https://github.com/turhancan97/VisBench/blob/main/examples/save_probe.py),
which demonstrates the mismatch on purpose.

## Reproducibility

**Twelve probes against six backbones, as records:
[LEADERBOARD.md](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md).**
Every board there — and every measured table below — is generated from
[`results/corpus/visbench.jsonl`](https://github.com/turhancan97/VisBench/blob/main/results/corpus/visbench.jsonl),
the committed corpus, by
[`scripts/render_tables.py`](https://github.com/turhancan97/VisBench/blob/main/scripts/render_tables.py).
A test in the fast suite fails if any of them drifts from the records, so a
published number and the run behind it cannot disagree.

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
pip install visbench                    # core: DINOv2, every task, the CLI
pip install 'visbench[clip,timm]'       # + CLIP and timm CNN backbones
pip install 'visbench[hub]'             # + push/pull probes to Hugging Face
```

`clip` and `timm` are optional extras. A backbone whose extra is missing stays
listed — `visbench list backbones` marks it — and constructing one tells you
which extra to install rather than pretending the name does not exist.

`hub` is needed only to *transfer* a probe. Saving one to a local file and
loading it back works in a core install.

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
visbench demo                       # a real probe on generated data, no setup
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

<!-- visbench:board task=similarity metrics=accuracy,f1 heading=4 -->
#### similarity

| backbone | `accuracy` | `f1` | `tie_rate` |
| --- | --- | --- | --- |
| `dinov2_vits14` | **0.8701** | **0.8687** | 0.0000 |
| `dinov2_vitb14` | 0.8580 | 0.8575 | 0.0000 |
| `clip_vitb32` | 0.8465 | 0.8443 | 0.0000 |
| `resnet18` | 0.8317 | 0.8307 | 0.0000 |
| `clip_vitb16` | 0.8284 | 0.8266 | 0.0000 |
| `resnet50` | 0.8273 | 0.8282 | 0.0000 |

Ordered by `accuracy`, which **disagrees with `f1`, `recall`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>similarity on nights/test, protocol=midvision_2afc, frozen [0cc388a0]</sub>
<!-- /visbench:board -->

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

<!-- visbench:board task=detection metrics=map_50,map_50_95 heading=4 -->
#### detection

| backbone | `map_50` | `map_50_95` | `classes_scored` | `detections_per_image` |
| --- | --- | --- | --- | --- |
| `dinov2_vitb14` | **0.2895** | **0.0978** | 20 | 88.5217 |
| `dinov2_vits14` | 0.2291 | 0.0702 | 20 | 83.0333 |
| `clip_vitb16` | 0.1894 | 0.0622 | 20 | 88.7500 |
| `clip_vitb32` | 0.1886 | 0.0584 | 20 | 91.3833 |
| `resnet50` | 0.1380 | 0.0420 | 20 | 48.2133 |
| `resnet18` | 0.0912 | 0.0270 | 20 | 57.1033 |

Ordered by `map_50`.

> **Read this first.** Absolute mAP is low by design: the head is anchor-free and single-scale, so it has no feature pyramid and small objects fall between cells. The board ranks representations, which is what it is for — it is not a detector benchmark.

<sub>detection on detection_folder/val, protocol=visbench_anchor_free_det, frozen [4d3fbeb4]</sub>
<!-- /visbench:board -->

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

<!-- visbench:board task=edge metrics=edge_correlation,rmse,mae heading=4 -->
#### edge

| backbone | `edge_correlation` | `rmse` | `mae` |
| --- | --- | --- | --- |
| `clip_vitb16` | **0.4565** | 0.9340 | **0.4882** |
| `dinov2_vits14` | 0.4558 | **0.9226** | 0.5028 |
| `dinov2_vitb14` | 0.4481 | 0.9265 | 0.4972 |
| `clip_vitb32` | 0.3834 | 0.9656 | 0.5080 |
| `resnet50` | 0.3549 | 0.9770 | 0.5056 |
| `resnet18` | 0.3430 | 0.9797 | 0.5153 |

Ordered by `edge_correlation`, which **disagrees with `mae`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>edge on taskonomy_edge_texture/val, protocol=visbench_edge_regression, frozen [f8d2af2a]</sub>
<!-- /visbench:board -->

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

The two probes share every line of their implementation and sit one tier apart,
so they render as two boards rather than one row each:

<!-- visbench:board task=keypoints2d metrics=keypoint_correlation,mae,rmse heading=4 -->
#### keypoints2d

| backbone | `keypoint_correlation` | `mae` | `rmse` |
| --- | --- | --- | --- |
| `dinov2_vits14` | **0.2356** | **1.1281** | **2.5472** |
| `dinov2_vitb14` | 0.2248 | 1.1294 | 2.5541 |
| `clip_vitb16` | 0.2175 | 1.1533 | 2.5770 |
| `clip_vitb32` | 0.1933 | 1.1474 | 2.5891 |
| `resnet50` | 0.1792 | 1.2374 | 2.6163 |
| `resnet18` | 0.1659 | 1.2579 | 2.6282 |

Ordered by `keypoint_correlation`, which **disagrees with `mae`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>keypoints2d on taskonomy_keypoints2d/val, protocol=visbench_keypoint2d_regression, frozen [e647c722]</sub>
<!-- /visbench:board -->

<!-- visbench:board task=occlusion_edge metrics=occlusion_edge_correlation,mae,rmse heading=4 -->
#### occlusion_edge

| backbone | `occlusion_edge_correlation` | `mae` | `rmse` |
| --- | --- | --- | --- |
| `dinov2_vitb14` | **0.3167** | **0.2061** | **0.4315** |
| `dinov2_vits14` | 0.2924 | 0.2205 | 0.4373 |
| `clip_vitb16` | 0.2558 | 0.2149 | 0.4415 |
| `clip_vitb32` | 0.2174 | 0.2203 | 0.4440 |
| `resnet50` | 0.1979 | 0.2294 | 0.4502 |
| `resnet18` | 0.1745 | 0.2418 | 0.4578 |

Ordered by `occlusion_edge_correlation`, which **disagrees with `mae`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>occlusion_edge on taskonomy_edge_occlusion/val, protocol=visbench_occlusion_edge_regression, frozen [d12a4923]</sub>
<!-- /visbench:board -->

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

<!-- visbench:board task=semantic_segmentation metrics=miou,miou_per_image,pixel_acc,mean_acc heading=4 -->
#### semantic_segmentation

| backbone | `miou` | `miou_per_image` | `pixel_acc` | `mean_acc` |
| --- | --- | --- | --- | --- |
| `dinov2_vitb14` | **0.7533** | **0.7161** | **0.9316** | **0.8403** |
| `dinov2_vits14` | 0.7328 | 0.6841 | 0.9267 | 0.8271 |
| `clip_vitb16` | 0.6546 | 0.6683 | 0.9019 | 0.7312 |
| `clip_vitb32` | 0.5813 | 0.6067 | 0.8731 | 0.6633 |
| `resnet50` | 0.4574 | 0.5163 | 0.8322 | 0.5248 |
| `resnet18` | 0.4212 | 0.4497 | 0.8205 | 0.4915 |

Ordered by `miou`.

<sub>semantic_segmentation on VOC2012/val, protocol=visbench_semantic_seg, frozen [e14b47db]</sub>
<!-- /visbench:board -->

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

3,925-image val split, one V100. Correspondence on 200 pairs at `max_warp=0.2`.

<!-- visbench:board task=classification metrics=top1,top5 heading=4 -->
#### classification

| backbone | `top1` | `top5` |
| --- | --- | --- |
| `resnet50` | **0.9980** | 0.9997 |
| `dinov2_vitb14` | 0.9975 | **1.0000** |
| `clip_vitb16` | 0.9954 | 0.9997 |
| `dinov2_vits14` | 0.9939 | 0.9997 |
| `clip_vitb32` | 0.9921 | 0.9992 |
| `resnet18` | 0.9888 | 0.9995 |

Ordered by `top1`, which **disagrees with `top5`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>classification on val/val, frozen [12e02eff]</sub>
<!-- /visbench:board -->

<!-- visbench:board task=retrieval metrics=mAP,recall@1,recall@5 heading=4 -->
#### retrieval

| backbone | `mAP` | `recall@1` | `recall@5` |
| --- | --- | --- | --- |
| `resnet50` | **0.9357** | 0.9901 | **0.9987** |
| `dinov2_vitb14` | 0.9171 | **0.9954** | 0.9977 |
| `clip_vitb16` | 0.9102 | 0.9893 | 0.9975 |
| `dinov2_vits14` | 0.8893 | 0.9921 | 0.9972 |
| `clip_vitb32` | 0.8680 | 0.9806 | 0.9941 |
| `resnet18` | 0.8648 | 0.9725 | 0.9944 |

Ordered by `mAP`, which **disagrees with `recall@1`, `recall@5`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>retrieval on val/val, frozen [eb312a7b]</sub>
<!-- /visbench:board -->

<!-- visbench:board task=correspondence metrics=recall@5px,recall@10px,auc@5px heading=4 -->
#### correspondence

| backbone | `recall@5px` | `recall@10px` | `auc@5px` | `ceiling_auc@5px` | `ceiling_recall@10px` | `ceiling_recall@5px` | `num_matches` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dinov2_vits14` | **0.3049** | **0.6526** | **0.1152** | 0.1454 | 0.9329 | 0.4123 | 23,439 |
| `dinov2_vitb14` | 0.2816 | 0.6260 | 0.1055 | 0.1389 | 0.9264 | 0.4005 | 27,590 |
| `clip_vitb16` | 0.2689 | 0.5725 | 0.1080 | 0.1333 | 0.9159 | 0.3519 | 12,798 |
| `resnet18` | 0.0973 | 0.3256 | 0.0335 | 0.0350 | 0.3653 | 0.1028 | 4,911 |
| `clip_vitb32` | 0.0897 | 0.2951 | 0.0321 | 0.0352 | 0.3633 | 0.1002 | 4,283 |
| `resnet50` | 0.0887 | 0.3003 | 0.0299 | 0.0350 | 0.3595 | 0.1038 | 4,373 |

Ordered by `recall@5px`, which **disagrees with `auc@1px`, `auc@2px`, `auc@5px`, `recall@10px`, `recall@1px`, `recall@2px`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

> **Read this first.** Thresholds are in **pixels**, which is the only unit two backbones can be compared in — a patch width is 14px on DINOv2/14 and 32px on a ResNet, so scoring in patch widths asks each backbone a different question. Read `ceiling_` beside every score: a 7x7 grid cannot place a match within 5px more than ~10% of the time whatever its features are, so part of this ordering is resolution rather than quality. `num_matches` is the denominator each backbone's own ratio test left, and it varies by more than 5x.

<sub>correspondence on val/val, frozen [7db23175]</sub>
<!-- /visbench:board -->

The patch grid at 224px is 16x16 for DINOv2 (patch 14), 14x14 for CLIP-B/16 and
7x7 for CLIP-B/32 and both ResNets. Hold that beside the correspondence board:
it is what `num_matches` tracks, and it moves the score without saying anything
about feature quality.

**Read the ResNet rows with care.** Imagenette's ten classes are ImageNet-1k
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

**Correspondence thresholds are in pixels (`px`), and this changed in v0.6.1.**
Patch widths were the default until then, on the reasoning that a match can
only land on a patch centre so patch spacing is the natural yardstick. That is
true *within* one backbone and wrong across several: a patch is 14px on
DINOv2/14, 16px on CLIP ViT-B/16 and 32px on ViT-B/32 or a ResNet stage, so
`recall@1p` asks a coarse-grid backbone to land within 32px and a fine-grid one
within 14px, then prints both under one name.

It inverted the board. On 200 pairs, `resnet18` read **0.8927** against
`dinov2_vits14`'s 0.7834 in patch widths — and **0.0973 against 0.3049** in
pixels. First and last place swap.

The quantisation floor is real, and the honest handling is the `ceiling_`
metrics that already travel beside every score: a 7×7 grid cannot place a match
within 5px more than ~10% of the time whatever its features are, against ~41%
for a 16×16 grid. That *states* the disadvantage instead of normalising it
away. Pass `--units patch` deliberately for a single-backbone study; do not
rank two backbones with it.

Degradation with viewpoint is gradual. Measured on **DINOv2 ViT-S/14 alone**,
50 pairs, in patch widths — a within-backbone sweep, which is what that unit is
for — as score/ceiling:

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
