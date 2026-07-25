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
  <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.9+">
</p>

---

> **Status: v0.1.0 released; v0.2 in progress.** Three backbone families
> (DINOv2, CLIP, timm CNNs) and all three v0.1 tasks run end-to-end on a local
> image folder. Not yet on PyPI. See [Build order](#build-order).

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
- [ ] **5e.** rest of v0.2 — surface normals, generic segmentation, semantic
      segmentation, mid-level similarity, CLI

## Roadmap

**v0.1** — prove the abstraction. DINOv2 + CLIP. Zero-shot or linear-probe-on-cached-features only; no fine-tuning, no dense training loops. Deferred: CLI, custom backbones, ResNet/timm, multi-layer extraction.

**v0.2** — ResNet/timm + custom backbones *(done)*, pluggable heads (linear + DPT) *(done)*, multi-layer extraction *(done)*, depth estimation *(done)*, remaining dense mid-level tasks (probe3d protocols), CLI.

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
pip install visbench          # not yet published — v0.1 target
```

Development:

```bash
git clone https://github.com/turhancan97/VisBench && cd VisBench
pip install -e ".[dev,clip]"
pytest              # fast tests, no weights downloaded
pytest -m slow      # also runs the real DINOv2 and CLIP checkpoints
```

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

These examples are not the CLI: the packaged `visbench` command stays deferred
to v0.2 until the Python API settles.

## License

MIT — see [LICENSE](LICENSE).

VisBench borrows evaluation protocols from prior work, all permissively
licensed and MIT-compatible; [NOTICE](NOTICE) records what came from where.
Backbone weights are downloaded at runtime, never redistributed here, and
carry their own upstream terms.
