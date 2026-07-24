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

> **Status: v0.1.0.dev0 — two of three v0.1 tasks work.** DINOv2, the feature
> cache, zero-shot retrieval and the classification linear probe all run
> end-to-end on a local image folder. Correspondence and CLIP are next. See
> [Build order](#build-order).

## What it is

VisBench answers one question with as little ceremony as possible: *what does
this vision backbone actually encode?*

Working today — folder to scored metrics, on any image folder laid out as
`root/<class_name>/<image>`:

```python
import visbench
from visbench.cache import FeatureCache
from visbench.data import ImageFolderDataset

backbone = visbench.get_backbone("dinov2_vitb14")      # frozen, eval mode
probe    = visbench.get_probe("retrieval")             # zero-shot
dataset  = ImageFolderDataset("data/tiny", split="val")

features = FeatureCache().extract_dataset(
    backbone, dataset, pooling=probe.pooling, keep="pooled"
)                                                      # one forward pass per image
probe.evaluate(features, dataset.labels())
# {"recall@1": 0.94, "recall@5": 0.99, "mAP": 0.87}
```

Re-running is much cheaper: the second call reads every feature from disk and
the backbone never executes. On Imagenette (13,394 images, DINOv2 ViT-S, one
V100) a cold run takes ~4 min and a fully cached one ~113 s — cheaper, not
free, because the cache still decodes each image to compute its content hash.
Results go to JSONL through `visbench.results.ResultWriter`, under one schema
from the first record.

Trained probes follow the same shape, with a `fit` on the training split. A
train/test split is just two datasets, so each half carries its own
fingerprint into its own record:

```python
probe = visbench.get_probe("classification")
probe.fit(train_features, train_dataset.labels())
probe.evaluate(test_features, test_dataset.labels())   # {"top1": ..., "top5": ...}

probe.train_top1     # 0.99 — if this is low, the probe underfitted,
                     # not the backbone. Raise `lr` or `epochs`.
```

The linear probe trains with AdamW on cached features, so its hyperparameters
are part of the reported number and travel with it in the record's
`task_params`.

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
- [ ] **4.** Next task, then next backbone — classification done; correspondence
      and CLIP remain
- [ ] **5.** v0.2 scope — only once all of v0.1 is implemented, tested, reviewed

## Roadmap

**v0.1** — prove the abstraction. DINOv2 + CLIP. Zero-shot or linear-probe-on-cached-features only; no fine-tuning, no dense training loops. Deferred: CLI, custom backbones, ResNet/timm, multi-layer extraction.

**v0.2** — ResNet/timm + custom backbones, dense mid-level tasks (probe3d protocols), pluggable heads (linear + DPT), multi-layer extraction, CLI.

**v0.3** — opt-in fine-tuning of the last N blocks, detection groundwork, HF Hub probe sharing and a public leaderboard.

## Reproducibility

Every run logs a structured JSON record — backbone, weights key, task, dataset,
pooling, feature mode, metrics, seed, timestamp — under one schema from v0.1,
so leaderboard tooling never needs a retrofit. Dependencies are pinned via a
lockfile; ranges in `pyproject.toml` carry upper bounds so a minor dependency
release cannot quietly move reported numbers.

Backbone weights are pinned the same way. DINOv2 loads from a fixed upstream
commit rather than the default branch, and that ref is part of the cache key —
so bumping it invalidates every stale entry instead of silently serving
features from the old weights. Pass `checkpoint=` to load local weights; the
cache key then carries a hash of that file instead.

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
pytest -m slow      # also runs the real DINOv2 checkpoint against torch.hub
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

This is an example, not the CLI: the packaged `visbench` command stays
deferred to v0.2 until the Python API settles.

## License

MIT — see [LICENSE](LICENSE).

VisBench borrows evaluation protocols from prior work, all permissively
licensed and MIT-compatible; [NOTICE](NOTICE) records what came from where.
Backbone weights are downloaded at runtime, never redistributed here, and
carry their own upstream terms.
