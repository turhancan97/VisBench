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
> and [the roadmap](https://github.com/turhancan97/VisBench/blob/main/docs/roadmap.md).

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

## Prior art

VisBench reuses established protocols rather than re-deriving them, and cites
them at the point of use in the code:

- **[probe3d](https://arxiv.org/abs/2404.08476)** (El Banani et al., CVPR 2024)
  — evaluation protocols for depth, surface normal and correspondence.
- **[Probing the Mid-level Vision Capabilities of Self-Supervised Learning](https://arxiv.org/abs/2411.17474)**
  (Chen, Marks & Cheng) — the task categorization used throughout.
- **[vismatch](https://github.com/gmberton/vismatch)** (Berton) — API
  philosophy, and the matching logic mirrored in the correspondence task.

## Where to go next

| | |
| --- | --- |
| Every probe, its data layout and its measured numbers | [docs/tasks.md](https://github.com/turhancan97/VisBench/blob/main/docs/tasks.md) |
| Twelve probes against six backbones, ranked | [LEADERBOARD.md](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md) |
| How it was built, and what might come next | [docs/roadmap.md](https://github.com/turhancan97/VisBench/blob/main/docs/roadmap.md) |
| What changed in each release | [CHANGELOG.md](https://github.com/turhancan97/VisBench/blob/main/CHANGELOG.md) |
| Borrowed evaluation protocols and their licences | [NOTICE](https://github.com/turhancan97/VisBench/blob/main/NOTICE) |

Questions and bug reports: [open an issue](https://github.com/turhancan97/VisBench/issues).

## License

MIT — see [LICENSE](https://github.com/turhancan97/VisBench/blob/main/LICENSE).

VisBench borrows evaluation protocols from prior work, all permissively
licensed and MIT-compatible; [NOTICE](https://github.com/turhancan97/VisBench/blob/main/NOTICE) records what came from where.
Backbone weights are downloaded at runtime, never redistributed here, and
carry their own upstream terms.
