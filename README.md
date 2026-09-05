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
  <a href="https://turhancan97.github.io/VisBench/"><img src="https://img.shields.io/badge/docs-visbench-3a7eab.svg" alt="Documentation"></a>
  <a href="https://doi.org/10.5281/zenodo.21822684"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21822684.svg" alt="DOI"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10+">
</p>

---

> **Status: v0.15.0.** Sixteen probes across high, mid and low level, thirteen
> backbones from three families, and a committed corpus of **192 records** —
> every one of them reproducible from the flags in its own record. The full
> documentation, including a generated API reference, is at
> **[https://turhancan97.github.io/VisBench](https://turhancan97.github.io/VisBench/)**.

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
pip install visbench                    # core: DINOv2, every probe, the CLI
pip install 'visbench[clip,timm]'       # + CLIP and timm backbones
pip install 'visbench[hub]'             # + push/pull probes to Hugging Face
```

A backbone whose extra is missing stays **listed** — `visbench list backbones`
marks it — and constructing one tells you which extra to install rather than
pretending the name does not exist.

Full instructions, including a source install and what each extra buys:
[installation](https://turhancan97.github.io/VisBench/getting-started/installation.html).

## What it is

The name is literal: **Vis**(ion) **Bench**(mark). Both halves are a scope
commitment. *Vis* — the subject is a vision backbone's features, not a
multimodal or language model's behaviour; a text tower is only ever used to
build the visual one. *Bench* — the output is a **comparable** number rather
than a score: every run writes a record saying which backbone, dataset,
pooling, layers and protocol produced it, and explicit rules decide which
records may be ranked against each other at all.

VisBench answers one question with as little ceremony as possible: *what does
this vision backbone actually encode?*

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
them separately — see
[the quickstart](https://turhancan97.github.io/VisBench/getting-started/quickstart.html).

Installing the package also puts a `visbench` command on your path, a thin
wrapper over the same call:

```bash
visbench demo                       # a real probe on generated data, no setup
visbench list                       # backbones, probes and heads that exist
visbench run retrieval --data /path/to/imagenette2 --split val
visbench show depth --data /path/to/nyuv2 --out panels.png
```

Each probe is its own subcommand, because they do not take the same data —
[the CLI reference](https://turhancan97.github.io/VisBench/getting-started/cli.html).

## Look before you measure

`visbench show <probe>` draws what a probe saw beside what it predicted. It
measures nothing — it exists because a dense target that has drifted from its
image fails **silently**: the probe trains, and the number merely comes out
mediocre. Two of the most expensive bugs in this project were exactly that, and
both are obvious in one frame.

![Depth panels: image, target, and magenta where there is no ground truth](https://raw.githubusercontent.com/turhancan97/VisBench/main/docs/_static/gallery/depth.png)

![Correspondence: two views with the matches between them](https://raw.githubusercontent.com/turhancan97/VisBench/main/docs/_static/gallery/correspondence.png)

Every probe is drawable, and every figure here is a real photograph run through
the real command — see [looking at a probe](https://turhancan97.github.io/VisBench/guides/visualising.html) for all
sixteen and how to read them.

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

## Documentation

| | |
| --- | --- |
| Install, first run, the CLI | [Getting started](https://turhancan97.github.io/VisBench/getting-started/installation.html) |
| Backbones, datasets, dense probes, sharing | [Guides](https://turhancan97.github.io/VisBench/guides/backbones.html) |
| Every probe, its data layout and its board | [The probes](https://turhancan97.github.io/VisBench/probes/overview.html) |
| **How to read a board before quoting one** | [Reading a board](https://turhancan97.github.io/VisBench/guides/reading-a-board.html) |
| Every class, function and attribute | [API reference](https://turhancan97.github.io/VisBench/api/index.html) |
| Sixteen probes against twelve backbones | [LEADERBOARD.md](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md) |
| What each board means, and what it does not | [CORPUS_FINDINGS.md](https://github.com/turhancan97/VisBench/blob/main/CORPUS_FINDINGS.md) |
| Setting up, the checks, adding a probe | [CONTRIBUTING.md](https://github.com/turhancan97/VisBench/blob/main/CONTRIBUTING.md) |
| How it was built, and what might come next | [docs/roadmap.md](https://github.com/turhancan97/VisBench/blob/main/docs/roadmap.md) |
| What changed, release by release | [CHANGELOG.md](https://github.com/turhancan97/VisBench/blob/main/CHANGELOG.md) |
| The reference, page by page | [docs/probes/overview.md](https://github.com/turhancan97/VisBench/blob/main/docs/probes/overview.md), [docs/api/index.md](https://github.com/turhancan97/VisBench/blob/main/docs/api/index.md) |

## Reproducibility

**Sixteen probes against twelve backbones, as records:
[LEADERBOARD.md](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md).** Every board is generated from
[`results/corpus/visbench.jsonl`](https://github.com/turhancan97/VisBench/blob/main/results/corpus/visbench.jsonl), the
committed corpus, by [`scripts/render_tables.py`](https://github.com/turhancan97/VisBench/blob/main/scripts/render_tables.py).
A test in the fast suite fails if any of them drifts, so a published number and
the run behind it cannot disagree.

Every run logs a structured JSON record — backbone, weights key, task, dataset,
pooling, feature mode, metrics, seed, timestamp — under one **additive-only**
schema, so leaderboard tooling never needs a retrofit. Dependencies are pinned
in [`uv.lock`](https://github.com/turhancan97/VisBench/blob/main/uv.lock), and CI fails if it drifts from `pyproject.toml`.

**Before quoting any board**, read
[how to read one](https://turhancan97.github.io/VisBench/guides/reading-a-board.html). The short version: "which
backbone is best" is not a well-formed question against this corpus —
`mae_vitb16` is first on six of the sixteen boards and last on four.

## Prior art

VisBench reuses established protocols rather than re-deriving them, and cites
them at the point of use in the code:

- **[probe3d](https://arxiv.org/abs/2404.08476)** (El Banani et al., CVPR 2024)
  — evaluation protocols for depth, surface normal and correspondence.
- **[Probing the Mid-level Vision Capabilities of Self-Supervised Learning](https://arxiv.org/abs/2411.17474)**
  (Chen, Marks & Cheng) — the task categorization used throughout.
- **[vismatch](https://github.com/gmberton/vismatch)** (Berton) — API
  philosophy, and the matching logic mirrored in the correspondence task.

## Citing VisBench

If VisBench contributed to work you are publishing, please cite it. GitHub's
**Cite this repository** button generates APA and BibTeX from
[CITATION.cff](https://github.com/turhancan97/VisBench/blob/main/CITATION.cff),
or use:

```bibtex
@software{kargin_visbench,
  author  = {Kargın, Turhan Can},
  title   = {VisBench: probing vision backbones across high-, mid- and low-level tasks},
  doi     = {10.5281/zenodo.21822684},
  url     = {https://doi.org/10.5281/zenodo.21822684},
  license = {MIT},
  version = {0.12.0}
}
```

**If you are reporting numbers, cite the version you ran.** Every VisBench
result record carries the schema, the resolved pooling, the layers and the
protocol that produced it, so a number is reproducible — but only against the
release that produced it. `visbench.__version__` is in every record.

The DOI above is the **concept DOI**: it always resolves to the newest release,
which is what you want when citing the software itself. Zenodo also mints a
**version DOI** per release, listed on the archive page under *Versions* — pin
that one in a paper reporting measured numbers, for the same reason every
record carries its schema and protocol.

## License

MIT — see [LICENSE](https://github.com/turhancan97/VisBench/blob/main/LICENSE).

VisBench borrows evaluation protocols from prior work, all permissively
licensed and MIT-compatible; [NOTICE](https://github.com/turhancan97/VisBench/blob/main/NOTICE) records what came from where.
Backbone weights are downloaded at runtime, never redistributed here, and
carry their own upstream terms.
