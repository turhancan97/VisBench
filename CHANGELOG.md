# Changelog

All notable changes to VisBench are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

Each released section is written to be pasted straight into a GitHub release,
so it stands on its own rather than assuming you have read the ones above it.

## [Unreleased]

Nothing yet. v0.2 scope — CLI, ResNet/timm and custom backbones, dense
mid-level tasks, pluggable heads, multi-layer extraction — starts once v0.1 has
been reviewed.

## [0.1.0] — 2026-07-24

**v0.1 — prove the abstraction.**

The first release: two backbones, three tasks, and the infrastructure they
share, all running end-to-end on a local image folder.

```python
import visbench
from visbench.data import ImageFolderDataset

result = visbench.run(
    "dinov2_vitb14",
    "retrieval",
    ImageFolderDataset("data/tiny", split="val"),
    results="results/visbench.jsonl",
)
result.metrics    # {"recall@1": 0.99, "recall@5": 1.00, "mAP": 0.91}
result.record     # the ResultRecord saying exactly how they were produced
```

### Backbones

| Name | Weights | Patch | Grid @224 |
| --- | --- | --- | --- |
| `dinov2_vits14`, `dinov2_vitb14` | torch.hub, pinned to a fixed upstream commit | 14 | 16×16 |
| `clip_vitb16`, `clip_vitb32` | open_clip, OpenAI weights (QuickGELU-correct) | 16 / 32 | 14×14 / 7×7 |

One method covers both: `extract_features(image, pooling=..., layers=...)`
returns `{"dense": (B,C,H,W), "pooled": (B,C), "grid_hw": (H,W)}` from a single
forward pass, with the same shape for every architecture family. Tasks choose
pooling; backbones just execute it.

CLIP returns the **pre-projection** CLS token by default. The 512-d image-text
projection is trained to discard whatever does not help match a caption —
exactly the visual detail a mid-level probe measures — and DINOv2 has no
equivalent head to compare against. `use_projection=True` gets the projected
vector, under its own cache key.

### Tasks

| Level | Task | Training |
| --- | --- | --- |
| high | `classification` | linear probe, AdamW on cached features |
| high | `retrieval` | none — leave-one-out cosine |
| mid | `correspondence` | none — dense feature matching + Lowe ratio test |

Correspondence reports error in **patch widths**, not pixels. A match can only
land on a patch centre, so patch spacing is a hard floor on achievable error —
in pixels that floor moves with resolution and patch size, and `recall@1px` on
DINOv2 ViT-S/14 at 224px has a *ceiling* of 0.015. It also reports that ceiling
beside every score, so a low number reads as "coarse grid" rather than "bad
backbone".

### Infrastructure

- **Feature cache**, mandatory rather than an optimisation. Content-addressed,
  so one forward pass per image per backbone; a cached image is never decoded
  again.
- **`ResultRecord` / JSONL** under one additive schema, carrying the weights
  ref, dataset fingerprint, resolved pooling, seed, duration and task
  hyperparameters — enough to reproduce the number, not just read it.
- **`visbench.run()`** — resolve pooling, extract, fit if the task trains,
  evaluate, append the record.
- **`uv.lock`** pinning 116 packages with hashes, extras included; CI fails if
  it drifts from `pyproject.toml`.
- MIT licence, plus a `NOTICE` recording the CC BY-NC parts of probe3d that are
  deliberately **not** reused.

### Measured on Imagenette

3,925-image val split, one V100. Correspondence on 50 pairs at `max_warp=0.2`.

| Task | Metric | DINOv2 ViT-S/14 | CLIP ViT-B/16 |
| --- | --- | --- | --- |
| classification | top1 | 0.9939 | **0.9954** |
| retrieval | recall@1 | **0.9921** | 0.9893 |
| retrieval | mAP | 0.8893 | **0.9102** |
| correspondence | recall@1p | **0.7650** | 0.6993 |
| correspondence | ceiling | 0.9408 | 0.9505 |

CLIP leads on both semantic tasks and trails on the geometric one despite a
*higher* ceiling — the high-level / mid-level split the task taxonomy exists to
expose.

Caching, 13,394 images: cold run 208 s, fully cached 26 s, 107 MB on disk.

### Install

Not on PyPI yet.

```bash
git clone https://github.com/turhancan97/VisBench && cd VisBench
uv sync --all-extras            # exact locked versions
# or
pip install -e ".[dev,clip]"    # ranges, for day-to-day work

pytest                          # 286 fast tests, no weights downloaded
pytest -m slow                  # 37 more, against the real checkpoints
```

### Known limits

- `run()` does not cover correspondence — it takes pairs plus geometry rather
  than images plus labels. See `examples/correspond.py`.
- Retrieval ranks with an N×N score matrix; ~10 GB at 50k images, and there is
  no chunked path yet.
- Correspondence ground truth comes from synthetic homographies, so it measures
  viewpoint robustness on a **plane**, not 3D correspondence across parallax.
  probe3d's ScanNet/NAVI protocol is the real test and lands in v0.2.

### Deferred to v0.2

CLI · ResNet/timm and user-supplied custom backbones · depth, surface normals,
generic segmentation, mid-level similarity · pluggable task heads (linear +
DPT) · multi-layer extraction

### Prior art

Protocols are reused and cited at the point of use, not re-derived —
[probe3d](https://arxiv.org/abs/2404.08476) (El Banani et al., CVPR 2024),
[Probing the Mid-level Vision Capabilities of Self-Supervised Learning](https://arxiv.org/abs/2411.17474)
(Chen, Marks & Cheng), and [vismatch](https://github.com/gmberton/vismatch) for
API philosophy.

[Unreleased]: https://github.com/turhancan97/VisBench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/turhancan97/VisBench/releases/tag/v0.1.0
