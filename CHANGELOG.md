# Changelog

All notable changes to VisBench are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

Each released section is written to be pasted straight into a GitHub release,
so it stands on its own rather than assuming you have read the ones above it.

## [Unreleased]

### Added

- **timm CNN backbones** (`resnet18`, `resnet50`, or any timm CNN via
  `TimmBackbone(model_name=...)`) — the first non-ViT family. Dense features
  are the last conv map before global pooling, flattened to a token sequence so
  `extract_features` needs no branch on architecture. Mean-pooling those tokens
  reproduces a ResNet's own `global_pool` output exactly, so the pooled vector
  means the same thing for a CNN as a CLS token does for a ViT. Behind a `timm`
  extra.
- Cache keys carry the timm pretrained tag: `resnet50.a1_in1k` and
  `resnet50.a3_in1k` are different weights under one architecture name.
- **`CustomBackbone`** — wrap any `nn.Module` plus a preprocessing callable.
  The grid is read from the module's output shape, `embed_dim` from the first
  forward pass, and the cache key from a hash of the weights, so a fine-tuned
  checkpoint cannot reuse its parent's cached features. Ambiguous output shapes
  raise rather than guess: a square token *count* from a non-square layout
  would otherwise misplace every patch silently.
- `visbench.register_backbone` / `register_task` are public, so a
  `BaseBackbone` subclass outside this package can claim a registry name.

- `extract_features` takes **`feature_mode`**, so `dense_cls_broadcast` and
  `dense_plus_cls` are reachable through the public API. They were declared,
  implemented and tested in v0.1 but `apply_feature_mode` had zero callers and
  no parameter exposed them — a DPT head is exactly the consumer that wants
  `dense_plus_cls`, so this had to exist before heads were designed against it.
  `dense_plus_cls` returns the global vector under a new `cls` key, and the
  cache both keys on the mode and stores `cls`.

### Fixed

- `FeatureCache.extract_dataset` refused nothing when handed a `PairDataset`:
  it read `item[0]` and silently discarded the second view and the geometry,
  returning features for half the data. It now raises.
- `cls` was produced by extraction with `dense_plus_cls` but never stored, so it
  existed on a cache miss and vanished on the next hit.
- `DPTHead(use_cls=True)` sized its CLS projection from the *last* layer's width
  while injecting the vector at the *first*, so any head built with per-layer
  `in_channels` raised a matmul shape error. It now follows the stage the vector
  actually reaches, and checks the vector's width with a message that names the
  expected one.
- `DPTHead` read `head((stage0, stage1))` — a tuple of two layers rather than a
  list — as one dense map plus a CLS vector, and reported it as "got a single
  tensor" when the caller had passed two. A `(stages, cls)` pair is now
  identified by its first element being a sequence.

### Changed

- mypy is **gating** in CI. It had `continue-on-error` from when everything was
  stubs, which made it a check that could never fail; 19 errors had accumulated,
  including the `PairDataset` variance violation above. Now clean.
- Removed the unused `visbench.utils.device.batched` helper.
- `BaseBackbone._forward_features` now returns a **list** of
  `(patch_tokens, cls, grid_hw)`, one per requested layer, and receives layer
  indices already resolved. Only affects code subclassing `BaseBackbone`
  directly; `extract_features` is unchanged for single-layer callers.
- A timm ViT is rejected when the backbone is constructed rather than at the
  first extraction. `forward_intermediates` reshapes a ViT's tokens into a grid
  when asked for NCHW, so from that point the output is indistinguishable from
  a conv map and nothing would notice the CLS token had been dropped while
  `has_cls_token` stayed False.

Still to come in v0.2: dense mid-level tasks, CLI.
  `LinearHead` (1x1 convolution over the dense grid, upsampled) and `DPTHead`
  (RefineNet-style multiscale fusion, following probe3d and Ranftl et al.).
  `register_head` makes this a real extension point. A head declares which
  feature modes it consumes and `check_feature_mode` rejects a mismatch at
  construction rather than as a shape error partway through training.
  `DPTHead` refuses a single feature map: fed one layer it is not multiscale,
  and duplicating the input would report a single-layer result as a DPT number.
- `DPTHead(cls_dim=...)`, for when a backbone's CLS width differs from the
  channel count of the layer the vector is injected alongside.

- **Multi-layer feature extraction.** `extract_features(layers=[2, 5, 8, 11])`
  returns `dense_layers` — one map per requested depth, from a **single**
  forward pass — plus the resolved `layer_indices`. Declared in the interface
  since v0.1 and wired up now that the single-layer path is proven; this is
  what `DPTHead` has been waiting for.

  `dense`, `pooled` and `cls` still describe the last requested layer, so a
  multi-layer call is a strict superset of a single-layer one and a task
  reading only `dense` is unaffected. `dense_layers` is a separate key rather
  than `dense` sometimes being a list: a type that depends on how many layers
  were requested would break every existing consumer the moment a layer list
  was widened.

  Layer indices are resolved once, in `BaseBackbone.resolve_layers`, instead of
  in each backbone: negatives count from the end, and the list must be strictly
  increasing, since a multiscale head reads the first layer it is given as the
  coarsest. A descending or repeated list is rejected rather than reordered.
- Each layer gets **its own cache entry**, keyed on the resolved index.
  Widening `[3, 7]` to `[3, 7, 11]` re-extracts one layer rather than three,
  and a later single-layer run at layer 7 reads what the multi-layer run
  stored. `layers=[-1]` and `layers=[11]` name the same entry on a 12-block
  model rather than storing identical features twice.
- `TimmBackbone.layer_channels([1, 2, 3, 4])`, because a CNN's stages differ in
  width — which is exactly why `DPTHead` accepts per-layer `in_channels`.
- `CustomBackbone(layer_feature_fn=..., num_layers=...)`. An arbitrary
  `nn.Module` has no `get_intermediate_layers` to call, so this is where a user
  says how their model exposes depth. Without it `num_layers` stays 1 and a
  multi-layer request is refused — returning the final map several times would
  let a multiscale head report a single-layer result.
- Result schema v4 adds `layers`. A record for a run over four depths is not
  the same run as one over the last, and widening `layer`'s type would have
  changed how every v1–v3 record on disk parses.

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
