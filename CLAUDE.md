# VisBench

Unified library for probing vision backbones (DINOv2, CLIP, custom) across
high-level, mid-level, and eventually low-level computer vision tasks, through
a `get_backbone()` / `get_probe()` API. Sibling project to vismatch (image
matching - https://github.com/gmberton/vismatch), same ergonomic philosophy, applied to representation probing
instead of matching.

**Distribution**: this ships as a pip-installable Python package on
[PyPI](https://pypi.org/) (`pip install visbench`), not just a research repo.
Packaging conventions (pyproject.toml, semantic versioning, a lockfile) apply
from v0.1 onward, not bolted on later.

**Prior art to credit explicitly, not re-derive**:
- [probe3d](https://arxiv.org/abs/2404.08476) (El Banani et al., CVPR 2024) —
  reuse its evaluation protocols for depth, surface normal, and correspondence.
- Chen, Marks & Cheng, ["Probing the Mid-level Vision Capabilities of
  Self-Supervised Learning"](https://arxiv.org/abs/2411.17474) — the task
  categorization below follows this paper directly.

---

## Critical: build order — read this before writing any code

Do not implement all tasks, backbones, and versions in one pass. This is a
multi-month roadmap; each session should complete one step below and stop for
review, not race ahead.

1. **Decide and scaffold the full project structure first.** Every folder,
   every module, with docstrings and `NotImplementedError`/`pass` stubs — no
   real logic yet. Get this reviewed before writing a single real function
   body.
2. Implement `BaseBackbone` + the feature cache + **one** backbone (DINOv2)
   fully, with tests.
3. Implement `BaseTask` + **one** task (image retrieval, zero-shot) fully,
   end-to-end, on a small local image folder.
4. Only once that full path works — scaffold → backbone → cache → task →
   passing test — move to the next task, then the next backbone.
5. Do not touch v0.2 or v0.3 scope until all of v0.1 is implemented, tested,
   and reviewed. If asked to "build VisBench," re-confirm which step is next
   rather than attempting the whole roadmap in one session.

---

## Architecture

### `BaseBackbone`

- One method, `.extract_features(image, pooling="default", layers=None)`,
  returns a dict: `{"dense": Tensor, "pooled": Tensor, "grid_hw": (H, W)}`.
- Returns **both** the dense spatial features and a pooled single vector from
  the same call — tasks pick whichever they need, backbones never expose
  separate methods per use case.
- `layers` accepts a list for future multi-layer extraction; only a single
  layer is wired up in v0.1 (see Feature extraction below).
- Same method signature for every backbone type (ViT or CNN) even though the
  internals differ completely — see CNN vs ViT handling below.

### `BaseTask` (a.k.a. probe)

- `.fit(features, labels)` — no-op for zero-shot tasks (retrieval,
  correspondence).
- `.evaluate(features, labels) -> dict` — always returns a flat metrics dict,
  never prints results directly (see structured logging below).
- `.predict(features)`.
- **Pooling strategy is chosen here, not on the backbone.** A task passes
  `pooling="cls"` or `pooling="mean"` (etc.) into `extract_features()`; the
  backbone just executes whatever is asked. This keeps backbones dumb and
  interchangeable, and keeps the "what representation does this task need"
  decision in one place.

### Feature cache

Mandatory in v0.1, not an optional speed-up added later. Disk-backed
key-value store keyed by `(image_hash, backbone_name, layer, pooling)`. Every
v0.1 task reads from the cache; the backbone forward pass runs at most once
per image per backbone.

---

## Feature extraction design — the most important decision in this codebase

Handle this consistently; don't improvise per-backbone.

### Default pooling rules
- ViT backbones with a CLS token → default single-vector representation is
  the **CLS token**.
- CNNs, and any backbone without a CLS token → default is **mean-pooling**
  over the dense feature map / patch tokens.
- Either default can be overridden per task call via the `pooling` argument.

### Dense-task feature modes (all three supported by the interface from day
one; mode 1 is the only one enabled by default in v0.1)

1. **`dense_only`** (v0.1 default) — just the spatial grid of patch/conv
   features, no CLS involved.
2. **`dense_cls_broadcast`** — the CLS token is broadcast spatially and
   concatenated onto every patch location, increasing channel dim uniformly
   across the grid.
3. **`dense_plus_cls`** — the dense grid and a single global CLS vector are
   kept **separate** and both handed to the task head, which decides how to
   fuse them (e.g. only at a bottleneck, or as a global conditioning vector),
   rather than broadcasting CLS into every spatial location.

Modes 2 and 3 exist in the interface starting v0.1 so no later refactor is
needed, but are opt-in — a task must explicitly request them.

### CNN vs ViT handling
- **CNNs**: "dense features" = the last conv feature map before global
  pooling (e.g. `layer4` output of a ResNet).
- **ViTs**: "dense features" = the patch token grid, reshaped from
  `(num_patches, dim)` to `(H, W, dim)` using the model's known patch size and
  input resolution.
- Both are exposed through the **identical** `.extract_features()` signature
  and return shape, even though the internal extraction logic is completely
  different per architecture family.

### Multi-layer extraction
Supported in the interface (`layers=[...]`) starting v0.1, but not actually
wired up for any backbone until v0.2 — get the single-layer path proven
first.

---

## Task categorization

Tasks are organized into three levels, following Chen, Marks & Cheng
(arXiv:2411.17474):

```
tasks/
  high_level/   classification, semantic (multi-class) segmentation, detection
  mid_level/    generic (binary) object segmentation, depth estimation,
                surface normal estimation, geometric correspondence,
                mid-level image similarity
  low_level/    placeholder only until v0.3+ — edge detection, optical flow,
                texture/reflectance, image quality
```

- **High-level** = semantic/category understanding.
- **Mid-level** = geometry and generic structure prior to semantic labeling —
  this is the paper's core contribution area, and it's where VisBench should
  be strongest relative to existing tools.
- **Mid-level image similarity is a distinct task class from high-level
  (semantic) retrieval** — mid-level similarity judges perceptual/geometric
  resemblance between candidates and a reference (scene layout, geometry),
  not category membership. Do not merge these two into one task even though
  both are "similarity"-flavored.
- **Low-level** is a folder with a README describing future scope only —
  nothing implemented there before v0.3, and possibly not even then without
  contributor bandwidth.

---

## v0.1 — prove the abstraction (zero / near-zero-training tasks only)

**Hard boundary: no fine-tuning, no dense-prediction training loops. Every
v0.1 task either needs no training (zero-shot) or trains a linear layer on
cached features.**

- **Backbones**: DINOv2 (ViT-S/B) and CLIP (OpenCLIP ViT-B) only. No
  ResNet/timm, no custom-backbone support yet — that's v0.2.
- **Tasks**:
  - High-level image classification — linear probe on cached pooled features.
  - High-level image retrieval — zero-shot, cosine similarity over cached
    pooled features (CLS default for ViT backbones).
  - Mid-level geometric correspondence — zero-shot, dense feature matching
    (conceptually reusing matching logic familiar from vismatch (https://github.com/gmberton/vismatch), applied to
    raw backbone features instead of dedicated matcher networks).
- **Required infrastructure before any task code**: reviewed folder skeleton
  → `BaseBackbone` with dual pooled+dense output → feature cache → `BaseTask`
  abstraction → structured JSON result logging (see below), from the very
  first task, not retrofitted later.
- **Explicitly deferred**: CLI, custom backbones, ResNet/timm, multi-layer
  extraction, any dense-prediction task, fine-tuning.

---

## v0.2 — dense mid-level tasks + broader backbone support

- Add ResNet/timm backbones and user-supplied custom-backbone support
  (arbitrary `nn.Module` + preprocessing function).
- Add the dense mid-level tasks: generic object segmentation, depth
  estimation, surface normal estimation — using probe3d's evaluation
  protocols directly rather than re-deriving metrics.
- Add mid-level image similarity as its own task, separate from high-level
  retrieval.
- Add high-level semantic (multi-class) segmentation alongside mid-level
  generic (binary) segmentation, so the two can be compared directly.
- **Task heads for dense tasks must be pluggable, not hardcoded to one
  architecture.** Support at minimum: (a) a simple linear probe head and
  (b) a DPT-style multiscale head, selectable per task run. Leave the head
  interface open for more options later (this is a known extension point for
  contributors).
- Wire up multi-layer feature extraction now that the single-layer path is
  proven.
- Add the CLI, now that the Python API is stable.

---

## v0.3 — fine-tuning + detection groundwork

- Fine-tuning mode: allow unfreezing the last N backbone blocks per task,
  opt-in, off by default.
- Begin high-level detection support (lightweight head). Expect this to take
  longer than any other single addition — it's the hardest task to do cheaply
  on limited compute.
- Low-level tasks get their first real entries if there's contributor
  bandwidth (edge detection, optical flow); otherwise the folder stays a
  placeholder.
- HF Hub integration for sharing pretrained probe heads and a public
  leaderboard, once there's enough task/backbone coverage for a leaderboard
  to be meaningful.

---

## Engineering conventions

- PyTorch, Python 3.9+.
- Pin exact dependency versions via a lockfile starting v0.1 — this is a
  reproducible benchmark library, not a moving-target research repo.
- Write at least minimal tests alongside every new module in v0.1; don't
  defer testing to "later."
- Every task run logs a structured JSON record — backbone, task, dataset,
  pooling mode, feature mode, metrics, timestamp — one schema from the start,
  so leaderboard tooling never needs a retrofit.
- Package for PyPI from v0.1: `pyproject.toml`, semantic versioning,
  `pip install visbench` as the eventual target install path.
- Cite prior art in code comments and docs wherever an evaluation protocol is
  borrowed, not just in the README.