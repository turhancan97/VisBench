# Roadmap, build order and future directions

This is the project's own record of how it was built and where it might go. It
is here rather than in the README because it answers "what is the plan", not
"how do I use this" — see the [README](../README.md) for the latter.

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
