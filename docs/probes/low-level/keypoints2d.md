# `keypoints2d`

**2D keypoint response, from Taskonomy's `keypoints2d`.**

The weakest-scoring low-level probe, at 0.18-0.26 — and it ranks backbones
perfectly well anyway, which is why the criterion is the spread rather than the
level. It reaches only 41% of its own oracle where `corner` reaches 80%.

```{figure} /_static/gallery/keypoints2d.png
:alt: keypoints2d — image, target and prediction

What `visbench show keypoints2d` draws. {doc}`How to read it </show>`.
```

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

## Its board

<!-- visbench:board task=keypoints2d metrics=keypoint_correlation,mae,rmse heading=3 -->
### keypoints2d

| backbone | `keypoint_correlation` | `mae` | `rmse` | `ceiling_keypoint_correlation` | `ceiling_mae` | `ceiling_rmse` |
| --- | --- | --- | --- | --- | --- | --- |
| `dino_vitb16` | **0.2850** | 1.0827 | **2.5143** | 0.6674 | 0.9238 | 1.9161 |
| `sam_vitb16` | 0.2696 | 1.1117 | 2.5589 | 0.6674 | 0.9238 | 1.9161 |
| `mae_vitb16` | 0.2626 | **1.0533** | 2.5342 | 0.6674 | 0.9238 | 1.9161 |
| `supervised_vitb16` | 0.2573 | 1.1242 | 2.5468 | 0.6674 | 0.9238 | 1.9161 |
| `dinov2_vits14` | 0.2356 | 1.1281 | 2.5472 | 0.6976 | 0.8873 | 1.8452 |
| `dinov2_vitb14` | 0.2248 | 1.1294 | 2.5541 | 0.6976 | 0.8873 | 1.8452 |
| `convnext_base` | 0.2187 | 1.1690 | 2.5633 | 0.4728 | 1.0934 | 2.2552 |
| `clip_vitb16` | 0.2175 | 1.1533 | 2.5770 | 0.6674 | 0.9238 | 1.9161 |
| `clip_vitb32` | 0.1933 | 1.1474 | 2.5891 | 0.4728 | 1.0934 | 2.2552 |
| `resnet50` | 0.1792 | 1.2374 | 2.6163 | 0.4728 | 1.0934 | 2.2552 |
| `resnet18` | 0.1659 | 1.2579 | 2.6282 | 0.4728 | 1.0934 | 2.2552 |
| `siglip_vitb16` | 0.1577 | 1.1789 | 2.6072 | 0.6674 | 0.9238 | 1.9161 |

Ordered by `keypoint_correlation`, which **disagrees with `mae`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>keypoints2d on taskonomy_keypoints2d/val, protocol=visbench_keypoint2d_regression, frozen [e647c722]</sub>
<!-- /visbench:board -->

## Run it

[`examples/keypoints.py`](https://github.com/turhancan97/VisBench/blob/main/examples/keypoints.py) is the whole path, end to end, on a real backbone.

```bash
python examples/keypoints.py --data /path/to/dataset
```
