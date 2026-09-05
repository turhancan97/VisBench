# `occlusion_edge`

**Depth discontinuities — where one surface ends and another begins behind it.**

This probe and {doc}`edge </probes/low-level/edge>` **share every line of their
implementation and sit one tier apart**, which is the cleanest statement of
what the tiers mean that this project has: recovering a depth discontinuity
needs scene geometry, recovering an intensity one does not.

```{figure} /_static/gallery/occlusion_edge.png
:alt: occlusion_edge — image, target and prediction

What `visbench show occlusion_edge` draws. {doc}`How to read it </guides/visualising>`.
```

## Its board

<!-- visbench:board task=occlusion_edge metrics=occlusion_edge_correlation,mae,rmse heading=3 -->
### occlusion_edge

| backbone | `occlusion_edge_correlation` | `mae` | `rmse` | `ceiling_mae` | `ceiling_occlusion_edge_correlation` | `ceiling_rmse` |
| --- | --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | **0.3273** | **0.1904** | **0.4261** | 0.1555 | 0.5150 | 0.3797 |
| `dinov2_vitb14` | 0.3167 | 0.2061 | 0.4315 | 0.1512 | 0.5301 | 0.3755 |
| `dino_vitb16` | 0.2928 | 0.2025 | 0.4338 | 0.1555 | 0.5150 | 0.3797 |
| `dinov2_vits14` | 0.2924 | 0.2205 | 0.4373 | 0.1512 | 0.5301 | 0.3755 |
| `sam_vitb16` | 0.2680 | 0.2108 | 0.4391 | 0.1555 | 0.5150 | 0.3797 |
| `clip_vitb16` | 0.2558 | 0.2149 | 0.4415 | 0.1555 | 0.5150 | 0.3797 |
| `siglip_vitb16` | 0.2254 | 0.2205 | 0.4423 | 0.1555 | 0.5150 | 0.3797 |
| `clip_vitb32` | 0.2174 | 0.2203 | 0.4440 | 0.1811 | 0.4336 | 0.3994 |
| `supervised_vitb16` | 0.1996 | 0.2435 | 0.4540 | 0.1555 | 0.5150 | 0.3797 |
| `resnet50` | 0.1979 | 0.2294 | 0.4502 | 0.1811 | 0.4336 | 0.3994 |
| `resnet18` | 0.1745 | 0.2418 | 0.4578 | 0.1811 | 0.4336 | 0.3994 |
| `convnext_base` | 0.1741 | 0.2533 | 0.4704 | 0.1811 | 0.4336 | 0.3994 |

Ordered by `occlusion_edge_correlation`, which **disagrees with `mae`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

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

## Run it

[`examples/occlusion_edges.py`](https://github.com/turhancan97/VisBench/blob/main/examples/occlusion_edges.py) is the whole path, end to end, on a real backbone.

```bash
python examples/occlusion_edges.py --data /path/to/dataset
```
