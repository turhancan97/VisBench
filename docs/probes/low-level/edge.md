# `edge`

**Intensity edges — gradient magnitude, from Taskonomy's `edge_texture`.**

The first low-level probe, and the template every magnitude probe follows. Its
`protocol` is `visbench_edge_regression`, **not** BSDS500's, which is a
correspondence metric this does not implement.

```{figure} /_static/gallery/edge.png
:alt: edge — image, target and prediction

What `visbench show edge` draws. {doc}`How to read it </guides/visualising>`.
```

[`examples/edges.py`](https://github.com/turhancan97/VisBench/blob/main/examples/edges.py) is the first
**low-level** probe: dense edge-magnitude regression on
[Taskonomy](https://arxiv.org/abs/1804.08328)'s `edge_texture` maps.

```bash
python examples/edges.py --data /path/to/taskonomy --limit 600
visbench run edge --data /path/to/taskonomy --limit 600
```

Taskonomy's splits are **disjoint by building** — 25 rooms train, 4 validate,
5 test — so a val number is measured in rooms the probe has never seen.

600 train / 600 val frames at 224px, linear head, ten epochs:

## Its board

<!-- visbench:board task=edge metrics=edge_correlation,rmse,mae heading=3 -->
### edge

| backbone | `edge_correlation` | `rmse` | `mae` | `ceiling_edge_correlation` | `ceiling_mae` | `ceiling_rmse` |
| --- | --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | **0.4982** | **0.9150** | **0.4687** | 0.6106 | 0.4560 | 0.7906 |
| `dino_vitb16` | 0.4817 | 0.9150 | 0.4789 | 0.6106 | 0.4560 | 0.7906 |
| `sam_vitb16` | 0.4734 | 0.9286 | 0.4784 | 0.6106 | 0.4560 | 0.7906 |
| `clip_vitb16` | 0.4565 | 0.9340 | 0.4882 | 0.6106 | 0.4560 | 0.7906 |
| `dinov2_vits14` | 0.4558 | 0.9226 | 0.5028 | 0.6336 | 0.4418 | 0.7727 |
| `dinov2_vitb14` | 0.4481 | 0.9265 | 0.4972 | 0.6336 | 0.4418 | 0.7727 |
| `supervised_vitb16` | 0.4420 | 0.9366 | 0.5086 | 0.6106 | 0.4560 | 0.7906 |
| `clip_vitb32` | 0.3834 | 0.9656 | 0.5080 | 0.4977 | 0.5175 | 0.8630 |
| `siglip_vitb16` | 0.3639 | 0.9785 | 0.5169 | 0.6106 | 0.4560 | 0.7906 |
| `resnet50` | 0.3549 | 0.9770 | 0.5056 | 0.4977 | 0.5175 | 0.8630 |
| `convnext_base` | 0.3485 | 0.9671 | 0.5224 | 0.4977 | 0.5175 | 0.8630 |
| `resnet18` | 0.3430 | 0.9797 | 0.5153 | 0.4977 | 0.5175 | 0.8630 |

Ordered by `edge_correlation`, which **disagrees with `mae`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>edge on taskonomy_edge_texture/val, protocol=visbench_edge_regression, frozen [f8d2af2a]</sub>
<!-- /visbench:board -->

DINOv2-S edges out DINOv2-B here, by 0.008 — the same ordering as mid-level
similarity and the opposite of segmentation and detection. The margin is small
enough to be worth stating as *consistent with* the level taxonomy rather than
as evidence for it.

Three things that are protocol rather than detail:

- **Quote `edge_correlation`, not `rmse`.** Edge magnitude is concentrated near
  zero, so a probe that ignores its input and predicts the split's mean
  everywhere gets a *small* RMSE while having learned nothing. Pearson
  correlation is invariant to scale and offset, so it asks only whether the
  representation knows **where** the edges are, and scores that probe 0. RMSE
  and MAE are reported alongside because correlation is blind to the opposite
  failure — right shape, wrong magnitude.
- **Nothing is masked.** Depth has holes and normals have zero-length vectors,
  both meaning "no ground truth". Here 0 means *no edge*, a real reading
  covering most of most frames. Masking it away would score the probe only
  where an edge already is.
- **This is not BSDS500's protocol and must not share a table with one.** BSDS's
  ODS/OIS/AP matches edge pixels by bipartite correspondence after non-maximum
  suppression, swept over thresholds, against several annotators. Records say
  `protocol: "visbench_edge_regression"`.

## Run it

[`examples/edges.py`](https://github.com/turhancan97/VisBench/blob/main/examples/edges.py) is the whole path, end to end, on a real backbone.

```bash
python examples/edges.py --data /path/to/dataset
```
