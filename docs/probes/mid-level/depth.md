# `depth`

**Monocular metric depth, via probe3d's 256-bin expectation.**

Bins rather than one number: regressing a scalar per pixel pushes a linear head
towards the dataset's mean depth almost everywhere, and predicting a
distribution lets a *linear* map express a multi-modal belief. That is most of
why probe3d's linear probe is a fair baseline rather than a straw man.

**This board is not ranking by metric accuracy.** A readout that discards scale
and shift entirely reproduces its ordering at Spearman **+1.000**, so what it
ranks is ordering plus feature resolution — reported in metres. Not a defect:
it reproduces probe3d's protocol, which is the only reason its numbers compare
to anything. See `results/controls/relative_depth.jsonl`.

```{figure} /_static/gallery/depth.png
:alt: depth — a photograph and a depth map in metres predicted from it; no target column

What `visbench show depth` draws, minus the target column: depth needs sensor or reconstruction geometry, which no redistributable photograph carries, so the prediction comes from a published Hub head instead. {doc}`How to read it </guides/visualising>`.
```

## Data layout

Images and per-pixel targets paired by filename stem under `train/` and `val/`.

```bash
python examples/depth.py --data /path/to/dataset --target-scale 1000
```

**`--target-scale` is load-bearing.** Depth datasets ship millimetres in a
16-bit container, so 1000 is right for a PNG distribution and **wrong** for
`.npy` files already in metres — pass 1.0 there. `depth_metrics` reports RMSE
in whatever unit it is handed, so the mistake produces a superb-looking number
that means nothing.

## Its board

<!-- visbench:board task=depth metrics=abs_rel,d1,d2,d3,rmse heading=3 -->
### depth

| backbone | `d1` | `abs_rel` | `d2` | `d3` | `rmse` |
| --- | --- | --- | --- | --- | --- |
| `dinov2_vitb14` | **0.7851** | **0.1538** | **0.9690** | **0.9951** | **0.5308** |
| `dinov2_vits14` | 0.7652 | 0.1639 | 0.9593 | 0.9931 | 0.5518 |
| `mae_vitb16` | 0.6930 | 0.1993 | 0.9260 | 0.9839 | 0.6292 |
| `dino_vitb8` | 0.6841 | 0.1987 | 0.9258 | 0.9848 | 0.6491 |
| `dino_vitb16` | 0.6777 | 0.2006 | 0.9225 | 0.9832 | 0.6629 |
| `clip_vitb32` | 0.6539 | 0.2092 | 0.9090 | 0.9828 | 0.7005 |
| `sam_vitb16` | 0.6344 | 0.2248 | 0.8918 | 0.9702 | 0.7309 |
| `clip_vitb16` | 0.6321 | 0.2158 | 0.9054 | 0.9807 | 0.7173 |
| `siglip_vitb16` | 0.6318 | 0.2276 | 0.8975 | 0.9750 | 0.7244 |
| `supervised_vitb16` | 0.6259 | 0.2286 | 0.8957 | 0.9735 | 0.7174 |
| `convnext_base` | 0.6232 | 0.2344 | 0.8951 | 0.9727 | 0.7386 |
| `resnet50` | 0.5380 | 0.3119 | 0.8283 | 0.9406 | 0.8691 |
| `resnet18` | 0.5276 | 0.3112 | 0.8177 | 0.9384 | 0.8853 |

Ordered by `d1`, which **disagrees with `abs_rel`, `d2`, `d3`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>depth on test/test, protocol=probe3d, frozen [ec5f45be]</sub>
<!-- /visbench:board -->

## Run it

[`examples/depth.py`](https://github.com/turhancan97/VisBench/blob/main/examples/depth.py) is the whole path, end to end, on a real backbone.

```bash
python examples/depth.py --data /path/to/dataset
```
