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
:alt: depth — image, target and prediction

What `visbench show depth` draws. {doc}`How to read it </guides/visualising>`.
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
| `mae_vitb16` | 0.6945 | 0.1986 | 0.9267 | 0.9833 | 0.6326 |
| `dino_vitb16` | 0.6748 | 0.2025 | 0.9209 | 0.9819 | 0.6705 |
| `clip_vitb32` | 0.6538 | 0.2092 | 0.9090 | 0.9828 | 0.7005 |
| `sam_vitb16` | 0.6356 | 0.2227 | 0.8945 | 0.9721 | 0.7317 |
| `clip_vitb16` | 0.6321 | 0.2158 | 0.9054 | 0.9807 | 0.7173 |
| `convnext_base` | 0.6215 | 0.2347 | 0.8932 | 0.9722 | 0.7414 |
| `supervised_vitb16` | 0.6195 | 0.2284 | 0.8950 | 0.9752 | 0.7214 |
| `siglip_vitb16` | 0.6169 | 0.2358 | 0.8881 | 0.9711 | 0.7498 |
| `resnet50` | 0.5395 | 0.3064 | 0.8330 | 0.9433 | 0.8599 |
| `resnet18` | 0.5315 | 0.3047 | 0.8241 | 0.9423 | 0.8772 |

Ordered by `d1`, which **disagrees with `abs_rel`, `d2`, `d3`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>depth on test/test, protocol=probe3d, frozen [ec5f45be]</sub>
<!-- /visbench:board -->

## Run it

[`examples/depth.py`](https://github.com/turhancan97/VisBench/blob/main/examples/depth.py) is the whole path, end to end, on a real backbone.

```bash
python examples/depth.py --data /path/to/dataset
```
