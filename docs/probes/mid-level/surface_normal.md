# `surface_normal`

**Per-pixel surface orientation, scored by angular error.**

probe3d's angular protocol, including its uncertainty-aware loss — kept exactly
as that paper wrote it, because silently substituting the plain one would make
these numbers incomparable with the published ones, which is the only reason to
borrow a protocol at all.

```{figure} /_static/gallery/surface_normal.png
:alt: surface_normal — image, target and prediction

What `visbench show surface_normal` draws. {doc}`How to read it </guides/visualising>`.
```

## Things that will bite

- Surface normals default to probe3d's uncertainty-aware loss, which has a
  failure mode near chance accuracy where it all but switches its own
  supervision off. VisBench detects it and warns; `--no-uncertainty` is the way
  out. See `SurfaceNormalTask.fit` for the measured dynamics.
- **Quote IoU, not pixel accuracy, for segmentation.** Objects are a minority of
  most frames, so a probe predicting background everywhere already scores high
  accuracy and zero IoU. `examples/segment.py` prints the foreground fraction
  and that baseline before it trains, so the comparison is unavoidable.
- **Two mIoUs are reported and they differ.** `miou` accumulates one confusion
  matrix over the whole split, which is what VOC and the literature define;
  `miou_per_image` averages each image's own mIoU, this codebase's convention

## Its board

**This probe disagrees with itself.** On Taskonomy normals DINOv2-S wins on
mean angular error while DINOv2-B wins on the 11.25° threshold, so quoting one
and dropping the other manufactures a result — read the whole row.

<!-- visbench:board task=surface_normal metrics=d1,d2,d3,mean,median,rmse heading=3 -->
### surface_normal

| backbone | `mean` | `d1` | `d2` | `d3` | `median` | `rmse` |
| --- | --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | **27.5219** | **0.2700** | **0.5453** | **0.6557** | **20.8780** | **35.0638** |
| `dinov2_vits14` | 29.4827 | 0.2185 | 0.4841 | 0.6107 | 23.8327 | 36.4513 |
| `dinov2_vitb14` | 30.1143 | 0.2104 | 0.4730 | 0.5979 | 24.4548 | 37.1147 |
| `dino_vitb16` | 34.3047 | 0.1719 | 0.3782 | 0.4999 | 30.1048 | 41.0298 |
| `sam_vitb16` | 34.5618 | 0.1574 | 0.3700 | 0.4924 | 30.6795 | 41.0009 |
| `clip_vitb16` | 36.1668 | 0.1380 | 0.3399 | 0.4640 | 32.3666 | 42.5307 |
| `supervised_vitb16` | 36.7236 | 0.1297 | 0.3357 | 0.4588 | 32.7257 | 43.1069 |
| `clip_vitb32` | 37.0217 | 0.1426 | 0.3342 | 0.4524 | 33.0977 | 43.6495 |
| `siglip_vitb16` | 38.3425 | 0.1203 | 0.3015 | 0.4211 | 34.6561 | 44.6580 |
| `convnext_base` | 38.3801 | 0.1338 | 0.3205 | 0.4341 | 34.0186 | 45.1984 |
| `resnet50` | 38.4520 | 0.1441 | 0.3262 | 0.4364 | 34.3090 | 45.4922 |
| `resnet18` | 38.4856 | 0.1290 | 0.3150 | 0.4287 | 34.5742 | 45.0914 |

Ordered by `mean`, which **disagrees with `d1`, `d2`, `d3`, `median`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>surface_normal on test/test, protocol=probe3d, frozen [fc483401]</sub>
<!-- /visbench:board -->

## Run it

[`examples/normals.py`](https://github.com/turhancan97/VisBench/blob/main/examples/normals.py) is the whole path, end to end, on a real backbone.

```bash
python examples/normals.py --data /path/to/dataset
```
