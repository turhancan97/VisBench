# `surface_normal`

**Per-pixel surface orientation, scored by angular error.**

probe3d's angular protocol, including its uncertainty-aware loss — kept exactly
as that paper wrote it, because silently substituting the plain one would make
these numbers incomparable with the published ones, which is the only reason to
borrow a protocol at all.

```{figure} /_static/gallery/surface_normal.png
:alt: surface_normal — a photograph and a normal map, RGB by the (n+1)/2 convention predicted from it; no target column

What `visbench show surface_normal` draws, minus the target column: surface_normal needs sensor or reconstruction geometry, which no redistributable photograph carries, so the prediction comes from a published Hub head instead. {doc}`How to read it </guides/visualising>`.
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
| `mae_vitb16` | **26.6901** | **0.2749** | **0.5557** | **0.6685** | **20.3589** | **33.9731** |
| `dinov2_vits14` | 29.4827 | 0.2185 | 0.4841 | 0.6107 | 23.8327 | 36.4513 |
| `dinov2_vitb14` | 30.1143 | 0.2104 | 0.4730 | 0.5979 | 24.4548 | 37.1147 |
| `dino_vitb8` | 33.3003 | 0.1873 | 0.3943 | 0.5139 | 29.2946 | 40.0145 |
| `dino_vitb16` | 34.3316 | 0.1696 | 0.3760 | 0.4979 | 30.1963 | 41.0143 |
| `sam_vitb16` | 34.6532 | 0.1537 | 0.3672 | 0.4905 | 30.7914 | 41.0376 |
| `clip_vitb16` | 36.1668 | 0.1380 | 0.3399 | 0.4640 | 32.3666 | 42.5307 |
| `supervised_vitb16` | 36.8491 | 0.1243 | 0.3318 | 0.4579 | 32.8659 | 43.1919 |
| `clip_vitb32` | 37.0217 | 0.1426 | 0.3342 | 0.4524 | 33.0977 | 43.6495 |
| `siglip_vitb16` | 38.0345 | 0.1216 | 0.3046 | 0.4253 | 34.4498 | 44.3042 |
| `convnext_base` | 38.3952 | 0.1303 | 0.3186 | 0.4337 | 34.0880 | 45.1367 |
| `resnet50` | 38.4200 | 0.1439 | 0.3266 | 0.4371 | 34.2610 | 45.4686 |
| `resnet18` | 38.5108 | 0.1262 | 0.3132 | 0.4281 | 34.6235 | 45.0627 |

Ordered by `mean`, which **disagrees with `d1`, `d2`, `d3`, `median`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>surface_normal on test/test, protocol=probe3d, frozen [fc483401]</sub>
<!-- /visbench:board -->

## Run it

[`examples/normals.py`](https://github.com/turhancan97/VisBench/blob/main/examples/normals.py) is the whole path, end to end, on a real backbone.

```bash
python examples/normals.py --data /path/to/dataset
```
