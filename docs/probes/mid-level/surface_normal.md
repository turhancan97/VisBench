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

The twelve-backbone board is in [`LEADERBOARD.md`](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md). Note that this probe can disagree with itself: on Taskonomy normals DINOv2-S wins on mean angular error while DINOv2-B wins on the 11.25° threshold, so quoting one and dropping the other manufactures a result.

## Run it

[`examples/normals.py`](https://github.com/turhancan97/VisBench/blob/main/examples/normals.py) is the whole path, end to end, on a real backbone.

```bash
python examples/normals.py --data /path/to/dataset
```
