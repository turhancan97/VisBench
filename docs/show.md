# Looking at what a probe saw

Every other command in VisBench turns pixels into numbers. `visbench show` turns
them back: it writes a grid of **image / target / prediction** panels to a file,
measures nothing, and records nothing.

```bash
visbench show depth --data /path/to/nyuv2 --out panels.png
```

## Why this exists

Not for convenience. **A dense target that has drifted from the image it belongs
to fails silently** — nothing raises, the probe trains, and the number merely
comes out mediocre, which reads as a hard task or a weak representation. Those
are the two explanations this library exists to tell apart, so a failure that
imitates one of them is the expensive kind.

Two of this project's most costly bugs were exactly that shape, and both were
found by reading code rather than by looking:

- geometric correspondence scoring `recall@1px = 0.003` — a homography expressed
  in original pixels while the features came from a 224 centre crop
- VOC's palette PNGs read through `convert("L")`, which resolves the palette and
  turns classes `[0, 1, 15, 255]` into `[0, 38, 147, 220]` — a target that
  loads, trains and scores against labels that mean nothing

Both are obvious in one frame. Neither is visible in a metric.

## The rules it keeps

**It displays what the dataset yielded.** No resize, no re-read of the source
file, no second crop. A viewer that applied its own geometry could make a
misaligned pipeline look fine and a correct one look broken — which would make
it worse than no viewer at all, because the entire evidence a panel carries is
whether the pair lines up. This is cheap to guarantee: dense datasets already
hand over a PIL image at the working resolution rather than a normalised tensor,
so there is nothing to invert.

**It draws an invalid pixel as invalid**, in magenta, per that probe's own
convention. There are four conventions across the nine drawable probes and none
of them is visible in a tensor's shape or dtype:

| convention | probes |
| --- | --- |
| `0` is invalid | `depth`; the zero *vector* for `surface_normal` |
| negative is invalid | both segmentations — `0` is a real class |
| nothing is invalid | `edge`, `keypoints2d`, `corner` |
| `NaN` is invalid | `occlusion_edge` |

Magenta is used because no colouriser here can produce it: greyscale has no hue,
the `(n + 1) / 2` normal-map convention cannot reach it for a unit vector, and
VOC's palette does not contain it. So a hole reads as a hole rather than as a
plausible dark pixel.

**It scales a prediction by the target's range.** Each row states the range it
was drawn against, computed from the target over valid pixels only. Scaling each
panel to its own extremes is the obvious implementation and it hides the most
common way a regression head is wrong: a prediction uniformly half the target's
magnitude would render *identically* to a correct one.

## What it can draw

Nine probes: `depth`, `surface_normal`, `generic_segmentation`,
`semantic_segmentation`, `edge`, `keypoints2d`, `occlusion_edge`, `corner` and
`detection`. Detection boxes are drawn straight onto the crop, in the
post-transform pixel coordinates the dataset returns.

`classification`, `retrieval` and `similarity` have no spatial target to draw.
`correspondence` needs a *pair* renderer with match lines rather than a panel
grid — a different layout, and not yet built, which is worth naming because it
is the probe whose historical bug is quoted above.

The flags are the same ones `visbench run` takes for that probe's data, minus
everything about the training schedule:

```bash
visbench show semantic_segmentation \
    --data /path/to/VOCdevkit/VOC2012 \
    --stems /path/to/ImageSets/Segmentation/val.txt \
    --num-classes 21 --frames 6 --out voc.png
```

Without `--predict-from` this needs **no backbone, no cache and no GPU** — which
is when you actually want to look, before spending a training budget on data you
have not seen.

## Adding the prediction column

`visbench run --save-probe` writes the head it just trained, with the backbone
identity beside it, to a local file. No Hub account is involved.

```bash
visbench run corner --data ./frames --backbone dinov2_vits14 \
    --save-probe heads/corner.pt

visbench show corner --data ./frames --backbone dinov2_vits14 \
    --predict-from heads/corner.pt --out corner.png
```

The artifact is loaded through {func}`visbench.hub.load_probe`, so the backbone
it was fitted on is **checked, not assumed** — a head fed the wrong pooling from
the same backbone scores 0.9620 against 0.9820 without raising. See
[sharing probes](hub.md) for what that check covers.

## From Python

```python
from visbench.data import DenseFolderDataset
from visbench.viz import render_probe_panels

dataset = DenseFolderDataset("data/val", target_dir="depths", image_size=224)
page = render_probe_panels(dataset, "depth", indices=[0, 1, 2])
page.save("panels.png")
```

`render_probe_panels` takes an optional `predictions` argument — whatever that
probe's `predict()` returned for exactly those indices. The lower-level pieces
(`style_for`, `display_range`, `target_to_rgb`, `draw_boxes`, `render_panels`)
are exported too, for a layout this command does not offer.

Pillow and numpy only. Nothing here adds a dependency, and there is no
matplotlib in the core install.
