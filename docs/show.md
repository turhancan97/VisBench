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
convention. There are four conventions across the eight panel probes and none of
them is visible in a tensor's shape or dtype:

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

Ten probes. Eight are the `image | target | prediction` grid above:
`depth`, `surface_normal`, `generic_segmentation`, `semantic_segmentation`,
`edge`, `keypoints2d`, `occlusion_edge` and `corner`.

Two have their own renderer. **`detection`** draws boxes straight onto the crop,
in the post-transform pixel coordinates the dataset returns.
**`correspondence`** draws two views side by side with the matches between them
— see below.

`classification`, `retrieval` and `similarity` have no spatial target to draw
and are absent from the subcommand list rather than drawn blank.

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

## Correspondence: the shape of the errors, not their size

```bash
visbench show correspondence --data /path/to/images --split val \
    --backbone dinov2_vits14 --frames 4 --out matches.png
```

Each row is both views with sampled matches drawn between them — green where a
match landed within `--threshold` pixels of where the geometry says it should
have, red otherwise, with a short amber segment from the expected position to
the actual one.

**This is the panel that would have caught `recall@1px = 0.003`.** That bug does
not look like noise. A homography expressed in original pixels while the
features came from a 224 centre crop makes every match wrong *in the same
direction* — and a coherent field of long errors is a broken pipeline, where
scattered short ones are merely a weak backbone. A recall figure cannot tell
those apart. A picture can, instantly.

So the row label states it as a number rather than leaving it to the eye:
**coherence**, the mean resultant length of the error directions, 1.0 when every
error points the same way and near 0 when they scatter. Measured on 224px
homography pairs with ResNet-18 features:

| geometry | median error | coherence |
| --- | --- | --- |
| correct | 10.2 px, 22.6 px | **0.40, 0.29** |
| homography in the wrong pixel frame | 293.9 px, 226.6 px | **0.98, 1.00** |

Coherence near 1 means the geometry, the crop or the units are wrong. It is a
diagnostic, never a score: it says nothing about a backbone and is not recorded.

Two things to know. This probe **always needs a backbone** — the matches are the
thing being looked at and they do not exist until features do — and
`--predict-from` is refused, because correspondence is zero-shot and has no
saved head. And matches are sampled **evenly** across the kept set rather than
taking the most confident few, which would draw a systematically better picture
than the score describes.

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
probe's `predict()` returned for exactly those indices. For correspondence,
`render_match_panels` and `draw_matches` take the regrouped pair features
`visbench.run` builds, and `error_coherence` returns the diagnostic on its own.
The lower-level pieces (`style_for`, `display_range`, `target_to_rgb`,
`draw_boxes`, `render_panels`) are exported too, for a layout this command does
not offer.

Pillow and numpy only. Nothing here adds a dependency, and there is no
matplotlib in the core install.
