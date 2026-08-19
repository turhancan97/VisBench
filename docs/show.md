# Looking at what a probe saw

Every other command in VisBench turns pixels into numbers. `visbench show` turns
them back: it writes a grid of **image / target / prediction** panels to a file,
measures nothing, and records nothing.

```bash
visbench show depth --data /path/to/nyuv2 --out panels.png
```

```{image} _static/gallery/depth.png
:alt: Depth panels: image, target, with magenta marking pixels that have no depth return
:class: visbench-figure
```

Every figure on this page is a **real photograph**, run through the real
`visbench show` by
[`scripts/render_gallery.py`](https://github.com/turhancan97/VisBench/blob/main/scripts/render_gallery.py).
The frames come from [Open Images](https://storage.googleapis.com/openimages/web/index.html)
and are every one CC BY 2.0, fetched and licence-checked by
[`scripts/fetch_gallery_frames.py`](https://github.com/turhancan97/VisBench/blob/main/scripts/fetch_gallery_frames.py)
and credited in
[`CREDITS.md`](https://github.com/turhancan97/VisBench/blob/main/assets/gallery_frames/CREDITS.md).
They are *not* the datasets the probes are scored on: VOC, ImageNet, NYUv2,
Taskonomy and NIGHTS each restrict redistribution, so none of them may appear on
a page this project ships.

That constraint decides what each figure can show, and the figures say which of
three they are:

- **exact ground truth, computed from the frame itself** — `corner` runs the
  probe's own Shi-Tomasi generator over the crop you are looking at, and
  `correspondence` warps by a homography the script chooses. Nothing is
  approximated.
- **real human annotation** — `detection`, `generic_segmentation` and
  `semantic_segmentation` draw Open Images' own boxes and instance masks, so the
  target column is what an annotator marked.
- **a prediction, labelled as one** — `depth`, `surface_normal`, `keypoints2d`
  and `occlusion_edge` need sensor or reconstruction geometry that no
  redistributable photograph carries. Rather than fabricate a target, those
  pages drop the target column and show what a *published* VisBench head
  predicts, named in the footer. A three-column figure with an invented middle
  column would teach the wrong convention to exactly the reader who came here to
  learn it.

`edge` is the one in between: Taskonomy's `edge_texture` is itself computed from
its RGB frame, so the same *kind* of target is computed here — a magnitude map
where 0 is a real reading and nothing is masked, which is the convention the
probe is scored under. It is not Taskonomy's generator and does not claim to
be.

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

**Every probe.** All thirteen, across four renderers — a test asserts
`show_probes() == list_probes()`, so a new probe cannot ship undrawable.

| renderer | probes | a row is |
| --- | --- | --- |
| panel grid | `depth`, `surface_normal`, `generic_segmentation`, `semantic_segmentation`, `edge`, `keypoints2d`, `occlusion_edge`, `corner` | image, target, prediction |
| boxes | `detection` | the crop with its boxes drawn on |
| matches | `correspondence` | two views and the matches between them |
| gallery | `classification`, `retrieval`, `similarity` | the decision the probe made |

The last three have no spatial target — nothing to lay beside the image at the
same resolution. What they have is a *decision*: which class, which neighbours,
which of two candidates. See [below](#the-three-probes-that-choose).

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

### The panel grid

Depth is above. The other seven differ only in how the target is coloured and
what counts as invalid.

```{image} _static/gallery/surface_normal.png
:alt: Surface normal panels, RGB by the (n+1)/2 convention
:class: visbench-figure
```

```{image} _static/gallery/semantic_segmentation.png
:alt: Semantic segmentation panels in VOC's palette
:class: visbench-figure
```

```{image} _static/gallery/generic_segmentation.png
:alt: Binary segmentation panels, white foreground
:class: visbench-figure
```

```{image} _static/gallery/occlusion_edge.png
:alt: Occlusion edge panels, NaN holes drawn magenta
:class: visbench-figure
```

```{image} _static/gallery/edge.png
:alt: Edge magnitude panels
:class: visbench-figure
```

```{image} _static/gallery/keypoints2d.png
:alt: 2D keypoint response panels
:class: visbench-figure
```

```{image} _static/gallery/corner.png
:alt: Corner response panels, the target computed from the image
:class: visbench-figure
```

Note what differs between the last three and the depth figure at the top:
`edge`, `keypoints2d` and `corner` show **no magenta at all**, because 0 is a
real reading for them. The same frame drawn under depth's convention would come
out looking like a target full of holes.

### Boxes

Drawn straight onto the crop, in the post-transform pixel coordinates the
dataset returns — so a box that had missed its rescale would sit visibly off its
object rather than merely scoring badly.

```{image} _static/gallery/detection.png
:alt: Detection: ground-truth boxes drawn on the crop the probe saw
:class: visbench-figure
```

## Correspondence: the shape of the errors, not their size

```bash
visbench show correspondence --data /path/to/images --split val \
    --backbone dinov2_vits14 --frames 4 --out matches.png
```


```{image} _static/gallery/correspondence.png
:alt: Correspondence: two views with matches, and the coherence figure
:class: visbench-figure
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

## The three probes that choose

### `classification` — a contact sheet

```bash
visbench show classification --data ./imagenette --split val --frames 12 --out cls.png
```


```{image} _static/gallery/classification.png
:alt: A contact sheet of frames and their class labels
:class: visbench-figure
```

Thumbnails several to a row, each captioned with its class and bordered green or
red once a head is given. Packed rather than one per row because the failures
worth catching here are *class-level patterns*, invisible in four frames.

**The failure it catches**: `subset(n)` on a labelled folder takes a prefix, and
the file list is grouped by class, so an Imagenette prefix is entirely class 0 —
and the run then **scores 1.0 while measuring nothing**. `balanced_subset` exists
because of this. The footer states the split's balance, so a collapsed split
reads `1 class` whichever frames were drawn:

```
4 classes, 24 items, 6-6 per class
1 class, 8 items, 8-8 per class - every item is 'tench', so any score here is an artefact
```

Frames are picked **spread across the split** rather than as a prefix, for the
same reason: drawing the first four rows of a class-grouped folder would
reproduce the very artefact the sheet exists to reveal.

Without `--predict-from` this is a *dataset* check and needs no backbone. With
one it becomes an error analysis: the caption reads `predicted != actual`.

### `retrieval` — a query and its neighbours

```bash
visbench show retrieval --data ./imagenette --split val \
    --backbone dinov2_vits14 --frames 4 --neighbours 5 --out ret.png
```


```{image} _static/gallery/retrieval.png
:alt: Each query with its nearest neighbours, green where the class matches
:class: visbench-figure
```

One row per query: the query frame, then its nearest neighbours in rank order,
each bordered green if it shares the query's class.

Two things follow from what retrieval *is*. It **always needs a backbone** — the
neighbours are the content and do not exist until features do. And it loads the
**whole split** regardless of `--frames`, because leave-one-out retrieval over
four images ranks each against three alternatives; shortening the split would
not shorten the drawing, it would destroy what is being drawn. `--limit` caps it
if the split is large.

### `similarity` — the triplet, and who chose what

```bash
visbench show similarity --data ./nights --frames 4 --out sim.png
```


```{image} _static/gallery/similarity.png
:alt: Reference and two candidates, with the human vote marked
:class: visbench-figure
```

Reference, left candidate, right candidate, with the human vote marked. Given a
backbone, the model's choice is marked too, so agreement and disagreement are
directly visible instead of pooled into an accuracy.

**The failure it catches**: the NIGHTS CSV is read by column *name* because the
reference implementation reads the vote positionally from column 2 and the paths
from 4/5/6. Reading the wrong field scores against the wrong column, and the
result *looks like a mediocre number rather than an error*. Drawn, it is
obvious — the "preferred" candidate is visibly the more distorted one. The
footer states it as a figure too:

```
240 triplets, humans chose right in 51% (far from 50% means the vote column is wrong)
```

The candidates are presented in arbitrary order, so a balanced vote is expected;
a figure far from 50% means the column is wrong. Backbone optional here: the
human vote alone is the check, and it needs no features.

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
