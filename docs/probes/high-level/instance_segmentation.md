# `instance_segmentation`

**Detect, then segment inside each detected box — over one frozen feature map.**

The seventeenth probe, and the only one that asks *which object* a pixel belongs
to. {doc}`semantic_segmentation </probes/high-level/semantic_segmentation>`
reads the same 1,449 VOC images and asks which **class** a pixel is; two
touching sheep are one region to that probe and two to this one.

```{figure} /_static/gallery/instance_segmentation.png
:alt: instance_segmentation — image and per-instance target masks

What `visbench show instance_segmentation` draws. One colour per instance,
arbitrary and **not** matched between panels — an instance index is only
annotation order. {doc}`How to read it </guides/visualising>`.
```

[`examples/segment_instances.py`](https://github.com/turhancan97/VisBench/blob/main/examples/segment_instances.py)
is the whole path. The head is a
{class}`~visbench.heads.detection.DetectionHead` for the boxes plus **one 1x1
convolution** for the masks, over RoI-aligned features:

```bash
python examples/segment_instances.py --data /path/to/pascal_voc
```

## Why VOC and not COCO

A dense probe reads one feature vector per patch, so an instance has to survive
the feature grid to be recoverable at all. Measured over all 1,449 VOC val
images at 224px on a 16x16 grid:

| | VOC 2012 | COCO |
| --- | --- | --- |
| median instance | **16.92 patches** | 2.27 patches |
| instance pairs sharing a grid cell | **0 of 3,207** | many |

No patch is contested on VOC, which is what makes a per-patch head viable.
COCO's median instance covers barely two patches, so most of its annotation is
below what the probe can see — a board built there would rank feature
resolution and call it instance segmentation.

**Read the mask AP against another backbone, never against the segmentation
literature.** The mask branch is a *single 1x1 convolution*, where Mask R-CNN's
is four 3x3 convolutions and a transposed convolution on an FPN. RoIAlign
carries no parameters, so the only learned thing between a backbone's features
and a predicted mask is that convolution — which is what makes a difference
between two rows a difference between two representations. Records say
`protocol: "visbench_anchor_free_instance"`, so the number cannot be mistaken
for a segmenter's.

The oracle a *perfect* per-patch predictor would reach on this split is mask
mAP@50 **0.6666**, against a connected-components floor of 0.1376 mean IoU. The
mask-AP harness scores exactly **1.0000** on perfect predictions, on real VOC
val and in the fast suite, which is what makes any lower number attributable to
the probe rather than to the scorer.

Measured on VOC 2012, the official 1,464 train / 1,449 val segmentation splits
at 224px, ten epochs:

## Its board

<!-- visbench:board task=instance_segmentation metrics=mask_map_50,mask_map_50_95,box_map_50 heading=3 -->
### instance_segmentation

| backbone | `mask_map_50` | `mask_map_50_95` | `box_map_50` | `classes_scored` | `detections_per_image` |
| --- | --- | --- | --- | --- | --- |
| `dinov2_vitb14` | **0.2861** | **0.1077** | **0.3067** | 20 | 83.8226 |
| `dinov2_vits14` | 0.2696 | 0.0994 | 0.2863 | 20 | 73.7081 |
| `mae_vitb16` | 0.2186 | 0.0779 | 0.2974 | 20 | 54.1477 |
| `dino_vitb16` | 0.2122 | 0.0782 | 0.2563 | 20 | 71.6674 |
| `sam_vitb16` | 0.2068 | 0.0704 | 0.2518 | 20 | 57.7129 |
| `clip_vitb16` | 0.1805 | 0.0617 | 0.2126 | 20 | 90.2740 |
| `clip_vitb32` | 0.1520 | 0.0448 | 0.2340 | 20 | 87.7453 |
| `siglip_vitb16` | 0.1491 | 0.0386 | 0.2087 | 20 | 99.9979 |
| `supervised_vitb16` | 0.1486 | 0.0454 | 0.2051 | 20 | 91.0366 |
| `resnet50` | 0.0973 | 0.0235 | 0.1796 | 20 | 59.5438 |
| `resnet18` | 0.0794 | 0.0193 | 0.1600 | 20 | 64.9607 |
| `convnext_base` | 0.0713 | 0.0169 | 0.1398 | 20 | 75.3741 |

Ordered by `mask_map_50`, which **disagrees with `box_map_50`, `mask_map_50_95`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>instance_segmentation on voc_instance/val, protocol=visbench_anchor_free_instance, frozen [ed4484fa]</sub>
<!-- /visbench:board -->

## `box_map_50` is beside `mask_map_50` on purpose

Mask AP falls when either half of a detect-then-segment probe fails, and the two
failures mean opposite things:

- **poor masks inside good boxes** is a statement about whether the features
  carry an *outline*;
- **boxes that miss** is a statement about localisation, and is a finding about
  the detection half rather than the mask half.

A single number cannot separate those, so both are recorded. Every box decision
here — FCOS-style assignment, focal and GIoU losses, per-class NMS — is
inherited from {doc}`detection </probes/high-level/detection>` unchanged, so a
gap between the two boards is about the split and the schedule, not about two
implementations of one idea.

**`mae_vitb16` is the row where that pays off.** It has the second-best
`box_map_50` on the board (0.2974, behind only `dinov2_vitb14`) and the third
`mask_map_50` (0.2186, well behind both DINOv2 rows). Its boxes land and its
outlines do not, which is a statement about what its features carry — and it is
invisible in mask AP alone, where it simply looks like third place.

**Do not read the two halves as two rankings, though.** Across all twelve
backbones the box and mask orderings agree at Spearman **+0.986**: the
disagreement is local to a few rows rather than a different view of the corpus.
Both halves rank with the *mid-level* geometry boards rather than with
high-level ones — which means the mask branch is not what puts this board
there. See {doc}`reading a board </guides/reading-a-board>` and
[`CORPUS_FINDINGS.md`](https://github.com/turhancan97/VisBench/blob/main/CORPUS_FINDINGS.md).

## Protocol details that move the number

- **The mask branch trains on ground-truth boxes and predicts on detected
  ones**, recorded as `mask_train_boxes: "ground_truth"`. The boxes come from
  the same head being trained, so early epochs would otherwise hand the mask
  branch RoIs containing no object at all.
- **The mask branch is class-agnostic** — one channel, not one per class. The
  class is already decided by the detection branch, and asking the mask branch
  to relearn it makes the readout less linear rather than more informative.
  Recorded as `mask_classes: "agnostic"`.
- **`mask_size` is 14, not Mask R-CNN's 28.** The features are on a 16x16 grid,
  so a larger RoI asks RoIAlign to invent detail between patch centres that the
  backbone never produced. It travels in `task_params`, because two runs at
  different values are not comparable.
- **`mask_threshold` is protocol, not display.** Raising it shrinks every mask
  and moves mask AP.
- **An instance's class is read from `SegmentationClass` exactly**, at that
  instance's pixels, and a mixed instance raises rather than voting. All 6,934
  instances in train and val carry exactly one class at purity 1.000000, so a
  vote would only ever paper over the two files disagreeing.
- **Boxes are derived from the cropped mask**, which deletes the
  rescale-and-shift hazard rather than re-testing it.

## Memory

Average precision is a **dataset-level** ranking, so every prediction is
collected before scoring — and masks make that expensive. At the defaults
`max_detections` masks at `image_size**2` is 5.0 MB per image, and DINOv2-S
decodes 74.4 detections an image over VOC val, so the collected predictions come
to about **5.4 GB**. They are stored as `bool` rather than `float32`, which is
what makes that feasible at all; the float version is 21.6 GB. Lower
`max_detections` or raise `score_threshold` if it does not fit — both are
recorded, because both move mask AP.

## Run it

```bash
# The example, on a VOC devkit root (the parent of VOCdevkit/)
python examples/segment_instances.py --data /path/to/pascal_voc

# The CLI, on the VOC2012 directory itself, as its siblings take it
visbench run instance_segmentation \
  --data /path/to/VOCdevkit/VOC2012 \
  --stems /path/to/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
  --train-stems /path/to/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt
```
