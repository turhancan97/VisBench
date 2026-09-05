# `detection`

**Anchor-free, single-scale boxes decoded from one frozen feature map.**

Absolute mAP is low **by design** — no feature pyramid, so small objects fall
between cells. The board ranks representations, which is what it is for; it is
not a detector benchmark.

```{figure} /_static/gallery/detection.png
:alt: detection — image, target and prediction

What `visbench show detection` draws. {doc}`How to read it </guides/visualising>`.
```

[`examples/detect.py`](https://github.com/turhancan97/VisBench/blob/main/examples/detect.py) trains an **anchor-free,
single-scale** box head on frozen dense features: two 1x1 convolutions over the
patch grid, FCOS-style centre-inside-box assignment, focal loss on the classes
and GIoU on the boxes. It reads the Pascal VOC devkit directly, using
`ImageSets/**Main**` — the detection split, roughly four times the segmentation
one:

```bash
python examples/detect.py --data /path/to/pascal_voc --voc
```

**Read these numbers against another backbone, never against published VOC
detectors.** A single-scale head has no feature pyramid, so small objects fall
between grid cells and are simply unrecoverable. That ceiling is the point: the
probe measures what a frozen representation carries, and every point an FPN
would add is a point about the FPN. Records say
`protocol: "visbench_anchor_free_det"` so the number cannot be mistaken for a
detector's.

Measured on VOC 2012, 600 train / 600 val images at 224px, ten epochs:

## Its board

<!-- visbench:board task=detection metrics=map_50,map_50_95 heading=3 -->
### detection

| backbone | `map_50` | `map_50_95` | `classes_scored` | `detections_per_image` |
| --- | --- | --- | --- | --- |
| `dinov2_vitb14` | **0.2895** | **0.0978** | 20 | 88.5217 |
| `dinov2_vits14` | 0.2291 | 0.0702 | 20 | 83.0333 |
| `clip_vitb16` | 0.1894 | 0.0622 | 20 | 88.7500 |
| `clip_vitb32` | 0.1886 | 0.0584 | 20 | 91.3833 |
| `siglip_vitb16` | 0.1871 | 0.0637 | 20 | 99.8550 |
| `sam_vitb16` | 0.1797 | 0.0544 | 20 | 56.0033 |
| `supervised_vitb16` | 0.1669 | 0.0563 | 20 | 83.5567 |
| `dino_vitb16` | 0.1660 | 0.0583 | 20 | 78.6033 |
| `resnet50` | 0.1380 | 0.0420 | 20 | 48.2133 |
| `mae_vitb16` | 0.1296 | 0.0460 | 20 | 63.7000 |
| `convnext_base` | 0.0912 | 0.0237 | 20 | 81.1533 |
| `resnet18` | 0.0912 | 0.0270 | 20 | 57.1033 |

Ordered by `map_50`, which **disagrees with `map_50_95`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

> **Read this first.** Absolute mAP is low by design: the head is anchor-free and single-scale, so it has no feature pyramid and small objects fall between cells. The board ranks representations, which is what it is for — it is not a detector benchmark.

<sub>detection on detection_folder/val, protocol=visbench_anchor_free_det, frozen [4d3fbeb4]</sub>
<!-- /visbench:board -->

`map_50` follows VOC's protocol as `VOCevaldet.m` defines it, cross-checked
against a literal transcription of that MATLAB over 3,060 generated APs with
zero mismatches. `map_50_95` averages COCO's ten IoU thresholds but integrates
all recall points at each, where COCO quantises recall to 101 — so it is
COCO-*style*, not a COCO number.

DINOv2-B leads DINOv2-S here by 4.9 mAP@50, the same direction as semantic
segmentation and the opposite of mid-level similarity. That is a recorded
observation, not a check the probe passed — see the similarity numbers above
for why "did the bigger model win?" is not a way to validate a task.

Two things that are protocol rather than detail:

- **`difficult` objects are ignored, not dropped.** VOC removes a detection
  matching one from the tally entirely; dropping those boxes from the ground
  truth instead scores **4.3 mAP lower** on VOC val and reads as a weaker
  detector. So the scored split is built with `include_difficult=True` and the
  training split without — the example and the CLI both do this.
- **`classes_scored` is mAP's real denominator.** A class with no non-difficult
  objects in the split has undefined AP and is excluded rather than scored 0.
  Check it matches before comparing two runs.
- **Quote detection to three decimals, not four.** Every VisBench number is
  deterministic given its seed except this one. Training on a GPU is not
  bit-reproducible — convolution backward accumulates atomically, so two runs
  from the same seed give head weights differing by ~7.5e-09 — and detection is
  the only probe whose metric can resolve that. The other twelve report
  continuous averages over ~10^5 pixels, where independent noise averages down;
  average precision is a discrete ranking over ~50,000 detections with hard
  thresholds, so a handful of borderline detections flip in or out instead.
  Measured on DINOv2-S/14: two back-to-back runs scored `map_50` 0.228834 and
  0.229836, with `detections_per_image` moving by about ten detections out of
  49,800. The spread is roughly **1e-3** and the board above is one draw from it.

  Whether it shows up depends on the backbone, so three decimals is a floor
  rather than a description of every row. The two DINOv2 rows drift; both CLIP
  rows are *bit*-identical across three independent runs, matching the recorded
  value to every digit. That matters for one pair in particular — CLIP-B/16
  leads CLIP-B/32 by 0.0008, less than the DINOv2 spread, and it is precisely
  those two rows that reproduce exactly, so the ordering is measured rather than
  assumed. Every other adjacent gap on the board is 0.04–0.06.

## Run it

[`examples/detect.py`](https://github.com/turhancan97/VisBench/blob/main/examples/detect.py) is the whole path, end to end, on a real backbone.

```bash
python examples/detect.py --data /path/to/dataset
```
