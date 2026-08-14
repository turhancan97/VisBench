# The probes, one by one

Every probe on this page can also be *drawn* — see
[looking at a probe](show.md) for a rendered example of each, image beside
target, with each probe's own convention for an invalid pixel.

Every measured number on this page is generated from
[`results/corpus/visbench.jsonl`](https://github.com/turhancan97/VisBench/blob/main/results/corpus/visbench.jsonl), the
committed record corpus, by [`scripts/render_tables.py`](https://github.com/turhancan97/VisBench/blob/main/scripts/render_tables.py).
A test in the fast suite fails if any of them drifts, so a number here and the
run behind it cannot disagree.

For the ranking rules — which records may sit in one table at all — see
[the leaderboard](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md).

## Classification and retrieval

The two probes that need only a labelled image folder.

[`examples/classify.py`](https://github.com/turhancan97/VisBench/blob/main/examples/classify.py) runs the whole path on any
folder laid out as `<data>/train/<class>/…` and `<data>/val/<class>/…`:

```bash
pip install -e .                                   # required: the script imports visbench
python examples/classify.py --data /path/to/dataset
python examples/classify.py --data /path/to/dataset --limit 20   # 20 images per class, quick
```

The first run extracts features; every later run on the same data reads them
from disk and the backbone never executes, so sweeping probe settings costs
only the probe:

```bash
python examples/classify.py --data /path/to/dataset --epochs 500 --lr 0.05
```

It prints `train top1` next to the validation score. If the validation number
is low *and* `train top1` is low, the probe underfitted — raise `--lr` or
`--epochs`. If `train top1` is near 1.0, the backbone genuinely does not
separate those classes.

[`examples/retrieve.py`](https://github.com/turhancan97/VisBench/blob/main/examples/retrieve.py) does the zero-shot version —
no training at all, every image queries every other by cosine similarity:

```bash
python examples/retrieve.py --data /path/to/dataset --split val
python examples/retrieve.py --data /path/to/dataset --split val --pooling mean
```

Both examples share one cache, so running retrieval after classification on
the same split costs nothing but the ranking.

[`examples/correspond.py`](https://github.com/turhancan97/VisBench/blob/main/examples/correspond.py) runs the mid-level task —
also zero-shot, and needing no annotation at all, since each image is warped by
a known homography:

```bash
python examples/correspond.py --data /path/to/folder --limit 50
```

It reports a **ceiling** beside every score: matches can only land on patch
centres, so with 14px patches a target falling between them cannot be hit
exactly. A low `recall@1px` almost always means the grid is coarse, not that
the backbone failed.

## Mid-level image similarity

[`examples/similarity.py`](https://github.com/turhancan97/VisBench/blob/main/examples/similarity.py) asks whether the backbone
agrees with a human about which of two candidates looks more like a reference —
a two-alternative forced choice over [NIGHTS](https://dreamsim-nights.github.io)
(Fu et al., *DreamSim*). Also zero-shot: the probe is two cosine similarities
and a comparison, with no head and no training split.

```bash
python examples/similarity.py --data /path/to/nights
python examples/similarity.py --data ... --split test_no_imagenet
```

**This is not retrieval.** The ground truth is perceptual — layout, pose,
structure — not category membership, which is why the two are separate tasks. A
backbone can be strong at one and ordinary at the other.

Measured on the NIGHTS test split (1,824 triplets, `min_votes=6`), pooled
features at 224px. Humans chose "right" 49.1% of the time, so chance is ~51%:

<!-- visbench:board task=similarity metrics=accuracy,f1 heading=3 -->
### similarity

| backbone | `accuracy` | `f1` | `tie_rate` |
| --- | --- | --- | --- |
| `dinov2_vits14` | **0.8701** | **0.8687** | 0.0000 |
| `dinov2_vitb14` | 0.8580 | 0.8575 | 0.0000 |
| `siglip_vitb16` | 0.8575 | 0.8552 | 0.0000 |
| `clip_vitb32` | 0.8465 | 0.8443 | 0.0000 |
| `resnet18` | 0.8317 | 0.8307 | 0.0000 |
| `clip_vitb16` | 0.8284 | 0.8266 | 0.0000 |
| `resnet50` | 0.8273 | 0.8282 | 0.0000 |
| `convnext_base` | 0.7725 | 0.7711 | 0.0000 |
| `mae_vitb16` | 0.6897 | 0.6827 | 0.0000 |

Ordered by `accuracy`, which **disagrees with `f1`, `precision`, `recall`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>similarity on nights/test, protocol=midvision_2afc, frozen [0cc388a0]</sub>
<!-- /visbench:board -->

**The small DINOv2 beats the base one here** — the reverse of semantic
segmentation, where B leads S (0.753 against 0.732). Two tasks, two orderings,
same four backbones: which is the entire reason for probing more than one level
rather than assuming a single ranking of representations.

Run `--split test_imagenet` and `test_no_imagenet` before quoting a number: they
partition the test set by whether the reference came from ImageNet, so a gap
between them is a contamination signal rather than a similarity result. For
`dinov2_vits14` that gap is **0.882 against 0.854** — worth knowing before
reading 0.870 as a clean measure of perceptual alignment.

## Detection

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

<!-- visbench:board task=detection metrics=map_50,map_50_95 heading=3 -->
### detection

| backbone | `map_50` | `map_50_95` | `classes_scored` | `detections_per_image` |
| --- | --- | --- | --- | --- |
| `dinov2_vitb14` | **0.2895** | **0.0978** | 20 | 88.5217 |
| `dinov2_vits14` | 0.2291 | 0.0702 | 20 | 83.0333 |
| `clip_vitb16` | 0.1894 | 0.0622 | 20 | 88.7500 |
| `clip_vitb32` | 0.1886 | 0.0584 | 20 | 91.3833 |
| `siglip_vitb16` | 0.1871 | 0.0637 | 20 | 99.8550 |
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

## Edge detection

[`examples/edges.py`](https://github.com/turhancan97/VisBench/blob/main/examples/edges.py) is the first
**low-level** probe: dense edge-magnitude regression on
[Taskonomy](https://arxiv.org/abs/1804.08328)'s `edge_texture` maps.

```bash
python examples/edges.py --data /path/to/taskonomy --limit 600
visbench run edge --data /path/to/taskonomy --limit 600
```

Taskonomy's splits are **disjoint by building** — 25 rooms train, 4 validate,
5 test — so a val number is measured in rooms the probe has never seen.

600 train / 600 val frames at 224px, linear head, ten epochs:

<!-- visbench:board task=edge metrics=edge_correlation,rmse,mae heading=3 -->
### edge

| backbone | `edge_correlation` | `rmse` | `mae` |
| --- | --- | --- | --- |
| `mae_vitb16` | **0.4982** | **0.9150** | **0.4687** |
| `clip_vitb16` | 0.4565 | 0.9340 | 0.4882 |
| `dinov2_vits14` | 0.4558 | 0.9226 | 0.5028 |
| `dinov2_vitb14` | 0.4481 | 0.9265 | 0.4972 |
| `clip_vitb32` | 0.3834 | 0.9656 | 0.5080 |
| `siglip_vitb16` | 0.3639 | 0.9785 | 0.5169 |
| `resnet50` | 0.3549 | 0.9770 | 0.5056 |
| `convnext_base` | 0.3485 | 0.9671 | 0.5224 |
| `resnet18` | 0.3430 | 0.9797 | 0.5153 |

Ordered by `edge_correlation`, which **disagrees with `mae`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>edge on taskonomy_edge_texture/val, protocol=visbench_edge_regression, frozen [f8d2af2a]</sub>
<!-- /visbench:board -->

DINOv2-S edges out DINOv2-B here, by 0.008 — the same ordering as mid-level
similarity and the opposite of segmentation and detection. The margin is small
enough to be worth stating as *consistent with* the level taxonomy rather than
as evidence for it.

Three things that are protocol rather than detail:

- **Quote `edge_correlation`, not `rmse`.** Edge magnitude is concentrated near
  zero, so a probe that ignores its input and predicts the split's mean
  everywhere gets a *small* RMSE while having learned nothing. Pearson
  correlation is invariant to scale and offset, so it asks only whether the
  representation knows **where** the edges are, and scores that probe 0. RMSE
  and MAE are reported alongside because correlation is blind to the opposite
  failure — right shape, wrong magnitude.
- **Nothing is masked.** Depth has holes and normals have zero-length vectors,
  both meaning "no ground truth". Here 0 means *no edge*, a real reading
  covering most of most frames. Masking it away would score the probe only
  where an edge already is.
- **This is not BSDS500's protocol and must not share a table with one.** BSDS's
  ODS/OIS/AP matches edge pixels by bipartite correspondence after non-maximum
  suppression, swept over thresholds, against several annotators. Records say
  `protocol: "visbench_edge_regression"`.

## Keypoints, and occlusion edges

Two more probes share the edge probe's implementation and differ only in what
they read — [`examples/keypoints.py`](https://github.com/turhancan97/VisBench/blob/main/examples/keypoints.py)
on Taskonomy's `keypoints2d` response maps, and
[`examples/occlusion_edges.py`](https://github.com/turhancan97/VisBench/blob/main/examples/occlusion_edges.py)
on its `edge_occlusion` maps.

```bash
visbench run keypoints2d     --data /path/to/taskonomy --limit 600
visbench run occlusion_edge  --data /path/to/taskonomy --limit 600
```

600 train / 600 val frames at 224px, linear head, ten epochs:

The two probes share every line of their implementation and sit one tier apart,
so they render as two boards rather than one row each:

<!-- visbench:board task=keypoints2d metrics=keypoint_correlation,mae,rmse heading=3 -->
### keypoints2d

| backbone | `keypoint_correlation` | `mae` | `rmse` |
| --- | --- | --- | --- |
| `mae_vitb16` | **0.2626** | **1.0533** | **2.5342** |
| `dinov2_vits14` | 0.2356 | 1.1281 | 2.5472 |
| `dinov2_vitb14` | 0.2248 | 1.1294 | 2.5541 |
| `convnext_base` | 0.2187 | 1.1690 | 2.5633 |
| `clip_vitb16` | 0.2175 | 1.1533 | 2.5770 |
| `clip_vitb32` | 0.1933 | 1.1474 | 2.5891 |
| `resnet50` | 0.1792 | 1.2374 | 2.6163 |
| `resnet18` | 0.1659 | 1.2579 | 2.6282 |
| `siglip_vitb16` | 0.1577 | 1.1789 | 2.6072 |

Ordered by `keypoint_correlation`, which **disagrees with `mae`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>keypoints2d on taskonomy_keypoints2d/val, protocol=visbench_keypoint2d_regression, frozen [e647c722]</sub>
<!-- /visbench:board -->

<!-- visbench:board task=occlusion_edge metrics=occlusion_edge_correlation,mae,rmse heading=3 -->
### occlusion_edge

| backbone | `occlusion_edge_correlation` | `mae` | `rmse` |
| --- | --- | --- | --- |
| `mae_vitb16` | **0.3273** | **0.1904** | **0.4261** |
| `dinov2_vitb14` | 0.3167 | 0.2061 | 0.4315 |
| `dinov2_vits14` | 0.2924 | 0.2205 | 0.4373 |
| `clip_vitb16` | 0.2558 | 0.2149 | 0.4415 |
| `siglip_vitb16` | 0.2254 | 0.2205 | 0.4423 |
| `clip_vitb32` | 0.2174 | 0.2203 | 0.4440 |
| `resnet50` | 0.1979 | 0.2294 | 0.4502 |
| `resnet18` | 0.1745 | 0.2418 | 0.4578 |
| `convnext_base` | 0.1741 | 0.2533 | 0.4704 |

Ordered by `occlusion_edge_correlation`, which **disagrees with `mae`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>occlusion_edge on taskonomy_edge_occlusion/val, protocol=visbench_occlusion_edge_regression, frozen [d12a4923]</sub>
<!-- /visbench:board -->

**The occlusion-edge probe is mid-level and the texture-edge probe is
low-level, and they are otherwise the same code.** An occlusion edge is a depth
discontinuity — a painted line on a wall is not one, and the silhouette of a
chair against a similarly-toned wall is one with almost no intensity gradient —
so recovering it needs scene geometry. Running both on the same frames is about
as direct a comparison of the two tiers as VisBench offers.

Two things worth knowing before adding a fourth probe of this shape:

- **`edge_occlusion` is loaded in log space, and nothing else is.** Its target
  holds 46% of its mass in the strongest 1% of pixels, against ~0.10 for the
  other two. At that tail the L1 loss (chosen so strong pixels cannot dominate)
  and the Pearson metric (dominated by exactly those pixels) pull apart: the
  linear-target probe scored 0.088 and stayed flat under four target scales, 4x
  the training budget and a 10x learning rate. `dataset_params` records
  `target_transform`, so a log-space correlation can never be pooled with a
  linear-space one.
- **Ask whether a probe ranks, not whether its number is high.** The linear
  occlusion probe's DINOv2-S-versus-B gap was 0.0035 — noise. A low score can be
  by design; failing to separate two backbones never is.

## Corner detection — the probe that brings its own target

[`examples/corners.py`](https://github.com/turhancan97/VisBench/blob/main/examples/corners.py)
is the only dense probe here that needs **no dataset**. Its target is computed
from the images at read time — Shi-Tomasi cornerness, the smaller eigenvalue of
the Gaussian-windowed structure tensor, compressed with `log1p`:

```bash
visbench run corner --data /path/to/any/images --limit 600
```

```text
<data>/train/images/*.jpg
<data>/val/images/*.jpg
```

That is the whole layout. Any folder of photographs runs it.

**The generator is part of the protocol.** A stored target is identified by the
dataset it came from; a derived one only by the code that computed it. So
`--corner-sigma`, `--corner-transform` and `--corner-scale` all land in
`dataset_params`, which puts two settings of the same operator into two
comparability groups automatically. Two records both saying "corners" are not
thereby comparable — check the fields.

600 train / 600 val Taskonomy frames at 224px, linear head, ten epochs — the
*same frames* the edge probe uses, so only the target differs:

<!-- visbench:board task=corner metrics=corner_correlation,mae,rmse heading=3 -->
### corner

| backbone | `corner_correlation` | `mae` | `rmse` |
| --- | --- | --- | --- |
| `mae_vitb16` | **0.6669** | **0.4187** | 0.6979 |
| `dinov2_vitb14` | 0.6526 | 0.4402 | **0.6899** |
| `dinov2_vits14` | 0.6512 | 0.4510 | 0.6919 |
| `clip_vitb16` | 0.6227 | 0.4508 | 0.7229 |
| `siglip_vitb16` | 0.5383 | 0.4866 | 0.7846 |
| `clip_vitb32` | 0.5367 | 0.4829 | 0.7825 |
| `convnext_base` | 0.5129 | 0.4852 | 0.7833 |
| `resnet18` | 0.5014 | 0.4706 | 0.8085 |
| `resnet50` | 0.4923 | 0.4661 | 0.8033 |

Ordered by `corner_correlation`, which **disagrees with `mae`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>corner on val/val, protocol=visbench_shi_tomasi_regression, frozen [94342f24]</sub>
<!-- /visbench:board -->

**The frames behind this board are pinned, and they have to be.** A probe whose
target is computed runs on *any* folder — which is its selling point and, for a
leaderboard, its problem: two people's corner numbers are comparable only if
they ran the same images. Nothing in the probe pins which. So the corpus names a
set, and
[`scripts/stage_corner_frames.py`](https://github.com/turhancan97/VisBench/blob/main/scripts/stage_corner_frames.py)
is what reconstructs it: the first 600 rows of Taskonomy's `tiny` split lists,
symlinked into the flat layout above. That is the *same* frame set the edge and
keypoint boards use, verified set-equal rather than assumed — which is what
makes the cross-probe comparison below exact rather than suggestive.

Read that as a property of this probe, not a requirement of it. Running `corner`
needs no download; ranking two backbones *against this board* needs these
frames.

**A corner score and an edge score are not independent evidence.** The two
targets correlate at **0.52** per image, against **0.147** between the two
Taskonomy probes above. The overlap is intrinsic, not an artifact of the
compression — it holds across eight transforms including near-linear ones,
because a corner is a pixel whose gradient is large in two directions and an
edge map is gradient magnitude.

They do rank differently, which is what earns the probe its place: the spread is
**0.1603** against the edge probe's 0.1136, and **CLIP-B/16 comes first on edges
and third on corners** while the two ResNets swap. See
[the low-level README](https://github.com/turhancan97/VisBench/blob/main/visbench/tasks/low_level/README.md)
for why the operator is Shi-Tomasi rather than Harris, and for the tail
measurements that chose the compression.

## Dense tasks

[`examples/depth.py`](https://github.com/turhancan97/VisBench/blob/main/examples/depth.py),
[`examples/normals.py`](https://github.com/turhancan97/VisBench/blob/main/examples/normals.py),
[`examples/segment.py`](https://github.com/turhancan97/VisBench/blob/main/examples/segment.py) and
[`examples/segment_semantic.py`](https://github.com/turhancan97/VisBench/blob/main/examples/segment_semantic.py) train a probe
head on frozen dense features. Depth and normals follow
[probe3d](https://arxiv.org/abs/2404.08476)'s protocols; both segmentation tasks
borrow only its optimiser schedule, since that paper has neither. They want
images and per-pixel targets paired by filename stem under `train/` and `val/`:

```bash
python examples/depth.py   --data /path/to/dataset --target-scale 1000
python examples/normals.py --data /path/to/dataset --normal-source geonet
python examples/segment.py --data /path/to/dataset
python examples/segment_semantic.py --data /path/to/dataset --num-classes 21
```

Semantic segmentation also reads the Pascal VOC devkit directly, using the
official split lists rather than whatever the folders contain:

```bash
python examples/segment_semantic.py --data /path/to/pascal_voc --voc
```

Measured on VOC 2012 val (1449 images), linear head, 224px, at the default
ten-epoch schedule:

<!-- visbench:board task=semantic_segmentation metrics=miou,miou_per_image,pixel_acc,mean_acc heading=3 -->
### semantic_segmentation

| backbone | `miou` | `miou_per_image` | `pixel_acc` | `mean_acc` |
| --- | --- | --- | --- | --- |
| `dinov2_vitb14` | **0.7533** | **0.7161** | **0.9316** | **0.8403** |
| `dinov2_vits14` | 0.7328 | 0.6841 | 0.9267 | 0.8271 |
| `clip_vitb16` | 0.6546 | 0.6683 | 0.9019 | 0.7312 |
| `clip_vitb32` | 0.5813 | 0.6067 | 0.8731 | 0.6633 |
| `siglip_vitb16` | 0.5405 | 0.3210 | 0.8539 | 0.6511 |
| `convnext_base` | 0.4880 | 0.4596 | 0.8310 | 0.5902 |
| `resnet50` | 0.4574 | 0.5163 | 0.8322 | 0.5248 |
| `resnet18` | 0.4212 | 0.4497 | 0.8205 | 0.4915 |
| `mae_vitb16` | 0.3350 | 0.4555 | 0.8269 | 0.3757 |

Ordered by `miou`, which **disagrees with `miou_per_image`, `pixel_acc`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>semantic_segmentation on VOC2012/val, protocol=visbench_semantic_seg, frozen [e14b47db]</sub>
<!-- /visbench:board -->

**Report the linear head.** It is the default and the only one under which a
difference between two backbones is a difference between two *feature maps*.
The DPT head is probe3d's own choice and scores higher for everyone, so run
both and say which:

```bash
python examples/normals.py --data ... --head dpt --layers 2 5 8 11
```

Features are shared between the three tasks when the images and `--image-size`
match, so probing all of them on one dataset costs one extraction. Splits larger
than memory are fine — dense features stream from the cache a batch at a time
rather than being stacked.

Things that will bite otherwise:

- **Say where surface normals came from.** NYU's are derived (GeoNet's
  extraction, or Ladicky's) rather than sensed, and the sources disagree enough
  to move every metric. `--normal-source` is recorded verbatim in the result.
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
  elsewhere. On VOC they sit five points apart. Quote `miou` against published
  numbers, and say which one you mean.
- **Label maps are read without mode conversion, and getting this wrong is
  silent.** VOC's PNGs are palette images whose raw bytes are the class indices;
  resolving the palette turns classes `[0, 1, 15]` into `[0, 38, 147]`, which
  trains and scores perfectly happily against labels that mean nothing. Use
  `load_label_map`, not `load_mask`, for anything multi-class — including
  binarising a VOC map, since `load_mask` would read its void border as
  foreground.
- **The ten-epoch schedule assumes a dataset the size of NYUv2.** On a small
  split it underfits badly — 80 training images gave 0.16 IoU at the defaults
  and 0.87 at `--epochs 40 --lr 5e-3`, on identical features. `train_loss` is
  printed for exactly this: a poor score with a high training loss means the
  probe did not converge, which is a different finding from a representation
  that does not carry the signal.

## Measured on Imagenette

3,925-image val split, one V100. Correspondence on 200 pairs at `max_warp=0.2`.

<!-- visbench:board task=classification metrics=top1,top5 heading=3 -->
### classification

| backbone | `top1` | `top5` |
| --- | --- | --- |
| `convnext_base` | **0.9997** | **1.0000** |
| `resnet50` | 0.9980 | 0.9997 |
| `dinov2_vitb14` | 0.9975 | **1.0000** |
| `clip_vitb16` | 0.9954 | 0.9997 |
| `dinov2_vits14` | 0.9939 | 0.9997 |
| `siglip_vitb16` | 0.9936 | 0.9995 |
| `clip_vitb32` | 0.9921 | 0.9992 |
| `resnet18` | 0.9888 | 0.9995 |
| `mae_vitb16` | 0.9582 | 0.9985 |

Ordered by `top1`, which **disagrees with `top5`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>classification on val/val, frozen [12e02eff]</sub>
<!-- /visbench:board -->

<!-- visbench:board task=retrieval metrics=mAP,recall@1,recall@5 heading=3 -->
### retrieval

| backbone | `mAP` | `recall@1` | `recall@5` |
| --- | --- | --- | --- |
| `convnext_base` | **0.9890** | **0.9987** | **0.9990** |
| `resnet50` | 0.9357 | 0.9901 | 0.9987 |
| `dinov2_vitb14` | 0.9171 | 0.9954 | 0.9977 |
| `clip_vitb16` | 0.9102 | 0.9893 | 0.9975 |
| `dinov2_vits14` | 0.8893 | 0.9921 | 0.9972 |
| `clip_vitb32` | 0.8680 | 0.9806 | 0.9941 |
| `resnet18` | 0.8648 | 0.9725 | 0.9944 |
| `siglip_vitb16` | 0.8525 | 0.9799 | 0.9936 |
| `mae_vitb16` | 0.1883 | 0.6741 | 0.8892 |

Ordered by `mAP`, which **disagrees with `recall@1`, `recall@5`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>retrieval on val/val, frozen [eb312a7b]</sub>
<!-- /visbench:board -->

<!-- visbench:board task=correspondence metrics=recall@5px,recall@10px,auc@5px heading=3 -->
### correspondence

| backbone | `recall@5px` | `recall@10px` | `auc@5px` | `ceiling_auc@5px` | `ceiling_recall@10px` | `ceiling_recall@5px` | `num_matches` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | **0.3577** | **0.7325** | **0.1370** | 0.1462 | 0.9212 | 0.3932 | 17,300 |
| `dinov2_vits14` | 0.3049 | 0.6526 | 0.1152 | 0.1454 | 0.9329 | 0.4123 | 23,439 |
| `dinov2_vitb14` | 0.2816 | 0.6260 | 0.1055 | 0.1389 | 0.9264 | 0.4005 | 27,590 |
| `clip_vitb16` | 0.2689 | 0.5725 | 0.1080 | 0.1333 | 0.9159 | 0.3519 | 12,798 |
| `siglip_vitb16` | 0.1461 | 0.3463 | 0.0560 | 0.1073 | 0.8715 | 0.3041 | 13,504 |
| `resnet18` | 0.0973 | 0.3256 | 0.0335 | 0.0350 | 0.3653 | 0.1028 | 4,911 |
| `clip_vitb32` | 0.0897 | 0.2951 | 0.0321 | 0.0352 | 0.3633 | 0.1002 | 4,283 |
| `resnet50` | 0.0887 | 0.3003 | 0.0299 | 0.0350 | 0.3595 | 0.1038 | 4,373 |
| `convnext_base` | 0.0824 | 0.2950 | 0.0280 | 0.0320 | 0.3575 | 0.0940 | 5,413 |

Ordered by `recall@5px`, which **disagrees with `auc@1px`, `auc@2px`, `auc@5px`, `recall@10px`, `recall@1px`, `recall@2px`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

> **Read this first.** Thresholds are in **pixels**, which is the only unit two backbones can be compared in — a patch width is 14px on DINOv2/14 and 32px on a ResNet, so scoring in patch widths asks each backbone a different question. Read `ceiling_` beside every score: a 7x7 grid cannot place a match within 5px more than ~10% of the time whatever its features are, so part of this ordering is resolution rather than quality. `num_matches` is the denominator each backbone's own ratio test left, and it varies by more than 5x.

<sub>correspondence on val/val, frozen [7db23175]</sub>
<!-- /visbench:board -->

The patch grid at 224px is 16x16 for DINOv2 (patch 14), 14x14 for CLIP-B/16,
MAE-B/16 and SigLIP-B/16, and 7x7 for CLIP-B/32, ConvNeXt-B and both ResNets.
Hold that beside the correspondence board: it is what `num_matches` tracks, and
it moves the score without saying anything about feature quality.

**Read the supervised rows with care.** Imagenette's ten classes are ImageNet-1k
wnids, and both `resnet50.a1_in1k` and `convnext_base.fb_in1k` were trained on
ImageNet-1k with labels — they have seen these exact categories, while DINOv2
and MAE are self-supervised and CLIP and SigLIP are image-text. Their semantic
scores are close to in-distribution recall, not a transfer result: ConvNeXt tops
both the classification board (0.9997) and the retrieval board (0.9890) and then
places seventh or lower on every dense geometric one. This says more about the
dataset than the backbone; a benchmark comparing supervised against
self-supervised features needs data the supervised model has not been trained
on.

**MAE is the sharpest tier separation this corpus has produced, and it is worth
reading before trusting any single board.** `mae_vitb16` is **first on six of
the thirteen** — all three low-level probes (edge, 2D keypoints, corner), plus
correspondence, occlusion edges and surface normals — and **last or
next-to-last on the four semantic ones**: ninth on classification (0.9582),
ninth on retrieval (0.1883), ninth on semantic segmentation (0.3350) and ninth
on mid-level similarity (0.6897). No other backbone here is simultaneously best
and worst.

Its correspondence win is not the grid. `mae_vitb16` scores 0.3577 against a
`ceiling_recall@5px` of 0.3932, where `dinov2_vits14` scores 0.3049 against a
*higher* ceiling of 0.4123 — so MAE reaches 91% of what its own quantisation
allows and DINOv2-S reaches 74%. This is the case the `ceiling_` prefix was
added for: read that way the gap is larger than the raw scores show, not
smaller.

That is what the taxonomy predicts of a pixel-reconstruction objective, and the
two numbers that look wrong are the ones that confirm it. **Retrieval's 0.1883
is barely above the 0.1 chance floor** for ten balanced classes — read alone it
looks like a broken run. It is not, and the check is internal to the corpus:
the *same* features score 0.9582 top-1 under a trained linear probe. A learned
projection recovers category structure that cosine similarity on the raw CLS
token cannot, which is precisely the documented behaviour of MAE without
fine-tuning. A genuine extraction bug would have taken the linear probe down
with it.

The practical consequence is that **"which backbone is best" is not a
well-formed question here**, and the corpus now says so loudly rather than by
implication. Six of the thirteen boards are headed by a model that is last on
the other four. Pick the tier that matches the downstream use.

Correspondence is less exposed to that (no labels are used), but comes with its
own caveat: ResNet's 7x7 grid means matching among 49 candidates against
DINOv2's 256. Patch-width thresholds make the *error* comparable across grids;
they do not make the matching problem equally hard.

The DINOv2/CLIP split is the cleaner comparison, and it lands where the task
taxonomy predicts: CLIP ahead on the semantic tasks, behind on the geometric
one despite a higher ceiling.

Retrieval with `--pooling mean` instead of CLS costs DINOv2 about 1.8 points of
recall@1 (0.9740, mAP 0.8314).

**Correspondence thresholds are in pixels (`px`), and this changed in v0.6.1.**
Patch widths were the default until then, on the reasoning that a match can
only land on a patch centre so patch spacing is the natural yardstick. That is
true *within* one backbone and wrong across several: a patch is 14px on
DINOv2/14, 16px on CLIP ViT-B/16 and 32px on ViT-B/32 or a ResNet stage, so
`recall@1p` asks a coarse-grid backbone to land within 32px and a fine-grid one
within 14px, then prints both under one name.

It inverted the board. On 200 pairs, `resnet18` read **0.8927** against
`dinov2_vits14`'s 0.7834 in patch widths — and **0.0973 against 0.3049** in
pixels. First and last place swap.

The quantisation floor is real, and the honest handling is the `ceiling_`
metrics that already travel beside every score: a 7×7 grid cannot place a match
within 5px more than ~10% of the time whatever its features are, against ~41%
for a 16×16 grid. That *states* the disadvantage instead of normalising it
away. Pass `--units patch` deliberately for a single-backbone study; do not
rank two backbones with it.

Degradation with viewpoint is gradual. Measured on **DINOv2 ViT-S/14 alone**,
50 pairs, in patch widths — a within-backbone sweep, which is what that unit is
for — as score/ceiling:

| `max_warp` | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|
| recall@1p | 0.872 | 0.834 | 0.765 | 0.744 | 0.732 |
| ceiling | 1.000 | 0.980 | 0.941 | 0.916 | 0.891 |
| matches kept / pair | 160 | 143 | 115 | 86 | 55 |

The ratio test rejects more as the warp grows (164 → 59 matches), which is the
behaviour it exists for: fewer matches, still mostly correct.

Chance recall@1 is 0.10. Retrieval reused the classification cache: 3,925
hits, 0 misses, 8 s end to end. Switching to `--pooling mean` is a genuine
re-extraction (3,925 misses, 56 s) because pooling is part of the cache key —
and it costs about 1.8 points of recall@1 here, which is the sort of question
these two lines of CLI exist to answer.

Every one of these examples has a `visbench run` equivalent — see
[the command-line section of the README](https://github.com/turhancan97/VisBench/blob/main/README.md#the-command-line). They stay because an example is readable
top to bottom and a subcommand is not: when you want to know *how* a probe is
wired up, the script is the answer.
