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

## Feature resolution is the strongest correlate of every dense board

Across the twelve backbones, the size of the feature grid correlates with every
dense board — +0.958 with `generic_segmentation`, +0.867 with `surface_normal`,
+0.818 with `depth` — while embedding width correlates with essentially nothing
(-0.44 to +0.43, no consistent sign). Reproduce with
`scripts/analyse_board_correlates.py --section structure`.

That was confounded, because the only backbones carrying 256 tokens are the two
DINOv2s: grid size, the DINOv2 training objective and LVD-142M pretraining were
one variable. The `dinov2_vitb14_196` control separates them by running the
same weights at 196px, giving DINOv2-B the same 14x14 grid as every ViT-B/16.

| board | 256 tokens | 196 tokens | change |
| --- | --- | --- | --- |
| `generic_segmentation` | 0.7556 | 0.7407 | -2.0% |
| `depth` | 0.7851 | 0.7791 | -0.8% |
| `surface_normal` (deg, lower better) | 30.1143 | 30.6556 | +1.8% |
| `edge` | 0.4481 | 0.4363 | -2.6% |
| `corner` | 0.6526 | 0.6349 | -2.7% |

**Matching the grid costs under 3% everywhere**, and DINOv2-B keeps its lead
over the whole ViT-B/16 pack on both boards it led. Resolution accounts for 21%
of its `generic_segmentation` lead and 7% of its `depth` lead — most of the gap
is not the grid. On the other three boards DINOv2-B never led (`mae_vitb16` is
ahead), so there was no lead for resolution to explain.

The control spans 256 to 196 tokens where the corpus correlation spans 49 to
256, so it bounds the comparison that was confounded and says nothing about the
49-token backbones. See
[`results/controls/`](https://github.com/turhancan97/VisBench/tree/main/results/controls)
for the records and the full write-up; they are deliberately kept out of the
corpus, so no table on this page contains them.

### How much of a target the grid puts out of reach, before any backbone

The five magnitude and orientation probes report a **ceiling** beside every
score, as `correspondence` does. It is the target itself pooled to that run's
feature grid and upsampled back — what a perfect backbone would make available,
since the head reads one feature vector per patch. Measured over the pinned 600
frames on the three grids the corpus backbones produce at 224px:

| target | 16 (ViT/14) | 14 (ViT/16) | 7 (ResNet) |
|---|---|---|---|
| `corner` | 0.8316 | 0.8053 | 0.6685 |
| `keypoints2d` | 0.6976 | 0.6674 | 0.4728 |
| `edge` | 0.6336 | 0.6106 | 0.4977 |
| `occlusion_edge` | 0.5301 | 0.5150 | 0.4336 |
| `orientation` (deg, lower better) | 11.02 | 12.18 | 18.57 |

So a ResNet is scored on `corner` against a ceiling a fifth lower than a
DINOv2's, and reading the board without that invites attributing a grid
difference to a representation. This is the resolution finding above arrived at
from the other side, with no weights involved at all — it is a property of the
targets.

**Do not rank on a ceiling, average one, or divide by one.** It says what was
available, not what the backbone recovered; because it falls with the grid, a
board ordered by ceiling is a board ordered by resolution. The ratio does not
behave either — `corner` reaches 80% of its ceiling and `keypoints2d` 41%, and
both rank backbones perfectly well. (Correspondence's ceiling *is* a proven
bound and is read as a fraction further down this page; this one is an
achievable reconstruction, not a proven maximum.) **The five low-level boards
below now carry their ceilings as columns**, the way the correspondence board
already did — never bolded, because "best ceiling" is not a thing to win. Note
the column is really a property of the *grid*: the two DINOv2s share a value and
so do the ViT/16s.

## The high-level tier is two clusters, not one

The probes are grouped into high-, mid- and low-level tiers following
[Chen, Marks & Cheng](https://arxiv.org/abs/2411.17474), and the natural
reading is that probes in one tier measure related things. Ranking each board
against every other and averaging within and across tiers:

| | mean rho |
| --- | --- |
| within low-level | **+0.839** |
| within mid-level | **+0.666** |
| within high-level | **+0.297** |
| across tiers | +0.266 |

Every within-tier mean now exceeds the cross-tier mean, but do not read that as
the high-level tier cohering. It sat *below* the cross-tier line at thirteen
boards and crossed it only when `scene_classification` (the fourteenth) was
added — and it crossed by pulling the *cross-tier* number down, because the
image-level high-level boards disagree sharply with the low-level tier (the most
negative pair in the corpus is `orientation` / `scene_classification` at
−0.51). The within-high-level mean barely moved.

What is stable across six, nine and twelve backbones is that high-level is
**two tight clusters that ignore each other**:

| pair | rho |
| --- | --- |
| detection / semantic_segmentation | **+0.804** |
| classification / retrieval | **+0.769** |
| detection / scene_classification | **+0.720** |
| scene_classification / semantic_segmentation | +0.524 |
| classification / scene_classification | +0.161 |
| classification / detection | +0.140 |
| classification / semantic_segmentation | +0.140 |
| detection / retrieval | −0.035 |
| retrieval / semantic_segmentation | −0.042 |
| retrieval / scene_classification | −0.217 |

Image-level categorisation on one side, localised prediction on VOC on the
other, and nothing linking them. **`scene_classification` is image-level
classification and still lands with the localised cluster** (+0.72 with
detection, −0.22 with retrieval) — a place category is recovered from layout
and context, which is what the VOC-dense probes reward and single-object
Imagenette does not. So **do not average a backbone's high-level results into
one figure of merit** — the five boards are not measuring one capability, and a
mean over them describes nothing. Read them individually.

This clustering is **not** an artefact of which images each probe reads, which
is the first thing to suspect since `detection` and `semantic_segmentation`
both run on VOC. `semantic_segmentation` and `generic_segmentation` read the
*same 1449 images* through the same head, and they agree **least** of the three
VOC pairs (+0.538, against +0.804 and +0.720 for pairs that read different
frames). `generic_segmentation`'s nearest neighbours are `surface_normal`,
`depth` and `corner` — NYUv2 and Taskonomy. And the three probes that share
Imagenette average +0.128, the lowest figure of any shared corpus here.

The low-level tier, by contrast, is *one* cluster and `orientation` (the
fifteenth board) tightened it, from +0.825 to +0.839. That is worth a note: the
orientation *target* is near-independent of every other probe's per pixel — it
is a phase, and `|r|` with the `edge` and `corner` targets is under 0.09 — yet
its *board* ranks backbones almost exactly like `keypoints2d` (rho **+0.95**),
`corner` (+0.82) and `edge` (+0.79). Target independence and board independence
are different things: a backbone that is good at localised geometric structure
is good at all of it, magnitude and phase alike, even though those are different
quantities.

Two caveats. This is n=12, so the coefficients are wide. And the tier means are
a statement about *this corpus*: at nine backbones the high-level tier mean was
above the cross-tier line (+0.497 against +0.340), at thirteen boards it was
below, and at fourteen and fifteen it is marginally above again — moved each
time by which backbones and which boards are in the set, not by any board's
features changing. The two-cluster structure is what survives all three.
Reproduce or re-test with
[`scripts/analyse_board_correlates.py`](https://github.com/turhancan97/VisBench/blob/main/scripts/analyse_board_correlates.py)
`--section agreement`.

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

### Scene classification

`scene_classification` is the same linear probe on a different question: the
category of the *place* in the frame — its layout and context — rather than of
an object in it. A backbone can be strong at one and weak at the other, so it
is a **distinct probe with its own leaderboard board**, not a dataset flag on
`classification`. (It has to be: a board renders one comparability group, and
records on a second dataset under one task name would make the object board
unrenderable.)

[`examples/scene_classify.py`](https://github.com/turhancan97/VisBench/blob/main/examples/scene_classify.py)
runs it on any labelled folder; the canonical dataset is Places365-standard,
which ships in exactly the `train/<class>/` + `val/<class>/` layout:

```bash
python examples/scene_classify.py --data /path/to/places365_standard --limit 100
```

Places365 scenes overlap what ImageNet-supervised backbones already saw, so for
those the number is closer to in-distribution recall than transfer. The
twelve-backbone board is under [Measured on Places365](#measured-on-places365),
and it ranks backbones almost independently of the object one.

### Fine-grained recognition

`fine_grained_classification` is the same linear probe again, and the third
distinct question on that one path. Object classification asks whether a
representation separates *basic-level* categories — a bird from a car — which
is the level ImageNet-1k supervision optimises directly, and the level at which
the Imagenette board is saturated. This one asks whether it separates
*subordinate* categories inside one basic-level class: 200 species of bird that
share a body plan, a pose distribution and a background, and differ in the shape
of a beak or the colour of a wing bar. The information either survived the
encoder or it did not.

It is a distinct probe for the same reason `scene_classification` is — a board
renders one comparability group, so a second dataset under one task name would
make the object board unrenderable.

[`examples/fine_grained_classify.py`](https://github.com/turhancan97/VisBench/blob/main/examples/fine_grained_classify.py)
runs it on any labelled folder; the canonical dataset is CUB-200-2011, whose
official 5994/5794 split ships in exactly the `train/<class>/` + `val/<class>/`
layout:

```bash
python examples/fine_grained_classify.py --data /path/to/CUB-200/images_train_test
```

Stanford Cars and Stanford Dogs are the same shape and run the same probe under
a different dataset fingerprint — which puts them in a different comparability
group, so they cannot be quoted beside a CUB number.

**Two things to read beside any score here**, both measured rather than
assumed, and both stated on the board itself.

**The probe does not underfit**, which is worth saying because 200 classes over
~6k training images is exactly where you would expect it to. `train top1` is
**1.0000 on every backbone measured**. A linear map from 384–2048 dimensions to
200 classes has the capacity to separate 5994 points, so the schedule saturates
and the whole gap to the validation score is generalisation. A low number here
is a property of the representation.

**The in-distribution confound does not carry over from the object board.**
ImageNet-1k holds 59 bird classes, so the ImageNet-1k-supervised backbones were
expected to be flattered here the way `convnext_base` and `supervised_vitb16`
are flattered by Imagenette's ImageNet-1k wnids. They are not — see
[Measured on CUB-200-2011](#measured-on-cub-200-2011), where they place 8th and
10th of twelve. Basic-level supervision appears to discard exactly the
within-class variation this board asks about.

The twelve-backbone board is under
[Measured on CUB-200-2011](#measured-on-cub-200-2011).

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
| `dino_vitb16` | **0.9019** | **0.9004** | 0.0000 |
| `dinov2_vits14` | 0.8701 | 0.8687 | 0.0000 |
| `sam_vitb16` | 0.8695 | 0.8675 | 0.0000 |
| `dinov2_vitb14` | 0.8580 | 0.8575 | 0.0000 |
| `siglip_vitb16` | 0.8575 | 0.8552 | 0.0000 |
| `clip_vitb32` | 0.8465 | 0.8443 | 0.0000 |
| `resnet18` | 0.8317 | 0.8307 | 0.0000 |
| `clip_vitb16` | 0.8284 | 0.8266 | 0.0000 |
| `resnet50` | 0.8273 | 0.8282 | 0.0000 |
| `supervised_vitb16` | 0.8202 | 0.8188 | 0.0000 |
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

| backbone | `edge_correlation` | `rmse` | `mae` | `ceiling_edge_correlation` | `ceiling_mae` | `ceiling_rmse` |
| --- | --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | **0.4982** | **0.9150** | **0.4687** | 0.6106 | 0.4560 | 0.7906 |
| `dino_vitb16` | 0.4817 | 0.9150 | 0.4789 | 0.6106 | 0.4560 | 0.7906 |
| `sam_vitb16` | 0.4734 | 0.9286 | 0.4784 | 0.6106 | 0.4560 | 0.7906 |
| `clip_vitb16` | 0.4565 | 0.9340 | 0.4882 | 0.6106 | 0.4560 | 0.7906 |
| `dinov2_vits14` | 0.4558 | 0.9226 | 0.5028 | 0.6336 | 0.4418 | 0.7727 |
| `dinov2_vitb14` | 0.4481 | 0.9265 | 0.4972 | 0.6336 | 0.4418 | 0.7727 |
| `supervised_vitb16` | 0.4420 | 0.9366 | 0.5086 | 0.6106 | 0.4560 | 0.7906 |
| `clip_vitb32` | 0.3834 | 0.9656 | 0.5080 | 0.4977 | 0.5175 | 0.8630 |
| `siglip_vitb16` | 0.3639 | 0.9785 | 0.5169 | 0.6106 | 0.4560 | 0.7906 |
| `resnet50` | 0.3549 | 0.9770 | 0.5056 | 0.4977 | 0.5175 | 0.8630 |
| `convnext_base` | 0.3485 | 0.9671 | 0.5224 | 0.4977 | 0.5175 | 0.8630 |
| `resnet18` | 0.3430 | 0.9797 | 0.5153 | 0.4977 | 0.5175 | 0.8630 |

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

| backbone | `keypoint_correlation` | `mae` | `rmse` | `ceiling_keypoint_correlation` | `ceiling_mae` | `ceiling_rmse` |
| --- | --- | --- | --- | --- | --- | --- |
| `dino_vitb16` | **0.2850** | 1.0827 | **2.5143** | 0.6674 | 0.9238 | 1.9161 |
| `sam_vitb16` | 0.2696 | 1.1117 | 2.5589 | 0.6674 | 0.9238 | 1.9161 |
| `mae_vitb16` | 0.2626 | **1.0533** | 2.5342 | 0.6674 | 0.9238 | 1.9161 |
| `supervised_vitb16` | 0.2573 | 1.1242 | 2.5468 | 0.6674 | 0.9238 | 1.9161 |
| `dinov2_vits14` | 0.2356 | 1.1281 | 2.5472 | 0.6976 | 0.8873 | 1.8452 |
| `dinov2_vitb14` | 0.2248 | 1.1294 | 2.5541 | 0.6976 | 0.8873 | 1.8452 |
| `convnext_base` | 0.2187 | 1.1690 | 2.5633 | 0.4728 | 1.0934 | 2.2552 |
| `clip_vitb16` | 0.2175 | 1.1533 | 2.5770 | 0.6674 | 0.9238 | 1.9161 |
| `clip_vitb32` | 0.1933 | 1.1474 | 2.5891 | 0.4728 | 1.0934 | 2.2552 |
| `resnet50` | 0.1792 | 1.2374 | 2.6163 | 0.4728 | 1.0934 | 2.2552 |
| `resnet18` | 0.1659 | 1.2579 | 2.6282 | 0.4728 | 1.0934 | 2.2552 |
| `siglip_vitb16` | 0.1577 | 1.1789 | 2.6072 | 0.6674 | 0.9238 | 1.9161 |

Ordered by `keypoint_correlation`, which **disagrees with `mae`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>keypoints2d on taskonomy_keypoints2d/val, protocol=visbench_keypoint2d_regression, frozen [e647c722]</sub>
<!-- /visbench:board -->

<!-- visbench:board task=occlusion_edge metrics=occlusion_edge_correlation,mae,rmse heading=3 -->
### occlusion_edge

| backbone | `occlusion_edge_correlation` | `mae` | `rmse` | `ceiling_mae` | `ceiling_occlusion_edge_correlation` | `ceiling_rmse` |
| --- | --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | **0.3273** | **0.1904** | **0.4261** | 0.1555 | 0.5150 | 0.3797 |
| `dinov2_vitb14` | 0.3167 | 0.2061 | 0.4315 | 0.1512 | 0.5301 | 0.3755 |
| `dino_vitb16` | 0.2928 | 0.2025 | 0.4338 | 0.1555 | 0.5150 | 0.3797 |
| `dinov2_vits14` | 0.2924 | 0.2205 | 0.4373 | 0.1512 | 0.5301 | 0.3755 |
| `sam_vitb16` | 0.2680 | 0.2108 | 0.4391 | 0.1555 | 0.5150 | 0.3797 |
| `clip_vitb16` | 0.2558 | 0.2149 | 0.4415 | 0.1555 | 0.5150 | 0.3797 |
| `siglip_vitb16` | 0.2254 | 0.2205 | 0.4423 | 0.1555 | 0.5150 | 0.3797 |
| `clip_vitb32` | 0.2174 | 0.2203 | 0.4440 | 0.1811 | 0.4336 | 0.3994 |
| `supervised_vitb16` | 0.1996 | 0.2435 | 0.4540 | 0.1555 | 0.5150 | 0.3797 |
| `resnet50` | 0.1979 | 0.2294 | 0.4502 | 0.1811 | 0.4336 | 0.3994 |
| `resnet18` | 0.1745 | 0.2418 | 0.4578 | 0.1811 | 0.4336 | 0.3994 |
| `convnext_base` | 0.1741 | 0.2533 | 0.4704 | 0.1811 | 0.4336 | 0.3994 |

Ordered by `occlusion_edge_correlation`, which **disagrees with `mae`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

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

| backbone | `corner_correlation` | `mae` | `rmse` | `ceiling_corner_correlation` | `ceiling_mae` | `ceiling_rmse` |
| --- | --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | **0.6669** | **0.4187** | 0.6979 | 0.8053 | 0.3078 | 0.5258 |
| `dino_vitb16` | 0.6657 | 0.4268 | **0.6837** | 0.8053 | 0.3078 | 0.5258 |
| `dinov2_vitb14` | 0.6526 | 0.4402 | 0.6899 | 0.8316 | 0.2849 | 0.4940 |
| `dinov2_vits14` | 0.6512 | 0.4510 | 0.6919 | 0.8316 | 0.2849 | 0.4940 |
| `sam_vitb16` | 0.6454 | 0.4305 | 0.7198 | 0.8053 | 0.3078 | 0.5258 |
| `clip_vitb16` | 0.6227 | 0.4508 | 0.7229 | 0.8053 | 0.3078 | 0.5258 |
| `supervised_vitb16` | 0.6204 | 0.4710 | 0.7291 | 0.8053 | 0.3078 | 0.5258 |
| `siglip_vitb16` | 0.5383 | 0.4866 | 0.7846 | 0.8053 | 0.3078 | 0.5258 |
| `clip_vitb32` | 0.5367 | 0.4829 | 0.7825 | 0.6685 | 0.4150 | 0.6579 |
| `convnext_base` | 0.5129 | 0.4852 | 0.7833 | 0.6685 | 0.4150 | 0.6579 |
| `resnet18` | 0.5014 | 0.4706 | 0.8085 | 0.6685 | 0.4150 | 0.6579 |
| `resnet50` | 0.4923 | 0.4661 | 0.8033 | 0.6685 | 0.4150 | 0.6579 |

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

## Gradient orientation — a derived target that is a direction, not a magnitude

[`examples/orientation.py`](https://github.com/turhancan97/VisBench/blob/main/examples/orientation.py).
The fourth low-level probe and the second whose target is computed from the
frame — but the first whose target is a *direction*: the local orientation
structure runs, read as `2θ = atan2(2·Ixy, Ixx − Iyy)` from the same
Gaussian-windowed structure tensor whose smaller eigenvalue is the corner
response.

```bash
visbench run orientation --data /path/to/any/images --limit 600
```

The angle is defined **modulo π** — an edge and its reverse run the same way —
so the target is the unit vector `(cos 2θ, sin 2θ)`, which is single-valued
under that wrap, with its length set to the **coherence**
`(λ_max − λ_min) / (λ_max + λ_min)`. Loss and metric both weight by that length,
so a flat isotropic patch contributes ~0 rather than being masked out by a
threshold nobody chose. Only 1.4% of Taskonomy tiny val pixels fall below
coherence 0.1. The metric, `orientation_error`, is the coherence-weighted mean
angular error in degrees, halved into `[0, 90]` (45 is chance).

**It measures phase, which no other probe here does.** Per-image `|r|` with the
`edge_texture` target is 0.07 and with `corner` 0.08, where `corner` and `edge`
themselves sit at 0.53 — so an orientation score is close to independent
evidence about a backbone, unlike a corner score beside an edge score. Its board
uses the same pinned Taskonomy frames as `corner` and `edge`; `--orientation-sigma`
travels in `dataset_params` and splits the comparability groups on its own.

<!-- visbench:board task=orientation metrics=orientation_error,d1,d2 heading=3 -->
### orientation

| backbone | `orientation_error` | `d1` | `d2` | `ceiling_d1` | `ceiling_d2` | `ceiling_orientation_error` |
| --- | --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | **18.8206** | **0.5820** | **0.7268** | 0.7047 | 0.8450 | 12.1822 |
| `dino_vitb16` | 21.4352 | 0.5244 | 0.6850 | 0.7047 | 0.8450 | 12.1822 |
| `sam_vitb16` | 21.7203 | 0.5231 | 0.6811 | 0.7047 | 0.8450 | 12.1822 |
| `dinov2_vits14` | 22.1286 | 0.4962 | 0.6688 | 0.7321 | 0.8652 | 11.0211 |
| `supervised_vitb16` | 24.2851 | 0.4608 | 0.6355 | 0.7047 | 0.8450 | 12.1822 |
| `dinov2_vitb14` | 24.5740 | 0.4646 | 0.6312 | 0.7321 | 0.8652 | 11.0211 |
| `convnext_base` | 28.2284 | 0.4194 | 0.5780 | 0.5699 | 0.7348 | 18.5669 |
| `clip_vitb16` | 28.2988 | 0.4097 | 0.5740 | 0.7047 | 0.8450 | 12.1822 |
| `resnet18` | 28.9932 | 0.4074 | 0.5672 | 0.5699 | 0.7348 | 18.5669 |
| `clip_vitb32` | 29.9416 | 0.3859 | 0.5494 | 0.5699 | 0.7348 | 18.5669 |
| `resnet50` | 29.9725 | 0.3956 | 0.5526 | 0.5699 | 0.7348 | 18.5669 |
| `siglip_vitb16` | 31.1453 | 0.3759 | 0.5317 | 0.7047 | 0.8450 | 12.1822 |

Ordered by `orientation_error`, which **disagrees with `d1`, `d2`, `median`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>orientation on val/val, protocol=visbench_structure_tensor_orientation_regression, frozen [38bd953b]</sub>
<!-- /visbench:board -->

`orientation_error` is degrees, lower is better; chance is 45. The board spans
18.8° to 31.2° — every backbone is well clear of chance, and the ordering is
unlike any other low-level board: `mae_vitb16` leads (as it does across the
low-level tier), but the image-text ViTs `siglip_vitb16` and `clip_vitb32` are
*last*, where a semantic board puts them near the top, and DINOv2-S beats
DINOv2-B. See `CORPUS_FINDINGS.md` for what that does to the low-level tier.

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
| `supervised_vitb16` | 0.5791 | 0.5877 | 0.8681 | 0.6761 |
| `siglip_vitb16` | 0.5405 | 0.3210 | 0.8539 | 0.6511 |
| `dino_vitb16` | 0.5063 | 0.3221 | 0.8632 | 0.5964 |
| `convnext_base` | 0.4880 | 0.4596 | 0.8310 | 0.5902 |
| `resnet50` | 0.4574 | 0.5163 | 0.8322 | 0.5248 |
| `resnet18` | 0.4212 | 0.4497 | 0.8205 | 0.4915 |
| `mae_vitb16` | 0.3350 | 0.4555 | 0.8269 | 0.3757 |
| `sam_vitb16` | 0.3339 | 0.3905 | 0.8146 | 0.3825 |

Ordered by `miou`, which **disagrees with `mean_acc`, `miou_per_image`, `pixel_acc`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>semantic_segmentation on VOC2012/val, protocol=visbench_semantic_seg, frozen [e14b47db]</sub>
<!-- /visbench:board -->

**What this board ranks by is not what the other dense boards rank by, and
that is worth knowing before quoting it.** Correlate each board's ordering
against the backbones' feature-grid area, over the twelve backbones in the
corpus, and every other dense board comes out between +0.73 and +0.96 — a
finer grid is most of what a dense probe rewards. Semantic segmentation is
**+0.545**, and that much is carried by DINOv2, which has both the finest grid
and the largest pretraining corpus: without those two rows it is +0.212, and
with the pretraining data held fixed it is **0.000**.

The control is already in this page. `generic_segmentation` runs on the *same
1449 VOC images* at the same resolution with the same linear head and the same
schedule, and differs only in whether the target has 2 classes or 21 — it
ranks by grid at **+0.958**. Same pixels, same probe, opposite behaviour, so
this is a property of the target rather than the data or the protocol.

What it rewards instead is, weakly, pretraining breadth: corpus size is its
best single correlate at +0.615, the highest of the thirteen boards. But
within the six *identical* ViT-B/16 backbones the spread is 0.3207 mIoU with
only a +0.314 correlation, so most of the variance is not structural. **Treat
this board as evidence about representations, not about training objectives**
— step 10e's recipe control showed it cannot separate those at all. Reproduce
any of it with `scripts/analyse_board_correlates.py`; at twelve backbones the
coefficients have wide error bars.

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
| `sam_vitb16` | 0.9972 | **1.0000** |
| `supervised_vitb16` | 0.9972 | 0.9997 |
| `clip_vitb16` | 0.9954 | 0.9997 |
| `dinov2_vits14` | 0.9939 | 0.9997 |
| `siglip_vitb16` | 0.9936 | 0.9995 |
| `dino_vitb16` | 0.9931 | 0.9997 |
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
| `supervised_vitb16` | **0.9947** | 0.9977 | 0.9987 |
| `sam_vitb16` | 0.9912 | 0.9944 | **0.9992** |
| `convnext_base` | 0.9890 | **0.9987** | 0.9990 |
| `resnet50` | 0.9357 | 0.9901 | 0.9987 |
| `dino_vitb16` | 0.9192 | 0.9868 | 0.9972 |
| `dinov2_vitb14` | 0.9171 | 0.9954 | 0.9977 |
| `clip_vitb16` | 0.9102 | 0.9893 | 0.9975 |
| `dinov2_vits14` | 0.8893 | 0.9921 | 0.9972 |
| `clip_vitb32` | 0.8680 | 0.9806 | 0.9941 |
| `resnet18` | 0.8648 | 0.9725 | 0.9944 |
| `siglip_vitb16` | 0.8525 | 0.9799 | 0.9936 |
| `mae_vitb16` | 0.1883 | 0.6741 | 0.8892 |

Ordered by `mAP`, which **disagrees with `recall@1`, `recall@10`, `recall@5`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>retrieval on val/val, frozen [eb312a7b]</sub>
<!-- /visbench:board -->

<!-- visbench:board task=correspondence metrics=recall@5px,recall@10px,auc@5px heading=3 -->
### correspondence

| backbone | `recall@5px` | `recall@10px` | `auc@5px` | `ceiling_auc@5px` | `ceiling_recall@10px` | `ceiling_recall@5px` | `num_matches` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | **0.3577** | 0.7325 | **0.1370** | 0.1462 | 0.9212 | 0.3932 | 17,300 |
| `dino_vitb16` | 0.3566 | **0.7842** | 0.1351 | 0.1418 | 0.9389 | 0.3847 | 18,027 |
| `sam_vitb16` | 0.3298 | 0.7369 | 0.1238 | 0.1342 | 0.9316 | 0.3689 | 18,539 |
| `supervised_vitb16` | 0.3232 | 0.6567 | 0.1259 | 0.1468 | 0.9371 | 0.3928 | 8,796 |
| `dinov2_vits14` | 0.3049 | 0.6526 | 0.1152 | 0.1454 | 0.9329 | 0.4123 | 23,439 |
| `dinov2_vitb14` | 0.2816 | 0.6260 | 0.1055 | 0.1389 | 0.9264 | 0.4005 | 27,590 |
| `clip_vitb16` | 0.2689 | 0.5725 | 0.1080 | 0.1333 | 0.9159 | 0.3519 | 12,798 |
| `siglip_vitb16` | 0.1461 | 0.3463 | 0.0560 | 0.1073 | 0.8715 | 0.3041 | 13,504 |
| `resnet18` | 0.0973 | 0.3256 | 0.0335 | 0.0350 | 0.3653 | 0.1028 | 4,911 |
| `clip_vitb32` | 0.0897 | 0.2951 | 0.0321 | 0.0352 | 0.3633 | 0.1002 | 4,283 |
| `resnet50` | 0.0887 | 0.3003 | 0.0299 | 0.0350 | 0.3595 | 0.1038 | 4,373 |
| `convnext_base` | 0.0824 | 0.2950 | 0.0280 | 0.0320 | 0.3575 | 0.0940 | 5,413 |

Ordered by `recall@5px`, which **disagrees with `auc@10px`, `auc@1px`, `auc@2px`, `auc@5px`, `recall@10px`, `recall@1px`, `recall@2px`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

> **Read this first.** Thresholds are in **pixels**, which is the only unit two backbones can be compared in — a patch width is 14px on DINOv2/14 and 32px on a ResNet, so scoring in patch widths asks each backbone a different question. Read `ceiling_` beside every score: a 7x7 grid cannot place a match within 5px more than ~10% of the time whatever its features are, so part of this ordering is resolution rather than quality. `num_matches` is the denominator each backbone's own ratio test left, and it varies by more than 5x.

<sub>correspondence on val/val, frozen [7db23175]</sub>
<!-- /visbench:board -->

The patch grid at 224px is 16x16 for DINOv2 (patch 14), 14x14 for CLIP-B/16,
MAE-B/16 and SigLIP-B/16, and 7x7 for CLIP-B/32, ConvNeXt-B and both ResNets.
Hold that beside the correspondence board: it is what `num_matches` tracks, and
it moves the score without saying anything about feature quality.

**Read the supervised rows with care.** Imagenette's ten classes are ImageNet-1k
wnids, and `resnet50.a1_in1k`, `convnext_base.fb_in1k` and
`supervised_vitb16` (`vit_base_patch16_224.augreg_in1k`) were all trained on
ImageNet-1k with labels — they have seen these exact categories, while DINOv2
and MAE are self-supervised and CLIP and SigLIP are image-text. Their semantic
scores are close to in-distribution recall, not a transfer result: ConvNeXt tops
both the classification board (0.9997) and the retrieval board (0.9890) and then
places seventh or lower on every dense geometric one. This says more about the
dataset than the backbone; a benchmark comparing supervised against
self-supervised features needs data the supervised model has not been trained
on.

**The one pair that isolates the objective is `supervised_vitb16` against
`mae_vitb16`.** They are the same timm architecture pretrained on the same
ImageNet-1k, differing only in whether the objective was labels or masked pixel
reconstruction — every other pair of backbones in this corpus varies at least
two things at once, so this is the only place a gap can be attributed. Over the
thirteen boards the winner tracks the tier, with exactly one crossing:

| tier | boards | split |
| --- | --- | --- |
| high-level (4) | classification 0.9972 v 0.9582, retrieval 0.9947 v 0.1883, semantic seg 0.5791 v 0.3350, detection 0.1669 v 0.1296 | **supervised wins all four** |
| mid-level (6) | depth, surface normals (27.52° v 36.72° mean), correspondence, occlusion edges, generic seg — and similarity 0.8202 v 0.6897 | **MAE wins five, supervised wins similarity** |
| low-level (3) | edge, keypoints2d, corner | **MAE wins all three** |

Two things not to round off. **The retrieval gap is the largest on the board and
is not a clean transfer result**: it is the in-distribution caveat above at its
strongest, since this backbone was trained with labels on the categories
Imagenette is drawn from.

And **mid-level similarity is the one board that crosses the tier line**, which
is worth stating rather than filing under "high-level". A hypothesis, untested:
NIGHTS' images are Stable Diffusion generations prompted with categories from
ImageNet, CIFAR-10/100, Flowers-102, Food-101 and SUN397, so a 2AFC over them
may reward category familiarity rather than the perceptual and geometric
resemblance the probe is meant to isolate — the in-distribution caveat again,
in a place it is easy to miss.

The low-level tier and the geometric half of the mid-level one carry no such
caveat, and are the boards worth quoting.

**MAE is the sharpest tier separation this corpus has produced, and it is worth
reading before trusting any single board.** `mae_vitb16` is **first on five of
the thirteen** — edge, corner, correspondence, occlusion edges and surface
normals — and **last or next-to-last on the four semantic ones**: twelfth on
classification (0.9582), twelfth on retrieval (0.1883), twelfth on mid-level
similarity (0.6897) and eleventh on semantic segmentation (0.3350). No other
backbone here is simultaneously best and worst.

Two of those figures moved when the corpus widened from nine backbones to
twelve, and how they moved is itself the lesson. MAE led **2D keypoints** and
is third now, behind `dino_vitb16` (0.2850) and `sam_vitb16` (0.2696), so its
count of firsts is five rather than six. And it is no longer last on semantic
segmentation, because `sam_vitb16` lands beneath it at 0.3339. **A count over a
corpus is a fact about that corpus, not about the backbone** — both numbers
moved without MAE's features changing at all.

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

## Measured on Places365

`scene_classification`, on Places365-standard: the full official validation
split (36,500 images, 100 per class across 365 scene categories) scored, with
100 training images per class (`--limit 100`). Linear probe on pooled features,
the same path and schedule as object `classification`.

<!-- visbench:board task=scene_classification metrics=top1,top5 heading=3 -->
### scene_classification

| backbone | `top1` | `top5` |
| --- | --- | --- |
| `siglip_vitb16` | **0.4035** | **0.6930** |
| `clip_vitb16` | 0.3934 | 0.6760 |
| `clip_vitb32` | 0.3890 | 0.6769 |
| `dinov2_vitb14` | 0.3865 | 0.6562 |
| `resnet50` | 0.3575 | 0.6525 |
| `dinov2_vits14` | 0.3529 | 0.6430 |
| `sam_vitb16` | 0.3430 | 0.6318 |
| `dino_vitb16` | 0.3410 | 0.6320 |
| `convnext_base` | 0.3356 | 0.6176 |
| `mae_vitb16` | 0.3111 | 0.6046 |
| `supervised_vitb16` | 0.3068 | 0.5855 |
| `resnet18` | 0.2712 | 0.5535 |

Ordered by `top1`, which **disagrees with `top5`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

> **Read this first.** This is *scene* category, not object category — a distinct question from the `classification` board, and a backbone's rank can move between the two. Places365 scenes overlap what ImageNet-supervised backbones already saw, so for those the number is closer to in-distribution recall than transfer.

<sub>scene_classification on val/val, protocol=visbench_scene_linear_probe, frozen [9f6f94e8]</sub>
<!-- /visbench:board -->

**This board ranks backbones almost independently of the object board.**
Spearman correlation between the two orderings is **+0.16** across the twelve
backbones. The image-text models take the top three places (`siglip_vitb16`
0.4035, both CLIPs behind it), where on Imagenette they sit mid-pack. The two
ImageNet-1k **supervised** backbones fall the hardest: `convnext_base` goes from
first on objects to ninth here, `supervised_vitb16` from fifth to eleventh —
Imagenette's classes are ImageNet-1k wnids, so their object numbers are close to
in-distribution recall, and Places365 is where that advantage does not apply.
`mae_vitb16` is tenth, consistent with its last-or-near-last placing on every
other semantic board.

The object board is also **saturated** — eleven of twelve backbones score above
0.988 top-1, a spread of 0.04 — where this one spans 0.271 to 0.404, a spread of
0.13. A saturated board cannot rank; this is the reason scene classification is
a separate probe rather than a note under the object one.

## Measured on CUB-200-2011

`fine_grained_classification`, on the official CUB split: 5,994 training and
5,794 validation images across 200 bird species, the whole split with no cap.
Linear probe on pooled features, the same path and schedule as object
`classification`.

<!-- visbench:board task=fine_grained_classification metrics=top1,top5 heading=3 -->
### fine_grained_classification

| backbone | `top1` | `top5` |
| --- | --- | --- |
| `dinov2_vitb14` | **0.8683** | **0.9757** |
| `dinov2_vits14` | 0.8652 | 0.9707 |
| `clip_vitb16` | 0.8045 | 0.9591 |
| `sam_vitb16` | 0.7927 | 0.9486 |
| `siglip_vitb16` | 0.7839 | 0.9427 |
| `dino_vitb16` | 0.7520 | 0.9253 |
| `clip_vitb32` | 0.7344 | 0.9289 |
| `convnext_base` | 0.7311 | 0.9210 |
| `resnet50` | 0.6943 | 0.9137 |
| `supervised_vitb16` | 0.6590 | 0.8873 |
| `resnet18` | 0.6177 | 0.8693 |
| `mae_vitb16` | 0.4696 | 0.7686 |

Ordered by `top1`, which **disagrees with `top5`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

> **Read this first.** This is *subordinate* category — 200 bird species that share a body plan — not the basic-level question the `classification` board asks, and a backbone's rank moves a long way between the two. **The in-distribution confound that shapes the object board does not carry over here**, which was measured rather than assumed: ImageNet-1k holds 59 bird classes, so the four ImageNet-1k-*supervised* backbones were expected to be flattered, and instead they take places 8, 9, 10 and 11 of twelve — `convnext_base`, `resnet50`, `supervised_vitb16`, `resnet18`, above only `mae_vitb16`. The controlled comparison says the same thing: among the four ViT-B/16 models, the supervised one is second-to-last, behind both `sam_vitb16` and `dino_vitb16`. Basic-level supervision appears to discard the within-class variation this board asks about. The probe also does **not** underfit despite 200 classes over ~6k training images — `train_top1` is 1.0000 on all six backbones it was measured on directly, including the board's last place — so the spread is generalisation and a low score is a property of the representation.

<sub>fine_grained_classification on val/val, protocol=visbench_fine_grained_linear_probe, frozen [a10a2fcf]</sub>
<!-- /visbench:board -->

Every one of these examples has a `visbench run` equivalent — see
[the command-line section of the README](https://github.com/turhancan97/VisBench/blob/main/README.md#the-command-line). They stay because an example is readable
top to bottom and a subcommand is not: when you want to know *how* a probe is
wired up, the script is the answer.
