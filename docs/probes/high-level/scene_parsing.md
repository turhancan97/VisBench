# `scene_parsing`

**Every pixel of an indoor scene, in forty classes — and the counterpart to
`semantic_segmentation` that asks about *stuff* rather than things.**

VOC's twenty classes are objects on a background. NYU40's are a room: wall,
floor, ceiling, and the furniture in it. Most of the pixels in this target
belong to categories that have no instances and no boundaries in the object
sense, which is what the literature calls scene parsing rather than object
segmentation.

```{figure} /_static/gallery/scene_parsing.png
:alt: scene_parsing — an image and its label map

What `visbench show scene_parsing` draws. {doc}`How to read it </guides/visualising>`.

**These are Open Images object labels, not NYU40.** The board reads NYUv2,
which cannot be redistributed, and unlike `depth` this probe has no published
head to predict from — so the figure demonstrates the renderer on labels that
*can* ship rather than inventing a forty-class indoor target. The panel format
is exactly what a real run draws; the classes are not.
```

[`examples/segment_semantic.py`](https://github.com/turhancan97/VisBench/blob/main/examples/segment_semantic.py)
runs the same implementation with `--num-classes`.

```bash
visbench run scene_parsing --data /path/to/nyuv2_new
```

## Why a distinct probe and not a flag

`board_for` renders exactly one table per task and **refuses a task with more
than one comparability group**, and `comparability_key` groups by dataset name
and fingerprint. An NYUv2 record under `task="semantic_segmentation"` would not
merge with the VOC board — it would make that board **unrenderable**. A separate
name gives this its own board, its own CLI row and its own leaderboard group,
the way `scene_classification` is separate from `classification`.

Everything mechanical is inherited unchanged: the linear head over frozen
features, the AdamW schedule, cross-entropy with `ignore_index`, and the two
mIoUs reported under distinct names. Only the identity of the number changes,
and `protocol` says so — `visbench_scene_parsing`, not `visbench_semantic_seg`.

## The label convention, measured rather than assumed

The maps are mode `L` with values **0–39 and 255**, and which of those is void
decides whether the probe trains on anything. Measured over all 1,449 maps:

| | share of pixels | at the border | in the interior |
| --- | --- | --- | --- |
| value 255 | 17.4% | **88.5%** | 13.7% |
| value 0 | 21.4% | 2.3% | **28.6%** |

**255 is void** — that border concentration is the depth-projection margin, and
a 255 pixel is 2.5× more likely than average to have depth exactly 0. **0 is a
real class**, `wall`, which dominates interiors and is nearly absent at the
frame edge. So this is VOC's shape exactly: contiguous zero-indexed classes plus
a 255 void, handled by `load_label_map` and `--ignore-index 255` with no new
loader and **no fifth validity convention**.

Getting that backwards would not raise. Treating 0 as unlabelled would discard a
fifth of every image and train the probe never to answer `wall`; leaving 255 as
a class would ask a forty-class head to predict index 255.

## It reads the frames `depth` and `surface_normal` already read

The canonical 795/654 labelled split ships as stem-matched folders, so
`images/` is shared with those two boards and only `segmentation_nyu40/` is
new. Three probes over **identical pixels** asking geometry, geometry and
semantics — the NYUv2 counterpart of the VOC trio, and the reason a cluster
comparison here is about the *question* rather than about the data. The split
control showed that cluster membership is partly a property of a board's split;
this is the other side of that experiment.

## Reading its number

**Not comparable with the VOC board as a number.** Forty classes on 795 training
images against twenty-one on 1,464: the absolute mIoU is lower by construction.
The two are comparable as *orderings*, which is what a board is for.

**Two mIoUs, and they disagree by more here than on VOC.** Dataset-level (one
confusion matrix over the split) is what the segmentation literature defines and
the only one comparable with published numbers; per-image-then-averaged is this
codebase's rule everywhere else. Both are reported under distinct names, and the
board is ordered by the first.

## Its board

<!-- visbench:board task=scene_parsing metrics=miou,miou_per_image,pixel_acc heading=3 -->
### scene_parsing

| backbone | `miou` | `miou_per_image` | `pixel_acc` |
| --- | --- | --- | --- |
| `dinov2_vitb14` | **0.4959** | **0.3362** | **0.7344** |
| `dinov2_vits14` | 0.4519 | 0.3014 | 0.7070 |
| `clip_vitb16` | 0.3490 | 0.2547 | 0.6458 |
| `siglip_vitb16` | 0.3241 | 0.1757 | 0.5971 |
| `dino_vitb8` | 0.3206 | 0.1542 | 0.6235 |
| `dino_vitb16` | 0.3105 | 0.1717 | 0.6118 |
| `clip_vitb32` | 0.2918 | 0.2278 | 0.5969 |
| `convnext_base` | 0.2894 | 0.1962 | 0.5781 |
| `supervised_vitb16` | 0.2817 | 0.2075 | 0.5810 |
| `sam_vitb16` | 0.1968 | 0.1790 | 0.5590 |
| `resnet50` | 0.1812 | 0.1728 | 0.5274 |
| `mae_vitb16` | 0.1484 | 0.1790 | 0.5457 |
| `resnet18` | 0.1419 | 0.1464 | 0.4924 |

Ordered by `miou`, which **disagrees with `mean_acc`, `miou_per_image`, `pixel_acc`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

> **Read this first.** **Forty classes on 795 training images**, where the VOC board has twenty-one on 1,464 — so the absolute mIoU here is lower by construction and the two boards are not comparable as numbers, only as orderings. Most of these pixels are *stuff* (wall, floor, ceiling) rather than things, which is the question this probe asks that `semantic_segmentation` does not. It reads the frames `depth` and `surface_normal` already read, so a cluster comparison against those two is about the question rather than about the data.

<sub>scene_parsing on test/test, protocol=visbench_scene_parsing, frozen [73d37f62]</sub>
<!-- /visbench:board -->

**It ranks with the semantic boards, not with the ones it shares images with.**
Mean Spearman over the same thirteen backbones is **+0.553** against the seven
high-level boards, +0.414 against mid-level and **+0.136** against the four
low-level ones. Its strongest partners are `fine_grained_classification`
(**+0.868**), `semantic_segmentation` (**+0.852**) and `detection` (+0.791);
its weakest are `orientation` (−0.022), `keypoints2d` (−0.027) and `retrieval`
(−0.038).

That is worth sitting with, because this probe reads the **identical frames**
`depth` and `surface_normal` read — and correlates with those two at +0.533 and
+0.423, against +0.852 with a VOC board that shares no image with it. On this
pair of boards the *question* dominates the *data*. It is the complement of
{doc}`the split control </guides/reading-a-board>` rather than a contradiction:
that control showed a board's cluster can move when its split moves; this shows
the split alone does not decide it.

**`mae_vitb16` is twelfth of thirteen here** and first on five other boards —
and its `train_loss` is 1.6169 against `dinov2_vitb14`'s 0.7238, so that is the
representation rather than an unconverged fit. Checking the fit before reading a
low score is the rule the `training` field exists for.

