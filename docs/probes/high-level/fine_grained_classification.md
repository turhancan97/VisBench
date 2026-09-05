# `fine_grained_classification`

**Subordinate category — which species of bird, not whether it is a bird.**

The third distinct question asked by one linear-probe implementation, and the
one that replicated {doc}`scene_classification <scene_classification>`'s
surprise rather than merely adding to it.

```{figure} /_static/gallery/fine_grained_classification.png
:alt: fine_grained_classification — image, target and prediction

What `visbench show fine_grained_classification` draws. {doc}`How to read it </guides/visualising>`.
```

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
the board below, where they place 8th and
10th of twelve. Basic-level supervision appears to discard exactly the
within-class variation this board asks about.

The twelve-backbone board is under
the board below.

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

`fine_grained_classification`, on the official CUB split: 5,994 training and
5,794 validation images across 200 bird species, the whole split with no cap.
Linear probe on pooled features, the same path and schedule as object
`classification`.

## Its board

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

## Run it

[`examples/fine_grained_classify.py`](https://github.com/turhancan97/VisBench/blob/main/examples/fine_grained_classify.py) is the whole path, end to end, on a real backbone.

```bash
python examples/fine_grained_classify.py --data /path/to/dataset
```
