# `scene_classification`

**The category of the *place* in the frame, not of an object in it.**

Mechanically identical to {doc}`classification <classification>`: the same
linear probe on the same pooled features. It is a **distinct probe** because a
backbone's rank moves between the two — and it does, almost independently
(Spearman +0.16).

```{figure} /_static/gallery/scene_classification.png
:alt: scene_classification — image, target and prediction

What `visbench show scene_classification` draws. {doc}`How to read it </guides/visualising>`.
```

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
those the number is closer to in-distribution recall than transfer. Its twelve-
backbone board is below, and it ranks backbones almost
independently of the object one.

`scene_classification`, on Places365-standard: the full official validation
split (36,500 images, 100 per class across 365 scene categories) scored, with
100 training images per class (`--limit 100`). Linear probe on pooled features,
the same path and schedule as object `classification`.

## Its board

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

## Run it

[`examples/scene_classify.py`](https://github.com/turhancan97/VisBench/blob/main/examples/scene_classify.py) is the whole path, end to end, on a real backbone.

```bash
python examples/scene_classify.py --data /path/to/dataset
```
