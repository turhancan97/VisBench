# `vehicle_classification`

**Subordinate category again, on a manufactured object — which car model, not which car.**

The fourth distinct question asked by one linear-probe implementation, and the
one that shows *granularity* is not what organises this family.

```{figure} /_static/gallery/vehicle_classification.png
:alt: vehicle_classification — a contact sheet of frames and their labels

What `visbench show vehicle_classification` draws. The photographs are Open
Images frames, as every figure in this gallery is, so the labels shown are
theirs rather than car models — the sheet illustrates the *viewer*, not the
board's data. {doc}`How to read it </guides/visualising>`.
```

`vehicle_classification` is the same linear probe as
{doc}`classification <classification>`,
{doc}`scene_classification <scene_classification>` and
{doc}`fine_grained_classification <fine_grained_classification>`: one linear
layer on pooled features, AdamW, top-1/top-5. It sits at the same *granularity*
as the bird board — subordinate categories inside one basic-level class — and
differs in what kind of object carries the evidence. Two bird species share a
body plan and differ in the colour of a wing bar: texture, spread over the
animal. Two car models share a silhouette and differ in badge, grille and lamp
geometry, at whatever scale the photograph happens to put them.

It is a distinct probe for the same reason `scene_classification` is: a Cars
record filed under `fine_grained_classification` would not join the CUB board,
it would make that board *unrenderable*, because `comparability_key` groups by
dataset and fingerprint and `board_for` refuses a task with two groups.

## The split is VisBench's own

**Numbers on this board are not comparable with published Stanford Cars
results**, and that is a property of the data available here rather than a
choice. The copy on this machine is not the official 8,144 / 8,041 split:

| | raw copy | staged |
| --- | --- | --- |
| train | 8,148 | **8,125** |
| val | 8,041 | **8,026** |

Eleven train images are *the same photograph filed under two class
directories*, and seven more pairs are in test — contradictory supervision on
one side and unanswerable items on the other, since whichever label a model
predicts, one copy scores wrong. One train image is a blank white placeholder.
The duplicate pairs sit on adjacent class indices (13/14, 124/125, 176/177),
which looks like a boundary artefact in whoever built the copy rather than
anything in the benchmark. `train.txt` and `test.txt` index a *different*
directory again — `train_cars_augmented`, 195,456 images — so they are not split
lists for this data.

`scripts/stage_cars_split.py` therefore pins a cleaned split as symlinks, on one
rule: **no image appears twice, under any label.** A cross-class duplicate loses
*both* copies, because there is no way to tell which label is right and keeping
either would be a guess; a within-class duplicate keeps the first. Every
exclusion is named with its reason in `data/cars_split_manifest.json`, so the
split is reproducible by anyone holding the same raw copy, and
`tests/scripts/test_stage_cars_split.py` recomputes the exclusions from the
dataset rather than trusting the file.

This is the same constraint `corner` carries: a probe whose data VisBench
defines has no board until the data is pinned.

## What the board says

<!-- visbench:board task=vehicle_classification metrics=top1,top5 heading=3 -->
### vehicle_classification

| backbone | `top1` | `top5` |
| --- | --- | --- |
| `siglip_vitb16` | **0.8603** | **0.9687** |
| `clip_vitb16` | 0.8074 | 0.9660 |
| `dinov2_vitb14` | 0.7953 | 0.9509 |
| `dinov2_vits14` | 0.7614 | 0.9370 |
| `clip_vitb32` | 0.7327 | 0.9335 |
| `sam_vitb16` | 0.6711 | 0.8831 |
| `dino_vitb8` | 0.6480 | 0.8560 |
| `resnet50` | 0.6221 | 0.8568 |
| `dino_vitb16` | 0.6131 | 0.8501 |
| `convnext_base` | 0.6005 | 0.8458 |
| `resnet18` | 0.5199 | 0.7815 |
| `supervised_vitb16` | 0.4578 | 0.7415 |
| `mae_vitb16` | 0.4353 | 0.7134 |

Ordered by `top1`, which **disagrees with `top5`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>vehicle_classification on val/val, protocol=visbench_vehicle_linear_probe, frozen [47c01df8]</sub>
<!-- /visbench:board -->

**`siglip_vitb16` leads at 0.8603**, where it is fifth on CUB. Web image-text
pretraining is ahead of everything else here, including both DINOv2 rows —
plausible for categories whose names saturate web alt-text, and the sharpest
single statement this board makes.

**It ranks differently from the bird board**, which is what earns it a place at
all: Spearman **+0.863** against `fine_grained_classification`, where a dozen
existing board pairs in this corpus are *more* correlated than that. Had it
merely reproduced CUB's ordering it should not have shipped, on the standard
relative depth was rejected against.

**Its closest neighbour is not the other subordinate board.** The strongest
partner is `scene_classification` at **+0.901**, ahead of CUB's +0.863 — so what
these boards share is not granularity, which is the property this probe was
built to isolate. Mean rho is +0.525 against the high-level tier, +0.149 against
mid-level and **−0.190** against low-level. See
{doc}`reading a board </guides/reading-a-board>` before quoting any of it.

**`train_top1` reaches 1.0000 on seven of thirteen rows and 0.985 or better on
the rest**, so the gap to the validation score is generalisation rather than an
unconverged probe. CUB reaches a flat 1.0000 where this board does not, which is
the one place the two differ mechanically — read `train_top1` before quoting a
low row here.

One caveat travels with every number: ImageNet-1k holds several car classes
("sports car", "convertible", "limousine") but none at *model* granularity, so
the in-distribution-recall confound `CORPUS_FINDINGS.md` records for the
Imagenette board is weaker here than on CUB rather than absent.
`supervised_vitb16` sits twelfth of thirteen, which is consistent with that and
does not on its own separate "the confound is weak" from "this backbone is weak
at this task".

## Running it

```bash
python scripts/stage_cars_split.py          # pins data/cars_split
visbench run vehicle_classification --data data/cars_split \
    --split val --train-split train --backbone dinov2_vits14
```

`examples/vehicle_classify.py` is the same thing in Python. A staged symlink
resolves to the same path as the original, so features extracted from the raw
copy are reused rather than recomputed.
