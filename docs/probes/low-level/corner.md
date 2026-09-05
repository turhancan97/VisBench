# `corner`

**Shi-Tomasi cornerness, computed from the frame rather than read from disk.**

The first probe whose target needs no dataset — which makes it the cheapest
kind to add and the easiest to fool yourself with. Three things had to be
measured before it was worth shipping, none of which a probe run would have
revealed on its own.

```{figure} /_static/gallery/corner.png
:alt: corner — image, target and prediction

What `visbench show corner` draws. {doc}`How to read it </guides/visualising>`.
```

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

## Its board

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

## Run it

[`examples/corners.py`](https://github.com/turhancan97/VisBench/blob/main/examples/corners.py) is the whole path, end to end, on a real backbone.

```bash
python examples/corners.py --data /path/to/dataset
```
