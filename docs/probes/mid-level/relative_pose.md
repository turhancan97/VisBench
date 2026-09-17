# `relative_pose`

**How the camera moved between two views of one rigid scene — and the only
board here whose head is not a linear map.**

Two photographs of the same object from different viewpoints, and the probe
regresses the rigid transform between the cameras: a wxyz quaternion and a
translation, seven numbers, under probe3d's pairwise protocol. It reads
**pooled** features, so there is no feature grid, no streaming and no dense
cache — the cheapest geometry probe here to run, once its frames are decoded.

```{figure} /_static/gallery/relative_pose.png
:alt: relative_pose — an anchor frame, a partner frame, and how far the camera turned

What `visbench show relative_pose` draws. {doc}`How to read it </guides/visualising>`.

**These pairs are made rather than captured.** Rotating a frame about its centre
*is* a camera rotation about the optical axis for a pinhole camera, so the
figure's ground truth is exact — but it is a degenerate pose, one axis and no
translation, where NAVI's are general. The gallery photographs are Open Images
frames under CC BY 2.0; NAVI's own frames are not redistributed here.
```

[`examples/pose.py`](https://github.com/turhancan97/VisBench/blob/main/examples/pose.py).

```bash
visbench run relative_pose --data /path/to/navi_v1
```

## Read every row against the floor, never against zero

Pairs are drawn within 120 degrees of rotation, so their median relative
rotation is about 65 degrees and **predicting the training set's mean pose — no
features at all — already scores about 67 degrees**. A backbone landing there is
*at chance*, which is a different claim about a representation from "weak", and
quoting its 66 degrees as a result states the wrong one.

Every record carries that number as `floor_rotation_error_deg`, beside the
score, on the same convention `ceiling_` uses for the dense probes. **Never rank
on it**: it describes the draw rather than a backbone, and two boards with
different floors cannot be compared by subtracting them.

That floor is also what makes this probe's own gauntlet run different from the
dense ones. There is no oracle here — pooling a target to the feature grid is
the question a *per-patch* head raises, and this head reads two vectors — so
what had to be established before building it was the opposite bound: that the
target is not something a constant already predicts.

## Two things that are protocol, not configuration

**The training pair count.** Each anchor can be paired with many eligible
partners, and rotation error keeps falling as more are drawn — from 44.96
degrees at one partner per anchor to 24.29 at eight and 20.47 at sixteen, on the
same 1,740 validation pairs. Nothing has converged at any count measured, so the
board **pins** eight partners (50,519 training pairs) and records the count in
`task_params`, which puts it in the comparability key. Two pose numbers drawn
from different pair sets are not two measurements of the same thing.

Eight rather than sixteen because separation, not absolute error, is what a
board is for: at eight the adjacent gaps are 7.12 / 5.98 / 4.95 degrees against
a widest seed range of 0.87, and at sixteen they are 5.03 / 5.96 / 5.88 against
2.30. More data narrowed the gaps *and* widened the seed spread.

**The head.** Every other board here is quoted with the least expressive head
that can express the task, and in practice that has always come out a linear
map — an affine layer, a `LinearHead`, or the 1x1 convolutions `DetectionHead`
and `InstanceHead` are built from. A single affine map cannot express this one.
Measured on the same 50,519 pairs:

| head | rotation error | vs floor | spread | `train_loss` |
| --- | --- | --- | --- | --- |
| probe3d's MLP (512/256/128) | 22.70–43.14 | +23.7 to +44.2 | 20.44 | 0.0020–0.0057 |
| one affine map | 62.94–65.99 | **+0.85 to +3.91** | 3.06 | 0.0680–0.0729 |

Measured over all thirteen backbones, on the same pairs and the same features.
**A linear head is at or near chance here**: it clears the no-feature floor by
under four degrees for every backbone, and `resnet50` at +0.85 is
indistinguishable from predicting a constant. It **underfits** — a flat training
loss 25x the MLP's — and its residual ordering is not the MLP's (Spearman
+0.681, with `mae_vitb16` third rather than first). Against this board's ~1
degree of reproducibility noise, a linear board would separate its rows by about
three times the noise where the MLP board separates them by twenty.

So this board departs from `hidden_dim=0` deliberately, `task_params["head"]`
records which head produced a number, and the linear run is committed beside the
board as
[a control](https://github.com/turhancan97/VisBench/blob/main/results/controls/README.md)
rather than described — the honest form of "we used a bigger head" is a
measurement of what the smaller one did.

**What this costs the reading.** A gap between two rows here is less purely a
gap between two representations than elsewhere in this corpus: a deeper head can
compensate for a weaker feature vector. That is the DPT control's lesson applied
to a board that ships, and it is why the linear control is published rather than
described.

## The data

NAVI (Jampani et al., NeurIPS 2023), multi-view captures of rigid objects with a
camera pose per frame. `NaviPoseDataset` presents the **unique frames** its
pairs refer to and puts the pairing in `labels()` as indices into itself, which
is the move `TwoAFCDataset` makes for triplets: at eight partners the split is
50,519 pairs over 8,217 frames, so pairing by presentation would extract every
frame eight times to hold one copy of it.

Two details that are silent when wrong. **Translation is millimetres on disk**
and becomes metres at the loader; omitting that does not read as a units bug but
as every backbone scoring worse than a constant, because an MSE loss over
`[quat, trans]` then optimises a translation four orders of magnitude larger
than the quaternion. And **327 of the 8,217 frames carry a half-turn EXIF tag
whose camera pose describes the turned image**, so they are loaded with that tag
applied — reading them as stored supervises 4.0% of the release against its own
negation.

**A pose board is decode-bound.** NAVI ships 12-megapixel JPEGs and extraction
runs at roughly 3 frames per second single-threaded, so the first run of a
backbone is about an hour and later ones are minutes.

## Its board

<!-- visbench:board task=relative_pose metrics=rotation_error_deg,rotation_median_deg,rotation_acc@30deg heading=3 -->
### relative_pose

| backbone | `rotation_error_deg` | `rotation_median_deg` | `rotation_acc@30deg` | `floor_rotation_acc@30deg` | `floor_rotation_error_deg` | `floor_rotation_median_deg` |
| --- | --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | **22.6984** | **14.9383** | **0.8046** | 0.1563 | 66.8464 | 65.6347 |
| `dinov2_vits14` | 25.3698 | 16.5657 | 0.7690 | 0.1563 | 66.8464 | 65.6347 |
| `dino_vitb8` | 27.0987 | 16.5571 | 0.7437 | 0.1563 | 66.8464 | 65.6347 |
| `dino_vitb16` | 29.0404 | 18.7155 | 0.7023 | 0.1563 | 66.8464 | 65.6347 |
| `dinov2_vitb14` | 29.0808 | 18.0390 | 0.7115 | 0.1563 | 66.8464 | 65.6347 |
| `sam_vitb16` | 35.0188 | 22.6655 | 0.6138 | 0.1563 | 66.8464 | 65.6347 |
| `resnet50` | 37.0125 | 24.6576 | 0.5845 | 0.1563 | 66.8464 | 65.6347 |
| `convnext_base` | 37.6031 | 25.0400 | 0.5770 | 0.1563 | 66.8464 | 65.6347 |
| `siglip_vitb16` | 38.2885 | 24.6272 | 0.5776 | 0.1563 | 66.8464 | 65.6347 |
| `clip_vitb16` | 40.5643 | 26.0582 | 0.5621 | 0.1563 | 66.8464 | 65.6347 |
| `supervised_vitb16` | 41.4298 | 29.3044 | 0.5098 | 0.1563 | 66.8464 | 65.6347 |
| `resnet18` | 43.0147 | 28.5317 | 0.5149 | 0.1563 | 66.8464 | 65.6347 |
| `clip_vitb32` | 43.1388 | 29.3872 | 0.5115 | 0.1563 | 66.8464 | 65.6347 |

Ordered by `rotation_error_deg`, which **disagrees with `rotation_acc@15deg`, `rotation_acc@30deg`, `rotation_median_deg`, `translation_error`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

> **Read this first.** **Read every row against `floor_rotation_error_deg`, never against zero.** Pairs are drawn within 120 degrees of rotation, so predicting the training set's mean pose — no features at all — already scores about 67 degrees, and a backbone landing there is *at chance* rather than weak. This is also the one board here whose head is **not a linear map**: probe3d's MLP, because a single affine layer underfits the task badly enough to nearly invert the ordering, so a gap between two rows is less purely a gap between two representations than elsewhere in this corpus. The **training pair count is part of the protocol** and is in `task_params`: error keeps falling as pairs are added, so two pose numbers are comparable only if they drew the same pairs. **Read it to whole degrees**: two extractions of the same features differ by ~1e-5, which thirty epochs of this head turn into about a degree of rotation error, so adjacent rows closer than that are ties.

<sub>relative_pose on navi_v1/val, protocol=probe3d_pose, frozen [67965158]</sub>
<!-- /visbench:board -->

**Nothing on this board is at chance**, which is the first thing to check here
and the reason the floor travels with the score: every row clears the
no-feature floor by between 23.7 and 44.2 degrees, so the 20-degree spread is
signal rather than thirteen different ways of failing.

**It is close to orthogonal to the semantic boards.** Mean Spearman over the
same thirteen backbones is **+0.701** against the four low-level boards,
**+0.642** against its own mid-level tier and **+0.099** against the seven
high-level ones — with `semantic_segmentation` at **−0.115**,
`scene_classification` at −0.104 and `retrieval` at −0.016. Its strongest
partners are `surface_normal` (+0.791), `occlusion_edge` (+0.764) and `corner`
(+0.758). A pose number is therefore close to independent evidence about a
backbone rather than another way of asking what the object boards already
answered. See {doc}`reading a board </guides/reading-a-board>` before quoting
any of that.

**Quote the objective gap against the recipe gap.** `mae_vitb16` and
`supervised_vitb16` share an architecture, a width and a pretraining set and
differ in what they were trained to do: **18.73 degrees**. The recipe control on
the same board, `sam_vitb16` against `supervised_vitb16`, is **6.41**. And the
grid pair `dino_vitb8` against `dino_vitb16` — same objective and data at four
times the tokens — is **1.94**, which is worth noting on a probe that reads
*pooled* features and so has no grid for its head to read.
