# The probes

Sixteen probes across three levels. Every one of them is reachable the same
three ways — `visbench.run()`, the
`visbench run` CLI, and an `examples/` script — and every one writes a record
saying exactly what produced its number.

Each probe page states what it measures, the data layout it expects, its
`protocol` string, and its twelve-backbone board. **Read
{doc}`the leaderboard page </probes/leaderboard>` first if you intend to quote
one.**

## High level — semantic and category understanding

| probe | what it measures |
| --- | --- |
| {doc}`classification </probes/high-level/classification>` | basic-level object category, linear probe on pooled features |
| {doc}`scene_classification </probes/high-level/scene_classification>` | the category of the *place*, not of an object in it |
| {doc}`fine_grained_classification </probes/high-level/fine_grained_classification>` | subordinate category — which species, not whether it is a bird |
| {doc}`retrieval </probes/high-level/retrieval>` | zero-shot nearest neighbours by cosine over pooled features |
| {doc}`semantic_segmentation </probes/high-level/semantic_segmentation>` | multi-class per-pixel labels |
| {doc}`detection </probes/high-level/detection>` | anchor-free single-scale boxes from one feature map |

Three of those six share one implementation and ask three different questions:
`classification` is basic-level, `scene_classification` is place,
`fine_grained_classification` is subordinate. Each is a distinct probe *name*
rather than a dataset flag, because a board is keyed on the task name — a second
dataset under one name does not merge into that board, it makes it
unrenderable.

## Mid level — geometry and structure, before naming anything

| probe | what it measures |
| --- | --- |
| {doc}`depth </probes/mid-level/depth>` | monocular metric depth, probe3d's 256-bin expectation |
| {doc}`surface_normal </probes/mid-level/surface_normal>` | per-pixel surface orientation, scored by angular error |
| {doc}`generic_segmentation </probes/mid-level/generic_segmentation>` | binary figure-ground — is this pixel an object at all? |
| {doc}`correspondence </probes/mid-level/correspondence>` | zero-shot geometric matching between two views, in pixels |
| {doc}`similarity </probes/mid-level/similarity>` | perceptual resemblance as a two-alternative forced choice |
| {doc}`occlusion_edge </probes/mid-level/occlusion_edge>` | depth discontinuities — needs scene geometry |

The tier the task taxonomy treats as its core contribution. Mid-level
**similarity** is deliberately not high-level retrieval: it judges resemblance
in layout and geometry rather than category membership.

## Low level — signal properties, recoverable without naming an object

| probe | what it measures |
| --- | --- |
| {doc}`edge </probes/low-level/edge>` | intensity edges — gradient magnitude |
| {doc}`keypoints2d </probes/low-level/keypoints2d>` | 2D keypoint response |
| {doc}`corner </probes/low-level/corner>` | Shi-Tomasi cornerness, computed from the frame |
| {doc}`orientation </probes/low-level/orientation>` | local gradient orientation — a direction, not a magnitude |

Two of the four compute their target **from the frame** rather than reading it
from disk. That makes them the cheapest kind of probe to add and the easiest to
fool yourself with — see the gauntlet in
[`visbench/tasks/low_level/README.md`](https://github.com/turhancan97/VisBench/blob/main/visbench/tasks/low_level/README.md),
which has now rejected three candidates.

`occlusion_edge` and `edge` **share every line of their implementation and sit
one tier apart.** That is the cleanest statement of what the tiers mean that
this project has: recovering a depth discontinuity needs scene geometry,
recovering an intensity one does not.

---

Two findings below cut across every board, and both change how a page further
down should be read.

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

## What every dense board shares

**Report the linear head.** It is the default, and the only one under which a
difference between two backbones is a difference between two *feature maps*. A
DPT head scores higher for everyone — and reorders 24 of 174 separable pairs,
changing the leader on two of five boards, so it is not a neutral magnifying
glass. Run both and say which:

```bash
python examples/normals.py --data ... --head dpt --layers 2 5 8 11
```

**Features are shared between dense probes** when the images and `--image-size`
match, so probing all of them on one dataset costs one extraction. Splits larger
than memory are fine: dense features stream from the cache a batch at a time
rather than being stacked.

**The ten-epoch schedule assumes a dataset the size of NYUv2.** On a small split
it underfits badly — 80 training images gave 0.16 IoU at the defaults and 0.87
at `--epochs 40 --lr 5e-3`, on identical features. `train_loss` is recorded for
exactly this: a poor score with a high training loss means the probe did not
converge, which is a different finding from a representation that does not carry
the signal.

```{toctree}
:hidden:
:caption: High level

high-level/classification
high-level/scene_classification
high-level/fine_grained_classification
high-level/retrieval
high-level/semantic_segmentation
high-level/detection
```

```{toctree}
:hidden:
:caption: Mid level

mid-level/depth
mid-level/surface_normal
mid-level/generic_segmentation
mid-level/correspondence
mid-level/similarity
mid-level/occlusion_edge
```

```{toctree}
:hidden:
:caption: Low level

low-level/edge
low-level/keypoints2d
low-level/corner
low-level/orientation
```

```{toctree}
:hidden:

leaderboard
```
