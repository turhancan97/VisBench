# `correspondence`

**Zero-shot geometric matching between two views, scored in pixels.**

Needs no annotation at all: the second view is a known random homography of the
first, so the ground-truth correspondence is exact by construction.

```{figure} /_static/gallery/correspondence.png
:alt: correspondence — image, target and prediction

What `visbench show correspondence` draws. {doc}`How to read it </guides/visualising>`.
```

## Its board

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

## Run it

[`examples/correspond.py`](https://github.com/turhancan97/VisBench/blob/main/examples/correspond.py) is the whole path, end to end, on a real backbone.

```bash
python examples/correspond.py --data /path/to/dataset
```
