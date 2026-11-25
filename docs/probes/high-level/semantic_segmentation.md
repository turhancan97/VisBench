# `semantic_segmentation`

**Multi-class per-pixel labels — cross-entropy over class indices.**

A dense probe whose board does not rank by what nearly every other dense board
ranks by. Read the section below before quoting it.

```{figure} /_static/gallery/semantic_segmentation.png
:alt: semantic_segmentation — image, target and prediction

What `visbench show semantic_segmentation` draws. {doc}`How to read it </guides/visualising>`.
```

## Its board

<!-- visbench:board task=semantic_segmentation metrics=miou,miou_per_image,pixel_acc,mean_acc heading=3 -->
### semantic_segmentation

| backbone | `miou` | `miou_per_image` | `pixel_acc` | `mean_acc` |
| --- | --- | --- | --- | --- |
| `dinov2_vitb14` | **0.7533** | **0.7161** | **0.9316** | **0.8403** |
| `dinov2_vits14` | 0.7328 | 0.6841 | 0.9267 | 0.8270 |
| `clip_vitb16` | 0.6546 | 0.6683 | 0.9019 | 0.7312 |
| `supervised_vitb16` | 0.6184 | 0.6103 | 0.8853 | 0.7068 |
| `siglip_vitb16` | 0.5885 | 0.3386 | 0.8744 | 0.6940 |
| `clip_vitb32` | 0.5813 | 0.6067 | 0.8731 | 0.6634 |
| `dino_vitb8` | 0.5304 | 0.2054 | 0.8776 | 0.6263 |
| `dino_vitb16` | 0.5199 | 0.3099 | 0.8704 | 0.6097 |
| `convnext_base` | 0.4967 | 0.4656 | 0.8335 | 0.5951 |
| `resnet50` | 0.4513 | 0.5146 | 0.8296 | 0.5164 |
| `resnet18` | 0.4234 | 0.4516 | 0.8213 | 0.4925 |
| `clip_convnext_base` | 0.4162 | 0.5112 | 0.8081 | 0.4722 |
| `mae_vitb16` | 0.3546 | 0.4596 | 0.8362 | 0.3960 |
| `sam_vitb16` | 0.3393 | 0.3787 | 0.8202 | 0.3896 |

Ordered by `miou`, which **disagrees with `miou_per_image`, `pixel_acc`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>semantic_segmentation on VOC2012/val, protocol=visbench_semantic_seg, frozen [e14b47db]</sub>
<!-- /visbench:board -->

**What this board ranks by is less settled than on any other dense board, and
that is worth knowing before quoting it.** Correlate each board's ordering
against the backbones' feature-grid area, over the fourteen backbones in the
corpus: across the twelve whose head reads one vector per patch, **eleven have
the grid as their strongest correlate**, and on this one it only just wins —
grid **+0.509**, against pretraining scale **+0.427**. At thirteen backbones it
was the other way round (+0.494 against +0.673), and the column that turned it
is `clip_convnext_base`: the largest pretraining corpus of any CNN, a coarse
7x7 grid, and a low score here. That low score is partly not the
representation's — see below — so read this board as tied between the two
properties, not as grid-ranked. Much of the grid coefficient is carried by
DINOv2, which has both the finest grid and a large curated corpus — drop those
two rows and it falls to **+0.276** — and with the pretraining data held fixed
(the eight IN1k backbones) it is **+0.300**: within one pretraining set the
finer grids do score a little higher, just far less than the rest of the
board's spread.

**Its fourteenth column is placed by its feature scale.** The linear heads fit
raw features at a fixed learning rate, and `clip_convnext_base`'s features are
~9x smaller than `convnext_base`'s. Re-fitted with the two moved onto each
other's scale — a constant, which removes nothing a linear head could use — the
pair **reverses**: `clip_convnext_base` goes from 12th to 9th (+126 seed
deviations) and `convnext_base` from 9th to 11th. So this board's placement of
that pair says how the fit met each feature scale, not which representation
carries more; the control is in
[`CORPUS_FINDINGS.md`](https://github.com/turhancan97/VisBench/blob/main/CORPUS_FINDINGS.md)
under 24b. Across all fourteen columns, scale still tracks this board with the
grid held fixed (partial rho +0.446), which no geometry board does.

The control is already on this page. `generic_segmentation` runs on the *same
1449 VOC images* at the same resolution with the same linear head and the same
schedule, and differs only in whether the target has 2 classes or 21 — it
ranks by grid at **+0.936** against pretraining's −0.129, an ordering this
board does not share. Same pixels, same probe, different behaviour, so this is
a property of the target rather than the data, the head or the protocol.

Pretraining breadth is still a real correlate here: corpus size reads
**+0.427**, sixth of the twenty boards, behind `scene_classification`
(+0.896), `vehicle_classification` (+0.876), `detection` (+0.661),
`fine_grained_classification` (+0.571) and `scene_parsing` (+0.534). But
within the six *identical* ViT-B/16 backbones — same shape, same width, same
196 tokens — the spread is **0.3153 mIoU** (`clip_vitb16` 0.6546 down to
`sam_vitb16` 0.3393) at a +0.541 correlation, so a large part of this board's
variance is not structural at all. **Treat this board as evidence about
representations, not about training objectives** — step 10e's recipe control
showed it cannot separate those at all.

Reproduce any of it with `scripts/analyse_board_correlates.py`. At fourteen
backbones the coefficients have wide error bars, **and where one board's
coefficient sits relative to another's is a fact about this corpus, not about
the task**: these read +0.545 and +0.615 against thirteen boards, before the
2026-09-13 tie fix, further boards and the 23a crop fix moved both.

## Things that will bite otherwise

  binarising a VOC map, since `load_mask` would read its void border as
  foreground.
- **The ten-epoch schedule assumes a dataset the size of NYUv2.** On a small
  split it underfits badly — 80 training images gave 0.16 IoU at the defaults
  and 0.87 at `--epochs 40 --lr 5e-3`, on identical features. `train_loss` is
  printed for exactly this: a poor score with a high training loss means the
  probe did not converge, which is a different finding from a representation
  that does not carry the signal.

## Run it

[`examples/segment_semantic.py`](https://github.com/turhancan97/VisBench/blob/main/examples/segment_semantic.py) is the whole path, end to end, on a real backbone.

```bash
python examples/segment_semantic.py --data /path/to/dataset
```
