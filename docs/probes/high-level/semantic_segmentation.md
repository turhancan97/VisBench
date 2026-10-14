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
| `mae_vitb16` | 0.3546 | 0.4596 | 0.8362 | 0.3960 |
| `sam_vitb16` | 0.3393 | 0.3787 | 0.8202 | 0.3896 |

Ordered by `miou`, which **disagrees with `miou_per_image`, `pixel_acc`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>semantic_segmentation on VOC2012/val, protocol=visbench_semantic_seg, frozen [e14b47db]</sub>
<!-- /visbench:board -->

**What this board ranks by is not what nearly every other dense board ranks
by, and that is worth knowing before quoting it.** Correlate each board's
ordering against the backbones' feature-grid area, over the thirteen
backbones in the corpus: across the twelve whose head reads one vector per
patch, **nine have the grid as their strongest correlate**, and this is one of
the three that do not. Grid **+0.494**, against pretraining scale **+0.673**.
Much of the +0.494 is carried by DINOv2, which has both the finest grid and the
largest pretraining corpus — drop those two rows and it falls to **+0.240** —
and with the pretraining data held fixed (the eight IN1k backbones) it is
**+0.300**: within one pretraining set the finer grids do score a little
higher, just far less than pretraining breadth separates the rest.

**Read that as "the grid does not win here", not as "the grid coefficient is
small".** `keypoints2d` and `orientation` read grid +0.658 — not far above this
board — and *are* grid-ranked, because on them nothing else comes close. What
singles this board out is that something does: pretraining scale beats the
grid on it, and among the twelve only `detection` (+0.761 against +0.550) and
`scene_parsing` (+0.692 against +0.647) do the same.

The control is already on this page. `generic_segmentation` runs on the *same
1449 VOC images* at the same resolution with the same linear head and the same
schedule, and differs only in whether the target has 2 classes or 21 — it
ranks by grid at **+0.935** against pretraining's +0.088, which is the
ordering this board reverses. Same pixels, same probe, opposite behaviour, so
this is a property of the target rather than the data, the head or the
protocol.

What it rewards instead is, weakly, pretraining breadth: corpus size is its
best single correlate at **+0.673** — fifth of the twenty boards, behind
`scene_classification` (+0.874), `vehicle_classification` (+0.849),
`detection` (+0.761) and `scene_parsing` (+0.692). But within the six
*identical* ViT-B/16 backbones — same shape, same width, same 196 tokens — the
spread is **0.3153 mIoU** (`clip_vitb16` 0.6546 down to `sam_vitb16` 0.3393)
at a +0.541 correlation, so a large part of this board's variance is not
structural at all. **Treat this board as evidence about representations, not
about training objectives** — step 10e's recipe control showed it cannot
separate those at all.

Reproduce any of it with `scripts/analyse_board_correlates.py`. At thirteen
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
