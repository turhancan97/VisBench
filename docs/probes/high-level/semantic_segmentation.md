# `semantic_segmentation`

**Multi-class per-pixel labels — cross-entropy over class indices.**

A dense probe, and the one whose board behaves unlike every other dense board
here. Read the section below before quoting it.

```{figure} /_static/gallery/semantic_segmentation.png
:alt: semantic_segmentation — image, target and prediction

What `visbench show semantic_segmentation` draws. {doc}`How to read it </show>`.
```

## Its board

<!-- visbench:board task=semantic_segmentation metrics=miou,miou_per_image,pixel_acc,mean_acc heading=3 -->
### semantic_segmentation

| backbone | `miou` | `miou_per_image` | `pixel_acc` | `mean_acc` |
| --- | --- | --- | --- | --- |
| `dinov2_vitb14` | **0.7533** | **0.7161** | **0.9316** | **0.8403** |
| `dinov2_vits14` | 0.7328 | 0.6841 | 0.9267 | 0.8271 |
| `clip_vitb16` | 0.6546 | 0.6683 | 0.9019 | 0.7312 |
| `clip_vitb32` | 0.5813 | 0.6067 | 0.8731 | 0.6633 |
| `supervised_vitb16` | 0.5791 | 0.5877 | 0.8681 | 0.6761 |
| `siglip_vitb16` | 0.5405 | 0.3210 | 0.8539 | 0.6511 |
| `dino_vitb16` | 0.5063 | 0.3221 | 0.8632 | 0.5964 |
| `convnext_base` | 0.4880 | 0.4596 | 0.8310 | 0.5902 |
| `resnet50` | 0.4574 | 0.5163 | 0.8322 | 0.5248 |
| `resnet18` | 0.4212 | 0.4497 | 0.8205 | 0.4915 |
| `mae_vitb16` | 0.3350 | 0.4555 | 0.8269 | 0.3757 |
| `sam_vitb16` | 0.3339 | 0.3905 | 0.8146 | 0.3825 |

Ordered by `miou`, which **disagrees with `mean_acc`, `miou_per_image`, `pixel_acc`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>semantic_segmentation on VOC2012/val, protocol=visbench_semantic_seg, frozen [e14b47db]</sub>
<!-- /visbench:board -->

**What this board ranks by is not what the other dense boards rank by, and
that is worth knowing before quoting it.** Correlate each board's ordering
against the backbones' feature-grid area, over the twelve backbones in the
corpus, and every other dense board comes out between +0.73 and +0.96 — a
finer grid is most of what a dense probe rewards. Semantic segmentation is
**+0.545**, and that much is carried by DINOv2, which has both the finest grid
and the largest pretraining corpus: without those two rows it is +0.212, and
with the pretraining data held fixed it is **0.000**.

The control is already in this page. `generic_segmentation` runs on the *same
1449 VOC images* at the same resolution with the same linear head and the same
schedule, and differs only in whether the target has 2 classes or 21 — it
ranks by grid at **+0.958**. Same pixels, same probe, opposite behaviour, so
this is a property of the target rather than the data or the protocol.

What it rewards instead is, weakly, pretraining breadth: corpus size is its
best single correlate at +0.615, the highest of the thirteen boards. But
within the six *identical* ViT-B/16 backbones the spread is 0.3207 mIoU with
only a +0.314 correlation, so most of the variance is not structural. **Treat
this board as evidence about representations, not about training objectives**
— step 10e's recipe control showed it cannot separate those at all. Reproduce
any of it with `scripts/analyse_board_correlates.py`; at twelve backbones the
coefficients have wide error bars.

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
