# `generic_segmentation`

**Binary figure-ground — is this pixel part of an object at all?**

The same 1449 VOC images as {doc}`semantic_segmentation </probes/high-level/semantic_segmentation>`, at the same resolution with the same linear head
and the same schedule, differing **only** in whether the target has 2 classes
or 21. That makes the pair a control, and they behave oppositely: this board
ranks by feature-grid area at **+0.958**, the semantic one at +0.545 — so that
difference is a property of the target, not of the data or the protocol.

```{figure} /_static/gallery/generic_segmentation.png
:alt: generic_segmentation — image, target and prediction

What `visbench show generic_segmentation` draws. {doc}`How to read it </guides/visualising>`.
```

## Things that will bite

  elsewhere. On VOC they sit five points apart. Quote `miou` against published
  numbers, and say which one you mean.
- **Label maps are read without mode conversion, and getting this wrong is
  silent.** VOC's PNGs are palette images whose raw bytes are the class indices;
  resolving the palette turns classes `[0, 1, 15]` into `[0, 38, 147]`, which
  trains and scores perfectly happily against labels that mean nothing. Use
  `load_label_map`, not `load_mask`, for anything multi-class — including

## Its board

<!-- visbench:board task=generic_segmentation metrics=f1,iou,pixel_acc heading=3 -->
### generic_segmentation

| backbone | `iou` | `f1` | `pixel_acc` |
| --- | --- | --- | --- |
| `dinov2_vitb14` | **0.7556** | **0.8408** | **0.9360** |
| `dinov2_vits14` | 0.7494 | 0.8338 | 0.9324 |
| `dino_vitb16` | 0.6838 | 0.7835 | 0.8999 |
| `clip_vitb16` | 0.6787 | 0.7818 | 0.9027 |
| `sam_vitb16` | 0.6667 | 0.7687 | 0.8913 |
| `mae_vitb16` | 0.6374 | 0.7384 | 0.8891 |
| `supervised_vitb16` | 0.6195 | 0.7267 | 0.8839 |
| `clip_vitb32` | 0.6019 | 0.7178 | 0.8793 |
| `siglip_vitb16` | 0.5912 | 0.7172 | 0.8682 |
| `convnext_base` | 0.5480 | 0.6732 | 0.8530 |
| `resnet50` | 0.5475 | 0.6676 | 0.8646 |
| `resnet18` | 0.5358 | 0.6597 | 0.8517 |

Ordered by `iou`, which **disagrees with `pixel_acc`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>generic_segmentation on voc_binary/val, protocol=visbench_binary_seg, frozen [5c4cf9a6]</sub>
<!-- /visbench:board -->

## Run it

[`examples/segment.py`](https://github.com/turhancan97/VisBench/blob/main/examples/segment.py) is the whole path, end to end, on a real backbone.

```bash
python examples/segment.py --data /path/to/dataset
```
