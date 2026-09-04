# `classification`

**Basic-level object category, from a linear probe on pooled features.**

The simplest probe here, and the one whose board is most nearly saturated —
eleven of twelve backbones score above 0.99 on Imagenette, so read the spread
rather than the winner.

```{figure} /_static/gallery/classification.png
:alt: classification — image, target and prediction

What `visbench show classification` draws. {doc}`How to read it </show>`.
```

The two probes that need only a labelled image folder.

[`examples/classify.py`](https://github.com/turhancan97/VisBench/blob/main/examples/classify.py) runs the whole path on any
folder laid out as `<data>/train/<class>/…` and `<data>/val/<class>/…`:

```bash
pip install -e .                                   # required: the script imports visbench
python examples/classify.py --data /path/to/dataset
python examples/classify.py --data /path/to/dataset --limit 20   # 20 images per class, quick
```

The first run extracts features; every later run on the same data reads them
from disk and the backbone never executes, so sweeping probe settings costs
only the probe:

```bash
python examples/classify.py --data /path/to/dataset --epochs 500 --lr 0.05
```

It prints `train top1` next to the validation score. If the validation number
is low *and* `train top1` is low, the probe underfitted — raise `--lr` or
`--epochs`. If `train top1` is near 1.0, the backbone genuinely does not
separate those classes.

3,925-image val split, one V100. Correspondence on 200 pairs at `max_warp=0.2`.

## Its board

<!-- visbench:board task=classification metrics=top1,top5 heading=3 -->
### classification

| backbone | `top1` | `top5` |
| --- | --- | --- |
| `convnext_base` | **0.9997** | **1.0000** |
| `resnet50` | 0.9980 | 0.9997 |
| `dinov2_vitb14` | 0.9975 | **1.0000** |
| `sam_vitb16` | 0.9972 | **1.0000** |
| `supervised_vitb16` | 0.9972 | 0.9997 |
| `clip_vitb16` | 0.9954 | 0.9997 |
| `dinov2_vits14` | 0.9939 | 0.9997 |
| `siglip_vitb16` | 0.9936 | 0.9995 |
| `dino_vitb16` | 0.9931 | 0.9997 |
| `clip_vitb32` | 0.9921 | 0.9992 |
| `resnet18` | 0.9888 | 0.9995 |
| `mae_vitb16` | 0.9582 | 0.9985 |

Ordered by `top1`, which **disagrees with `top5`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>classification on val/val, frozen [12e02eff]</sub>
<!-- /visbench:board -->

## Run it

[`examples/classify.py`](https://github.com/turhancan97/VisBench/blob/main/examples/classify.py) is the whole path, end to end, on a real backbone.

```bash
python examples/classify.py --data /path/to/dataset
```
