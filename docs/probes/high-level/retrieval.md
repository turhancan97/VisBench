# `retrieval`

**Zero-shot: every image queries every other by cosine similarity over pooled features.**

Nothing is trained, so this measures the pooled representation directly. It
shares its dataset layout with {doc}`classification <classification>` — a
labelled image folder and nothing else.

```{figure} /_static/gallery/retrieval.png
:alt: retrieval — image, target and prediction

What `visbench show retrieval` draws. {doc}`How to read it </show>`.
```

## Its board

<!-- visbench:board task=retrieval metrics=mAP,recall@1,recall@5 heading=3 -->
### retrieval

| backbone | `mAP` | `recall@1` | `recall@5` |
| --- | --- | --- | --- |
| `supervised_vitb16` | **0.9947** | 0.9977 | 0.9987 |
| `sam_vitb16` | 0.9912 | 0.9944 | **0.9992** |
| `convnext_base` | 0.9890 | **0.9987** | 0.9990 |
| `resnet50` | 0.9357 | 0.9901 | 0.9987 |
| `dino_vitb16` | 0.9192 | 0.9868 | 0.9972 |
| `dinov2_vitb14` | 0.9171 | 0.9954 | 0.9977 |
| `clip_vitb16` | 0.9102 | 0.9893 | 0.9975 |
| `dinov2_vits14` | 0.8893 | 0.9921 | 0.9972 |
| `clip_vitb32` | 0.8680 | 0.9806 | 0.9941 |
| `resnet18` | 0.8648 | 0.9725 | 0.9944 |
| `siglip_vitb16` | 0.8525 | 0.9799 | 0.9936 |
| `mae_vitb16` | 0.1883 | 0.6741 | 0.8892 |

Ordered by `mAP`, which **disagrees with `recall@1`, `recall@10`, `recall@5`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>retrieval on val/val, frozen [eb312a7b]</sub>
<!-- /visbench:board -->

## Run it

[`examples/retrieve.py`](https://github.com/turhancan97/VisBench/blob/main/examples/retrieve.py) is the whole path, end to end, on a real backbone.

```bash
python examples/retrieve.py --data /path/to/dataset
```
