# `similarity`

**Perceptual and geometric resemblance, as a two-alternative forced choice.**

Deliberately **not** high-level retrieval: this judges whether two candidates
resemble a reference in layout and geometry, not whether they share a category.
Zero-shot — nothing is trained, despite what the source paper's own README
says.

```{figure} /_static/gallery/similarity.png
:alt: similarity — image, target and prediction

What `visbench show similarity` draws. {doc}`How to read it </guides/visualising>`.
```

[`examples/similarity.py`](https://github.com/turhancan97/VisBench/blob/main/examples/similarity.py) asks whether the backbone
agrees with a human about which of two candidates looks more like a reference —
a two-alternative forced choice over [NIGHTS](https://dreamsim-nights.github.io)
(Fu et al., *DreamSim*). Also zero-shot: the probe is two cosine similarities
and a comparison, with no head and no training split.

```bash
python examples/similarity.py --data /path/to/nights
python examples/similarity.py --data ... --split test_no_imagenet
```

**This is not retrieval.** The ground truth is perceptual — layout, pose,
structure — not category membership, which is why the two are separate tasks. A
backbone can be strong at one and ordinary at the other.

Measured on the NIGHTS test split (1,824 triplets, `min_votes=6`), pooled
features at 224px. Humans chose "right" 49.1% of the time, so chance is ~51%:

## Its board

<!-- visbench:board task=similarity metrics=accuracy,f1 heading=3 -->
### similarity

| backbone | `accuracy` | `f1` | `tie_rate` |
| --- | --- | --- | --- |
| `dino_vitb16` | **0.9019** | **0.9004** | 0.0000 |
| `dinov2_vits14` | 0.8701 | 0.8687 | 0.0000 |
| `sam_vitb16` | 0.8695 | 0.8675 | 0.0000 |
| `dinov2_vitb14` | 0.8580 | 0.8575 | 0.0000 |
| `siglip_vitb16` | 0.8575 | 0.8552 | 0.0000 |
| `clip_vitb32` | 0.8465 | 0.8443 | 0.0000 |
| `resnet18` | 0.8317 | 0.8307 | 0.0000 |
| `clip_vitb16` | 0.8284 | 0.8266 | 0.0000 |
| `resnet50` | 0.8273 | 0.8282 | 0.0000 |
| `supervised_vitb16` | 0.8202 | 0.8188 | 0.0000 |
| `convnext_base` | 0.7725 | 0.7711 | 0.0000 |
| `mae_vitb16` | 0.6897 | 0.6827 | 0.0000 |

Ordered by `accuracy`, which **disagrees with `f1`, `precision`, `recall`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>similarity on nights/test, protocol=midvision_2afc, frozen [0cc388a0]</sub>
<!-- /visbench:board -->

**The small DINOv2 beats the base one here** — the reverse of semantic
segmentation, where B leads S (0.753 against 0.732). Two tasks, two orderings,
same four backbones: which is the entire reason for probing more than one level
rather than assuming a single ranking of representations.

Run `--split test_imagenet` and `test_no_imagenet` before quoting a number: they
partition the test set by whether the reference came from ImageNet, so a gap
between them is a contamination signal rather than a similarity result. For
`dinov2_vits14` that gap is **0.882 against 0.854** — worth knowing before
reading 0.870 as a clean measure of perceptual alignment.

## Run it

[`examples/similarity.py`](https://github.com/turhancan97/VisBench/blob/main/examples/similarity.py) is the whole path, end to end, on a real backbone.

```bash
python examples/similarity.py --data /path/to/dataset
```
