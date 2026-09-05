# Datasets

Three tiers, and the first needs no code at all.

## A folder

```text
<root>/train/<class>/<image>.jpg      # labelled: classification, retrieval
<root>/val/<class>/<image>.jpg

<root>/train/images/<stem>.jpg        # dense: depth, normals, segmentation
<root>/train/targets/<stem>.png       # paired by filename STEM
```

A folder layout needs no code (`ImageFolderDataset`, `DenseFolderDataset`,
`DetectionFolderDataset`), and anything else is a `BaseDataset` subclass with two
methods. When the data already lives in a `torch.utils.data` dataset or a
Hugging Face `datasets.Dataset`, there is a bridge:

```python
from torchvision.datasets import CIFAR10
from visbench.data import TorchvisionDataset

raw = CIFAR10("./data", train=False, download=True)
visbench.run("dinov2_vits14", "classification", TorchvisionDataset(raw, split="test"),
             train_dataset=TorchvisionDataset(CIFAR10("./data", train=True, download=True)))
```

`HuggingFaceDataset` is the same shape and needs `pip install visbench[datasets]`.
Both derive a real per-item `cache_identity` from the fact that the wrapped
dataset is immutable in index order, so a cached re-run still skips the
backbone. See [`examples/custom_dataset.py`](https://github.com/turhancan97/VisBench/blob/main/examples/custom_dataset.py).

Sibling project to [vismatch](https://github.com/gmberton/vismatch) — same
ergonomic philosophy, applied to representation probing instead of image
matching.

## A `torch` or Hugging Face dataset

```python
from visbench.data import TorchvisionDataset, HuggingFaceDataset
```

or from a shell, on the image-level probes:

```bash
visbench run classification --dataset torchvision:CIFAR10 --split test
visbench run retrieval --dataset hf:cifar100:name=cifar100
```

Dense, pair and triplet probes stay folder-only: an HF dataset carrying a dense
target is a much larger surface — per-probe target-column plumbing, loader and
dtype selection, and the four validity conventions below.

## Anything else: subclass `BaseDataset`

Two abstract methods, `__len__` and `__getitem__`. Four optional ones, and
**every one of them fails silently when omitted**:

| method | omitted | what you get |
| --- | --- | --- |
| `labels()` | supervised probes have no targets | an error, eventually |
| `cache_identity()` | **every run re-decodes every image, forever** | it works, slowly |
| `fingerprint()` | records cannot tell your dataset from another | wrong comparability |
| `describe()` | `dataset_params` is empty | an unreproducible record |

**`cache_identity` is the one that fails invisibly.** Return `None` and nothing
raises and nothing is wrong — the cache simply never hits. That is not
hypothetical: `view_identity` was written and tested from v0.1 and had no caller
for a year, because `examples/correspond.py` passed the cache bare PIL images,
which have no identity. A fully "cached" run still decoded, cropped and warped
everything. 16.4 s cold against 8.2 s warm on 200 pairs, once `run()` used it.

Both shipped bridges derive a real one by leaning on a single property: the
wrapped dataset is immutable in index order.

## Four validity conventions, and they are not interchangeable

A dense target has to say which pixels are unmeasured, and there is no single
answer — check which one a new task needs rather than inheriting the nearest.

| target | invalid is | why not the others |
| --- | --- | --- |
| depth | `0` | a real reading is never 0 |
| normals | zero length | same |
| **label maps** | **negative**, `IGNORE_INDEX = -1` | 0 is a real class (background); the depth rule would discard every background pixel and train the probe to answer foreground everywhere |
| edge maps | *nothing* | 0 means "no edge", a real reading covering most of a frame |
| magnitude maps | `NaN` | the third case has no spare value, so validity travels out of band — and `NaN` is the **loud** choice, making an unmasked loss `NaN` on the first step where a fabricated 0 trains quietly and merely scores badly |

**Image and target must survive the same resize and crop**, applied by the
dataset, and a target resamples **nearest-neighbour**. Bilinear averages across
a depth discontinuity and turns a hole's zeros into a halo of plausible wrong
values the valid mask no longer excludes.

**Shorten a labelled folder with `balanced_subset(n)`, never `subset(n)`.** The
file list is grouped by class, so a prefix is entirely class 0 — and a
single-class retrieval scores 1.0 while measuring nothing.
