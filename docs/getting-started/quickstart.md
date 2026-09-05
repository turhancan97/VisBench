# Quickstart

## Thirty seconds, no dataset

No dataset, no configuration, no large download:

```bash
pip install visbench
visbench demo
```

```text
drawing 20 images per class for 4 shapes...
loading resnet18 (torchvision, ~45 MB on first run)...
running the classification probe...

  top1         0.8125

  chance is 0.25 — the shapes differ in outline only.
```

That is a real probe, on a real pretrained backbone, through the same code path
every other run uses. The images are generated: four shapes with **colour,
size, position and rotation randomised**, so only geometry identifies a class
and a backbone that has not learned shape scores about chance.

The number is deliberately not 1.0. Turn the difficulty up and watch it fall:

```bash
visbench demo --noise 90      # top1 ~0.31, against a chance of 0.25
```

| `--noise` | 28 | **45** (default) | 60 | 75 | 90 |
|---|---|---|---|---|---|
| top1 | 0.975 | **0.812** | 0.550 | 0.438 | 0.312 |

A probe whose score does not move when you destroy the signal is not measuring
the signal. That slide into chance is the demo's actual point.

## Your own folder

Folder to scored, logged metrics, on any image folder laid out as
`root/<class_name>/<image>`:

as `root/<class_name>/<image>`:

```python
import visbench
from visbench.data import ImageFolderDataset

result = visbench.run(
    "dinov2_vitb14",
    "retrieval",
    ImageFolderDataset("data/tiny", split="val"),
    results="results/visbench.jsonl",
)
result.metrics    # {"recall@1": 0.94, "recall@5": 0.99, "mAP": 0.87}
result.record     # the ResultRecord that says exactly how they were produced
```

`run()` resolves pooling, extracts through the cache, fits the probe if it
trains, evaluates, and appends the record. The pieces are public if you want
them separately:

```python
from visbench.cache import FeatureCache

backbone = visbench.get_backbone("dinov2_vitb14")      # frozen, eval mode
probe    = visbench.get_probe("retrieval")             # zero-shot
features = FeatureCache().extract_dataset(
    backbone, dataset, pooling=probe.pooling, keep="pooled"
)                                                      # one forward pass per image
probe.evaluate(features, dataset.labels())
```

Re-running is cheap. On Imagenette (13,394 images, DINOv2 ViT-S, one V100):

| | cold | cached |
|---|---|---|
| wall time | 208 s | **26 s** |
| on-disk cache | 107 MB | — |
| val top1 | 0.9939 | 0.9939 |

A cached image is resolved from its file identity and never decoded, and
`keep="pooled"` also stops dense features being written — storing them for a
task that never reads them cost 5 GB instead of 107 MB. Results go to JSONL
through `visbench.results.ResultWriter`, under one schema from the first
record.

Trained probes take the same call with a training split. A train/test split is
just two datasets, so each half carries its own fingerprint:

```python
result = visbench.run(
    "dinov2_vitb14", "classification", val_dataset, train_dataset=train_dataset
)
result.metrics             # {"top1": ..., "top5": ...}
result.probe.train_top1    # 0.99 — if this is low, the probe underfitted,
                           # not the backbone. Raise `lr` or `epochs`.
```

Passing `train_dataset` to a zero-shot task raises rather than being ignored:
silently dropping it would leave the caller's intent and the result
disagreeing.

The linear probe trains with AdamW on cached features, so its hyperparameters
are part of the reported number and travel with it in the record's
`task_params`.

## What to read next

| | |
| --- | --- |
| Every probe and what it measures | {doc}`the probes </probes/overview>` |
| Running from a shell instead | {doc}`the command line </getting-started/cli>` |
| A backbone VisBench has never heard of | {doc}`backbones </guides/backbones>` |
| A dataset that is not a folder | {doc}`datasets </guides/datasets>` |
| Seeing what a probe saw | {doc}`looking at a probe </guides/visualising>` |
| Every class and function | {doc}`API reference </api/index>` |
