# `visbench.metrics`

Every metric this library reports, and — in the docstrings — what each number
is comparable to. That is the part worth reading: a metric borrowed from a paper
carries that paper's conventions, and two of the decisions here cost real
measurements before they were pinned.

**Per image, then averaged**, everywhere except detection's AP. Pooling every
pixel of a split lets uneven hole coverage silently reweight the dataset. AP is
the exception because it is a *ranking*, and "per image then averaged" does not
apply to one.

## The package

```{eval-rst}
.. automodule:: visbench.metrics
   :no-members:
```

## Classification

```{eval-rst}
.. automodule:: visbench.metrics.classification
   :members:
```

## Retrieval

```{eval-rst}
.. automodule:: visbench.metrics.retrieval
   :members:
```

## Correspondence

Thresholds are in **pixels**, not patch widths. A patch is 14px on DINOv2/14 and 32px on a ResNet, so scoring in patch widths asks each backbone a different question — it inverted the published board, swapping first and last place.

```{eval-rst}
.. automodule:: visbench.metrics.correspondence
   :members:
```

## Similarity

```{eval-rst}
.. automodule:: visbench.metrics.similarity
   :members:
```

## Dense

Depth, normals, the magnitude probes and orientation.

```{eval-rst}
.. automodule:: visbench.metrics.dense
   :members:
```

## Boundaries

BSDS500's ODS/OIS/AP, written from the paper and validated against its published human agreement — 0.8030 against 0.80.

```{eval-rst}
.. automodule:: visbench.metrics.boundary
   :members:
```

## Detection

VOC's protocol, including the subtlety that `difficult` objects are *ignored* rather than dropped. Measured on oracle predictions the difference is 4.3 mAP, and the wrong one is **lower** — so it reads as a weak detector rather than as a scoring bug.

```{eval-rst}
.. automodule:: visbench.metrics.detection
   :members:
```
