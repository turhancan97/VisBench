# `visbench.data`

Datasets. A folder layout needs no code at all; anything else is a
`BaseDataset` subclass with two abstract methods.

**`cache_identity` is the method not to skip.** Return `None` and every run
re-decodes every image, forever, while appearing to work — which is exactly what
happened for a year before `run()` started calling `view_identity`. The four
optional methods all fail silently when omitted, and that one fails
*invisibly*.

## The package

```{eval-rst}
.. automodule:: visbench.data
   :no-members:
```

## The base class

```{eval-rst}
.. automodule:: visbench.data.base
   :members:
```

## Labelled folders

```{eval-rst}
.. automodule:: visbench.data.image_folder
   :members:
```

## Dense targets

Image and target must survive the *same* resize and crop, applied here rather than by a caller, and a target resamples **nearest-neighbour** — bilinear averages across a depth discontinuity and turns a hole's zeros into a halo of plausible wrong values that the valid mask no longer excludes.

```{eval-rst}
.. automodule:: visbench.data.dense
   :members:
```

## Taskonomy

Building-nested, indexed from split lists. `_DOMAIN_SPECS` records per domain which invalid-pixel convention it uses — measured, not inferred from whether the domain is geometric.

```{eval-rst}
.. automodule:: visbench.data.taskonomy
   :members:
```

## Targets computed from the frame

There is no second geometry here and no resampling of the response, which *deletes* the alignment hazard every other dense probe has to test for.

```{eval-rst}
.. automodule:: visbench.data.derived
   :members:
```

## Pairs

```{eval-rst}
.. automodule:: visbench.data.pair_dataset
   :members:
```

## Triplets

```{eval-rst}
.. automodule:: visbench.data.triplet
   :members:
```

## Boxes

Boxes are `xyxy`, absolute post-transform pixels, 0-indexed.

```{eval-rst}
.. automodule:: visbench.data.detection
   :members:
```

## BSDS500

```{eval-rst}
.. automodule:: visbench.data.bsds
   :members:
```

## torchvision and Hugging Face bridges

Both derive a real `cache_identity` by leaning on one property: the wrapped dataset is immutable in index order.

```{eval-rst}
.. automodule:: visbench.data.bridges
   :members:
```
