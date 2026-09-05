# `visbench.cache`

Mandatory from v0.1, not a speed-up bolted on later. A disk-backed store keyed
on `backbone_key | layer | pooling | feature_mode | image_hash`, so a backbone's
forward pass runs at most once per image per backbone however many probes read
it.

Two front doors, and the choice is not stylistic. `extract_dataset` stacks
everything into one `FeatureDict` — right for pooled features, impossible for
dense ones, which are ~250x larger. `materialise` returns a `CachedFeatures`, an
ordinary `torch.utils.data.Dataset` over files already on disk.

## The package

```{eval-rst}
.. automodule:: visbench.cache
   :no-members:
```

## Keys

```{eval-rst}
.. automodule:: visbench.cache.keys
   :members:
```

## The cache

```{eval-rst}
.. automodule:: visbench.cache.feature_cache
   :members:
```

## Streaming

Random-access rather than a generator: training reshuffles every epoch, and a generator can only shuffle *within* a batch.

```{eval-rst}
.. automodule:: visbench.cache.streaming
   :members:
```

## Frozen prefixes

For fine-tuning: the blocks below the cut never change, so they are cached separately.

```{eval-rst}
.. automodule:: visbench.cache.prefix_cache
   :members:
```
