# `visbench.results`

Every run writes a structured record — backbone, task, dataset, pooling, feature
mode, layers, metrics, timestamp — under one **additive-only** schema. A field
is never removed or repurposed, because old records must keep parsing.

This is where *Bench* in the name is enforced. `comparability_key` decides which
records may be ranked together at all, and `HEADLINE_METRICS` /
`METRIC_DIRECTIONS` are **listed tables** that raise on an unlisted entry: a
board ordered by whichever metric sorted first asserts a ranking nobody chose,
and `mean`/`median` are angular *error*, so a heuristic reading them as scores
ranks that board upside down and the output reads as a finding rather than a
bug.

## The package

```{eval-rst}
.. automodule:: visbench.results
   :no-members:
```

## The record

```{eval-rst}
.. automodule:: visbench.results.schema
   :members:
```

## Writing

```{eval-rst}
.. automodule:: visbench.results.writer
   :members:
```

## Comparability and ranking

Also `latest_per_backbone`, which **raises** when one name arrives under two `backbone_key`s — a reconfigured backbone reporting its old name would not produce a wrong row, it would silently *delete* the row it was built to be compared against.

```{eval-rst}
.. automodule:: visbench.results.leaderboard
   :members:
```

## Rendering

```{eval-rst}
.. automodule:: visbench.results.render
   :members:
```
