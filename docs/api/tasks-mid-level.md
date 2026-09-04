# `visbench.tasks.mid_level`

Geometry and generic structure, prior to naming anything. This is the tier the
task taxonomy treats as its core contribution.

Mid-level **similarity** is deliberately not high-level retrieval: it judges
perceptual and geometric resemblance rather than category membership, and the
two are kept apart even though both are "similarity"-flavoured.

## The `mid_level` package

```{eval-rst}
.. automodule:: visbench.tasks.mid_level
   :members:
```

## `correspondence`

```{eval-rst}
.. automodule:: visbench.tasks.mid_level.correspondence
   :members:
```

## `depth`

```{eval-rst}
.. automodule:: visbench.tasks.mid_level.depth
   :members:
```

## `surface_normal`

```{eval-rst}
.. automodule:: visbench.tasks.mid_level.surface_normal
   :members:
```

## `generic_segmentation`

```{eval-rst}
.. automodule:: visbench.tasks.mid_level.generic_segmentation
   :members:
```

## `similarity`

```{eval-rst}
.. automodule:: visbench.tasks.mid_level.similarity
   :members:
```

## `occlusion_edge`

```{eval-rst}
.. automodule:: visbench.tasks.mid_level.occlusion_edge
   :members:
```

## `relative_depth`

**Not a registered probe.** Relative depth ordering was built as a candidate, measured against the `depth` board it subclasses, and rejected: Spearman between the two readouts is **+1.000** at 38% of the spread. It is kept, unregistered, because the rejection is a finding about a board that *does* ship — `depth` is ranking ordering plus feature resolution, not metric accuracy. Construct the class directly; its records are in `results/controls/`.

```{eval-rst}
.. automodule:: visbench.tasks.mid_level.relative_depth
   :members:
```
