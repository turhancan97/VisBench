# `visbench.tasks.high_level`

Semantic and category understanding. Six probes — and three of them share one
implementation while asking three different questions: `classification` is
basic-level, `scene_classification` is place, `fine_grained_classification` is
subordinate. Each is a distinct probe *name* because a board is keyed on the
task name, so a second dataset under one name makes that board unrenderable
rather than merely mixed.

## The `high_level` package

```{eval-rst}
.. automodule:: visbench.tasks.high_level
   :members:
```

## `classification`

```{eval-rst}
.. automodule:: visbench.tasks.high_level.classification
   :members:
```

## `scene_classification`

```{eval-rst}
.. automodule:: visbench.tasks.high_level.scene_classification
   :members:
```

## `fine_grained_classification`

```{eval-rst}
.. automodule:: visbench.tasks.high_level.fine_grained_classification
   :members:
```

## `retrieval`

```{eval-rst}
.. automodule:: visbench.tasks.high_level.retrieval
   :members:
```

## `semantic_segmentation`

```{eval-rst}
.. automodule:: visbench.tasks.high_level.semantic_segmentation
   :members:
```

## `detection`

```{eval-rst}
.. automodule:: visbench.tasks.high_level.detection
   :members:
```
