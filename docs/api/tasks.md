# `visbench.tasks`

A probe. `BaseTask` is the whole interface — `fit`, `evaluate`, `predict`,
`describe` — and everything else here exists because a *dense* probe needs far
more than that and sixteen of them should not each reimplement it.

Subclass `DenseTrainingTask` for a new dense probe: it supplies feature sources
(in-memory or streaming), batching, head construction, probe3d's optimiser
schedule, the training loop and per-image metric averaging. A subclass supplies
`out_channels`, `_activate`, `_loss` and `_batch_metrics`, and little else.

## The package

```{eval-rst}
.. automodule:: visbench.tasks
   :no-members:
```

## The interface

```{eval-rst}
.. automodule:: visbench.tasks.base
   :members:
```

## Dense probes

Also home to `evaluate_oracle`, the recoverability gate: what a probe could score if the features contained the answer. It needs no backbone and no fitted head, so it costs one pass over a split rather than a board — and it has refused two candidate probes.

```{eval-rst}
.. automodule:: visbench.tasks.dense_base
   :members:
```

## Magnitude probes

`edge`, `keypoints2d` and `occlusion_edge` share every line of this, and `corner` is it under a different name. The `_activate` is the identity, and a test pins that: both ways of imposing non-negativity destroy the score.

```{eval-rst}
.. automodule:: visbench.tasks.magnitude_base
   :members:
```

## The schedule

probe3d's, shared by every trained probe here.

```{eval-rst}
.. automodule:: visbench.tasks.schedule
   :members:
```
