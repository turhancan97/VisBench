# `visbench.tasks`

A probe. `BaseTask` is the whole interface — `fit`, `evaluate`, `predict`,
`describe` — and everything else on this page exists because a *dense* probe
needs far more than that and sixteen of them should not each reimplement it.

## The interface

```{eval-rst}
.. automodule:: visbench.tasks.base
   :members:
```
