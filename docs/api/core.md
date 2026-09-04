# `visbench`

The five calls almost every use of this library goes through, and the three
modules behind them.

`run()` is the one that matters: it seeds, resolves a backbone name, extracts
features through the cache, fits the probe, evaluates it, and writes a record
saying exactly what produced the number. Everything else on this page is
something `run()` does for you.

**Pass a backbone by name, not as an object.** `run()` seeds *before* it
constructs, so handing it a pre-built backbone fits the head from a different
RNG state while every recorded field — the seed included — stays identical. Take
the constructed object back off `RunResult.backbone` if you need it.

## The top-level API

```{eval-rst}
.. automodule:: visbench
   :members: get_backbone, get_probe, list_backbones, list_probes, run
```

## `visbench.runner`

What `run()` actually does, in order, and why each step is where it is.

```{eval-rst}
.. automodule:: visbench.runner
   :no-members:
```

## `visbench.registry`

Name to class, for backbones, probes and heads. `register_backbone` and
`register_task` take `name` **positional-only**, so a decorator argument cannot
shadow a constructor parameter of the same name — which is how a resolution
control once reported itself under the name of the row it was built to be
compared against.

```{eval-rst}
.. automodule:: visbench.registry
   :no-members:
```

## `visbench.demo`

What `visbench demo` runs: generated shapes and a wrapped ResNet-18, through the
same `run()`, the same cache and the same record as everything else. Its score is
deliberately not 1.0, and colour deliberately carries no information — a first
pass with fixed colours scored a flat 1.0, which would have demonstrated a
colour shortcut while claiming to demonstrate shape recognition.

```{eval-rst}
.. automodule:: visbench.demo
   :members:
```

## `visbench.types`

The shapes that cross module boundaries.

```{eval-rst}
.. automodule:: visbench.types
   :no-members:
```
