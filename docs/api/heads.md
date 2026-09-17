# `visbench.heads`

What maps features to a task output. Pluggable from v0.2, never hardcoded: a
head declares which feature modes it consumes and rejects a mismatch at
construction rather than mid-training.

**Report a headline dense number with `LinearHead`.** A deeper head can
compensate for a weak feature map and narrow the very gap a probe exists to
measure — measured, not asserted: across five probes and nine ViTs a DPT head
reorders 24 of 174 separable pairs and changes the leader on two of five boards.

## The package

```{eval-rst}
.. automodule:: visbench.heads
   :no-members:
```

## Registration

```{eval-rst}
.. automodule:: visbench.heads.base
   :members:
```

## Linear

One 1x1 convolution per patch, then a bilinear upsample — which is literally what `evaluate_oracle` models.

```{eval-rst}
.. automodule:: visbench.heads.linear
   :members:
```

## DPT

Genuinely multiscale, and the reason multi-layer extraction exists. It refuses a single feature map rather than duplicating it.

```{eval-rst}
.. automodule:: visbench.heads.dpt
   :members:
```

## Detection

```{eval-rst}
.. automodule:: visbench.heads.detection
   :members:
```

## Pose

The one head here that is deliberately **not** linear, and the only one that reads *pooled* vectors rather than a feature map. A single affine map cannot express pairwise pose: it underfits, clears the no-feature floor by 2.7-3.5 degrees where this head clears it by 24.5-42.6, and produces an ordering that nearly inverts this one's top two. The linear run is kept as a committed control rather than as an argument.

```{eval-rst}
.. automodule:: visbench.heads.pose
   :members:
```
