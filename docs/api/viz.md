# `visbench.viz`

Drawing what a probe saw beside what it predicted. This package exists because
of this project's own bug history — a correspondence misalignment that scored
`recall@1px = 0.003`, and VOC palette PNGs read through `convert("L")` that
turned classes `[0, 1, 15, 255]` into `[0, 38, 147, 220]`. Both were found by
reading code. Both are obvious in one rendered frame.

**A viewer that applies its own geometry is worse than no viewer**, and that is
the single rule here. A panel's entire evidential content is whether the image
and the target line up, so a viewer that resizes for layout or re-crops can make
a misaligned pipeline look fine and a correct one look broken.

## The package

```{eval-rst}
.. automodule:: visbench.viz
   :no-members:
```

## Styles

One row per drawable probe, and `style_for` **raises** on an unlisted one. There are four validity conventions and none of them is visible in a tensor's shape or dtype, so a "scalar map, mask the zeros" default is right for depth and silently wrong for the four probes where 0 is a real reading.

```{eval-rst}
.. automodule:: visbench.viz.styles
   :members:
```

## Colour

```{eval-rst}
.. automodule:: visbench.viz.colour
   :members:
```

## Panels

```{eval-rst}
.. automodule:: visbench.viz.panels
   :members:
```

## Matches

Two views and the errors between them. `error_coherence` is the number that separates a weak backbone from a broken pipeline, which a median error cannot do.

```{eval-rst}
.. automodule:: visbench.viz.matches
   :members:
```

## Decisions rather than maps

The probes whose answer is a choice among images. Each states the diagnostic its own history calls for — `class_balance` is the prefix bug as a figure, `vote_balance` the CSV-column bug.

```{eval-rst}
.. automodule:: visbench.viz.gallery
   :members:
```
