# `visbench.backbones`

A frozen feature extractor with one method. `extract_features` returns **both**
the dense spatial grid and a pooled vector from the same forward pass, under the
identical signature and return shape for a ViT and a CNN — which is the whole
point, and the reason a probe can be written once and run against either.

**`CLIP` and `TimmBackbone` are documented at their own modules below, not on
the package.** They are served by a module `__getattr__` that imports the
optional extra on attribute access, so they are absent from
`dir(visbench.backbones)` and autodoc cannot see them there. That is deliberate:
a backbone whose extra is missing stays registered and stays listed, and
constructing it raises `ImportError: ... pip install visbench[clip]` rather than
the registry claiming the name is unknown.

## The package

```{eval-rst}
.. automodule:: visbench.backbones
   :no-members:
```

## The base class

```{eval-rst}
.. automodule:: visbench.backbones.base
   :members:
```

## Pooling and feature modes

How a pooled vector is derived from a dense grid, and the three ways a CLS token can reach a dense head.

```{eval-rst}
.. automodule:: visbench.backbones.pooling
   :members:
```

## DINOv2

```{eval-rst}
.. automodule:: visbench.backbones.dinov2
   :members:
```

## CLIP

Needs `visbench[clip]`.

```{eval-rst}
.. automodule:: visbench.backbones.clip
   :members:
```

## timm — CNNs and ViTs

Needs `visbench[timm]`. This class reads a model's own structure rather than assuming a CNN's, which is what makes a timm ViT usable *and* honest about whether it had a CLS token to keep.

```{eval-rst}
.. automodule:: visbench.backbones.timm_backbone
   :members:
```

## Your own `nn.Module`

The escape hatch. `hash_weights` keys the cache on the parameters, so a fine-tuned checkpoint cannot collide with the model it came from.

```{eval-rst}
.. automodule:: visbench.backbones.custom
   :members:
```
