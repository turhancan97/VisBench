# Backbones

Thirteen registered names, three families, one method. `extract_features`
returns **both** the dense spatial grid and a pooled vector from the same
forward pass, under the identical signature and return shape for a ViT and a
CNN — which is why a probe is written once and runs against either.

```python
import visbench
visbench.list_backbones()
```

| family | names | extra |
| --- | --- | --- |
| DINOv2 | `dinov2_vits14`, `dinov2_vitb14` | *(core)* |
| CLIP | `clip_vitb16`, `clip_vitb32` | `clip` |
| timm CNNs | `resnet18`, `resnet50`, `convnext_base` | `timm` |
| timm ViTs | `mae_vitb16`, `dino_vitb16`, `sam_vitb16`, `supervised_vitb16`, `siglip_vitb16` | `timm` |

`dinov2_vitb14_196` is also registered and is **not** a corpus column: it is
the same weights at 196px, a resolution control that exists to separate "finer
grid" from "better representation".

## A missing extra is not a missing backbone

A backbone whose extra is absent stays **registered and listed** —
`visbench list backbones` marks it — and constructing it raises
`ImportError: ... pip install visbench[clip]`. That is better than the registry
claiming the name is unknown, so do not "fix" it by importing at module scope.
`registry.missing_extra(name)` asks without importing.

## Pass a name, not an object

```python
result = visbench.run("dinov2_vitb14", "retrieval", dataset)
backbone = result.backbone          # the constructed object, if you need it
```

**`run()` seeds before it constructs.** Handing it a pre-built backbone fits the
head from a different RNG state while every recorded field — the seed included —
stays identical, so two runs disagree with nothing in the record to explain it.
That shipped once, in `--push-to`, and was found by publishing a whole board and
diffing it against the corpus: 20 of 26 records differed and the 6 that
reproduced were exactly the zero-shot probes, which train no head.

## Your own `nn.Module`

Any `nn.Module` works, without adding anything to this package:

```python
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
backbone = visbench.CustomBackbone(
    convnext_tiny(weights=weights).features,
    preprocess=weights.transforms(),
    name="convnext_tiny",
)
visbench.run(backbone, "retrieval", dataset)
```

The grid comes from the module's output shape, `embed_dim` from the first
forward pass, and the cache key from a hash of the weights — so a fine-tuned
checkpoint never reuses its parent's cached features. Where the output shape is
genuinely ambiguous VisBench raises rather than guesses; pass `patch_size=`,
`has_cls_token=` or a `feature_fn=` to say what it cannot infer.

To give a custom backbone a registry name, subclass `BaseBackbone` and apply
`@visbench.register_backbone("my_model")` — the same path the built-ins use.

[`examples/custom_backbone.py`](https://github.com/turhancan97/VisBench/blob/main/examples/custom_backbone.py)
runs all of this end to end and needs no dataset:

```bash
python examples/custom_backbone.py --finetune --register
```

`hash_weights()` keys the cache on the parameters, so a fine-tuned checkpoint
cannot collide with the model it was fine-tuned from.

One measured note, because the hazard sounds worse than it is: a wrapped model
is constructed **before** `run()` seeds rather than after, so nothing the caller
did with the RNG can reach the head — the wrapped path is *perfectly*
reproducible across seeds. Its numbers are comparable with other numbers from
the same wrapped model, and not with a registered backbone's to the last
decimal. [`examples/custom_backbone.py`](https://github.com/turhancan97/VisBench/blob/main/examples/custom_backbone.py) has
the measurement.

## Pooling is chosen by the probe, not the backbone

A task passes `pooling="cls"` or `"mean"` into `extract_features`; the backbone
executes whatever it is asked. `"default"` means the CLS token on a ViT and the
mean over patch tokens on a CNN — which is why a record stores pooling
**resolved**, since the literal word "default" does not say what produced the
number.

`default` is read from what a model hands its *own* classifier, not inferred
from whether a CLS token exists: MAE reports `token` and SigLIP-GAP reports
`avg`, so `default` means different things for two models of identical shape.
