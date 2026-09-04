# `visbench.hub`

Serialising a trained head with the backbone identity beside it, and moving it
to and from the Hugging Face Hub. Pushing and pulling need `visbench[hub]`;
saving and loading a local path do not.

**A head is only meaningful against the features it was fitted on**, so the
artifact records four identity fields and `load_probe` refuses a mismatch. That
is not defensive coding for its own sake: the same head loaded under the wrong
pooling scores 0.9620 against 0.9895 on Imagenette, and neither number looks
wrong on its own.

`weights_only=True` on every load, and nothing enters the payload that needs
unpickling to reconstruct — these are fetched from a hub, so an unrestricted
`torch.load` is arbitrary code execution.

## The package

```{eval-rst}
.. automodule:: visbench.hub
   :no-members:
```

## The artifact

```{eval-rst}
.. automodule:: visbench.hub.artifact
   :members:
```

## Push and pull

`push_probe` defaults to `private=True`: a push is not reversible the way a local write is.

```{eval-rst}
.. automodule:: visbench.hub.remote
   :members:
```
