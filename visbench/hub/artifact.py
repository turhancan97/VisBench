"""Saving and loading a trained probe head, with the backbone it belongs to.

A probe head is only meaningful against the exact features it was fitted on. A
linear layer trained on DINOv2-S CLS tokens will happily consume DINOv2-S
*mean-pooled* tokens — same rank, same width, no error — and produce confident
nonsense. So an artifact that stores weights alone is not shareable; what makes
it shareable is everything recorded beside them, and a loader that refuses a
mismatch instead of trusting the caller.

Four things are checked on load, and each one is a way to be silently wrong:

- ``backbone_key`` — the weights and resolution, from
  :meth:`BaseBackbone.cache_key`. Two DINOv2-S checkpoints have identical
  shapes and different features; a fine-tuned one differs from its parent in no
  other visible way.
- ``pooling`` — resolved, so ``cls`` and ``mean`` cannot be confused. This is
  the case above, and it is the one that raises nothing on its own.
- ``feature_mode`` — ``dense_cls_broadcast`` doubles the channel width, so a
  mismatch here usually *does* raise. Checked anyway, since "usually" is doing
  a lot of work when the head is a ``1x1`` convolution.
- ``layers`` — a head fitted on block 11 fed block 5 has the right shape and
  the wrong depth.

The file is a ``torch.save`` of tensors and JSON primitives, and it is read back
with ``weights_only=True``. That is not a detail: step 6e-5 fetches these from a
hub, and an unrestricted ``torch.load`` on a downloaded file is arbitrary code
execution. Nothing here may put an object in the payload that requires
unpickling to reconstruct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from visbench.heads import build_head

__all__ = [
    "ARTIFACT_VERSION",
    "IncompatibleProbe",
    "load_probe",
    "probe_metadata",
    "save_probe",
]

#: Bumped when the payload layout changes in a way an older loader would
#: misread. Separate from ``SCHEMA_VERSION``, which versions *result records* --
#: the two move for unrelated reasons, and tying them would force a needless
#: migration on one side every time the other changed.
ARTIFACT_VERSION = 1

#: Fields that must agree between an artifact and the backbone it is loaded
#: against. Listed rather than "everything in the metadata": the corpus digest,
#: the metrics and the timestamp describe *provenance* and must not gate a load.
IDENTITY_FIELDS = ("backbone_key", "pooling", "feature_mode", "layers")


class IncompatibleProbe(ValueError):
    """A saved probe does not match the backbone it is being loaded against."""


def _rebuild_head(spec: dict) -> nn.Module:
    """Reconstruct an unfitted head from :meth:`BaseTask.head_spec`."""
    kind = spec.get("kind")
    if kind == "registered":
        return build_head(spec["name"], **spec["kwargs"])
    if kind == "linear":
        return nn.Linear(int(spec["in_features"]), int(spec["out_features"]))
    raise IncompatibleProbe(
        f"Unknown head kind {kind!r}. This artifact was written by a newer "
        "VisBench than the one loading it."
    )


def probe_metadata(task: Any, backbone: Any) -> dict:
    """Everything that decides whether these weights mean anything.

    Split out from :func:`save_probe` so a caller can inspect what would be
    written — and so the identity fields have exactly one definition, rather
    than one at save time and a second, drifting one at load time.
    """
    described = task.describe()
    resolved = getattr(backbone, "default_pooling", lambda: task.pooling)()
    pooling = resolved if task.pooling == "default" else task.pooling
    return {
        "artifact_version": ARTIFACT_VERSION,
        "backbone": backbone.name,
        "backbone_key": backbone.cache_key(),
        "task": described["task"],
        "level": described["level"],
        "pooling": pooling,
        "pooling_requested": str(task.pooling),
        "feature_mode": described["feature_mode"],
        "layers": list(task.layers) if task.layers is not None else None,
        "task_params": described["task_params"],
    }


def save_probe(
    task: Any,
    path: str | Path,
    *,
    backbone: Any,
    notes: str | None = None,
) -> Path:
    """Write ``task``'s trained head, and the identity it is only valid against.

    Parameters
    ----------
    task:
        A **fitted** probe. An unfitted one is refused rather than written
        empty: an artifact holding an untrained head is indistinguishable from
        one holding a trained head that happened to learn nothing.
    backbone:
        The backbone the head was fitted on. Required, and positional-only by
        keyword, because the whole point of the artifact is that the two travel
        together — a default here would let a caller write an unusable file
        without noticing.

    Raises
    ------
    ValueError
        If the probe is zero-shot, or has not been fitted.
    """
    path = Path(path)
    if getattr(task, "zero_shot", False):
        raise ValueError(
            f"{task.name!r} is zero-shot: it trains nothing, so there is no head "
            "to share. Nothing about the probe needs distributing -- the "
            "backbone alone reproduces it."
        )

    spec = task.head_spec()
    head = getattr(task, "head", None)
    if spec is None or head is None:
        raise ValueError(
            f"{task.name!r} has not been fitted, so its head has no weights. "
            "Call fit() before saving: an artifact holding an untrained head "
            "cannot be told apart from one that learned nothing."
        )

    payload = {
        "meta": probe_metadata(task, backbone),
        "head_spec": spec,
        "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "probe_state": {k: v.detach().cpu() for k, v in task.probe_state().items()},
    }
    if notes:
        payload["meta"]["notes"] = notes

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_probe(
    path: str | Path,
    *,
    backbone: Any,
    task: Any | None = None,
    strict: bool = True,
) -> Any:
    """Load a saved head onto ``task``, refusing a backbone it was not fitted on.

    Parameters
    ----------
    task:
        A probe of the same kind, constructed with the same hyperparameters.
        Built here from the registry when omitted — but a probe whose
        constructor took arguments that shape the head (``num_classes``,
        ``epochs``, a custom ``head_kwargs``) should be passed in, since the
        registry default may not match what the artifact expects.
    strict:
        When ``False``, an identity mismatch warns instead of raising. Provided
        because deliberately probing how far a head transfers is a legitimate
        experiment — but it is off by default, and a number produced this way
        is not comparable with anything, because ``run()`` would record the
        backbone actually used and nothing would say the head came from
        another.

    Raises
    ------
    IncompatibleProbe
        If the artifact was written by a newer format, or its identity does not
        match ``backbone`` and ``strict`` is set.
    """
    import warnings

    import visbench

    # weights_only: this file may have come from a hub. See the module
    # docstring -- an unrestricted load here is remote code execution.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)

    meta = payload["meta"]
    version = meta.get("artifact_version")
    if not isinstance(version, int) or version > ARTIFACT_VERSION:
        raise IncompatibleProbe(
            f"Artifact version {version!r} is newer than this VisBench "
            f"understands (v{ARTIFACT_VERSION}). Upgrade rather than guessing "
            "at a layout that may have moved."
        )

    if task is None:
        task = visbench.get_probe(meta["task"])

    expected = probe_metadata(task, backbone)
    mismatched = {
        field: (meta.get(field), expected[field])
        for field in IDENTITY_FIELDS
        if meta.get(field) != expected[field]
    }
    if mismatched:
        detail = "; ".join(
            f"{field}: artifact {saved!r} vs this run {current!r}"
            for field, (saved, current) in mismatched.items()
        )
        message = (
            f"This head was fitted on {meta['backbone']!r} and does not match "
            f"what it is being loaded against -- {detail}. The shapes may still "
            "line up, in which case it would score without complaint against "
            "features it has never seen."
        )
        if strict:
            raise IncompatibleProbe(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    head = _rebuild_head(payload["head_spec"])
    head.load_state_dict(payload["head_state"])
    head.eval()
    task.head = head.to(getattr(task, "device", "cpu"))
    task.load_probe_state({k: v for k, v in payload["probe_state"].items()})
    return task
