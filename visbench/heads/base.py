"""Pluggable task-head interface.

Dense task heads must be selectable per run, never hardcoded to one
architecture (CLAUDE.md, v0.2). This is a known extension point for
contributors, so the interface is intentionally small: declare which feature
modes you consume, and map features to an output.

Heads are registered by name for the same reason backbones and tasks are — a
task run should be able to say ``head="dpt"`` without importing anything.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from visbench.types import FEATURE_MODE_CHOICES

__all__ = ["BaseHead", "register_head", "get_head", "build_head", "list_heads"]

_HEADS: dict[str, type] = {}


def register_head(name: str) -> Callable[[type], type]:
    """Class decorator registering a :class:`BaseHead` subclass under ``name``.

    Raises on a duplicate rather than overwriting, so a contributor's typo
    cannot shadow the linear baseline every comparison depends on.
    """

    def decorator(cls: type) -> type:
        if name in _HEADS:
            existing = _HEADS[name]
            raise ValueError(
                f"Head name {name!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        _HEADS[name] = cls
        return cls

    return decorator


def _ensure_imported() -> None:
    """Import the head modules so their decorators have run."""
    import importlib

    for module in ("visbench.heads.linear", "visbench.heads.dpt", "visbench.heads.detection"):
        importlib.import_module(module)


def get_head(name: str) -> type:
    """Look up a registered head class."""
    _ensure_imported()
    if name not in _HEADS:
        known = ", ".join(sorted(_HEADS)) or "(none registered)"
        raise KeyError(f"Unknown head {name!r}. Available: {known}")
    return _HEADS[name]


def build_head(name: str, **kwargs: Any) -> "BaseHead":
    """Instantiate a registered head by name."""
    return get_head(name)(**kwargs)  # type: ignore[no-any-return]


def list_heads() -> list[str]:
    """Names accepted by :func:`get_head`."""
    _ensure_imported()
    return sorted(_HEADS)


class BaseHead(nn.Module, ABC):
    """Maps backbone features to a task output.

    A head declares which feature modes it accepts; a task requesting an
    unsupported mode should fail at construction, not mid-training.
    """

    #: Feature modes this head can consume, from :class:`FeatureMode`.
    supported_feature_modes: tuple[str, ...] = ()

    #: Whether the head needs features from several backbone depths. A head
    #: that says yes cannot be fed a single layer — see :class:`DPTHead`.
    multiscale: bool = False

    def __init__(self) -> None:
        """A head is an ordinary trainable :class:`torch.nn.Module`.

        Defined only so that it carries a docstring. ``autoclass_content =
        "both"`` appends ``__init__``'s documentation to the class's, and
        Sphinx filters exactly one inherited default — :meth:`object.__init__`
        — so without this the rendered page ends with *"Initialize internal
        Module state, shared by both nn.Module and ScriptModule"*, inherited
        from :class:`torch.nn.Module`. Turning off
        ``autodoc_inherit_docstrings`` would fix it and cost far more: sixteen
        probe classes rely on inheriting :class:`BaseTask`'s method
        documentation, and an un-inherited method is dropped from its page
        entirely rather than merely left undocumented.
        """
        super().__init__()

    @abstractmethod
    def forward(self, features) -> torch.Tensor:
        """Produce the task output from backbone features.

        ``features`` is a dense tensor for most modes, or a
        ``(dense, cls)`` tuple under ``dense_plus_cls``.
        """
        raise NotImplementedError

    @classmethod
    def check_feature_mode(cls, feature_mode: str) -> None:
        """Raise unless this head can consume ``feature_mode``.

        Called at task construction. The alternative is a shape error partway
        through training, by which point the traceback points at a conv layer
        rather than at the incompatible pair of choices that caused it.
        """
        if feature_mode not in FEATURE_MODE_CHOICES:
            raise ValueError(
                f"Unknown feature_mode {feature_mode!r}; expected one of {FEATURE_MODE_CHOICES}"
            )
        if feature_mode not in cls.supported_feature_modes:
            raise ValueError(
                f"{cls.__name__} does not accept feature_mode={feature_mode!r}; "
                f"it supports {cls.supported_feature_modes}."
            )

    @staticmethod
    def _resize(tensor: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
        """Bilinearly resize a ``(B, C, H, W)`` map, or pass it through.

        ``align_corners=False`` throughout: it is the convention that keeps a
        resized prediction aligned with the pixel grid it is scored against, and
        mixing conventions between a head and its metric shifts every value by
        half a pixel.
        """
        if size is None or tuple(tensor.shape[-2:]) == tuple(size):
            return tensor
        return nn.functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
