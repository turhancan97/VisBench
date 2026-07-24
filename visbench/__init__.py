"""VisBench — probing vision backbones across CV tasks through a small API.

Two entry points carry the whole library::

    backbone = visbench.get_backbone("dinov2_vitb14")
    probe    = visbench.get_probe("classification")

Sibling project to vismatch (https://github.com/gmberton/vismatch): same
ergonomic philosophy, applied to representation probing instead of matching.

Task categorization follows Chen, Marks & Cheng, "Probing the Mid-level Vision
Capabilities of Self-Supervised Learning" (arXiv:2411.17474). Evaluation
protocols for depth, surface normal and correspondence follow probe3d
(El Banani et al., CVPR 2024, arXiv:2404.08476).
"""

from typing import Any

from visbench import registry
from visbench.backbones.custom import CustomBackbone
from visbench.registry import register_backbone, register_task
from visbench.types import FeatureDict, FeatureMode, Pooling

__version__ = "0.1.0"

__all__ = [
    "get_backbone",
    "get_probe",
    "list_backbones",
    "list_probes",
    "run",
    "register_backbone",
    "register_task",
    "CustomBackbone",
    "FeatureDict",
    "FeatureMode",
    "Pooling",
    "__version__",
]


def get_backbone(name: str, **kwargs: Any) -> "Any":
    """Instantiate a registered backbone by name.

    Parameters
    ----------
    name:
        Registered backbone name, e.g. ``"dinov2_vitb14"``. See
        :func:`list_backbones`.
    **kwargs:
        Forwarded to the backbone constructor (device, weights variant, ...).

    Returns
    -------
    BaseBackbone
        A frozen, eval-mode backbone exposing ``.extract_features()``.
    """
    return registry.build_backbone(name, **kwargs)


def get_probe(name: str, **kwargs: Any) -> "Any":
    """Instantiate a registered task/probe by name.

    Parameters
    ----------
    name:
        Registered task name, e.g. ``"classification"``, ``"retrieval"``,
        ``"correspondence"``. See :func:`list_probes`.
    **kwargs:
        Forwarded to the task constructor (head type, pooling override, ...).

    Returns
    -------
    BaseTask
        A task exposing ``.fit()`` / ``.predict()`` / ``.evaluate()``.

    Notes
    -----
    No task is registered yet — tasks land at build step 3, so this currently
    raises ``KeyError`` for every name.
    """
    return registry.build_task(name, **kwargs)


def list_backbones() -> list[str]:
    """Names accepted by :func:`get_backbone`."""
    return registry.list_backbones()


def list_probes() -> list[str]:
    """Names accepted by :func:`get_probe`."""
    return registry.list_tasks()


def run(*args: Any, **kwargs: Any) -> Any:
    """Backbone + task + data -> scored, logged result. See :mod:`visbench.runner`.

    Imported lazily so that ``import visbench`` stays cheap; the module pulls in
    the cache and result writer, which a caller reaching only for
    ``get_backbone`` does not need.
    """
    from visbench.runner import run as _run

    return _run(*args, **kwargs)
