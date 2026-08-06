"""Structural tests for the scaffold.

These verify the *shape* of the package, not behaviour — every real function
still raises ``NotImplementedError``. They exist so that step 1 of the build
order is verifiable, and so an import cycle or a renamed module fails loudly
rather than at step 2.
"""

import importlib

import pytest

MODULES = [
    "visbench",
    "visbench.types",
    "visbench.registry",
    "visbench.backbones",
    "visbench.backbones.base",
    "visbench.backbones.pooling",
    "visbench.backbones.dinov2",
    "visbench.backbones.clip",
    "visbench.tasks",
    "visbench.tasks.base",
    "visbench.tasks.high_level",
    "visbench.tasks.high_level.classification",
    "visbench.tasks.high_level.retrieval",
    "visbench.tasks.mid_level",
    "visbench.tasks.mid_level.correspondence",
    "visbench.tasks.low_level",
    "visbench.heads",
    "visbench.cache",
    "visbench.data",
    "visbench.results",
    "visbench.metrics",
    "visbench.utils",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    """Every module imports cleanly — no cycles, no missing names."""
    importlib.import_module(module)


def test_public_api_exposed():
    """The two documented entry points exist on the package root."""
    import visbench

    assert hasattr(visbench, "get_backbone")
    assert hasattr(visbench, "get_probe")


def test_feature_modes_declared():
    """All three dense feature modes exist in the interface from v0.1."""
    from visbench.types import FEATURE_MODE_CHOICES

    assert set(FEATURE_MODE_CHOICES) == {
        "dense_only",
        "dense_cls_broadcast",
        "dense_plus_cls",
    }


def test_low_level_holds_its_tasks():
    """A documented placeholder from v0.1 until step 6d-1 filled it.

    Was `test_low_level_is_empty`, asserting the folder stayed a placeholder
    "until v0.3+". That is now the wrong assertion rather than a broken one, so
    it becomes its complement: the low-level probes are here, and the level each
    declares is what puts it in the third tier of the task taxonomy.
    """
    import visbench.tasks.low_level as low

    assert low.__all__ == ["CornerTask", "EdgeTask", "Keypoint2DTask"]
    assert low.EdgeTask.level == "low_level"
    assert low.Keypoint2DTask.level == "low_level"
    assert low.CornerTask.level == "low_level"


def test_the_occlusion_edge_probe_is_mid_level():
    """Its low-level twin shares every line of implementation with it.

    The two differ in what they read and therefore in which tier they belong to:
    recovering a depth discontinuity needs scene geometry, recovering an
    intensity one does not. Since the classes are otherwise identical, the level
    attribute is the only thing keeping the taxonomy honest.
    """
    from visbench.tasks.low_level import EdgeTask
    from visbench.tasks.mid_level import OcclusionEdgeTask

    assert OcclusionEdgeTask.level == "mid_level"
    assert EdgeTask.level == "low_level"
    assert OcclusionEdgeTask.protocol != EdgeTask.protocol
