"""Registry behaviour.

The registry is the reason ``get_backbone("dinov2_vitb14")`` works without
``__init__.py`` knowing about DINOv2. Its sharp edge is that one class may hold
several names, so most of these tests are about names not bleeding into each
other.
"""

import pytest

import visbench
from visbench import registry


def test_dinov2_variants_are_both_registered():
    assert {"dinov2_vits14", "dinov2_vitb14"} <= set(visbench.list_backbones())


def test_list_backbones_is_sorted():
    names = visbench.list_backbones()
    assert names == sorted(names)


def test_unknown_backbone_error_lists_known_names():
    with pytest.raises(KeyError) as excinfo:
        visbench.get_backbone("dinov3_vitl14")
    assert "dinov2_vitb14" in str(excinfo.value)


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):

        @registry.register_backbone("dinov2_vitb14")
        class Impostor:
            pass


def test_registered_defaults_are_overridable():
    """Registered kwargs are defaults, not constraints."""
    cls, defaults = registry._BACKBONES["dinov2_vits14"]
    assert defaults == {"variant": "dinov2_vits14"}

    merged = {**defaults, "variant": "dinov2_vitb14"}
    assert merged["variant"] == "dinov2_vitb14"


def test_registration_does_not_write_class_name():
    """Two names on one class: a class-level ``name`` would let the last win.

    The instance sets its own ``name``, so ``cache_key`` cannot silently report
    the wrong variant — which would serve one model's cached features as
    another's.
    """
    from visbench.backbones.dinov2 import DINOv2

    assert DINOv2.name == ""
    assert registry._BACKBONES["dinov2_vits14"][0] is registry._BACKBONES["dinov2_vitb14"][0]


class TestImportTolerance:
    """A missing extra hides one name; a real bug must not hide as one.

    The failure mode being guarded: a typo'd import inside dinov2.py used to
    surface as "Unknown backbone 'dinov2_vitb14'", which sends the reader into
    the registry looking for a registration bug that isn't there.
    """

    def _arrange(self, monkeypatch, optional_dependency, missing_name):
        """Point the registry at one module whose import fails on ``missing_name``."""
        import importlib

        module = "visbench._fake_module"
        monkeypatch.setattr(registry, "_REGISTRATION_MODULES", {module: optional_dependency})
        monkeypatch.setattr(registry, "_IMPORTED", False)

        def fake_import_module(name, *args, **kwargs):
            if name == module:
                raise ImportError(f"No module named {missing_name!r}", name=missing_name)
            return importlib.__dict__["__original_import_module"](name, *args, **kwargs)

        importlib.__dict__["__original_import_module"] = importlib.import_module
        monkeypatch.setattr(importlib, "import_module", fake_import_module)

    def test_missing_optional_dependency_is_skipped(self, monkeypatch):
        self._arrange(monkeypatch, optional_dependency="open_clip", missing_name="open_clip")
        registry._ensure_imported()  # must not raise

    def test_missing_submodule_of_optional_dep_is_skipped(self, monkeypatch):
        self._arrange(monkeypatch, optional_dependency="open_clip", missing_name="open_clip.model")
        registry._ensure_imported()  # must not raise

    def test_real_import_error_propagates(self, monkeypatch):
        """A typo'd import inside the module is a bug, not a missing extra."""
        self._arrange(monkeypatch, optional_dependency="open_clip", missing_name="torhc")
        with pytest.raises(ImportError, match="real import error"):
            registry._ensure_imported()

    def test_module_without_optional_dep_must_import(self, monkeypatch):
        self._arrange(monkeypatch, optional_dependency=None, missing_name="open_clip")
        with pytest.raises(ImportError, match="real import error"):
            registry._ensure_imported()


def test_retrieval_is_registered():
    assert "retrieval" in visbench.list_probes()


def test_get_probe_builds_a_task():
    probe = visbench.get_probe("retrieval")
    assert probe.name == "retrieval"
    assert probe.level == "high_level"


def test_get_probe_forwards_kwargs():
    probe = visbench.get_probe("retrieval", topk=(1, 3), metric="l2")
    assert probe.topk == (1, 3)
    assert probe.metric == "l2"


def test_all_v01_tasks_are_registered():
    """The v0.1 task list from CLAUDE.md, complete."""
    assert set(visbench.list_probes()) >= {"classification", "retrieval", "correspondence"}


def test_depth_is_registered():
    """The first dense task, v0.2."""
    assert "depth" in visbench.list_probes()
    assert visbench.get_probe("depth").level == "mid_level"


def test_surface_normal_is_registered():
    """The second dense task, v0.2."""
    assert "surface_normal" in visbench.list_probes()
    assert visbench.get_probe("surface_normal").level == "mid_level"


def test_generic_segmentation_is_registered():
    """The third dense task, v0.2 — mid-level, not high-level: figure-ground
    separation without naming either side."""
    assert "generic_segmentation" in visbench.list_probes()
    assert visbench.get_probe("generic_segmentation").level == "mid_level"


def test_semantic_segmentation_is_registered():
    """The fourth dense task, v0.2 — the multi-class counterpart to the binary one.

    ``num_classes`` is required, so this cannot be built bare: a default would
    silently size the head for someone else's dataset.
    """
    assert "semantic_segmentation" in visbench.list_probes()
    assert visbench.get_probe("semantic_segmentation", num_classes=21).level == "high_level"
    with pytest.raises(TypeError):
        visbench.get_probe("semantic_segmentation")


def test_unregistered_task_fails_informatively():
    """The error must list what does exist.

    ``detection`` is v0.3 and the longest-lived remaining stub; when it lands,
    point this at whatever is still a stub rather than deleting it — the check
    is that an unknown name is *helpfully* rejected, not which name.
    """
    with pytest.raises(KeyError, match="Unknown task") as excinfo:
        visbench.get_probe("detection")
    assert "correspondence" in str(excinfo.value)
