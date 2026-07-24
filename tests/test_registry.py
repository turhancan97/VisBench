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


def test_no_probes_registered_until_step_three():
    assert visbench.list_probes() == []


def test_get_probe_fails_informatively():
    with pytest.raises(KeyError, match="Unknown task"):
        visbench.get_probe("classification")
