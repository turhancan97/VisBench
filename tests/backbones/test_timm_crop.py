"""The timm double crop (22c), and the opt-in that measures it, without timm.

Every VisBench dataset resizes and centre-crops to the working size before a
backbone sees an image. A timm backbone then applies its checkpoint's eval
transform -- resize to ``size / crop_pct``, crop ``size`` -- so at the default
``crop_pct`` it crops a **second** time. The control re-runs the timm rows with
``crop_pct=1.0``.

Three things must stay true, and each fails silently rather than loudly:

- **the default key must not move**, or every published timm record stops
  matching its cache on every machine and every timm cell re-extracts;
- **an override must move the key**, or the full-frame run is served the
  published (cropped) features from cache and measures nothing;
- **an override must carry its own name**, or ``latest_per_backbone`` evicts
  the published row it was built to be compared against.

The fast suite has no timm (CI installs ``.[dev]``), so these exercise the pure
helper and a key computed on an instance built without ``__init__``. The
pixel-level claim -- that ``crop_pct=1.0`` is an exact no-op on a 224px image --
needs real timm transforms and is made by ``run_crop_control.py --check``.
"""

import inspect

import pytest

from visbench.backbones.timm_backbone import DEFAULT_CROP_PCT, TimmBackbone, eval_config

_CONVNEXT = {
    "input_size": (3, 224, 224),
    "interpolation": "bicubic",
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "crop_pct": 0.875,
    "crop_mode": "center",
}


def _keyed(crop_pct):
    """A backbone holding only what ``cache_key`` reads -- no timm, no weights."""
    backbone = TimmBackbone.__new__(TimmBackbone)
    backbone.model_name = "convnext_base"
    backbone.pretrained_tag = "fb_in1k"
    backbone.image_size = 224
    backbone.crop_pct = crop_pct
    return backbone


class TestEvalConfig:
    def test_no_override_is_the_checkpoints_own_config(self):
        """The published corpus was produced this way; it must not move."""
        assert eval_config(_CONVNEXT, None, None) == _CONVNEXT

    def test_the_override_replaces_only_the_crop_ratio(self):
        resolved = eval_config(_CONVNEXT, None, 1.0)
        assert resolved["crop_pct"] == 1.0
        assert {k: v for k, v in resolved.items() if k != "crop_pct"} == {
            k: v for k, v in _CONVNEXT.items() if k != "crop_pct"
        }

    def test_the_input_is_not_mutated(self):
        before = dict(_CONVNEXT)
        eval_config(_CONVNEXT, 256, 1.0)
        assert before == _CONVNEXT

    def test_an_image_size_override_still_works(self):
        assert eval_config(_CONVNEXT, 256, None)["input_size"] == (3, 256, 256)

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
    def test_a_crop_ratio_outside_the_unit_interval_is_refused(self, bad):
        with pytest.raises(ValueError, match="crop_pct"):
            eval_config(_CONVNEXT, None, bad)


class TestTheKey:
    def test_the_legacy_crop_keeps_the_pre_23a_key_byte_for_byte(self):
        """Pinned as a literal: every timm record before 23a carries it, and
        ``crop_pct=None`` is how those records stay reproducible."""
        assert _keyed(None).cache_key() == "timm/convnext_base/fb_in1k/224"

    def test_the_default_key_is_the_one_supersession_names(self):
        from visbench.results.leaderboard import SUPERSEDED_KEYS

        assert _keyed(DEFAULT_CROP_PCT).cache_key() == SUPERSEDED_KEYS[_keyed(None).cache_key()]

    def test_the_full_frame_key_differs_from_the_legacy_one(self):
        assert _keyed(1.0).cache_key() == "timm/convnext_base/fb_in1k/224/crop1"
        assert _keyed(1.0).cache_key() != _keyed(None).cache_key()

    def test_two_overrides_do_not_collide(self):
        assert _keyed(1.0).cache_key() != _keyed(0.95).cache_key()


def test_the_default_is_the_full_frame():
    """23a: preprocessing is a no-op on a dataset-cropped image unless asked."""
    assert DEFAULT_CROP_PCT == 1.0
    default = inspect.signature(TimmBackbone.__init__).parameters["crop_pct"].default
    assert default == DEFAULT_CROP_PCT


@pytest.mark.parametrize("crop_pct", [None, 0.875])
def test_a_non_default_crop_without_a_name_is_refused_before_any_download(crop_pct):
    """The guard fires before timm is imported, so it is fast and offline.

    ``None`` is included deliberately: the legacy crop reproduces published
    records, and under the variant's own name it would collide with the
    corrected row that now carries that name.
    """
    with pytest.raises(ValueError, match="distinct name"):
        TimmBackbone(variant="convnext_base", crop_pct=crop_pct)
