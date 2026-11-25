"""`publish_collection.py`'s probe list covers every probe that trains a head.

It was a fixed tuple of ten from the first push (2026-08-07), and seven later
trained probes never joined it, so the script refused to collect their heads.
Pinned here against the registry, the source neither the script nor its author
controls.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import visbench

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "publish_collection.py"


def _module():
    spec = importlib.util.spec_from_file_location("publish_collection", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_trained_probe_can_be_collected():
    module = _module()
    trained = set(visbench.list_probes()) - set(module.ZERO_SHOT_PROBES)
    assert set(module.TRAINED_PROBES) == trained
    assert len(module.TRAINED_PROBES) == len(set(module.TRAINED_PROBES))


def test_the_zero_shot_probes_are_registered():
    assert set(_module().ZERO_SHOT_PROBES) <= set(visbench.list_probes())
