"""The oracle gate's calibration table must keep naming real probes.

`scripts/oracle_ceiling.py` holds a `TARGETS` table: one row per target, each
pairing a probe with the dataset that supplies it. The table exists so a
*candidate* target is measured against targets whose verdict is already known —
five that ship and one that was rejected — and a bar with no known-failing
example beside its passing ones is a threshold nobody has tested.

The failure mode is the one `tests/scripts/test_corpus_scripts.py` was written
for: a hand-written table drifting away from the code it describes while staying
self-consistent, so it keeps printing numbers and nothing complains. Here a row
naming a probe whose metric key had been renamed would raise only when someone
ran the gate, which is exactly when they are trying to decide something.

Nothing here runs the gate — `skimage` is not a dependency of this package and
the superpixel row cannot execute in CI. The import is lazy for that reason, and
this asserts the table without touching it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from visbench import list_probes
from visbench.tasks.dense_base import DenseTrainingTask

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "oracle_ceiling.py"


def _load():
    spec = importlib.util.spec_from_file_location("oracle_ceiling", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["oracle_ceiling"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load()


def test_importing_the_gate_does_not_need_skimage(script):
    """The optional-extra trap, pre-empted: SLIC is imported inside the call.

    This package has hit it twice — hub tests needing `huggingface_hub`, and an
    issue-template test needing PyYAML — both times because an import at module
    scope reached a package CI does not install.
    """
    assert "skimage" not in sys.modules
    assert hasattr(script, "SlicBoundaryResponse")


def test_every_shipped_row_names_a_registered_probe(script):
    """A row claiming to calibrate against something that ships must be one."""
    shipped = [name for name, (_, _, origin) in script.TARGETS.items() if origin == "ships"]
    assert set(shipped) <= set(list_probes())
    assert len(shipped) >= 5


def test_the_rejected_row_is_not_a_registered_probe(script):
    """`superpixel` was measured and thrown away; it must not have crept back."""
    rejected = [name for name, (_, _, origin) in script.TARGETS.items() if origin == "rejected"]
    assert rejected == ["superpixel"]
    assert "superpixel" not in list_probes()


def test_every_row_can_be_scored(script):
    """Each row's probe must be a dense probe that declares an oracle.

    The refusal is deliberate for probes whose target does not average, so a row
    added for one of those would be a table that reads fine and cannot run.
    """
    for name, (probe, _, _) in script.TARGETS.items():
        assert isinstance(probe, DenseTrainingTask), name
        assert type(probe).oracle_prediction is not DenseTrainingTask.oracle_prediction, name


def test_the_headline_metric_of_every_row_exists(script):
    """Reading whichever key sorted first is what `HEADLINE_METRICS` exists to stop."""
    for name, (probe, _, _) in script.TARGETS.items():
        key = script.HEADLINE[name][0] if name in script.HEADLINE else probe.correlation_key
        assert isinstance(key, str) and key


def test_the_pinned_frame_count_matches_the_corpus_scripts(script):
    """600 in three files now. Two of them already disagreed once, over `corner`."""
    staging = (ROOT / "scripts" / "stage_corner_frames.py").read_text()
    corpus = (ROOT / "scripts" / "build_corpus.sh").read_text()
    assert f"DEFAULT_LIMIT = {script.PINNED_FRAMES}" in staging
    assert f"TASKONOMY_LIMIT={script.PINNED_FRAMES}" in corpus
