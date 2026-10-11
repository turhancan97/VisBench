"""The double-crop control's script and array agree, and it cannot reach a board.

Two files define this control's matrix, which is the shape that left the corpus
array short by a probe for a whole release (10b): each was self-consistent and
they disagreed. So the sbatch array is read here and compared against the
script, which reads `_VARIANTS` -- a source neither file controls.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

import visbench

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_crop_control.py"
SBATCH = ROOT / "slurm" / "crop_control.sbatch"


@pytest.fixture(scope="module")
def control():
    spec = importlib.util.spec_from_file_location("run_crop_control", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _array(name: str) -> list[str]:
    text = SBATCH.read_text(encoding="utf-8")
    body = re.search(rf"^{name}=\(([^)]*)\)", text, re.MULTILINE)
    assert body, f"no {name}=( ... ) in {SBATCH.name}"
    return body.group(1).replace("\\", " ").split()


def test_the_array_covers_every_timm_column(control):
    assert sorted(_array("BACKBONE_LIST")) == sorted(control.TIMM_COLUMNS)


def test_the_array_probes_exist():
    assert set(_array("PROBES")) <= set(visbench.list_probes())


def test_importing_the_script_registers_nothing(control):
    """Full-frame rows exist only inside a run, never in the listed registry."""
    assert not [b for b in visbench.list_backbones() if b.endswith(control.SUFFIX)]


def test_the_corpus_is_refused_as_a_destination(control):
    with pytest.raises(SystemExit):
        control.main(["depth", "--results", str(control.CORPUS)])


def test_a_non_timm_backbone_is_refused(control):
    with pytest.raises(SystemExit):
        control.main(["depth", "--backbones", "dinov2_vits14", "--results", "x.jsonl"])


def test_the_analysis_names_the_same_timm_columns(control):
    spec = importlib.util.spec_from_file_location(
        "analyse_crop_control", ROOT / "scripts" / "analyse_crop_control.py"
    )
    assert spec and spec.loader
    analysis = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analysis
    spec.loader.exec_module(analysis)
    assert sorted(analysis.TIMM_COLUMNS) == sorted(control.TIMM_COLUMNS)
