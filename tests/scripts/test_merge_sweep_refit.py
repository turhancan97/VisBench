"""23b's sweep merge replaces re-fitted columns and refuses a sweep with holes.

`scripts/merge_sweep_refit.py` is the one place a committed sweep is rewritten
rather than appended to, so its refusals are the guard: a sweep short of a
column or a seed still parses, still analyses, and silently describes fewer
pairs than the board has.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "merge_sweep_refit.py"


@pytest.fixture(scope="module")
def merge():
    spec = importlib.util.spec_from_file_location("merge_sweep_refit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = "timm/resnet18/a1_in1k/224"


def _row(backbone: str, seed: int, key: str, score: float = 0.5) -> dict:
    return {"task": "edge", "backbone": backbone, "backbone_key": key, "seed": seed, "x": score}


def _sweep(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> list[dict]:
    return [_row("dinov2_vits14", s, "dinov2/x") for s in seeds] + [
        _row("resnet18", s, LEGACY) for s in seeds
    ]


def test_a_refit_column_is_replaced_and_the_rest_kept_in_order(merge):
    new = [_row("resnet18", s, LEGACY + "/crop1", 0.9) for s in (4, 3, 2, 1, 0)]
    merged = merge.splice(_sweep(), new)
    assert merged[:5] == _sweep()[:5]
    assert [r["seed"] for r in merged[5:]] == [0, 1, 2, 3, 4]
    assert {r["backbone_key"] for r in merged if r["backbone"] == "resnet18"} == {LEGACY + "/crop1"}


def test_a_missing_seed_is_refused(merge):
    new = [_row("resnet18", s, LEGACY + "/crop1") for s in range(4)]
    with pytest.raises(ValueError, match="differ"):
        merge.splice(_sweep(), new)


def test_a_duplicated_seed_is_refused(merge):
    new = [_row("resnet18", s, LEGACY + "/crop1") for s in (0, 1, 2, 3, 4, 4)]
    with pytest.raises(ValueError, match="twice"):
        merge.splice(_sweep(), new)


def test_a_refit_under_two_keys_is_refused(merge):
    new = [_row("resnet18", s, LEGACY + "/crop1") for s in range(4)]
    new.append(_row("resnet18", 4, LEGACY + "/crop0.9"))
    with pytest.raises(ValueError, match="two keys"):
        merge.splice(_sweep(), new)


def test_a_backbone_the_sweep_never_held_is_refused(merge):
    new = [_row("dino_vitb8", s, "timm/x/crop1") for s in range(5)]
    with pytest.raises(ValueError, match="shape"):
        merge.splice(_sweep(), new)


def test_targets_name_the_files_the_analyses_read(merge):
    """`analyse_placements.py` finds the two sweeps outside `seeds/` by these names."""
    assert merge.target_for("edge") == Path("seeds/edge.jsonl")
    for probe, name in merge.SPECIAL.items():
        assert (ROOT / "results" / "controls" / name).is_file(), probe


def test_an_uncorrected_part_is_refused_before_anything_is_written(merge, tmp_path):
    row = _row("resnet18", 0, LEGACY) | {"hardware": {"gpu": "Tesla V100-SXM2-32GB"}}
    (tmp_path / "seeds_edge__resnet18.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(SystemExit, match="not a corrected key"):
        merge.staged(tmp_path)


def test_a_part_off_a_v100_is_refused(merge, tmp_path):
    row = _row("resnet18", 0, LEGACY + "/crop1") | {"hardware": {"gpu": "NVIDIA A100"}}
    (tmp_path / "seeds_edge__resnet18.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(SystemExit, match="V100"):
        merge.staged(tmp_path)


NEW = "timm/convnext_base/clip_laion2b/224/crop1"


def test_a_new_column_is_appended_and_nothing_else_moves(merge):
    new = [_row("clip_convnext_base", s, NEW, 0.7) for s in (3, 1, 4, 0, 2)]
    merged = merge.append_column(_sweep(), new)
    assert merged[:10] == _sweep()
    assert [r["seed"] for r in merged[10:]] == [0, 1, 2, 3, 4]


def test_adding_a_column_the_sweep_already_holds_is_refused(merge):
    new = [_row("resnet18", s, LEGACY + "/crop1") for s in range(5)]
    with pytest.raises(ValueError, match="already in the sweep"):
        merge.append_column(_sweep(), new)


def test_a_new_column_short_of_a_seed_is_refused(merge):
    new = [_row("clip_convnext_base", s, NEW) for s in range(4)]
    with pytest.raises(ValueError, match="differ"):
        merge.append_column(_sweep(), new)


def test_add_mode_reads_the_live_sweep_and_never_the_archive(merge, tmp_path, monkeypatch):
    """The archive holds the cropped sweeps; appending onto it would undo 23b."""
    controls = tmp_path / "controls"
    (controls / "seeds").mkdir(parents=True)
    (controls / "pre23a" / "seeds").mkdir(parents=True)
    live = [_row("dinov2_vits14", s, "dinov2/x", 0.9) for s in range(5)]
    cropped = [_row("dinov2_vits14", s, "dinov2/x", 0.1) for s in range(5)]
    (controls / "seeds" / "edge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in live))
    (controls / "pre23a" / "seeds" / "edge.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in cropped)
    )
    parts = tmp_path / "parts"
    parts.mkdir()
    gpu = {"hardware": {"gpu": "Tesla V100-SXM2-32GB"}}
    (parts / "seeds_edge__clip_convnext_base.jsonl").write_text(
        "".join(json.dumps(_row("clip_convnext_base", s, NEW) | gpu) + "\n" for s in range(5))
    )
    monkeypatch.setattr(merge, "CONTROLS", controls)
    monkeypatch.setattr(merge, "ARCHIVE", controls / "pre23a")

    assert merge.main([str(parts), "--add", "--write"]) == 0
    written = [
        json.loads(line) for line in (controls / "seeds" / "edge.jsonl").read_text().splitlines()
    ]
    assert [r["x"] for r in written if r["backbone"] == "dinov2_vits14"] == [0.9] * 5
    assert {r["backbone"] for r in written} == {"dinov2_vits14", "clip_convnext_base"}
