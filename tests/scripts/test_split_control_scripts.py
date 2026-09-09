"""The split control's matrix is defined in two files, and they must agree.

`scripts/build_split_control.sh` holds the configs and backbones; so does
`slurm/split_control.sbatch`, because #SBATCH directives are read before any of
the script runs and the array cannot derive its own range. That is the same
two-file matrix that left the corpus array short by a probe for a whole release
-- and the guard inside the array could not see it, because it read its own
short list. A guard that derives its expectation from the data it is checking
is not a guard, so the check lives here.

What the control asks is in the build script's own header. What this file pins
is that it cannot silently drift from the board it exists to explain: the flags
must stay equal to `probe_detection`'s apart from the split list and the limit,
and no record may reach the corpus.
"""

import json
import re
from pathlib import Path

import pytest

from visbench import list_probes
from visbench.results.render import HEADLINE_METRICS as LIBRARY_HEADLINE_METRICS

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "scripts" / "build_split_control.sh"
SBATCH = ROOT / "slurm" / "split_control.sbatch"
BUILD_CORPUS = ROOT / "scripts" / "build_corpus.sh"
MERGE = ROOT / "scripts" / "merge_controls.sh"
ANALYSE = ROOT / "scripts" / "analyse_split_control.py"
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
CONTROL = ROOT / "results" / "controls" / "detection_split.jsonl"

#: The two configs, listed here rather than parsed, so that adding a third is a
#: deliberate edit to this test as well as to both scripts.
CONFIGS = ["full", "limit600"]


def _inline_array(text: str, name: str) -> list[str]:
    """Read a `NAME=(a b c \\\n d)` array, comments stripped."""
    match = re.search(rf"{name}=\(\n?(.*?)\n?\)", text, re.DOTALL)
    assert match is not None, f"no {name}=( ... ) found"
    body = "\n".join(line.split("#", 1)[0] for line in match.group(1).splitlines())
    return body.replace("\\\n", " ").split()


@pytest.fixture(scope="module")
def build_text() -> str:
    return BUILD.read_text()


@pytest.fixture(scope="module")
def sbatch_text() -> str:
    return SBATCH.read_text()


def test_the_two_backbone_lists_are_identical(build_text, sbatch_text):
    """Same backbones, same order.

    Order matters as much as membership: the sbatch picks its backbone with
    `index % len(backbones)`, so a reordering silently relabels which run
    produced which record while every task still succeeds.
    """
    assert _inline_array(build_text, "BACKBONE_LIST") == _inline_array(sbatch_text, "BACKBONE_LIST")


def test_the_two_config_lists_are_identical(build_text, sbatch_text):
    build = _inline_array(build_text, "ALL_CONFIGS")
    sbatch = _inline_array(sbatch_text, "CONFIGS")
    assert build == sbatch == CONFIGS


def test_the_matrix_covers_every_corpus_backbone(sbatch_text):
    """A control read against a published board must cover the same rows.

    The figures this control explains (+0.804, +0.650, +0.958) are Spearman over
    twelve backbones. A control over three would be a different statistic
    wearing the same name, and the comparison would be silently meaningless.
    """
    corpus_backbones = {json.loads(line)["backbone"] for line in CORPUS.read_text().splitlines()}
    listed = set(_inline_array(sbatch_text, "BACKBONE_LIST"))
    assert listed == corpus_backbones


def test_the_array_size_matches_the_matrix(sbatch_text):
    """The documented `--array` must match configs x backbones.

    Too small a range omits tasks and that is invisible afterwards -- every
    config present would still hold every backbone.
    """
    backbones = _inline_array(sbatch_text, "BACKBONE_LIST")
    expected = len(CONFIGS) * len(backbones)
    assert f"--array=0-{expected - 1}" in sbatch_text


def test_it_reads_the_segmentation_split_not_main(build_text):
    """The whole control is which split list it names.

    `ImageSets/Main` would make it a re-run of the published board, and
    `SegmentationObject` covers 2913 images only -- which is why the control had
    to be inverted from the way it was first written down.
    """
    assert "ImageSets/Segmentation/val.txt" in build_text
    assert "ImageSets/Segmentation/train.txt" in build_text
    assert "ImageSets/Main" not in build_text.split("USAGE")[-1]


def test_the_dataset_flags_match_the_corpus_detection_probe(build_text):
    """Only the split and the limit may differ from `probe_detection`.

    If the control changed the image directory, the annotation directory or the
    resolution as well, a ranking difference would have more than one candidate
    cause and the experiment would answer nothing.
    """
    corpus = BUILD_CORPUS.read_text()
    detection = corpus.split("probe_detection()")[1].split("}")[0]
    for flag in ("--image-dir JPEGImages", "--annotation-dir Annotations"):
        assert flag in detection, f"{flag} left probe_detection; this control is now stale"
        assert flag in build_text
    # The corpus board's own limit, which config B holds so that A vs B varies
    # size alone.
    assert "DETECTION_LIMIT=600" in build_text
    assert "DETECTION_LIMIT=600" in corpus


def test_the_control_never_writes_to_the_corpus(build_text, sbatch_text):
    """A `task=detection` record over another split makes the published board
    UNRENDERABLE -- `board_for` refuses a task with more than one comparability
    group. So this must not be one typo from the corpus file."""
    for text in (build_text, sbatch_text):
        assert "results/corpus" not in text
    assert "results/controls/detection_split.jsonl" in build_text


def test_the_merge_script_routes_this_group(build_text):
    """merge_controls.sh routes by filename prefix from a hand-written list, so a
    new control merges into nothing until it is added."""
    merge = MERGE.read_text()
    assert "merge_group detection_split results/controls/detection_split.jsonl" in merge
    prefix = "detection_split__"
    assert prefix in SBATCH.read_text(), "the parts must carry the prefix the merge routes on"


def test_the_analysers_copied_headline_table_matches_the_library():
    """The copy is deliberate -- these scripts read a fixed table -- but a probe
    added with a new headline metric must not leave it behind."""
    source = ANALYSE.read_text()
    block = source.split("HEADLINE_METRICS: dict[str, str] = {")[1].split("}")[0]
    copied = dict(re.findall(r'"(\w+)":\s*"([^"]+)"', block))
    assert copied == LIBRARY_HEADLINE_METRICS


def test_the_control_covers_the_probe_it_explains():
    assert "detection" in list_probes()
    assert "instance_segmentation" in list_probes()


@pytest.mark.skipif(not CONTROL.exists(), reason="control has not been run here")
def test_the_recorded_control_holds_both_configs_at_full_width():
    """The file on disk is what a claim is read off, so pin its shape.

    Both configs, twelve backbones each, and every record `task=detection` --
    the control is the same probe on other images, so a record under another
    task name would mean the wrong thing ran.
    """
    records = [json.loads(line) for line in CONTROL.read_text().splitlines() if line.strip()]
    assert {r["task"] for r in records} == {"detection"}
    by_config: dict[str, set[str]] = {"full": set(), "limit600": set()}
    for record in records:
        name = "limit600" if record["dataset_size"] <= 600 else "full"
        by_config[name].add(record["backbone"])
    corpus_backbones = {json.loads(line)["backbone"] for line in CORPUS.read_text().splitlines()}
    for name, backbones in by_config.items():
        assert backbones == corpus_backbones, (
            f"{name} is short: {sorted(corpus_backbones - backbones)}"
        )
