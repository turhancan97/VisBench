"""The DPT control's matrix is defined in two files, and they must agree.

`scripts/build_dpt_control.sh` holds one `probe_<name>` function per probe and
lists them in `ALL_PROBES`; `slurm/dpt_control.sbatch` holds its own `PROBES`
array, which it multiplies by a backbone list to size the array job. Neither
file can see the other, and the failure when they disagree is the one the
sbatch's own guard cannot catch: a probe missing from `PROBES` is never
scheduled, the matrix stays *self-consistently* the wrong size, and every group
in the resulting control still holds every backbone.

That is `corner`'s history in the corpus array -- added to one list and not the
other, unschedulable for a whole release -- and the reason
`tests/scripts/test_corpus_scripts.py` exists. This is the same test for the
control, written when the control was widened rather than after it went wrong.

Two things here that the corpus equivalent does not have to check:

- **the probe list is not `list_probes()`**, it is the subset that declares an
  oracle. `evaluate_oracle` is opt-in, and every other dense probe raises, so a
  DPT run against them would have nothing to be measured against.
- **the layer spec is per backbone**, because `num_layers` counts what timm's
  `feature_info` exposes and that is 5 for a ResNet and 4 for ConvNeXt. A
  single shared spec was the first draft and is out of range on ConvNeXt.
"""

import json
import re
from pathlib import Path

import pytest

from visbench import list_probes

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "scripts" / "build_dpt_control.sh"
SBATCH = ROOT / "slurm" / "dpt_control.sbatch"
BUILD_CORPUS = ROOT / "scripts" / "build_corpus.sh"
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"

#: Probes that opt into `DenseTrainingTask.evaluate_oracle`. Listed here rather
#: than imported, so that a probe gaining an oracle is a deliberate addition to
#: this control rather than something that silently widens it.
ORACLE_PROBES = ["edge", "keypoints2d", "occlusion_edge", "corner", "orientation"]


def _bash_array(text: str, name: str) -> list[str]:
    r"""Read `NAME=(\n  a\n  b\n)` out of a shell script, comments stripped."""
    match = re.search(rf"^{name}=\(\n(.*?)^\)$", text, re.MULTILINE | re.DOTALL)
    assert match is not None, f"no {name}=( ... ) array found"
    entries = []
    for line in match.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def _inline_array(text: str, name: str) -> list[str]:
    """Read a `NAME=(a b c \\\n d)` one-or-two-line array."""
    match = re.search(rf"{name}=\((.*?)\)", text, re.DOTALL)
    assert match is not None, f"no {name}=( ... ) found"
    body = match.group(1).replace("\\\n", " ")
    return body.split()


@pytest.fixture(scope="module")
def build_text() -> str:
    return BUILD.read_text()


@pytest.fixture(scope="module")
def sbatch_text() -> str:
    return SBATCH.read_text()


def test_the_two_probe_lists_are_identical(build_text, sbatch_text):
    """Same probes, same order.

    Order matters as well as membership: the sbatch derives its probe from
    `index / len(backbones)`, so a reordering silently relabels which task ran
    which probe when a partial array is resubmitted against logged indices.
    """
    assert _bash_array(build_text, "ALL_PROBES") == _bash_array(sbatch_text, "PROBES")


def test_the_probes_are_the_ones_with_an_oracle(build_text):
    """Not every probe -- the subset whose target averages under pooling.

    `evaluate_oracle` is opted into per probe and raises otherwise, so running
    this control against `depth` or either segmentation would fail rather than
    produce a number. Pinning the list is what stops it being widened by
    someone reading `list_probes()` and assuming.
    """
    assert _bash_array(build_text, "ALL_PROBES") == ORACLE_PROBES


def test_every_oracle_probe_is_a_registered_probe():
    """A typo here would be a probe silently absent from the control."""
    registered = set(list_probes())
    assert set(ORACLE_PROBES) <= registered


def test_every_listed_probe_has_a_builder(build_text):
    """The script exits on an unknown probe, but only if it is *named*; a typo
    in `ALL_PROBES` would otherwise be a probe missing from the control."""
    for probe in _bash_array(build_text, "ALL_PROBES"):
        assert f"probe_{probe}()" in build_text, f"ALL_PROBES names {probe}, no builder"


def test_both_groups_cover_every_corpus_backbone(build_text, sbatch_text):
    """The two groups together must be every backbone the corpus has a board for.

    A backbone in neither group is one the control says nothing about, and the
    reason for two groups is a layer spec rather than a decision to leave a
    backbone out.

    **Read off the committed corpus, not `list_backbones()`.** The registry is
    process-global and other test modules register fakes into it, so a first
    draft of this test passed alone and failed in the full suite with
    `fake_cli_vit` in the expected set. The corpus file is also the better
    definition: "a corpus backbone" means one with records, which is exactly
    what the control has to cover.
    """
    corpus = {
        json.loads(line)["backbone"] for line in CORPUS.read_text().splitlines() if line.strip()
    }
    # `dinov2_vitb14_196` lives in results/controls/, so it is not in here --
    # asserted rather than filtered, since a resolution control appearing in the
    # corpus would be a different and much worse problem.
    assert "dinov2_vitb14_196" not in corpus

    specs = re.findall(r'"([a-z0-9_]+)=[-\d ]+"', build_text)
    assert set(specs) == corpus, "a corpus backbone is in neither DPT group"

    sbatch_names = set(_inline_array(sbatch_text, "BACKBONE_LIST")) | set(
        re.findall(r"BACKBONE_LIST=\((.*?)\)", sbatch_text, re.DOTALL)[1].split()
    )
    assert sbatch_names == corpus, "the sbatch's two groups do not cover the corpus"


def test_a_convnext_spec_is_in_range(build_text):
    """ConvNeXt exposes four feature stages, not five.

    `1 2 3 4` is a ResNet's last four stages and is **out of range** on
    ConvNeXt, which raises rather than scoring badly -- caught by a pre-check
    before an array burned on it. The specs are per backbone for this reason,
    so a levelling edit that gives every CNN one spec must fail here.
    """
    specs = dict(re.findall(r'"([a-z0-9_]+)=([-\d ]+)"', build_text))
    assert specs["convnext_base"].split() == ["0", "1", "2", "3"]
    assert specs["resnet50"].split() == ["1", "2", "3", "4"]
    assert specs["resnet18"].split() == ["1", "2", "3", "4"]


def test_every_vit_reads_four_evenly_spaced_blocks(build_text):
    """All nine are twelve-block, so they share one spec -- and must, or they
    would not be one comparability group."""
    specs = dict(re.findall(r'"([a-z0-9_]+)=([-\d ]+)"', build_text))
    vits = {
        name: spec
        for name, spec in specs.items()
        if "resnet" not in name and "convnext" not in name
    }
    assert len(vits) == 9, f"expected nine ViTs, got {sorted(vits)}"
    assert set(vits.values()) == {"2 5 8 11"}, "the ViT group must share one layer spec"


def test_the_dataset_flags_match_the_corpus(build_text):
    """Only the head may differ from the linear board.

    A control whose dataset flags have drifted is not a control: it answers
    "what happens when the head AND the frames change", and nothing in the
    record says which. These five probes' flags are copied from
    `build_corpus.sh`, and this is what keeps the copy honest.
    """
    corpus_text = BUILD_CORPUS.read_text()
    for probe in ORACLE_PROBES:
        for text, label in ((build_text, "control"), (corpus_text, "corpus")):
            body = re.search(rf"probe_{probe}\(\) \{{(.*?)^\}}", text, re.DOTALL | re.MULTILINE)
            assert body is not None, f"no probe_{probe} body in the {label} script"
        pattern = rf"probe_{probe}\(\) \{{(.*?)^\}}"
        control = re.search(pattern, build_text, re.DOTALL | re.MULTILINE)
        corpus = re.search(pattern, corpus_text, re.DOTALL | re.MULTILINE)
        assert control is not None and corpus is not None

        def flags(body: str) -> list[str]:
            line = " ".join(
                part.split("#", 1)[0].strip()
                for part in body.splitlines()
                if part.strip() and not part.strip().startswith("#")
            )
            line = line.replace("\\", " ")
            return [token for token in line.split() if token.startswith("--")]

        assert flags(control.group(1)) == flags(corpus.group(1)), (
            f"{probe}: the control's dataset flags have drifted from the corpus's"
        )


def test_the_control_never_writes_to_the_corpus(build_text):
    """`--head dpt` must never reach a corpus record.

    The head and the layers are both in `comparability_key`, so a DPT record
    under `task=edge` does not merge with the linear board -- it makes that
    board *unrenderable*, since `board_for` refuses a task with more than one
    group. Two scripts is what makes that structural.
    """
    assert "results/corpus" not in build_text
    assert "--head dpt" in build_text
    assert "--head dpt" not in BUILD_CORPUS.read_text()


# -- against the real models -------------------------------------------------
#
# Everything above reads the scripts as text, which pins what they *say*. What
# they say is only right if it matches what the backbones expose, and that
# needs real weights -- so this one is `slow`, and CI does not run it.
#
# That is acceptable here, and the reason is worth stating because the standing
# rule points the other way: a guard against a silently wrong *number* belongs
# in the fast suite. A wrong layer spec is not silent. `resolve_layers` raises
# "out of range", the run dies, and no record is written -- which is how
# ConvNeXt's spec was caught in the first place, by a pre-check rather than by
# reading a board afterwards. The cost of learning this a day late is a
# re-submitted array, not a published wrong claim.


#: What each backbone exposes and which four layers the control reads. A second
#: copy of the script's specs on purpose -- `test_the_table_matches_the_script`
#: is what stops the copy drifting, which is `corner`'s history in the corpus
#: array (added to one list, not the other, unschedulable for a release).
LAYER_SPECS = {
    "dinov2_vits14": ([2, 5, 8, 11], 12),
    "dinov2_vitb14": ([2, 5, 8, 11], 12),
    "clip_vitb16": ([2, 5, 8, 11], 12),
    "clip_vitb32": ([2, 5, 8, 11], 12),
    "mae_vitb16": ([2, 5, 8, 11], 12),
    "siglip_vitb16": ([2, 5, 8, 11], 12),
    "supervised_vitb16": ([2, 5, 8, 11], 12),
    "dino_vitb16": ([2, 5, 8, 11], 12),
    "sam_vitb16": ([2, 5, 8, 11], 12),
    # A ResNet's `feature_info` carries the stem at index 0, so its last four
    # stages are 1..4. ConvNeXt has no separate stem entry and exposes four,
    # whose last four are 0..3. Reading the second off the first is exactly the
    # mistake this pins -- and it raises rather than scoring badly.
    "resnet18": ([1, 2, 3, 4], 5),
    "resnet50": ([1, 2, 3, 4], 5),
    "convnext_base": ([0, 1, 2, 3], 4),
}


def test_the_table_matches_the_script():
    """The fast half: this file's table against the script's, string to string.

    Runs in CI, unlike the two below, so a spec edited in one place and not the
    other fails immediately rather than a day later.
    """
    specs = dict(re.findall(r'"([a-z0-9_]+)=([-\d ]+)"', BUILD.read_text()))
    expected = {name: " ".join(str(i) for i in spec) for name, (spec, _) in LAYER_SPECS.items()}
    assert specs == expected


@pytest.mark.slow
@pytest.mark.parametrize(
    ("backbone", "spec", "depth"),
    [(name, spec, depth) for name, (spec, depth) in LAYER_SPECS.items()],
)
def test_the_layer_spec_is_in_range_on_the_real_model(backbone, spec, depth):
    """What the control script says, against what the backbone exposes."""
    import visbench

    model = visbench.get_backbone(backbone, device="cpu")
    assert model.num_layers == depth, f"{backbone} exposes {model.num_layers}, not {depth}"
    assert model.resolve_layers(spec) == spec, "the spec must already be absolute and in range"


@pytest.mark.slow
def test_a_dpt_head_fits_every_group(monkeypatch):
    """The four maps a spec selects must actually build a head.

    A ViT's four blocks share one width; a CNN's four stages do not, which is
    why `layer_channels` exists. This is the check that both shapes reach a
    working `DPTHead` -- the failure it guards is a control that dies per
    backbone after the features are already extracted.
    """
    import torch

    import visbench
    from visbench.heads import DPTHead

    for backbone_name in ("clip_vitb32", "resnet18", "convnext_base"):
        spec, _ = LAYER_SPECS[backbone_name]
        model = visbench.get_backbone(backbone_name, device="cpu")
        with torch.no_grad():
            features = model.extract_features(torch.rand(1, 3, 224, 224), layers=spec)
        widths = [stage.shape[1] for stage in features["dense_layers"]]
        head = DPTHead(
            in_channels=widths, out_channels=1, num_layers=4, hidden_dim=16, output_size=224
        )
        with torch.no_grad():
            assert head(features["dense_layers"]).shape == (1, 1, 224, 224)
