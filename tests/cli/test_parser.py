"""The command line's shape.

A CLI's flag names are an API — they end up in shell scripts and papers'
appendices — so they get the same treatment as any other public surface. These
tests read the parser directly rather than through a subprocess, which is why
:func:`build_parser` returns it instead of parsing in place.
"""

import pytest

import visbench
from visbench.cli import build_parser
from visbench.cli.datasets import SPECS, spec_for, supported_probes


@pytest.fixture
def parser():
    return build_parser()


class TestStructure:
    def test_no_command_prints_help_rather_than_failing(self, run_cli):
        result = run_cli()
        assert result.code == 1
        assert "usage: visbench" in result.out

    @pytest.mark.parametrize(
        ("command", "rest"),
        [("list", []), ("run", ["retrieval", "--data", "x"]), ("cache", ["stats"])],
    )
    def test_the_three_commands_exist(self, parser, command, rest):
        assert parser.parse_args([command, *rest]).command == command

    def test_version_is_the_package_version(self, parser, capsys):
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])
        assert visbench.__version__ in capsys.readouterr().out


class TestProbeCoverage:
    """Every registered probe should be reachable, or knowingly not."""

    def test_every_registered_probe_has_a_subcommand(self):
        """If this fails, a probe shipped without a way to run it from a shell.

        A deliberate omission is fine — but it has to be deliberate, which means
        removing the name here and explaining why, not letting it drift out.
        """
        assert set(supported_probes()) == set(visbench.list_probes())

    @pytest.mark.parametrize("probe", sorted(SPECS))
    def test_each_subcommand_parses_its_minimum(self, parser, probe):
        argv = ["run", probe, "--data", "somewhere"]
        if probe == "semantic_segmentation":
            argv += ["--num-classes", "21"]
        args = parser.parse_args(argv)
        assert args.probe == probe
        assert str(args.data) == "somewhere"

    @pytest.mark.parametrize("probe", sorted(SPECS))
    def test_each_subcommand_documents_its_layout(self, probe):
        assert "<data>" in spec_for(probe).layout


class TestRequiredArguments:
    def test_data_is_required(self, parser, capsys):
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "retrieval"])
        assert "--data" in capsys.readouterr().err

    def test_semantic_segmentation_requires_num_classes(self, parser, capsys):
        """A wrong class count does not raise anywhere downstream — it trains a
        head that cannot express some categories — so the CLI will not guess."""
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "semantic_segmentation", "--data", "x"])
        assert "--num-classes" in capsys.readouterr().err

    def test_a_probe_name_is_required(self, parser, capsys):
        with pytest.raises(SystemExit):
            parser.parse_args(["run"])
        assert "probe" in capsys.readouterr().err


class TestDefaults:
    def test_extraction_and_training_batch_sizes_are_separate(self, parser):
        """One number cannot serve both: extraction wants 32, a dense head 8."""
        args = parser.parse_args(["run", "depth", "--data", "x"])
        assert args.batch_size == 32
        assert args.train_batch_size == 8

    def test_dense_probes_default_to_probe3d_s_schedule(self, parser):
        args = parser.parse_args(["run", "depth", "--data", "x"])
        assert (args.epochs, args.lr) == (10, 5e-4)

    def test_similarity_defaults_to_the_reference_vote_filter(self, parser):
        args = parser.parse_args(["run", "similarity", "--data", "x"])
        assert args.min_votes == 6
        assert args.split == "test"

    def test_label_maps_default_to_ignoring_255(self, parser):
        """VOC, ADE20K and Cityscapes all use it; leaving it unmapped would make
        255 a class the probe is trained and scored on."""
        args = parser.parse_args(
            ["run", "semantic_segmentation", "--data", "x", "--num-classes", "21"]
        )
        assert args.ignore_index == 255

    def test_binary_masks_default_to_no_ignore_region(self, parser):
        """The opposite default, and deliberately: every pixel of a plain
        foreground/background mask is labelled."""
        args = parser.parse_args(["run", "generic_segmentation", "--data", "x"])
        assert args.ignore_index < 0

    def test_target_directory_defaults_per_probe(self, parser):
        for probe, expected in [
            ("depth", "depths"),
            ("surface_normal", "normals"),
            ("generic_segmentation", "masks"),
        ]:
            args = parser.parse_args(["run", probe, "--data", "x"])
            assert args.target_dir == expected


class TestSpecLookup:
    def test_unknown_probe_lists_what_exists(self):
        with pytest.raises(KeyError, match="Unknown probe"):
            spec_for("nonsense")

    def test_a_registered_but_unsupported_probe_says_so(self, monkeypatch):
        """The message must not imply the user mistyped a real name."""
        monkeypatch.setattr(visbench, "list_probes", lambda: ["detection"])
        with pytest.raises(KeyError, match="registered probe but the CLI"):
            spec_for("detection")
