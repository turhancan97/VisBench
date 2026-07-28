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


class TestMissingExtras:
    """A missing extra does not unregister a backbone — it only makes one raise.

    CLAUDE.md claimed the opposite from v0.1 until the v0.2.0 wheel test put a
    core-only install in front of a listing that named all six backbones under a
    footer promising they would be absent. The behaviour is the good one; the
    listing has to say so itself rather than by omission.
    """

    def test_an_installed_backbone_needs_nothing(self):
        from visbench import registry

        assert registry.missing_extra("dinov2_vits14") is None

    def test_it_names_the_extra_not_the_module(self, monkeypatch):
        """`open_clip` ships in a distribution called open_clip_torch, which is
        not a name a user should need to know to read an error."""
        import importlib.util

        from visbench import registry

        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **k: None if name == "open_clip" else object(),
        )
        assert registry.missing_extra("clip_vitb16") == "clip"

    def test_it_does_not_import_the_dependency(self, monkeypatch):
        """Asking what you can run must not cost what running it would."""
        import builtins

        from visbench import registry

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name in ("open_clip", "timm"):
                raise AssertionError(f"missing_extra imported {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        for name in visbench.list_backbones():
            registry.missing_extra(name)

    def test_an_unknown_name_is_not_an_error(self):
        from visbench import registry

        assert registry.missing_extra("no_such_backbone") is None

    def test_the_listing_marks_a_missing_extra(self, monkeypatch, run_cli):
        from visbench import registry

        monkeypatch.setattr(
            registry, "missing_extra", lambda name: "clip" if "clip" in name else None
        )
        result = run_cli("list", "backbones")
        assert "clip_vitb16   (needs the 'clip' extra)" in result.out
        assert "pip install 'visbench[clip]'" in result.out
        # And an available backbone carries no note.
        assert "dinov2_vits14\n" in result.out

    def test_the_listing_is_clean_when_everything_is_installed(self, monkeypatch, run_cli):
        from visbench import registry

        monkeypatch.setattr(registry, "missing_extra", lambda name: None)
        result = run_cli("list", "backbones")
        assert "extra" not in result.out
        assert "pip install" not in result.out
