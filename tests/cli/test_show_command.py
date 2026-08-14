"""``visbench show``, and the ``run --save-probe`` that feeds its third column."""

import argparse

import pytest

from visbench.cli.datasets import SCHEDULE_DEFAULTS, showable_probes, spec_for, supported_probes
from visbench.cli.main import build_parser
from visbench.viz import show_probes


def _subparsers(parser, command):
    action = next(
        a for a in parser._subparsers._group_actions if isinstance(a, argparse._SubParsersAction)
    )
    nested = action.choices[command]
    inner = next(
        a for a in nested._subparsers._group_actions if isinstance(a, argparse._SubParsersAction)
    )
    return inner.choices


def _options(parser):
    return {option for action in parser._actions for option in action.option_strings}


class TestTheCommand:
    def test_show_is_a_command(self):
        parser = build_parser()
        args = parser.parse_args(["show", "depth", "--data", "x"])
        assert args.command == "show"
        assert args.probe == "depth"

    def test_only_probes_with_a_spatial_target_get_a_subcommand(self):
        """A probe with nothing to put in a panel is absent, not drawn blank."""
        choices = set(_subparsers(build_parser(), "show"))
        assert choices == set(showable_probes()) == set(show_probes())
        assert "retrieval" not in choices
        assert "similarity" not in choices
        assert choices < set(supported_probes())

    def test_an_undrawable_probe_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["show", "retrieval", "--data", "x"])

    @pytest.mark.parametrize("probe", showable_probes())
    def test_each_subcommand_parses_its_minimum(self, probe):
        extra = ["--num-classes", "3"] if probe == "semantic_segmentation" else []
        args = build_parser().parse_args(["show", probe, "--data", "x", *extra])
        assert args.probe == probe


class TestFlagsItMustNotOffer:
    """The reason ``show_arguments`` is a separate callable at all."""

    @pytest.mark.parametrize("probe", showable_probes())
    def test_no_schedule_flag_appears_in_help(self, probe):
        offered = _options(_subparsers(build_parser(), "show")[probe])
        for flag in ("--epochs", "--lr", "--train-batch-size", "--finetune-blocks"):
            assert flag not in offered, f"{probe} offers {flag}, which it cannot honour"

    @pytest.mark.parametrize("probe", showable_probes())
    def test_but_probe_kwargs_still_has_everything_it_reads(self, probe):
        """``show`` builds a probe to load a head onto, using ``run``'s own table.

        Dropping the flags without supplying the defaults would make this a
        ``TypeError`` at the point a user has already waited for extraction.
        """
        extra = ["--num-classes", "3"] if probe == "semantic_segmentation" else []
        args = build_parser().parse_args(["show", probe, "--data", "x", *extra])
        assert spec_for(probe).probe_kwargs(args)

    def test_schedule_defaults_match_the_flags_they_stand_in_for(self):
        """Two copies of one set of defaults, pinned against each other."""
        depth = _subparsers(build_parser(), "run")["depth"]
        actual = {
            action.dest: action.default
            for action in depth._actions
            if action.dest in SCHEDULE_DEFAULTS
        }
        assert actual == SCHEDULE_DEFAULTS


class TestRunIsUnchanged:
    def test_run_flags_are_unchanged_by_the_split(self):
        """The flag helpers were re-cut; ``run``'s surface must not have moved.

        ``--image-size`` in particular moved out of the head group into the data
        group, because it decides the dataset's resize and centre crop. It has
        to still be there, on every probe that had it, with the same default.
        """
        run = _subparsers(build_parser(), "run")
        for probe in ("depth", "semantic_segmentation", "edge", "corner", "detection"):
            offered = _options(run[probe])
            assert {"--image-size", "--head", "--epochs", "--lr"} <= offered
        for probe in ("depth", "edge", "corner"):
            sizes = [a for a in run[probe]._actions if a.dest == "image_size"]
            assert len(sizes) == 1 and sizes[0].default == 224


class TestDrawing:
    def test_it_writes_a_page(self, run_cli, dense_folder, tmp_path):
        out = tmp_path / "panels.png"
        result = run_cli(
            "show", "generic_segmentation", "--data", str(dense_folder), "--out", str(out)
        )
        assert result.code == 0, result.err
        assert out.is_file()

        from PIL import Image

        with Image.open(out) as page:
            assert page.size[0] > 0 and page.mode == "RGB"

    def test_it_needs_no_backbone_and_no_cache(self, run_cli, dense_folder, tmp_path, monkeypatch):
        """Which is when you actually want to look: before spending a budget.

        Enforced by making any backbone construction fail — a viewer that
        quietly built one would take a GPU and a download to draw two panels.
        """
        import visbench

        def refuse(*_args, **_kwargs):
            raise AssertionError("show built a backbone without --predict-from")

        monkeypatch.setattr(visbench, "get_backbone", refuse)
        result = run_cli(
            "show",
            "generic_segmentation",
            "--data",
            str(dense_folder),
            "--out",
            str(tmp_path / "p.png"),
        )
        assert result.code == 0, result.err

    def test_it_draws_boxes_for_detection(self, run_cli, voc_folder, tmp_path):
        out = tmp_path / "det.png"
        result = run_cli("show", "detection", "--data", str(voc_folder), "--out", str(out))
        assert result.code == 0, result.err
        assert out.is_file()

    def test_frames_and_start_select_rows(self, run_cli, dense_folder, tmp_path):
        from PIL import Image

        sizes = []
        for frames in (1, 2):
            out = tmp_path / f"{frames}.png"
            result = run_cli(
                "show",
                "generic_segmentation",
                "--data",
                str(dense_folder),
                "--frames",
                str(frames),
                "--out",
                str(out),
            )
            assert result.code == 0, result.err
            with Image.open(out) as page:
                sizes.append(page.height)
        assert sizes[1] > sizes[0]

    def test_a_start_past_the_end_is_one_line_not_a_traceback(
        self, run_cli, dense_folder, tmp_path
    ):
        result = run_cli(
            "show",
            "generic_segmentation",
            "--data",
            str(dense_folder),
            "--start",
            "99",
            "--out",
            str(tmp_path / "p.png"),
        )
        assert result.code == 2
        assert "past the end" in result.err
        assert "Traceback" not in result.err

    def test_zero_frames_is_refused(self, run_cli, dense_folder, tmp_path):
        result = run_cli(
            "show",
            "generic_segmentation",
            "--data",
            str(dense_folder),
            "--frames",
            "0",
            "--out",
            str(tmp_path / "p.png"),
        )
        assert result.code == 2

    def test_a_missing_folder_is_one_line(self, run_cli, tmp_path):
        result = run_cli("show", "depth", "--data", str(tmp_path / "nope"))
        assert result.code == 2
        assert "Traceback" not in result.err


class TestSaveProbe:
    def test_it_writes_a_loadable_artifact(self, run_cli, dense_folder, tmp_path, cache_dir):
        """The link between the two commands: ``run`` writes, ``show`` reads."""
        artifact = tmp_path / "head.pt"
        result = run_cli(
            "run",
            "generic_segmentation",
            "--data",
            str(dense_folder),
            "--backbone",
            "fake_cli_vit",
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--train-batch-size",
            "2",
            "--image-size",
            "64",
            "--cache",
            str(cache_dir),
            "--results",
            "none",
            "--save-probe",
            str(artifact),
        )
        assert result.code == 0, result.err
        assert artifact.is_file()
        assert "saved the trained head" in result.out

        import visbench
        from visbench.hub import load_probe

        backbone = visbench.get_backbone("fake_cli_vit")
        probe = load_probe(artifact, backbone=backbone)
        assert probe.head is not None

    def test_the_round_trip_draws_a_prediction_column(
        self, run_cli, dense_folder, tmp_path, cache_dir
    ):
        """``run --save-probe`` then ``show --predict-from``, the whole loop.

        The point of ``--save-probe`` existing at all: before it, the only way
        to get an artifact from a shell was ``--push-to``, which needs a Hub
        account, so the prediction column had no CLI-producible input.
        """
        from PIL import Image

        artifact = tmp_path / "head.pt"
        trained = run_cli(
            "run",
            "generic_segmentation",
            "--data",
            str(dense_folder),
            "--backbone",
            "fake_cli_vit",
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--train-batch-size",
            "2",
            "--image-size",
            "64",
            "--cache",
            str(cache_dir),
            "--results",
            "none",
            "--save-probe",
            str(artifact),
        )
        assert trained.code == 0, trained.err

        without = tmp_path / "without.png"
        assert (
            run_cli(
                "show",
                "generic_segmentation",
                "--data",
                str(dense_folder),
                "--image-size",
                "64",
                "--out",
                str(without),
            ).code
            == 0
        )

        with_prediction = tmp_path / "with.png"
        drawn = run_cli(
            "show",
            "generic_segmentation",
            "--data",
            str(dense_folder),
            "--backbone",
            "fake_cli_vit",
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--cache",
            str(cache_dir),
            "--predict-from",
            str(artifact),
            "--out",
            str(with_prediction),
        )
        assert drawn.code == 0, drawn.err
        assert "prediction" in drawn.out

        with Image.open(without) as a, Image.open(with_prediction) as b:
            assert b.width > a.width  # the third column is really there

    def test_a_head_from_another_backbone_is_refused(
        self, run_cli, dense_folder, tmp_path, cache_dir
    ):
        """``load_probe``'s identity check, reached through the command.

        A head fed the wrong features scores 0.9620 against 0.9820 without
        raising, so this refusal is the reason ``show`` goes through the
        artifact module rather than ``torch.load``.
        """
        import torch

        artifact = tmp_path / "head.pt"
        assert (
            run_cli(
                "run",
                "generic_segmentation",
                "--data",
                str(dense_folder),
                "--backbone",
                "fake_cli_vit",
                "--device",
                "cpu",
                "--epochs",
                "2",
                "--train-batch-size",
                "2",
                "--image-size",
                "64",
                "--cache",
                str(cache_dir),
                "--results",
                "none",
                "--save-probe",
                str(artifact),
            ).code
            == 0
        )

        payload = torch.load(artifact, weights_only=True)
        payload["meta"]["backbone_key"] = "someone-elses-weights"
        torch.save(payload, artifact)

        result = run_cli(
            "show",
            "generic_segmentation",
            "--data",
            str(dense_folder),
            "--backbone",
            "fake_cli_vit",
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--cache",
            str(cache_dir),
            "--predict-from",
            str(artifact),
            "--out",
            str(tmp_path / "p.png"),
        )
        assert result.code == 2
        assert "backbone_key" in result.err

    def test_a_zero_shot_probe_is_refused_before_the_run(self, run_cli, flat_folder, cache_dir):
        """Refused by the CLI, not by ``save_probe`` after training.

        ``save_probe`` would raise the same thing — having spent the whole run
        to do it. This is the rule ``--push-to`` already follows.
        """
        result = run_cli(
            "run",
            "correspondence",
            "--data",
            str(flat_folder),
            "--backbone",
            "fake_cli_vit",
            "--device",
            "cpu",
            "--split",
            "test",
            "--cache",
            str(cache_dir),
            "--results",
            "none",
            "--save-probe",
            "/tmp/never-written.pt",
        )
        assert result.code == 2
        assert "zero-shot" in result.err
        assert "save" in result.err
