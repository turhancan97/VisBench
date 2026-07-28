"""What the commands do, not just how they parse.

``run`` is exercised end to end against a registered fake backbone (see
``conftest``), so these cover the whole path a user takes — flags to dataset to
:func:`visbench.run` to a written record — without downloading weights.
"""

import json

import pytest
from PIL import Image

import visbench
from tests.cli.conftest import FAKE_BACKBONE
from visbench.cli.datasets import spec_for
from visbench.results import read_records


class TestList:
    def test_it_shows_all_three_registries(self, run_cli):
        result = run_cli("list")
        assert result.code == 0
        for section in ("backbones:", "probes:", "heads:"):
            assert section in result.out
        assert "dinov2_vits14" in result.out and "linear" in result.out

    def test_it_can_show_one(self, run_cli):
        result = run_cli("list", "probes")
        assert "correspondence" in result.out
        assert "backbones:" not in result.out

    def test_every_backbone_is_listed_whether_or_not_its_extra_is_installed(self, run_cli):
        """A missing extra does not unregister a backbone. Marking the ones that
        need one is `TestMissingExtras` in test_parser.py; here the point is only
        that the set is complete."""
        result = run_cli("list", "backbones")
        for name in visbench.list_backbones():
            assert name in result.out


class TestCache:
    def test_stats_reports_an_empty_cache(self, run_cli, cache_dir):
        result = run_cli("cache", "stats", "--cache", str(cache_dir))
        assert result.code == 0
        assert "entries: 0" in result.out

    def test_clear_refuses_without_confirmation(self, run_cli, cache_dir, image_folder):
        run_cli(
            "run",
            "retrieval",
            "--data",
            str(image_folder),
            "--split",
            "val",
            "--backbone",
            FAKE_BACKBONE,
            "--device",
            "cpu",
            "--cache",
            str(cache_dir),
            "--results",
            "none",
        )
        result = run_cli("cache", "clear", "--cache", str(cache_dir))
        assert result.code == 1
        assert "--yes" in result.out
        # And nothing was deleted.
        stats = run_cli("cache", "stats", "--cache", str(cache_dir))
        assert "entries: 0" not in stats.out

    def test_clear_with_yes_removes_entries(self, run_cli, cache_dir, image_folder):
        run_cli(
            "run",
            "retrieval",
            "--data",
            str(image_folder),
            "--split",
            "val",
            "--backbone",
            FAKE_BACKBONE,
            "--device",
            "cpu",
            "--cache",
            str(cache_dir),
            "--results",
            "none",
        )
        result = run_cli("cache", "clear", "--cache", str(cache_dir), "--yes")
        assert result.code == 0
        assert "removed 9 entries" in result.out

    def test_clear_scoped_to_a_backbone_resolves_its_key(self, run_cli, cache_dir, image_folder):
        """--backbone takes the registered name; the directory is named after
        the cache key, which is a different string."""
        run_cli(
            "run",
            "retrieval",
            "--data",
            str(image_folder),
            "--split",
            "val",
            "--backbone",
            FAKE_BACKBONE,
            "--device",
            "cpu",
            "--cache",
            str(cache_dir),
            "--results",
            "none",
        )
        result = run_cli(
            "cache", "clear", "--cache", str(cache_dir), "--backbone", FAKE_BACKBONE, "--yes"
        )
        assert result.code == 0
        assert "removed 9 entries" in result.out


def _run(run_cli, probe, data, cache_dir, results, *extra):
    return run_cli(
        "run",
        probe,
        "--data",
        str(data),
        "--backbone",
        FAKE_BACKBONE,
        "--device",
        "cpu",
        "--cache",
        str(cache_dir),
        "--results",
        str(results),
        *extra,
    )


class TestRun:
    def test_zero_shot_probe_scores_and_writes(self, run_cli, image_folder, cache_dir, tmp_path):
        results = tmp_path / "out.jsonl"
        result = _run(run_cli, "retrieval", image_folder, cache_dir, results, "--split", "val")

        assert result.code == 0
        assert "recall@1" in result.out
        records = read_records(results)
        assert len(records) == 1
        assert records[0].task == "retrieval"
        # The instance's own name, not the registry key it was looked up under —
        # one class may claim several names, so the instance is what knows.
        assert records[0].backbone == "fake_vit"

    def test_a_trained_probe_uses_both_splits(self, run_cli, image_folder, cache_dir, tmp_path):
        results = tmp_path / "out.jsonl"
        result = _run(run_cli, "classification", image_folder, cache_dir, results, "--epochs", "5")
        assert result.code == 0
        assert "training: 12 items" in result.out
        assert "scoring: 9 items" in result.out
        assert read_records(results)[0].split == "val"

    def test_correspondence_runs_and_reports_its_ceiling(
        self, run_cli, flat_folder, cache_dir, tmp_path
    ):
        """The probe run() could not express before step 5j."""
        results = tmp_path / "out.jsonl"
        result = _run(
            run_cli,
            "correspondence",
            flat_folder,
            cache_dir,
            results,
            "--split",
            "test",
            "--image-size",
            "64",
        )
        assert result.code == 0
        metrics = read_records(results)[0].metrics
        assert "recall@1p" in metrics and "ceiling_recall@1p" in metrics
        assert "ceiling" in result.out

    def test_correspondence_records_its_warp(self, run_cli, flat_folder, cache_dir, tmp_path):
        results = tmp_path / "out.jsonl"
        _run(
            run_cli,
            "correspondence",
            flat_folder,
            cache_dir,
            results,
            "--split",
            "test",
            "--image-size",
            "64",
            "--max-warp",
            "0.35",
        )
        assert read_records(results)[0].dataset_params["max_warp"] == 0.35

    def test_similarity_runs_from_a_nights_folder(
        self, run_cli, two_afc_folder, cache_dir, tmp_path
    ):
        root = tmp_path / "nights"
        two_afc_folder(root, triplets=6, construct=False)
        results = tmp_path / "out.jsonl"
        result = _run(run_cli, "similarity", root, cache_dir, results)
        assert result.code == 0
        assert "accuracy" in result.out

    def test_a_dense_probe_runs(self, run_cli, dense_folder, cache_dir, tmp_path):
        results = tmp_path / "out.jsonl"
        result = _run(
            run_cli,
            "generic_segmentation",
            dense_folder,
            cache_dir,
            results,
            "--image-size",
            "64",
            "--epochs",
            "2",
            "--train-batch-size",
            "2",
        )
        assert result.code == 0
        assert "iou" in result.out
        assert read_records(results)[0].dataset_params["image_size"] == 64

    def test_results_none_writes_nothing(self, run_cli, image_folder, cache_dir, tmp_path):
        results = tmp_path / "out.jsonl"
        result = _run(
            run_cli, "retrieval", image_folder, cache_dir, results.name and "none", "--split", "val"
        )
        assert result.code == 0
        assert not results.exists()

    def test_json_prints_the_whole_record(self, run_cli, image_folder, cache_dir, tmp_path):
        result = _run(
            run_cli,
            "retrieval",
            image_folder,
            cache_dir,
            tmp_path / "o.jsonl",
            "--split",
            "val",
            "--json",
        )
        payload = json.loads(result.out[result.out.index("{") :])
        assert payload["task"] == "retrieval"
        assert payload["schema_version"] == 5

    def test_no_cache_still_runs(self, run_cli, image_folder, cache_dir, tmp_path):
        result = _run(
            run_cli,
            "retrieval",
            image_folder,
            cache_dir,
            tmp_path / "o.jsonl",
            "--split",
            "val",
            "--no-cache",
        )
        assert result.code == 0
        assert not cache_dir.exists() or not list(cache_dir.rglob("*.pt"))


class TestLimit:
    def test_limit_is_per_class_for_a_labelled_folder(
        self, run_cli, image_folder, cache_dir, tmp_path
    ):
        """A prefix would take all three images from one class, and a
        single-class retrieval scores 1.0 while measuring nothing."""
        results = tmp_path / "out.jsonl"
        _run(
            run_cli, "retrieval", image_folder, cache_dir, results, "--split", "val", "--limit", "1"
        )
        record = read_records(results)[0]
        assert record.dataset_size == 3
        assert record.dataset_params["num_classes"] == 3

    def test_limit_shortens_a_triplet_split_by_triplet(
        self, run_cli, two_afc_folder, cache_dir, tmp_path
    ):
        """Not by image: the triplets index into the image list."""
        root = tmp_path / "nights"
        two_afc_folder(root, triplets=6, construct=False)
        results = tmp_path / "out.jsonl"
        _run(run_cli, "similarity", root, cache_dir, results, "--limit", "2")
        assert read_records(results)[0].dataset_params["num_triplets"] == 2

    def test_limit_keeps_a_dense_split_aligned(self, run_cli, dense_folder, cache_dir, tmp_path):
        results = tmp_path / "out.jsonl"
        _run(
            run_cli,
            "generic_segmentation",
            dense_folder,
            cache_dir,
            results,
            "--image-size",
            "64",
            "--epochs",
            "2",
            "--train-batch-size",
            "2",
            "--limit",
            "2",
        )
        assert read_records(results)[0].dataset_size == 2


class TestErrors:
    def test_a_missing_folder_is_one_line_not_a_traceback(self, run_cli, tmp_path):
        result = run_cli(
            "run",
            "retrieval",
            "--data",
            str(tmp_path / "nope"),
            "--backbone",
            FAKE_BACKBONE,
            "--results",
            "none",
        )
        assert result.code == 2

    def test_traceback_flag_re_raises(self, run_cli, tmp_path):
        """So a bug inside a training loop is still debuggable."""
        with pytest.raises(NotADirectoryError):
            run_cli(
                "run",
                "retrieval",
                "--data",
                str(tmp_path / "nope"),
                "--backbone",
                FAKE_BACKBONE,
                "--results",
                "none",
                "--traceback",
            )

    def test_an_unknown_backbone_names_the_known_ones(self, run_cli, image_folder):
        result = run_cli(
            "run",
            "retrieval",
            "--data",
            str(image_folder),
            "--split",
            "val",
            "--backbone",
            "not_a_backbone",
            "--results",
            "none",
        )
        assert result.code == 2
        assert "Available" in result.err


class TestProbeKwargs:
    """The flags have to reach the probe, or they are decoration."""

    def test_correspondence_units_reach_the_probe(self):
        args = visbench.cli.build_parser().parse_args(
            ["run", "correspondence", "--data", "x", "--units", "pixel"]
        )
        assert spec_for("correspondence").probe_kwargs(args)["threshold_units"] == "pixel"

    def test_the_dense_schedule_reaches_the_probe(self):
        args = visbench.cli.build_parser().parse_args(
            ["run", "depth", "--data", "x", "--epochs", "3", "--lr", "0.01", "--head", "dpt"]
        )
        kwargs = spec_for("depth").probe_kwargs(args)
        assert kwargs["epochs"] == 3 and kwargs["lr"] == 0.01 and kwargs["head"] == "dpt"

    def test_plain_loss_turns_off_the_uncertainty_aware_one(self):
        args = visbench.cli.build_parser().parse_args(
            ["run", "surface_normal", "--data", "x", "--plain-loss"]
        )
        assert spec_for("surface_normal").probe_kwargs(args)["uncertainty_aware"] is False


class TestStemsLayout:
    """Splits named by a file rather than by a directory.

    How every real benchmark expresses an official split — VOC ships 17k images
    beside 2.9k segmentation labels — and without it the CLI could not run the
    one dataset semantic segmentation is proved on.
    """

    @pytest.fixture
    def devkit(self, tmp_path):
        """One flat root with a listing per split, VOC-style."""
        import numpy as np

        root = tmp_path / "devkit"
        (root / "JPEGImages").mkdir(parents=True)
        (root / "Labels").mkdir()
        (root / "Sets").mkdir()
        for index in range(6):
            stem = f"2007_{index:06d}"
            Image.new("RGB", (64, 64), (30 * index, 120, 200)).save(
                root / "JPEGImages" / f"{stem}.jpg"
            )
            label = np.zeros((64, 64), dtype=np.uint8)
            label[: 20 + 6 * index] = 1
            np.save(root / "Labels" / f"{stem}.npy", label)
        (root / "Sets" / "train.txt").write_text(
            "\n".join(f"2007_{i:06d}" for i in range(4)) + "\n"
        )
        (root / "Sets" / "val.txt").write_text("\n".join(f"2007_{i:06d}" for i in range(4, 6)))
        return root

    def _argv(self, devkit, **overrides):
        argv = {
            "--image-dir": "JPEGImages",
            "--target-dir": "Labels",
            "--stems": str(devkit / "Sets" / "val.txt"),
            "--train-stems": str(devkit / "Sets" / "train.txt"),
            "--num-classes": "2",
            "--image-size": "64",
            "--epochs": "2",
            "--train-batch-size": "2",
        }
        argv.update(overrides)
        return [item for pair in argv.items() for item in pair]

    def test_a_split_list_selects_the_named_stems(self, run_cli, devkit, cache_dir, tmp_path):
        results = tmp_path / "out.jsonl"
        result = _run(
            run_cli, "semantic_segmentation", devkit, cache_dir, results, *self._argv(devkit)
        )
        assert result.code == 0, result.err
        records = read_records(results)
        assert records[0].dataset_size == 2  # val.txt names two, of six on disk
        assert records[0].split == "val"

    def test_limit_truncates_the_stem_list(self, run_cli, devkit, cache_dir, tmp_path):
        results = tmp_path / "out.jsonl"
        _run(
            run_cli,
            "semantic_segmentation",
            devkit,
            cache_dir,
            results,
            *self._argv(devkit),
            "--limit",
            "1",
        )
        assert read_records(results)[0].dataset_size == 1

    def test_one_stems_file_without_the_other_is_refused(
        self, run_cli, devkit, cache_dir, tmp_path
    ):
        """Naming one split by file and the other by directory mixes layouts."""
        argv = self._argv(devkit)
        del argv[argv.index("--train-stems") : argv.index("--train-stems") + 2]
        result = _run(run_cli, "semantic_segmentation", devkit, cache_dir, "none", *argv)
        assert result.code == 2
        assert "together, or neither" in result.err

    def test_a_missing_listing_says_where_it_looked(self, run_cli, devkit, cache_dir, tmp_path):
        result = _run(
            run_cli,
            "semantic_segmentation",
            devkit,
            cache_dir,
            "none",
            *self._argv(devkit, **{"--stems": str(devkit / "Sets" / "nope.txt")}),
        )
        assert result.code == 2
        assert "No split list at" in result.err
