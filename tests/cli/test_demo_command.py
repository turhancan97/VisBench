"""`visbench demo` — the command a new user runs first.

Its failure mode is not a red test: it is someone concluding the library does
not work and leaving. Lives here rather than in tests/test_demo.py because the
CLI fixtures (a registered fake backbone, a scratch cache) are defined in this
package's conftest.
"""

import pytest

from visbench.demo import SHAPES  # noqa: F401  - imported for the layout it defines


class TestDemoCommand:
    """The command, with the torchvision backbone swapped for a fake one.

    Patched rather than skipped: the interesting failures here are in the
    plumbing — argument defaults, the dataset layout, the run call — and none of
    them needs 45 MB of real weights to catch. `TestDemoOnRealWeights` covers
    the part that does.
    """

    @pytest.fixture
    def patched(self, monkeypatch, fake_vit):
        monkeypatch.setattr("visbench.demo.demo_backbone", lambda device=None: fake_vit)
        return fake_vit

    def test_it_runs_and_reports_a_score(self, run_cli, patched, cache_dir, tmp_path):
        result = run_cli(
            "demo", "--cache", str(cache_dir), "--images", "3", "--data", str(tmp_path / "d")
        )
        assert result.code == 0
        assert "top1" in result.out
        assert "chance is 0.25" in result.out

    def test_retrieval_is_also_offered(self, run_cli, patched, cache_dir, tmp_path):
        result = run_cli(
            "demo",
            "--probe",
            "retrieval",
            "--cache",
            str(cache_dir),
            "--images",
            "3",
            "--data",
            str(tmp_path / "d"),
        )
        assert result.code == 0
        assert "mAP" in result.out

    def test_it_needs_no_arguments_at_all(self, patched, monkeypatch, tmp_path):
        """The whole point: `visbench demo` with nothing after it."""
        from visbench.cli.main import build_parser

        args = build_parser().parse_args(["demo"])
        assert args.data is None, "the default must not require a path"
        assert args.probe == "classification"
        assert args.images == 20

    def test_the_images_land_where_asked(self, run_cli, patched, cache_dir, tmp_path):
        target = tmp_path / "kept"
        run_cli("demo", "--cache", str(cache_dir), "--images", "2", "--data", str(target))
        assert (target / "train" / "circle").is_dir()
        assert len(list((target / "val" / "cross").glob("*.png"))) == 2

    def test_it_writes_no_results_file_by_default(self, run_cli, patched, cache_dir, tmp_path):
        """A first run must not litter the user's working directory."""
        monkeypatched_cwd = tmp_path / "cwd"
        monkeypatched_cwd.mkdir()
        import os

        previous = os.getcwd()
        os.chdir(monkeypatched_cwd)
        try:
            run_cli("demo", "--cache", str(cache_dir), "--images", "2")
        finally:
            os.chdir(previous)
        assert not (monkeypatched_cwd / "results").exists()
