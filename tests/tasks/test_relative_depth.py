"""The relative-depth control: a rejected probe kept unregistered.

`RelativeDepthTask` is not a registered probe. It was built as a candidate,
measured against the `depth` board it subclasses, and rejected: Spearman between
the two readouts is +1.000 over five backbones, at 38% of the spread, with a
smallest adjacent gap of 0.0007 where `depth` manages 0.0707. Its records are a
control (`results/controls/relative_depth.jsonl`), because the rejection is a
finding about a board that ships.

These tests cover the two things that could still go wrong: the class quietly
becoming a probe, and the metric quietly changing what it measures.
"""

import json
from pathlib import Path

import pytest
import torch

import visbench
from visbench.metrics.dense import ordinal_metrics
from visbench.results.leaderboard import (
    DIAGNOSTIC_METRICS,
    METRIC_DIRECTIONS,
    UnknownMetric,
    metric_direction,
)
from visbench.tasks.mid_level.relative_depth import RelativeDepthTask, relative_depth_loss

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "results" / "controls" / "relative_depth.jsonl"


class TestItIsNotAProbe:
    """Unregistered is load-bearing, not an oversight.

    A registered probe is pinned by test to carry a corpus board, a CLI row, a
    `TARGET_STYLES` entry and a committed gallery figure. This readout earned
    none of the four, so registering it would either break those tests or --
    worse -- be "fixed" by giving it a board it did not earn.
    """

    def test_it_is_absent_from_the_registry(self):
        assert "relative_depth" not in visbench.list_probes()

    def test_the_registry_refuses_it_by_name(self):
        with pytest.raises(Exception, match="relative_depth|[Uu]nknown"):
            visbench.get_probe("relative_depth")

    def test_it_is_absent_from_the_cli(self):
        """`visbench run relative_depth` must not resolve."""
        from visbench.cli.datasets import SPECS

        assert "relative_depth" not in SPECS

    def test_it_is_absent_from_the_corpus_scripts(self):
        """A corpus board is exactly what it did not earn."""
        for path in ("scripts/build_corpus.sh", "slurm/corpus.sbatch"):
            assert "relative_depth" not in (ROOT / path).read_text(), path

    def test_the_class_still_names_itself(self):
        """Unregistered, but the record must still say what produced it."""
        assert RelativeDepthTask().name == "relative_depth"
        assert RelativeDepthTask().protocol == "visbench_relative_depth"


class TestTheMetric:
    def test_a_perfect_ranking_scores_one(self):
        """A prediction monotone in depth orders every pair correctly.

        The convention is that a LOWER score means NEARER, fixed by
        `relative_depth_loss` (it minimises `softplus(score_nearer -
        score_farther)`) and read back the same way by the metric. So a correct
        head's score increases with depth, and the depth map itself is a perfect
        prediction. The first draft of this test negated it and failed, which is
        the argument for one of the two fixing the sign rather than both
        choosing.
        """
        depth = torch.rand(2, 32, 32) * 5 + 0.5
        scores = ordinal_metrics(depth, depth, pairs_per_image=200)
        assert scores["ordinal_accuracy"] == pytest.approx(1.0)

    def test_the_wrong_sign_scores_zero(self):
        """Which is what makes the convention testable rather than assumed."""
        depth = torch.rand(2, 32, 32) * 5 + 0.5
        assert ordinal_metrics(-depth, depth, pairs_per_image=200)[
            "ordinal_accuracy"
        ] == pytest.approx(0.0)

    def test_a_constant_prediction_ties_everything(self):
        """No information, and a tie is scored as wrong rather than as half.

        `predicted = score_a < score_b` is False for every pair, so a constant
        head scores whatever fraction of pairs happen to have `depth_a > depth_b`
        -- about half. Pinning "not near 1.0" is the claim that matters: a head
        that has learned nothing cannot look good.
        """
        depth = torch.rand(4, 32, 32) * 5 + 0.5
        accuracy = ordinal_metrics(torch.zeros(4, 32, 32), depth, pairs_per_image=500)[
            "ordinal_accuracy"
        ]
        assert 0.3 < accuracy < 0.7

    def test_invalid_pixels_are_never_sampled(self):
        """A pixel with depth <= 0 is invalid -- the depth convention.

        Here only a 4x4 corner is valid and it is ordered correctly, so a run
        that sampled the zeros could not score 1.0.
        """
        depth = torch.zeros(1, 32, 32)
        depth[0, :4, :4] = torch.arange(1, 17).reshape(4, 4).float()
        assert ordinal_metrics(depth, depth, pairs_per_image=300)[
            "ordinal_accuracy"
        ] == pytest.approx(1.0)

    def test_the_vertical_baseline_is_reported_and_beats_chance_on_a_floor(self):
        """The number the whole rejection turned on.

        A synthetic frame whose depth increases downward is what an indoor floor
        looks like, and "the lower point is nearer" is then WRONG everywhere --
        the point is that the baseline is sensitive to that structure at all,
        which is why it must travel beside the score.
        """
        depth = torch.arange(1, 33).float().reshape(1, 32, 1).expand(1, 32, 32).contiguous()
        scores = ordinal_metrics(depth, depth, pairs_per_image=400)
        assert scores["ordinal_accuracy"] == pytest.approx(1.0)
        assert scores["ordinal_vertical"] == pytest.approx(0.0)

    def test_it_is_seeded_per_image_so_a_rerun_reproduces(self):
        depth = torch.rand(3, 24, 24) * 5 + 0.5
        pred = torch.rand(3, 24, 24)
        first = ordinal_metrics(pred, depth, pairs_per_image=250)
        assert first == ordinal_metrics(pred, depth, pairs_per_image=250)

    def test_an_images_score_does_not_depend_on_what_preceded_it(self):
        """Seeded per image *index*, so batching cannot move a number.

        A shared generator would make image 2's pairs depend on images 0 and 1,
        and `evaluate`'s batch-weighted mean would then differ from the
        whole-split number for a reason nothing recorded.
        """
        depth = torch.rand(4, 24, 24) * 5 + 0.5
        pred = torch.rand(4, 24, 24)
        whole = ordinal_metrics(pred[2:3], depth[2:3], pairs_per_image=200)
        # Index 2 of a full batch is index 0 of a slice, so the seeds differ by
        # design; what must hold is that a given (index, content) pair is fixed.
        again = ordinal_metrics(pred[2:3], depth[2:3], pairs_per_image=200)
        assert whole == again

    def test_it_rejects_a_nonsense_pair_count(self):
        with pytest.raises(ValueError, match="pairs_per_image"):
            ordinal_metrics(torch.rand(1, 8, 8), torch.rand(1, 8, 8) + 1, pairs_per_image=0)


class TestTheMetricTables:
    """`ordinal_vertical` is a diagnostic and must be refused as a ranking key."""

    def test_the_accuracy_is_rankable(self):
        assert metric_direction("ordinal_accuracy") == "higher"

    def test_the_baseline_is_not(self):
        assert "ordinal_vertical" in DIAGNOSTIC_METRICS
        assert "ordinal_vertical" not in METRIC_DIRECTIONS
        with pytest.raises(UnknownMetric, match="diagnostic"):
            metric_direction("ordinal_vertical")


class TestTheLoss:
    def test_a_correct_ranking_costs_less_than_a_wrong_one(self):
        depth = torch.rand(2, 16, 16) * 5 + 0.5
        right = relative_depth_loss(depth, depth, pairs_per_image=200)
        wrong = relative_depth_loss(-depth, depth, pairs_per_image=200)
        assert float(right) < float(wrong)

    def test_it_is_finite_on_a_confident_prediction(self):
        """softplus, not log1p(exp(...)).

        The hand-written form overflows to inf past a margin of about 88 in
        float32, and every later loss in the epoch is then nan while the run
        reports a plausible 0.5 accuracy -- detection's unclamped `exp` again.
        """
        depth = torch.rand(2, 16, 16) * 5 + 0.5
        # The WRONG sign at huge magnitude: that is the branch whose margin is
        # large and positive, where a hand-written log1p(exp(...)) overflows.
        loss = relative_depth_loss(-depth * 1e4, depth, pairs_per_image=100)
        assert torch.isfinite(loss)

    def test_it_carries_a_gradient(self):
        depth = torch.rand(1, 16, 16) * 5 + 0.5
        pred = torch.zeros(1, 16, 16, requires_grad=True)
        relative_depth_loss(pred, depth, pairs_per_image=100).backward()
        assert pred.grad is not None and float(pred.grad.abs().sum()) > 0

    def test_an_all_invalid_target_costs_nothing_and_still_has_a_graph(self):
        """No pair to score, so no signal -- but `backward` must not raise."""
        pred = torch.zeros(1, 8, 8, requires_grad=True)
        loss = relative_depth_loss(pred, torch.zeros(1, 8, 8), pairs_per_image=50)
        assert float(loss.detach()) == 0.0
        loss.backward()


class TestTheTask:
    def test_activation_is_the_identity(self):
        """Both obvious activations destroy the ordering rather than shape it.

        A sigmoid saturates and flattens the gradient for exactly the confident
        pairs the loss is still separating; a ReLU makes every negative score
        equal, so every pair among them ties. `edge`'s `_activate` is pinned the
        same way for the parallel reason.
        """
        raw = torch.randn(2, 1, 8, 8)
        assert torch.equal(RelativeDepthTask()._activate(raw), raw)

    def test_it_emits_one_channel_not_bins(self):
        """A ranking loss only ever sees differences, so the constant a
        mean-predictor would learn cancels and `depth`'s 256 bins are not
        needed."""
        assert RelativeDepthTask().out_channels == 1

    def test_it_declares_an_oracle(self):
        """Pooling a depth map averages meaningfully, so the gate applies."""
        target = torch.rand(2, 1, 32, 32) * 5 + 0.5
        oracle = RelativeDepthTask().oracle_prediction(target, (8, 8))
        assert oracle.shape == target.shape

    def test_the_diagnostic_gets_no_ceiling(self):
        """`ceiling_ordinal_vertical` was emitted and was meaningless.

        The base `context_metrics` prefixes every key the oracle returned, and
        one of them is a baseline that does not depend on the prediction -- so
        the value came out bit-identical to `ordinal_vertical`, under a name
        claiming to bound it. Found by reading a run's output, not by a test,
        which is why this one exists.
        """
        task = RelativeDepthTask(batch_size=2)
        context = task.context_metrics(torch.rand(2, 384, 8, 8), torch.rand(2, 1, 32, 32) * 5 + 0.5)
        assert "ceiling_ordinal_accuracy" in context
        assert "ceiling_ordinal_vertical" not in context

    def test_it_rejects_a_nonsense_pair_count(self):
        with pytest.raises(ValueError, match="pairs_per_image"):
            RelativeDepthTask(pairs_per_image=0)


class TestTheControlRecords:
    """The committed control, and the claim it exists to support."""

    @pytest.fixture(scope="class")
    def records(self):
        assert CONTROL.exists(), f"no control at {CONTROL}"
        text = CONTROL.read_text().splitlines()
        rows = [json.loads(line) for line in text if line.strip()]
        return {r["backbone"]: r for r in rows}

    def test_every_record_is_this_readout(self, records):
        for backbone, record in records.items():
            assert record["task"] == "relative_depth", backbone
            assert record["task_params"]["protocol"] == "visbench_relative_depth"
            assert record["task_params"]["activation"] == "identity"

    def test_every_record_carries_its_ceiling_and_its_baseline(self, records):
        """A score read without both says the wrong thing -- see the CAVEAT this
        probe would have carried if it had shipped."""
        for backbone, record in records.items():
            metrics = record["metrics"]
            assert "ceiling_ordinal_accuracy" in metrics, backbone
            assert "ordinal_vertical" in metrics, backbone
            assert "ceiling_ordinal_vertical" not in metrics, backbone

    def test_the_ranking_matches_the_depth_board_which_is_why_it_was_rejected(self, records):
        """The measurement the rejection rests on, pinned.

        If a future change made these two orderings disagree, the rejection's
        premise would be gone and the control's write-up would be wrong -- so
        this fails rather than letting the prose drift from the records.
        """
        corpus = ROOT / "results" / "corpus" / "visbench.jsonl"
        d1 = {}
        for line in corpus.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["task"] == "depth":
                d1[row["backbone"]] = row["metrics"]["d1"]

        shared = sorted(set(records) & set(d1))
        assert len(shared) >= 4, f"only {shared} in both"
        ordinal = [records[b]["metrics"]["ordinal_accuracy"] for b in shared]
        metric = [d1[b] for b in shared]
        by_ordinal = [b for _, b in sorted(zip(ordinal, shared, strict=True), reverse=True)]
        by_metric = [b for _, b in sorted(zip(metric, shared, strict=True), reverse=True)]
        assert by_ordinal == by_metric, "the two readouts no longer agree"

    def test_it_separates_less_than_the_board_it_subclasses(self, records):
        """The other half of the rejection: identical order AND less spread."""
        corpus = ROOT / "results" / "corpus" / "visbench.jsonl"
        d1 = {}
        for line in corpus.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row["task"] == "depth":
                    d1[row["backbone"]] = row["metrics"]["d1"]
        shared = sorted(set(records) & set(d1))
        ordinal = [records[b]["metrics"]["ordinal_accuracy"] for b in shared]
        metric = [d1[b] for b in shared]
        assert max(ordinal) - min(ordinal) < max(metric) - min(metric)
