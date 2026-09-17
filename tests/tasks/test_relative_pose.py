"""Relative camera pose — the head, and the probe that fits it.

The probe is unusual here in three ways, and the tests are mostly about those.

**Its head is not linear**, which every other board here is. That is a measured
departure rather than a convenience, so `hidden_dims=()` stays reachable as the
control and the record says which was used.

**Its labels are structure, not supervision** — the pairs say which two of the
extracted frames form a comparison — so `fit`, `predict` and `evaluate` all
have to assemble the same pairing, in the same column order. They share one
helper for that reason: the anchor is column 0, and the reverse is the same
magnitude with the wrong sign.

**Its score is uninterpretable without the floor.** Pairs drawn within 120
degrees have a median near 65, so a constant already scores about 67 and a
backbone landing there is at chance rather than weak. The floor is fitted on
the training targets, travels as `floor_*`, and must survive an artifact round
trip — a reloaded probe that can predict but cannot say what beats a constant
is a probe whose numbers cannot be read.
"""

import math

import pytest
import torch

from visbench.heads import build_head, list_heads
from visbench.heads.pose import PoseHead
from visbench.metrics.pose import rotation_angle_deg, rotation_error_deg
from visbench.tasks.mid_level.pose import RelativePoseTask
from visbench.utils.seed import set_seed


def about_z(degrees: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` wxyz quaternions rotating about z by ``degrees``."""
    half = torch.deg2rad(degrees) / 2
    zero = torch.zeros_like(half)
    return torch.stack([half.cos(), zero, zero, half.sin()], dim=-1)


def learnable_split(frames: int = 60, pairs: int = 500, seed: int = 0):
    """Features that carry the answer, and pairs whose pose follows from them.

    Frame ``k`` holds its own angle in channel 0 and noise elsewhere; a pair's
    relative pose is the difference of the two angles. So a head that reads
    both halves can recover it exactly and one that cannot is failing at
    something this test can name.
    """
    generator = torch.Generator().manual_seed(seed)
    angles = torch.rand(frames, generator=generator)
    features = torch.randn(frames, 8, generator=generator) * 0.1
    features[:, 0] = angles

    index = torch.randint(0, frames, (pairs, 2), generator=generator)
    index = index[index[:, 0] != index[:, 1]]
    turn = (angles[index[:, 1]] - angles[index[:, 0]]) * 90.0
    pose = torch.cat([about_z(turn), torch.zeros(len(index), 3)], dim=1)
    return features, (index, pose)


class TestTheHead:
    def test_it_is_registered(self):
        assert "pose" in list_heads()
        assert isinstance(build_head("pose", embed_dim=4), PoseHead)

    def test_it_reads_a_pair_not_one_view(self):
        head = PoseHead(embed_dim=16)
        assert head.in_features == 32
        assert head(torch.randn(4, 32)).shape == (4, 7)

    def test_the_wrong_width_names_both_readings(self):
        """`embed_dim` is one view's width, and passing the concatenated one
        builds a head twice as wide with every shape still checking out."""
        with pytest.raises(ValueError, match="2 x embed_dim"):
            PoseHead(embed_dim=16)(torch.randn(4, 16))

    def test_a_dense_map_is_refused(self):
        with pytest.raises(ValueError, match="pooled vectors"):
            PoseHead(embed_dim=4)(torch.randn(2, 8, 7, 7))

    def test_no_hidden_layers_is_one_affine_map(self):
        """The control, reachable by name rather than by a second class."""
        head = PoseHead(embed_dim=4, hidden_dims=())
        assert sum(1 for module in head.mlp if isinstance(module, torch.nn.Linear)) == 1
        assert head(torch.randn(3, 8)).shape == (3, 7)

    def test_the_batchnorm_is_fitted_state_and_rides_in_the_state_dict(self):
        """It is the `DetectionTask.grid_hw` class of trap: state outside the
        parameters that decides what a reloaded probe predicts."""
        assert "mlp.0.running_mean" in PoseHead(embed_dim=4).state_dict()


class TestFitting:
    def test_it_learns_a_pose_its_features_determine(self):
        """End to end on features that carry the answer, so a failure here is
        the probe rather than the representation."""
        set_seed(0)
        features, labels = learnable_split()
        probe = RelativePoseTask(epochs=40, batch_size=32).fit(features, labels)
        metrics = probe.evaluate(features, labels)
        floor = probe.context_metrics(features, labels)
        assert metrics["rotation_error_deg"] < floor["floor_rotation_error_deg"] / 2

    def test_the_fit_is_reported_and_is_not_a_score(self):
        set_seed(0)
        features, labels = learnable_split(pairs=200)
        probe = RelativePoseTask(epochs=5, batch_size=32)
        assert probe.training_summary() is None
        probe.fit(features, labels)
        assert set(probe.training_summary()) == {"train_loss"}
        assert probe.training_summary()["train_loss"] > 0

    def test_the_same_seed_gives_the_same_probe(self):
        features, labels = learnable_split(pairs=200)
        scores = []
        for _ in range(2):
            set_seed(3)
            probe = RelativePoseTask(epochs=5, batch_size=32).fit(features, labels)
            scores.append(probe.evaluate(features, labels)["rotation_error_deg"])
        assert scores[0] == pytest.approx(scores[1], abs=1e-9)

    def test_a_singleton_trailing_batch_does_not_crash_the_run(self):
        """The head starts with a BatchNorm, which cannot normalise one row."""
        set_seed(0)
        features, labels = learnable_split(pairs=200)
        index, pose = labels
        keep = (len(index) // 32) * 32 + 1
        probe = RelativePoseTask(epochs=2, batch_size=32)
        probe.fit(features, (index[:keep], pose[:keep]))
        assert probe.train_pairs == keep

    def test_a_batch_of_one_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="batch_size must be >= 2"):
            RelativePoseTask(batch_size=1)


class TestThePairing:
    def test_the_anchor_is_the_first_column(self):
        """Pinned because reversing it gives the same magnitude and the wrong
        sign, which is a plausible number rather than an error."""
        features = torch.eye(4)
        index = torch.tensor([[0, 1]])
        pose = torch.zeros(1, 7)
        pose[:, 0] = 1.0
        probe = RelativePoseTask()
        paired, _ = probe._paired(features, (index, pose))
        assert torch.equal(paired[0, :4], features[0])
        assert torch.equal(paired[0, 4:], features[1])

    def test_pairs_are_required(self):
        with pytest.raises(ValueError, match="pass dataset.labels"):
            RelativePoseTask()._paired(torch.randn(4, 8), None)

    def test_an_index_outside_the_features_is_refused(self):
        index = torch.tensor([[0, 9]])
        pose = torch.zeros(1, 7)
        with pytest.raises(IndexError, match="outside the 4 extracted features"):
            RelativePoseTask()._paired(torch.randn(4, 8), (index, pose))

    @pytest.mark.parametrize(
        "labels, message",
        [
            ((torch.zeros(3, 3, dtype=torch.long), torch.zeros(3, 7)), r"\(P, 2\)"),
            ((torch.zeros(3, 2, dtype=torch.long), torch.zeros(3, 6)), r"\(P, 7\)"),
            ((torch.zeros(3, 2, dtype=torch.long), torch.zeros(2, 7)), "3 pairs for 2"),
        ],
    )
    def test_malformed_pairs_are_named(self, labels, message):
        with pytest.raises(ValueError, match=message):
            RelativePoseTask()._paired(torch.randn(4, 8), labels)

    def test_a_non_pair_label_says_what_was_expected(self):
        with pytest.raises(TypeError, match="PosePairs"):
            RelativePoseTask()._paired(torch.randn(4, 8), 7)


class TestScoring:
    def test_predict_returns_the_raw_seven_vector(self):
        """Unnormalised, because nothing in this protocol normalises it: the
        loss is MSE over the 7-vector and the metric projects through the
        matrix. Normalising here alone would make three functions of two."""
        set_seed(0)
        features, labels = learnable_split(pairs=200)
        probe = RelativePoseTask(epochs=2, batch_size=32).fit(features, labels)
        predicted = probe.predict(features, labels)
        assert predicted.shape == (len(labels[0]), 7)
        assert not torch.allclose(predicted[:, :4].norm(dim=-1), torch.ones(len(predicted)))

    def test_predict_before_fit_says_so(self):
        features, labels = learnable_split(pairs=10)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            RelativePoseTask().predict(features, labels)

    def test_features_from_another_backbone_are_refused(self):
        set_seed(0)
        features, labels = learnable_split(pairs=100)
        probe = RelativePoseTask(epochs=2, batch_size=32).fit(features, labels)
        wider = torch.randn(len(features), 16)
        with pytest.raises(ValueError, match="same backbone"):
            probe.predict(wider, labels)

    def test_the_metrics_are_the_pose_ones(self):
        set_seed(0)
        features, labels = learnable_split(pairs=100)
        probe = RelativePoseTask(epochs=2, batch_size=32).fit(features, labels)
        assert set(probe.evaluate(features, labels)) == {
            "rotation_error_deg",
            "rotation_median_deg",
            "rotation_acc@15deg",
            "rotation_acc@30deg",
            "translation_error",
        }


class TestTheFloor:
    def test_it_is_empty_before_the_fit(self):
        """It is fitted like anything else; a floor taken from the split being
        scored would be unbeatable in a way no head is."""
        features, labels = learnable_split(pairs=50)
        assert RelativePoseTask().context_metrics(features, labels) == {}

    def test_it_travels_as_floor_and_cannot_collide_with_a_score(self):
        set_seed(0)
        features, labels = learnable_split(pairs=200)
        probe = RelativePoseTask(epochs=2, batch_size=32).fit(features, labels)
        floor = probe.context_metrics(features, labels)
        assert set(floor) == {f"floor_{key}" for key in probe.evaluate(features, labels)}

    def test_it_is_what_a_constant_scores(self):
        set_seed(0)
        features, labels = learnable_split(pairs=300)
        probe = RelativePoseTask(epochs=2, batch_size=32).fit(features, labels)
        index, pose = labels
        constant = probe._train_mean.expand(len(pose), 7)
        expected = float(rotation_error_deg(constant[:, :4], pose[:, :4]).mean())
        assert probe.context_metrics(features, labels)["floor_rotation_error_deg"] == pytest.approx(
            expected, abs=1e-4
        )

    def test_the_draw_is_what_sets_it(self):
        """Not a property of a backbone: it is the mean pose of whatever pairs
        were drawn, so a wider draw moves it and two boards with different
        floors cannot be compared by subtracting them."""
        set_seed(0)
        features, labels = learnable_split(pairs=300)
        index, pose = labels
        narrow = rotation_angle_deg(pose[:, :4]) < 20
        probe = RelativePoseTask(epochs=2, batch_size=32).fit(features, labels)
        wide_floor = probe.context_metrics(features, labels)["floor_rotation_error_deg"]
        narrow_floor = probe.context_metrics(features, (index[narrow], pose[narrow]))[
            "floor_rotation_error_deg"
        ]
        assert narrow_floor != pytest.approx(wide_floor, abs=1.0)


class TestTheRecord:
    def test_task_params_carry_what_decides_the_number(self):
        set_seed(0)
        features, labels = learnable_split(pairs=120)
        probe = RelativePoseTask(epochs=2, batch_size=32).fit(features, labels)
        params = probe.describe()["task_params"]
        assert params["protocol"] == "probe3d_pose"
        assert params["head"] == "mlp"
        assert params["hidden_dims"] == [512, 256, 128]
        assert params["train_pairs"] == len(labels[0])
        assert params["optimizer"] == "adamw" and params["schedule"] == "cosine"

    def test_the_pair_count_is_in_the_key_because_the_score_has_not_converged(self):
        """Two pose numbers drawn from different pair sets are not two
        measurements of the same thing — error falls by degrees between pair
        counts, so they must not share a comparability group."""
        set_seed(0)
        features, labels = learnable_split(pairs=200)
        index, pose = labels
        many = RelativePoseTask(epochs=1, batch_size=32).fit(features, labels)
        few = RelativePoseTask(epochs=1, batch_size=32).fit(features, (index[:100], pose[:100]))
        assert (
            many.describe()["task_params"]["train_pairs"]
            != few.describe()["task_params"]["train_pairs"]
        )

    def test_the_linear_control_is_recorded_as_one(self):
        set_seed(0)
        features, labels = learnable_split(pairs=120)
        probe = RelativePoseTask(hidden_dims=(), epochs=2, batch_size=32).fit(features, labels)
        params = probe.describe()["task_params"]
        assert params["head"] == "linear" and params["hidden_dims"] == []

    def test_it_is_a_frozen_probe(self):
        assert RelativePoseTask().finetune() is None

    def test_the_head_spec_rebuilds_the_head(self):
        set_seed(0)
        features, labels = learnable_split(pairs=120)
        probe = RelativePoseTask(epochs=2, batch_size=32).fit(features, labels)
        spec = probe.head_spec()
        assert spec["kind"] == "registered" and spec["name"] == "pose"
        rebuilt = build_head(spec["name"], **spec["kwargs"])
        rebuilt.load_state_dict(probe.head.state_dict())
        assert rebuilt.in_features == probe.head.in_features

    def test_the_floor_survives_an_artifact_round_trip(self):
        """A reloaded probe that can predict and cannot say what beats a
        constant has numbers that cannot be read."""
        set_seed(0)
        features, labels = learnable_split(pairs=200)
        probe = RelativePoseTask(epochs=2, batch_size=32).fit(features, labels)

        reloaded = RelativePoseTask(epochs=2, batch_size=32)
        # Rebuilt exactly as `visbench.hub.load_probe` does, device included —
        # the same head on two devices agrees only to about 1e-5, which would
        # make an equality here a test of float arithmetic rather than of the
        # state that was saved.
        reloaded.head = build_head("pose", **probe.head_spec()["kwargs"]).to(probe.device)
        reloaded.head.load_state_dict(probe.head.state_dict())
        reloaded.head.eval()
        reloaded.load_probe_state(probe.probe_state())

        assert reloaded.evaluate(features, labels) == probe.evaluate(features, labels)
        assert reloaded.context_metrics(features, labels) == probe.context_metrics(features, labels)

    @pytest.mark.parametrize(
        "state, message",
        [
            ({"nonsense": torch.zeros(1, 7)}, "train_pose_mean"),
            ({"train_pose_mean": torch.zeros(7)}, r"\(1, 7\)"),
        ],
    )
    def test_bad_probe_state_is_refused_rather_than_dropped(self, state, message):
        with pytest.raises(ValueError, match=message):
            RelativePoseTask().load_probe_state(state)


def test_a_rotation_helper_sanity_check():
    """The fixture's own maths, so a failure above is the probe."""
    assert float(rotation_angle_deg(about_z(torch.tensor(45.0)))) == pytest.approx(45.0, abs=1e-3)
    assert math.isclose(float(about_z(torch.tensor(0.0))[0]), 1.0)
