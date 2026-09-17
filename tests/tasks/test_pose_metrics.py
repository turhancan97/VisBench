"""Relative camera pose — the geometry, before any probe reads it.

Three things here are silent rather than loud when wrong, and each has a test
that would catch it on a number rather than on a traceback.

**The quaternion branch.** `matrix_to_quaternion` picks its branch from the
largest component because the naive ``w = sqrt(1 + trace) / 2`` form loses
every digit near 180 degrees — and pose pairs are drawn to 120, with relative
transforms that exceed it. `test_shepperd_beats_the_naive_form` runs both and
pins that the naive one actually fails on this input distribution, so the
expensive branch is justified by a measurement rather than by folklore.

**The direction of the relative pose.** ``target @ inverse(source)`` and its
opposite are both valid relative poses of the same magnitude, so a probe
trained on one and scored against the other reports a plausible number about
nothing.

**The floor.** A rotation error means nothing on its own: pairs within 120
degrees have a median near 65, so a constant already scores about 67. The floor
is fitted on the *training* targets and scored on the evaluation ones, like any
other predictor, and the tests pin both halves.
"""

import math

import pytest
import torch

from visbench.metrics.pose import (
    POSE_COLUMNS,
    ROTATION_ACC_THRESHOLDS,
    matrix_to_quaternion,
    mean_pose_floor,
    pose_metrics,
    pose_vector,
    quaternion_to_matrix,
    relative_pose,
    rotation_angle_deg,
    rotation_error_deg,
    translation_error,
)


def random_quaternions(count: int, seed: int = 0) -> torch.Tensor:
    """Uniform unit quaternions, wxyz."""
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(count, 4, generator=generator)
    return q / q.norm(dim=-1, keepdim=True)


def about_z(degrees: float) -> torch.Tensor:
    """A rotation of ``degrees`` about the z axis, as a wxyz quaternion."""
    half = math.radians(degrees) / 2
    return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])


def naive_matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """The form this module deliberately does not use — see the test below."""
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    w = torch.sqrt((1.0 + trace).clamp(min=1e-12)) / 2
    x = (m[..., 2, 1] - m[..., 1, 2]) / (4 * w)
    y = (m[..., 0, 2] - m[..., 2, 0]) / (4 * w)
    z = (m[..., 1, 0] - m[..., 0, 1]) / (4 * w)
    return torch.stack([w, x, y, z], dim=-1)


class TestTheConversion:
    def test_round_trips_to_float32_precision(self):
        """matrix -> quaternion -> matrix, over 5000 random rotations."""
        matrices = quaternion_to_matrix(random_quaternions(5000, seed=1))
        recovered = quaternion_to_matrix(matrix_to_quaternion(matrices))
        assert (matrices - recovered).abs().max() < 1e-5

    def near_180(self, count: int = 2000) -> torch.Tensor:
        """Rotations between 170 and 180 degrees — where the naive form dies."""
        angles = torch.linspace(170.0, 179.9, count)
        axes = torch.nn.functional.normalize(
            torch.randn(count, 3, generator=torch.Generator().manual_seed(2)), dim=-1
        )
        half = torch.deg2rad(angles) / 2
        return torch.cat([half.cos().unsqueeze(-1), axes * half.sin().unsqueeze(-1)], dim=-1)

    def test_shepperd_beats_the_naive_form_near_180_degrees(self):
        """The reason for the branch, measured rather than asserted.

        Both forms are correct in exact arithmetic. The naive one divides by
        ``w``, which goes to zero as the rotation approaches 180, so it loses
        precision exactly where these pairs live.

        **Measured on components, not on the angle** — see the next test for
        why, which is the one that stops this being simplified back.
        """
        truth = self.near_180()
        matrices = quaternion_to_matrix(truth)

        def component_error(q: torch.Tensor) -> torch.Tensor:
            q = q / q.norm(dim=-1, keepdim=True)
            sign = torch.where((q * truth).sum(-1, keepdim=True) < 0, -1.0, 1.0)
            return (q * sign - truth).abs().amax(dim=-1)

        ours = component_error(matrix_to_quaternion(matrices))
        naive = component_error(naive_matrix_to_quaternion(matrices))

        assert ours.max() < 1e-6
        assert naive.max() > 1e-5
        # Not a tie: about three orders of magnitude, which is what makes the
        # branch worth its cost rather than a stylistic preference.
        assert naive.max() > ours.max() * 100

    def test_the_angle_cannot_tell_the_two_forms_apart(self):
        """Why the test above reads components, pinned so it stays that way.

        ``acos`` of the trace is ill-conditioned as the relative rotation goes
        to zero, which is exactly where a round-trip check sits. Scored that
        way the broken conversion looks fine — 0.056 degrees against 0.049 —
        and a tolerance tight enough to fail it fails the correct one too.
        """
        truth = self.near_180()
        matrices = quaternion_to_matrix(truth)
        ours = rotation_error_deg(matrix_to_quaternion(matrices), truth)
        naive = rotation_error_deg(naive_matrix_to_quaternion(matrices), truth)

        assert ours.max() < 0.1 and naive.max() < 0.1
        assert (ours > 1e-3).sum() > 100 and (naive > 1e-3).sum() > 100

    def test_the_angle_metrics_own_noise_floor(self):
        """A quaternion against *itself* is not exactly zero degrees.

        0.028 on this sample. It is the same ill-conditioning, it is why every
        tolerance here is 0.05 rather than 1e-3, and it does not grow at the
        20-60 degree scale a pose board reports.
        """
        q = random_quaternions(64, seed=3)
        assert 0.0 < float(rotation_error_deg(q, q).max()) < 0.05

    def test_a_non_rotation_is_projected_not_rejected(self):
        """A head's raw output is not a unit quaternion and must still score."""
        raw = about_z(90.0) * 7.5
        assert float(rotation_error_deg(raw, about_z(90.0))) < 0.05

    def test_the_double_cover_is_not_an_error(self):
        """``q`` and ``-q`` are one rotation, so the error between them is nil."""
        q = random_quaternions(64, seed=3)
        # 0.05 rather than 1e-3: see test_the_angle_metrics_own_noise_floor.
        assert float(rotation_error_deg(q, -q).max()) < 0.05

    def test_shapes_are_checked(self):
        with pytest.raises(ValueError, match="wxyz"):
            quaternion_to_matrix(torch.zeros(4, 3))
        with pytest.raises(ValueError, match=r"\(\.\.\., 3, 3\)"):
            matrix_to_quaternion(torch.zeros(4, 4))


class TestTheAngle:
    @pytest.mark.parametrize("degrees", [0.0, 15.0, 90.0, 179.0])
    def test_rotation_angle_matches_the_construction(self, degrees):
        assert float(rotation_angle_deg(about_z(degrees))) == pytest.approx(degrees, abs=1e-3)

    def test_error_is_symmetric(self):
        first, second = random_quaternions(50, seed=4), random_quaternions(50, seed=5)
        forward = rotation_error_deg(first, second)
        backward = rotation_error_deg(second, first)
        assert (forward - backward).abs().max() < 1e-3

    def test_error_is_the_angle_between(self):
        assert float(rotation_error_deg(about_z(30.0), about_z(100.0))) == pytest.approx(
            70.0, abs=1e-3
        )


class TestTheRelativePose:
    def test_a_camera_against_itself_is_the_identity(self):
        rt = torch.eye(4, dtype=torch.float64)
        rt[:3, :3] = quaternion_to_matrix(random_quaternions(1, seed=6)[0].double())
        rt[:3, 3] = torch.tensor([0.4, -1.2, 3.0], dtype=torch.float64)
        assert (relative_pose(rt, rt) - torch.eye(4, dtype=torch.float64)).abs().max() < 1e-9

    def test_direction_is_target_then_inverse_source(self):
        """Pinned because the opposite composition is equally plausible.

        It is the same magnitude and the wrong sign, so a probe trained on one
        and scored against the other reports a number rather than an error.
        """
        source = torch.eye(4, dtype=torch.float64)
        source[:3, :3] = quaternion_to_matrix(about_z(40.0).double())
        target = torch.eye(4, dtype=torch.float64)
        target[:3, :3] = quaternion_to_matrix(about_z(100.0).double())

        pose = pose_vector(relative_pose(source, target))
        assert float(rotation_angle_deg(pose[:4])) == pytest.approx(60.0, abs=1e-3)
        # And it is the composition, not merely something 60 degrees away.
        expected = quaternion_to_matrix(about_z(60.0))
        assert (quaternion_to_matrix(pose[:4]) - expected).abs().max() < 1e-5

    def test_translation_survives_the_vectorisation(self):
        source = torch.eye(4, dtype=torch.float64)
        target = torch.eye(4, dtype=torch.float64)
        target[:3, 3] = torch.tensor([0.0, 0.0, 2.5], dtype=torch.float64)
        pose = pose_vector(relative_pose(source, target))
        assert pose[4:].tolist() == pytest.approx([0.0, 0.0, 2.5], abs=1e-6)
        assert len(POSE_COLUMNS) == 7


class TestPoseMetrics:
    def exact(self, count: int = 32, seed: int = 7) -> torch.Tensor:
        return torch.cat(
            [random_quaternions(count, seed=seed), torch.randn(count, 3) * 0.1], dim=-1
        )

    def test_a_perfect_prediction_scores_zero_and_one(self):
        target = self.exact()
        metrics = pose_metrics(target, target)
        assert metrics["rotation_error_deg"] < 0.05
        assert metrics["rotation_median_deg"] < 0.05
        assert metrics["translation_error"] < 1e-6
        for threshold in ROTATION_ACC_THRESHOLDS:
            assert metrics[f"rotation_acc_{threshold:g}"] == 1.0

    def test_the_keys_are_what_a_board_will_carry(self):
        target = self.exact()
        assert set(pose_metrics(target, target)) == {
            "rotation_error_deg",
            "rotation_median_deg",
            "rotation_acc_15",
            "rotation_acc_30",
            "translation_error",
        }

    def test_mean_and_median_can_disagree(self):
        """One catastrophic pair moves the mean and not the median.

        Both are reported for this reason: a board quoting only the mean would
        rank on how often a backbone fails badly, which is a different question
        from how well it usually does.
        """
        target = torch.zeros(10, 7)
        target[:, 0] = 1.0
        pred = target.clone()
        pred[0, :4] = about_z(170.0)
        metrics = pose_metrics(pred, target)
        assert metrics["rotation_error_deg"] > 15.0
        assert metrics["rotation_median_deg"] < 0.05
        assert metrics["rotation_acc_30"] == pytest.approx(0.9)

    def test_accuracy_counts_pairs_within_the_threshold(self):
        target = torch.zeros(4, 7)
        target[:, 0] = 1.0
        pred = target.clone()
        for row, degrees in enumerate((5.0, 20.0, 40.0, 100.0)):
            pred[row, :4] = about_z(degrees)
        metrics = pose_metrics(pred, target)
        assert metrics["rotation_acc_15"] == pytest.approx(0.25)
        assert metrics["rotation_acc_30"] == pytest.approx(0.5)

    def test_translation_error_is_a_distance(self):
        assert float(
            translation_error(torch.tensor([3.0, 4.0, 0.0]), torch.zeros(3))
        ) == pytest.approx(5.0)

    def test_a_diverged_head_raises_rather_than_scoring(self):
        """NaN must not reach a record: it is unparseable JSON, and from the
        metric alone a diverged head looks like a hard task."""
        target = self.exact(4)
        pred = target.clone()
        pred[1, 0] = float("nan")
        with pytest.raises(ValueError, match="diverged"):
            pose_metrics(pred, target)

    @pytest.mark.parametrize(
        "pred, target, message",
        [
            (torch.zeros(4, 6), torch.zeros(4, 7), r"\(P, 7\)"),
            (torch.zeros(4, 7), torch.zeros(3, 7), "3"),
            (torch.zeros(0, 7), torch.zeros(0, 7), "empty split"),
        ],
    )
    def test_shape_and_emptiness_are_refused(self, pred, target, message):
        with pytest.raises(ValueError, match=message):
            pose_metrics(pred, target)


class TestTheFloor:
    def constant_train(self, count: int = 20) -> torch.Tensor:
        train = torch.zeros(count, 7)
        train[:, :4] = about_z(50.0)
        train[:, 4:] = torch.tensor([0.0, 0.0, 1.0])
        return train

    def test_it_is_prefixed_so_it_can_travel_beside_a_score(self):
        """``floor_`` is the convention ``ceiling_`` set: one flat dict, no
        collision, and nothing to rank on."""
        val = torch.zeros(5, 7)
        val[:, 0] = 1.0
        floor = mean_pose_floor(self.constant_train(), val)
        assert set(floor) == {f"floor_{key}" for key in pose_metrics(val, val)}

    def test_it_is_what_predicting_the_training_mean_scores(self):
        val = torch.zeros(5, 7)
        val[:, 0] = 1.0
        floor = mean_pose_floor(self.constant_train(), val)
        assert floor["floor_rotation_error_deg"] == pytest.approx(50.0, abs=1e-3)
        assert floor["floor_translation_error"] == pytest.approx(1.0, abs=1e-6)

    def test_it_is_fitted_on_train_and_scored_on_val(self):
        """Fitting it on the split being scored would make it unbeatable in a
        way no head is, and the gap it defines meaningless."""
        val = torch.zeros(5, 7)
        val[:, :4] = about_z(50.0)
        assert mean_pose_floor(self.constant_train(), val)[
            "floor_rotation_error_deg"
        ] == pytest.approx(0.0, abs=1e-3)

    def test_an_empty_training_split_raises(self):
        with pytest.raises(ValueError, match="fitted, not assumed"):
            mean_pose_floor(torch.zeros(0, 7), torch.zeros(3, 7))
