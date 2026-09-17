"""Relative camera pose — the geometry a pairwise pose probe is scored on (16a-1).

Two views of one rigid scene, and the question is the rigid transform between
the cameras that took them. The target is a 7-vector ``[qw, qx, qy, qz, tx, ty,
tz]``: a **wxyz** quaternion and a translation, in that order and in those
units, matching probe3d's pairwise pose protocol as the reference
implementation writes it.

Three conventions, each of which is silent rather than loud when broken:

* **Quaternions are wxyz**, because that is what NAVI's ``annotations.json``
  stores and what :func:`quaternion_to_matrix` therefore reads. An xyzw
  quaternion loads, scores and reports a plausible angle.
* **Translation is in metres by the time it reaches here.** Scaling is the
  dataset's job (:data:`visbench.data.navi.TRANSLATION_SCALE`), not this
  module's, so a metric never silently rescales what a loss optimised.
* **A rotation error is uninterpretable on its own**, which is why
  :func:`mean_pose_floor` lives beside :func:`pose_metrics` rather than in a
  script. Pairs drawn within 120 degrees have a median relative rotation near
  65, so predicting the training mean — no features at all — already scores
  about 67 degrees. A backbone reading 66 is *at chance*, which is a different
  claim about a representation from "weak", and quoting its error against zero
  states the wrong one.

**The conversion is exact and the metric is not, and they are separate
facts.** :func:`matrix_to_quaternion` picks its branch from the largest of the
four components (Shepperd's method) and round-trips to float32 precision even
at 180 degrees, where the naive ``w = sqrt(1 + trace) / 2`` form loses every
digit. :func:`rotation_error_deg` then takes ``acos`` of the trace, whose
derivative is unbounded as the relative rotation approaches zero, so a
float-exact answer can still print as several hundredths of a degree. That is
ill-conditioning where the answer is already right — the same family as
``orientation``'s recorded ill-conditioning — and it does not grow at the
20-60 degree scale this probe reports. Do not read it as conversion error, and
do not tighten a tolerance against it.

**So this metric cannot check that conversion, and measuring it that way says
the branch does not matter.** On 2000 rotations between 170 and 180 degrees,
the two forms differ by 1000x in what they actually get wrong — 1.2e-07 against
1.2e-04 per quaternion component — while the angle between each and the truth
reads 0.0485 against 0.0560 degrees, with both exceeding a 1e-3 tolerance on
about 200 of the 2000. The angle's own noise floor is the giveaway: comparing a
quaternion with **itself** returns up to 0.028 degrees here. Validate a
conversion on components or matrices; the angle is for scoring a probe, where
the scale is 20-60 degrees and a hundredth does not signify.

``acos`` is kept rather than replaced by a well-conditioned ``atan2`` form
because it is the metric of record — probe3d's, as the reference implements it
— and the only reason to borrow a protocol is to stay comparable with the
numbers published under it.
"""

import torch

from visbench.types import MetricsDict

__all__ = [
    "POSE_COLUMNS",
    "ROTATION_ACC_THRESHOLDS",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "relative_pose",
    "pose_vector",
    "rotation_angle_deg",
    "rotation_error_deg",
    "translation_error",
    "pose_metrics",
    "mean_pose_floor",
]

#: Column order of the 7-vector every function here reads and returns. Written
#: down because the quaternion convention is the one thing about this target
#: that cannot be spotted from a printed number.
POSE_COLUMNS = ("qw", "qx", "qy", "qz", "tx", "ty", "tz")

#: Angular thresholds reported beside the mean, in degrees. A mean over pairs
#: drawn to 120 degrees is pulled about by the failures; the fraction within a
#: threshold says how often the answer was usable, and the two can disagree.
ROTATION_ACC_THRESHOLDS = (15.0, 30.0)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrices from wxyz quaternions.

    Parameters
    ----------
    q : torch.Tensor
        ``(..., 4)`` quaternions, **wxyz**. Normalised internally, so a raw
        head output — which is not a unit quaternion and need not be a rotation
        at all — is projected onto one rather than rejected.

    Returns
    -------
    torch.Tensor
        ``(..., 3, 3)`` rotation matrices.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"quaternions are (..., 4) wxyz, got {tuple(q.shape)}")
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = torch.unbind(q, dim=-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """wxyz quaternions from rotation matrices, by Shepperd's method.

    The branch is chosen per element from whichever of ``w, x, y, z`` is
    largest. The naive ``w = sqrt(1 + trace) / 2`` form then divides the
    off-diagonals by ``w``, which loses all precision as the rotation
    approaches 180 degrees — and pose pairs are drawn to 120 degrees, with a
    relative transform that can exceed it.

    Measured on 2000 random rotations, the naive form fails to round-trip
    within 1e-3 degrees on **66** of them. This one round-trips to 2.4e-07 per
    quaternion component and 4.8e-07 per matrix entry over 5000 rotations,
    which is float32 exact. A wrong target quaternion is a silently wrong
    number rather than an error, which is why the expensive branch is the one
    that ships.

    Parameters
    ----------
    m : torch.Tensor
        ``(..., 3, 3)`` rotation matrices.

    Returns
    -------
    torch.Tensor
        ``(..., 4)`` unit quaternions, wxyz, in float32. Computed in float64
        regardless of the input dtype: the intermediate ``0.5 / sqrt(...)`` is
        where the precision goes, not the storage.
    """
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"rotations are (..., 3, 3), got {tuple(m.shape)}")
    m = m.to(torch.float64)
    xx, yy, zz = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    candidates = torch.stack(
        [1.0 + xx + yy + zz, 1.0 + xx - yy - zz, 1.0 - xx + yy - zz, 1.0 - xx - yy + zz],
        dim=-1,
    )
    branch = candidates.argmax(dim=-1)
    scale = torch.sqrt(candidates.gather(-1, branch.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12))
    half = 0.5 / scale

    zy, yz = m[..., 2, 1], m[..., 1, 2]
    xz, zx = m[..., 0, 2], m[..., 2, 0]
    yx, xy = m[..., 1, 0], m[..., 0, 1]
    options = torch.stack(
        [
            torch.stack([0.5 * scale, (zy - yz) * half, (xz - zx) * half, (yx - xy) * half], -1),
            torch.stack([(zy - yz) * half, 0.5 * scale, (xy + yx) * half, (xz + zx) * half], -1),
            torch.stack([(xz - zx) * half, (xy + yx) * half, 0.5 * scale, (yz + zy) * half], -1),
            torch.stack([(yx - xy) * half, (xz + zx) * half, (yz + zy) * half, 0.5 * scale], -1),
        ],
        dim=-2,
    )
    q = options.gather(-2, branch[..., None, None].expand(*branch.shape, 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return q.to(torch.float32)


def relative_pose(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The transform taking the ``source`` camera's frame to the ``target``'s.

    ``target @ inverse(source)``, both world-to-camera. **The direction is part
    of the protocol**: the opposite composition is also a valid relative pose,
    of the same magnitude and the wrong sign, so a probe trained on one and
    scored against the other reports a plausible number about nothing.

    Parameters
    ----------
    source : torch.Tensor
        ``(..., 4, 4)`` world-to-camera matrix of the first view.
    target : torch.Tensor
        ``(..., 4, 4)`` world-to-camera matrix of the second view.

    Returns
    -------
    torch.Tensor
        ``(..., 4, 4)``, in the dtype it was given.
    """
    for name, matrix in (("source", source), ("target", target)):
        if matrix.shape[-2:] != (4, 4):
            raise ValueError(f"{name} must be (..., 4, 4), got {tuple(matrix.shape)}")
    return target @ torch.linalg.inv(source)


def pose_vector(rt: torch.Tensor) -> torch.Tensor:
    """``(..., 4, 4)`` transform to the ``(..., 7)`` target this probe regresses.

    Returns
    -------
    torch.Tensor
        ``[qw, qx, qy, qz, tx, ty, tz]`` in float32 — see :data:`POSE_COLUMNS`.
    """
    if rt.shape[-2:] != (4, 4):
        raise ValueError(f"transforms are (..., 4, 4), got {tuple(rt.shape)}")
    return torch.cat([matrix_to_quaternion(rt[..., :3, :3]), rt[..., :3, 3].float()], dim=-1)


def rotation_angle_deg(q: torch.Tensor) -> torch.Tensor:
    """How far a rotation turns, in degrees — the magnitude of ``q``.

    What the pair sampler thresholds on, and what makes a pair set's difficulty
    reportable: a median relative rotation is the context a mean error is read
    against.

    Returns
    -------
    torch.Tensor
        ``(...)`` angles in ``[0, 180]``.
    """
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0
    return rotation_error_deg(q, identity)


def rotation_error_deg(pred_q: torch.Tensor, true_q: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two rotations, in degrees.

    Computed from the matrices rather than from ``2 * acos(|<q1, q2>|)``
    because a predicted quaternion is not normalised and need not be a rotation
    at all; projecting through the matrix is what the metric of record does,
    and it makes the double cover (``q`` and ``-q`` are one rotation) a
    non-issue rather than a sign convention to remember.

    Parameters
    ----------
    pred_q : torch.Tensor
        ``(..., 4)`` predicted quaternions, wxyz.
    true_q : torch.Tensor
        ``(..., 4)`` ground-truth quaternions, wxyz.

    Returns
    -------
    torch.Tensor
        ``(...)`` angles in ``[0, 180]``.
    """
    rel = quaternion_to_matrix(pred_q) @ quaternion_to_matrix(true_q).transpose(-1, -2)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    return torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)))


def translation_error(pred_t: torch.Tensor, true_t: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between two translations, in the units given.

    Returns
    -------
    torch.Tensor
        ``(...)`` distances. Metres, for every dataset in this project — see
        the module docstring on where the scaling happens.
    """
    return (pred_t - true_t).norm(dim=-1)


def _check_poses(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Both ``(P, 7)``, non-empty and finite, as float."""
    for name, value in (("pred", pred), ("target", target)):
        if value.ndim != 2 or value.shape[-1] != 7:
            raise ValueError(
                f"{name} must be (P, 7) — {'/'.join(POSE_COLUMNS)} — got {tuple(value.shape)}"
            )
    if len(pred) != len(target):
        raise ValueError(f"pred has {len(pred)} pairs and target has {len(target)}")
    if len(pred) == 0:
        raise ValueError("no pairs to score; an empty split is not a score of 0")
    if not torch.isfinite(pred).all():
        # A NaN reaches the record as an unparseable JSON literal, and a head
        # that has diverged looks from the metric alone like a hard task.
        raise ValueError(
            "pred contains non-finite values, so the head diverged rather than "
            "scoring badly — check the loss, not the representation"
        )
    return pred.float(), target.float()


def pose_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: tuple[float, ...] = ROTATION_ACC_THRESHOLDS,
) -> MetricsDict:
    """Rotation and translation error over a set of pairs.

    Parameters
    ----------
    pred : torch.Tensor
        ``(P, 7)`` predictions — see :data:`POSE_COLUMNS`. Raw head output is
        expected: the quaternion is normalised on the way through.
    target : torch.Tensor
        ``(P, 7)`` ground truth, same layout.
    thresholds : tuple of float
        Angular thresholds for the accuracy terms, in degrees.

    Returns
    -------
    dict
        ``rotation_error_deg`` (mean), ``rotation_median_deg``,
        ``rotation_acc_<t>`` per threshold, and ``translation_error`` (mean).

    Notes
    -----
    **Per pair, then averaged** — the project's standing convention, and here
    it is simply what a pair set is: every pair contributes one rotation and
    one translation, so there is no second reduction to choose and no
    dataset-level variant to disagree with it, unlike the two mIoUs.

    The mean and the median can rank two backbones differently, and the
    accuracy terms can disagree with both. That is the ordinary case in this
    corpus, not a defect; quote whichever the board's headline metric is and do
    not drop the others.
    """
    pred, target = _check_poses(pred, target)
    rotation = rotation_error_deg(pred[:, :4], target[:, :4])
    metrics: MetricsDict = {
        "rotation_error_deg": float(rotation.mean()),
        "rotation_median_deg": float(rotation.median()),
    }
    for threshold in thresholds:
        # ``@`` because the threshold is a *parameter* of the metric, not part
        # of its identity — the same shape as ``recall@5px`` and ``auc@0.5p``,
        # which is what lets the leaderboard direct a threshold it has never
        # seen from the stem alone. ``rotation_acc_30`` would need one listed
        # entry per threshold, and an unlisted one is silently dropped from a
        # board rather than refused.
        metrics[f"rotation_acc@{threshold:g}deg"] = float((rotation <= threshold).float().mean())
    metrics["translation_error"] = float(translation_error(pred[:, 4:], target[:, 4:]).mean())
    return metrics


def mean_pose_floor(
    train_target: torch.Tensor,
    target: torch.Tensor,
    thresholds: tuple[float, ...] = ROTATION_ACC_THRESHOLDS,
) -> MetricsDict:
    """What predicting the training mean scores — no features at all.

    The constant an MSE-trained head converges to when its input carries
    nothing about the answer, so it is the yardstick every row of a pose board
    is read against rather than a baseline someone chose. A backbone at this
    value is **at chance**; one two degrees below it has cleared it by two
    degrees, and calling that a score of 65 states something else entirely.

    The mean of a set of 7-vectors is not a unit quaternion and generally not a
    rotation. That is deliberate and not a bug to fix by averaging on the
    manifold: this function reports what the *trained head* would do, and the
    head regresses seven numbers under an MSE loss whose optimum, given
    uninformative features, is exactly this arithmetic mean.
    :func:`rotation_error_deg` projects it through the matrix, as it does any
    raw prediction.

    Parameters
    ----------
    train_target : torch.Tensor
        ``(N, 7)`` training targets — the constant is fitted on these, never on
        the split being scored.
    target : torch.Tensor
        ``(P, 7)`` targets of the split being scored.
    thresholds : tuple of float
        Passed through to :func:`pose_metrics`.

    Returns
    -------
    dict
        The keys :func:`pose_metrics` returns, each prefixed ``floor_``. The
        prefix is what lets a floor travel in the same flat metrics dict as the
        score without colliding with it — the convention ``ceiling_`` already
        follows. **Never rank on one**: it describes the pair set, not a
        backbone, and two boards with different floors are not comparable by
        subtracting them.
    """
    if train_target.ndim != 2 or train_target.shape[-1] != 7:
        raise ValueError(f"train_target must be (N, 7), got {tuple(train_target.shape)}")
    if len(train_target) == 0:
        raise ValueError("no training pairs to average; the floor is fitted, not assumed")
    constant = train_target.float().mean(dim=0, keepdim=True).expand(len(target), 7)
    return {
        f"floor_{key}": value
        for key, value in pose_metrics(constant, target, thresholds=thresholds).items()
    }
