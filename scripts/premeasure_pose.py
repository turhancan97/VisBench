#!/usr/bin/env python
"""Pre-measure relative camera pose: does it separate backbones that differ only in objective?

    scripts/premeasure_pose.py
    scripts/premeasure_pose.py --backbones mae_vitb16,dino_vitb16 --seeds 1

Run before building a pose probe. It trains the probe3d-style pairwise head on
NAVI over a handful of backbones and prints what separates them -- nothing is
registered, nothing reaches the corpus, and no board is spent.

**The question it asks is not the one the gauntlet usually asks.** The oracle
gate asks whether a target is *recoverable*; `premeasure_ordering.py` asks
whether a trivial shortcut already reaches it. Pose looked as though the first
was already answered: a working probe3d-style implementation in a sibling
project ranks seven backbones from 37.1 to 65.2 deg at a per-cell seed std of
0.1-0.7.

**Read that range against the floor this script measures, which is 65.4 deg.**
The sibling spread is not 28 deg of signal above zero -- its *weakest* two rows
(SigLIP 62.7, Perception Encoder 65.2) sit at chance, and only its strongest
backbones clear the floor at all. An earlier draft of this docstring quoted the
28.2 deg spread as evidence that pose "clearly ranks something", which treated
the whole range as signal and is exactly the mistake the floor rule exists to
stop.

What that evidence cannot answer is whether it ranks **this** corpus. Those
seven backbones differ in architecture, scale, data and objective all at once
(DINOv3 / V-JEPA2 / SigLIP / Perception Encoder, ViT-S through ViT-L). VisBench's
corpus is mostly the opposite: six ViT-B/16s at identical architecture, width
and token count, differing only in training objective or recipe. A probe can
separate the first group and not the second, and this corpus holds both
precedents -- `semantic_segmentation` spreads 0.3207 mIoU across those six
identical ViT-B/16s, while `classification` reads 0.9972 for two of them and
separates nothing.

So the default backbone set is four ViT-B/16s that hold everything fixed except
what they were trained to do:

    mae_vitb16          masked pixel reconstruction, IN1k
    dino_vitb16         self-distillation, IN1k
    sam_vitb16          promptable segmentation, IN1k  (recipe control for the above)
    clip_vitb16         language supervision, WIT-400M

**Read the spread against the seed noise, not against zero.** Three seeds are
run per backbone by default for exactly that reason: a spread smaller than the
seed-to-seed range is not a ranking, and `duration_seconds` in this project is
the standing example of a number that looked solid and was not.

The floor is reported beside the scores because a pose error means nothing on
its own. `mean_pose` predicts the training set's mean rotation and translation
for every pair -- no features at all -- and any backbone near it is *at chance*
rather than merely weak, which is a different claim about a backbone and the
reason this project refuses to quote a gap against zero.

Data: NAVI multiview (`--data`), whose per-image `annotations.json` carries the
camera quaternion and translation. The pairing follows probe3d's protocol as
implemented in the sibling AMDF project (`data/navi_camera_pose.py`): one anchor
per image, a partner drawn uniformly among views whose relative rotation is in
(0, --max-angle] degrees, seeded so the pair set is fixed across backbones.
**The same pairs must reach every backbone**, or the comparison measures the
sampler.

Nothing here is a metric, nothing is written back, and the head is deliberately
probe3d's MLP rather than this project's `LinearHead` -- the point is to
reproduce the protocol that is known to rank, not to pre-judge the head a probe
would ship with.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visbench import get_backbone  # noqa: E402
from visbench.cache import FeatureCache  # noqa: E402
from visbench.data.base import BaseDataset  # noqa: E402

#: NAVI as staged on this machine. Overridable; nothing here assumes the path.
DEFAULT_DATA = Path("/shared/sets/datasets/vision/probing_3D/navi_v1")

#: Four ViT-B/16s differing only in what they were trained to do. See module
#: docstring: this is the comparison the sibling evidence cannot make.
DEFAULT_BACKBONES = ("mae_vitb16", "dino_vitb16", "sam_vitb16", "clip_vitb16")


# --------------------------------------------------------------------------
# Pose maths. Quaternions are (w, x, y, z), matching NAVI's annotations.
# --------------------------------------------------------------------------


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` wxyz quaternion -> ``(..., 3, 3)`` rotation matrix."""
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
    """``(..., 3, 3)`` -> ``(..., 4)`` wxyz, choosing the largest component.

    The naive ``w = sqrt(1 + trace) / 2`` form then divides the off-diagonals
    by ``w``, which loses all precision as the rotation approaches 180 degrees
    -- and these pairs sample to ``--max-angle``, 120 by default, with a
    fallback that can exceed it. A wrong target quaternion is a silently wrong
    number, not an error, so the branch is chosen per element from whichever of
    ``w, x, y, z`` is largest (Shepperd's method).

    Measured on 2000 random rotations, the naive form fails to round-trip
    within 1e-3 degrees on **66** of them. This one round-trips to **2.4e-07**
    per quaternion component and 4.8e-07 per matrix entry over 5000 rotations,
    which is float32 exact.

    **It does not follow that the reported angle round-trips that well, and the
    difference is the metric rather than the conversion.** `rotation_error_deg`
    takes ``acos`` of the trace, whose derivative is unbounded as the relative
    rotation approaches zero, so a 5e-07 perturbation of an exact matrix shows
    up as up to **0.05 degrees** when the true error is nil. That is
    ill-conditioning where the answer is already right, it does not grow at the
    20-60 degree scale this probe reports, and it is the same family as
    `orientation`'s recorded ill-conditioning. Do not read it as conversion
    error, and do not tighten a tolerance against it.
    """
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


def rotation_error_deg(pred_q: torch.Tensor, true_q: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two rotations, in degrees.

    Computed from the matrices rather than from ``|<q1, q2>|`` because a
    predicted quaternion is not normalised and need not be a rotation at all;
    projecting through the matrix is what the metric of record does.
    """
    rel = quaternion_to_matrix(pred_q) @ quaternion_to_matrix(true_q).transpose(-1, -2)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    return torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)))


#: NAVI stores translation in millimetres. The reference implementation divides
#: by this, and **it is not cosmetic**: the loss is MSE over the raw 7-vector
#: ``[quat, trans]``, so a translation of magnitude ~200 outweighs a unit
#: quaternion by ~4 orders of magnitude, the head optimises translation alone,
#: and rotation is never learned.
#:
#: Omitting it is what the first run of this script did, and the failure is
#: worth keeping because it does not look like a units bug: every backbone
#: landed at 107-109 deg rotation error against a no-feature floor of 65.4,
#: i.e. **worse than predicting a constant**, with train_loss 466-1001. A
#: trained head cannot lose to a constant unless the loss is not optimising
#: that term -- which is the tell, and is the same failure as 6d-1's
#: ``target_scale``: a gradient does not rescale itself to match its target.
TRANSLATION_SCALE = 1000.0


def camera_matrix(ann: dict) -> torch.Tensor:
    """A NAVI annotation's 4x4 world-to-camera matrix, translation in metres."""
    q = torch.tensor(ann["camera"]["q"], dtype=torch.float64)
    t = torch.tensor(ann["camera"]["t"], dtype=torch.float64) / TRANSLATION_SCALE
    rt = torch.eye(4, dtype=torch.float64)
    rt[:3, :3] = quaternion_to_matrix(q)
    rt[:3, 3] = t
    return rt


# --------------------------------------------------------------------------
# Pairs
# --------------------------------------------------------------------------


def build_pairs(root: Path, max_angle: float, stride: int, seed: int) -> list[dict]:
    """One anchor per image, partner within ``max_angle`` degrees of rotation.

    Seeded and independent of the backbone, so every backbone is scored on the
    *same* pairs. Returns dicts carrying both image paths, the 7D relative pose
    and the split the anchor belongs to.
    """
    generator = torch.Generator().manual_seed(seed)
    pairs: list[dict] = []

    for ann_path in sorted(root.glob("*/multiview_*/annotations.json")):
        scene = ann_path.parent
        records = json.loads(ann_path.read_text())
        usable = [r for r in records if (scene / "images" / r["filename"]).is_file()]
        if len(usable) < 2:
            continue

        rts = torch.stack([camera_matrix(r) for r in usable])
        rotations = rts[:, :3, :3]

        for i, anchor in enumerate(usable):
            rel = rotations[i].unsqueeze(0) @ rotations.transpose(-1, -2)
            trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
            angle = torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)))

            eligible = (angle > 0) & (angle <= max_angle)
            eligible[i] = False
            if not bool(eligible.any()):
                continue
            j = int(torch.multinomial(eligible.double(), 1, generator=generator).item())

            rt01 = rts[j] @ torch.linalg.inv(rts[i])
            pose = torch.cat([matrix_to_quaternion(rt01[:3, :3]), rt01[:3, 3]]).float()
            pairs.append(
                {
                    "a": scene / "images" / anchor["filename"],
                    "b": scene / "images" / usable[j]["filename"],
                    "pose": pose,
                    "split": anchor.get("split", "train"),
                }
            )

    return pairs[::stride] if stride > 1 else pairs


class _Frames(BaseDataset):
    """The unique images the pairs reference, so each is extracted once.

    A pair task is a flat image dataset plus indices -- the rule
    ``TwoAFCDataset`` and ``PairViewDataset`` already follow. Presenting the
    unique frames keeps the cache, and the pairing travels by index.
    """

    def __init__(self, paths: list[Path], image_size: int = 224) -> None:
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        """Short side to ``image_size`` (BICUBIC), then centre crop.

        Deliberately `DenseFolderDataset._crop_image`'s geometry rather than a
        crop-then-resize of my own: if a probe follows this measurement it must
        read the same pixels, or the pre-measurement stops predicting it.
        """
        image = Image.open(self.paths[index]).convert("RGB")
        width, height = image.size
        scale = self.image_size / min(width, height)
        image = image.resize(
            (
                max(self.image_size, round(width * scale)),
                max(self.image_size, round(height * scale)),
            ),
            Image.Resampling.BICUBIC,
        )
        left = (image.width - self.image_size) // 2
        top = (image.height - self.image_size) // 2
        return image.crop((left, top, left + self.image_size, top + self.image_size)), 0

    def cache_identity(self, index: int) -> str:
        stat = self.paths[index].stat()
        return f"{self.paths[index]}|{stat.st_size}|{stat.st_mtime_ns}"


# --------------------------------------------------------------------------
# The head: probe3d's, as the sibling implementation uses it
# --------------------------------------------------------------------------


def linear_pose_head(width: int) -> nn.Module:
    """``[B, 2D] -> [B, 7]``, one affine map -- what VisBench would ship.

    The MLP below is probe3d's and is what the sibling evidence used, but at
    this data scale it has ~1.2M parameters against 1612 training pairs and
    drives `train_loss` to ~5e-4 while validation rotation error sits at the
    no-feature floor. That is the *mirror* of the binary-segmentation case in
    `CLAUDE.md`: there a low score came with a high training loss and meant
    underfitting; here a low score comes with a training loss of essentially
    zero and means the head memorised the split. Both are statements about the
    fit, not about the representation, which is why the fit diagnostic is
    printed beside every score.
    """
    return nn.Sequential(nn.BatchNorm1d(2 * width), nn.Linear(2 * width, 7))


def pose_head(width: int) -> nn.Module:
    """``[B, 2D] -> [B, 7]``, BatchNorm then 512/256/128.

    Deliberately not `LinearHead`: this reproduces the protocol known to rank
    rather than pre-judging what a shipped probe would use. If a probe follows,
    that choice is its own decision -- and note the BatchNorm carries running
    statistics, which are fitted state a `probe_state()` would have to save.
    """
    return nn.Sequential(
        nn.BatchNorm1d(2 * width),
        nn.Linear(2 * width, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 7),
    )


def fit_and_score(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    seed: int,
    epochs: int,
    device: torch.device,
    head_kind: str = "mlp",
) -> dict:
    """Train the head on one seed and return validation errors."""
    torch.manual_seed(seed)
    build = linear_pose_head if head_kind == "linear" else pose_head
    head = build(train_x.shape[1] // 2).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    order = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(seed))
    train_x, train_y = train_x[order].to(device), train_y[order].to(device)

    head.train()
    for _ in range(epochs):
        for start in range(0, len(train_x), 128):
            batch_x = train_x[start : start + 128]
            if len(batch_x) < 2:  # BatchNorm needs more than one row
                continue
            loss = nn.functional.mse_loss(head(batch_x), train_y[start : start + 128])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        schedule.step()

    head.eval()
    with torch.no_grad():
        predicted = head(val_x.to(device)).cpu()
    return {
        "rot_err_deg": float(rotation_error_deg(predicted[:, :4], val_y[:, :4]).mean()),
        "trans_err": float((predicted[:, 4:] - val_y[:, 4:]).norm(dim=-1).mean()),
        "train_loss": float(loss.detach()),
    }


def floor(train_y: torch.Tensor, val_y: torch.Tensor) -> dict:
    """What predicting the training mean scores -- no features at all.

    Reported beside every backbone because a rotation error is uninterpretable
    on its own: a backbone landing here is *at chance*, which is a different
    statement from being weak.
    """
    constant = train_y.mean(dim=0, keepdim=True).expand_as(val_y)
    return {
        "rot_err_deg": float(rotation_error_deg(constant[:, :4], val_y[:, :4]).mean()),
        "trans_err": float((constant[:, 4:] - val_y[:, 4:]).norm(dim=-1).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES))
    parser.add_argument("--max-angle", type=float, default=120.0)
    parser.add_argument("--stride", type=int, default=4, help="subsample pairs")
    parser.add_argument("--pair-seed", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3, help="head seeds per backbone")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--head",
        choices=("mlp", "linear"),
        default="mlp",
        help="mlp reproduces probe3d's; linear is what VisBench would ship",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="extraction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if not args.data.is_dir():
        print(f"NAVI not found at {args.data}; pass --data", file=sys.stderr)
        return 1

    pairs = build_pairs(args.data, args.max_angle, args.stride, args.pair_seed)
    if not pairs:
        print("no pairs built -- check --data and --max-angle", file=sys.stderr)
        return 1

    frames = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    index_of = {path: i for i, path in enumerate(frames)}
    poses = torch.stack([p["pose"] for p in pairs])
    is_train = torch.tensor([p["split"] == "train" for p in pairs])
    left = torch.tensor([index_of[p["a"]] for p in pairs])
    right = torch.tensor([index_of[p["b"]] for p in pairs])

    print(f"=== NAVI pairs ===\n  {len(pairs)} pairs over {len(frames)} unique frames")
    print(f"  {int(is_train.sum())} train / {int((~is_train).sum())} val")
    angles = rotation_error_deg(poses[:, :4], torch.tensor([[1.0, 0, 0, 0]]).expand(len(poses), 4))
    print(f"  relative rotation: median {angles.median():.1f} deg, max {angles.max():.1f}")

    base = floor(poses[is_train], poses[~is_train])
    print("\n=== the floor: predict the training mean, no features ===")
    print(
        f"  mean_pose             rot {base['rot_err_deg']:7.2f} deg"
        f"   trans {base['trans_err']:.3f}"
    )

    cache = FeatureCache()
    dataset = _Frames(frames)
    results: dict[str, list[dict]] = {}

    for name in args.backbones.split(","):
        name = name.strip()
        if not name:
            continue
        backbone = get_backbone(name, device=args.device)
        pooled = cache.extract_dataset(
            backbone, dataset, batch_size=args.batch_size, keep="pooled"
        )["pooled"]
        features = torch.cat([pooled[left], pooled[right]], dim=1).float()
        device = torch.device(args.device)
        runs = [
            fit_and_score(
                features[is_train],
                poses[is_train],
                features[~is_train],
                poses[~is_train],
                seed=seed,
                epochs=args.epochs,
                device=device,
                head_kind=args.head,
            )
            for seed in range(args.seeds)
        ]
        results[name] = runs
        errors = [r["rot_err_deg"] for r in runs]
        print(
            f"  {name:20s} rot {np.mean(errors):7.2f} deg  "
            f"(seed range {max(errors) - min(errors):.2f})  "
            f"trans {np.mean([r['trans_err'] for r in runs]):.3f}  "
            f"train_loss {np.mean([r['train_loss'] for r in runs]):.4f}"
        )

    print("\n=== does it separate them? ===")
    means = {n: float(np.mean([r["rot_err_deg"] for r in v])) for n, v in results.items()}
    noise = max(
        (max(r["rot_err_deg"] for r in v) - min(r["rot_err_deg"] for r in v))
        for v in results.values()
    )
    spread = max(means.values()) - min(means.values())
    for name, value in sorted(means.items(), key=lambda kv: kv[1]):
        margin = base["rot_err_deg"] - value
        print(f"  {name:20s} {value:7.2f} deg   {margin:+6.2f} vs the floor")
    if min(means.values()) >= base["rot_err_deg"]:
        print(
            "\n  *** Every backbone is at or below the no-feature floor. ***\n"
            "  The ordering below is not a ranking of representations: a row that\n"
            "  loses to a constant carries no usable signal, so their differences\n"
            "  are differences in how they fail. Two causes seen here, and the\n"
            "  fit diagnostic tells them apart:\n"
            "    - the loss is not optimising the scored term (train_loss large;\n"
            "      an unconverted millimetre translation did this, and it printed\n"
            "      a perfectly plausible table), or\n"
            "    - the head has nothing to work with at this data scale\n"
            "      (train_loss ~0 with the score at the floor is memorisation).\n"
            "  Fix the first. The second is a finding about the probe."
        )

    print(f"\n  spread {spread:.2f} deg over {len(means)} backbones")
    print(f"  widest seed range {noise:.2f} deg")
    print(f"  spread / noise {spread / noise:.1f}x" if noise > 0 else "")
    print(
        "\n  A spread at or below the seed range is not a ranking. A backbone at "
        "the floor\n  is at chance, not weak -- quote the margin, never the score "
        "against zero."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
