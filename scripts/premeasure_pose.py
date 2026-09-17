#!/usr/bin/env python
"""Pre-measure relative camera pose: does it separate backbones that differ only in objective?

    scripts/premeasure_pose.py
    scripts/premeasure_pose.py --backbones mae_vitb16,dino_vitb16 --seeds 1
    scripts/premeasure_pose.py --partners 8 --head linear

Run before building a pose probe. It trains the probe3d-style pairwise head on
NAVI over a handful of backbones and prints what separates them -- nothing is
registered, nothing reaches the corpus, and no board is spent.

**Since 16a-1 it reads the pairs out of the library** rather than building its
own. :class:`~visbench.data.NaviPoseDataset` and
:mod:`visbench.metrics.pose` own the sampler, the quaternion maths, the
millimetre scaling and the floor; what stays here is the head, the training
loop and the reporting. A pre-measurement whose data is assembled by a second
copy of the loader stops predicting the probe the moment either copy moves --
and this one had already diverged: the script opened frames without applying
their EXIF orientation, which is wrong on 327 of NAVI's 8,217 images. **The
tables in `visbench/tasks/low_level/README.md` were measured that way**, on
4.0% different pixels.

**The question it asks is not the one the gauntlet usually asks.** The oracle
gate asks whether a target is *recoverable*; `premeasure_ordering.py` asks
whether a trivial shortcut already reaches it. Pose looked as though the first
was already answered: a working probe3d-style implementation in a sibling
project ranks seven backbones from 37.1 to 65.2 deg at a per-cell seed std of
0.1-0.7.

**Read that range against the floor this script prints, which is 66.9 deg.**
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
its own. It predicts the training set's mean pose for every pair -- no features
at all -- and any backbone near it is *at chance* rather than merely weak,
which is a different claim about a backbone and the reason this project refuses
to quote a gap against zero.

**`--partners` is the lever and it is protocol, not tuning.** One partner per
training anchor is where the head overfits at ``n ~ d`` and two of four
backbones read as at chance; eight is where every row clears the floor by 24
deg or more. Nothing has converged at any pair count measured, so a board must
pin the one it used.

Nothing here is a metric, nothing is written back, and the head is deliberately
probe3d's MLP rather than this project's `LinearHead` -- the point is to
reproduce the protocol that is known to rank, not to pre-judge the head a probe
would ship with. `--head linear` measures what that choice costs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visbench import get_backbone  # noqa: E402
from visbench.cache import FeatureCache  # noqa: E402
from visbench.data import NaviPoseDataset  # noqa: E402
from visbench.metrics.pose import mean_pose_floor, pose_metrics  # noqa: E402

#: NAVI as staged on this machine. Overridable; nothing here assumes the path.
DEFAULT_DATA = Path("/shared/sets/datasets/vision/probing_3D/navi_v1")

#: Four ViT-B/16s differing only in what they were trained to do. See module
#: docstring: this is the comparison the sibling evidence cannot make.
DEFAULT_BACKBONES = ("mae_vitb16", "dino_vitb16", "sam_vitb16", "clip_vitb16")


def linear_pose_head(width: int) -> nn.Module:
    """``[B, 2D] -> [B, 7]``, one affine map -- what VisBench ships everywhere else.

    Measured at 50,519 training pairs it **underfits**: ``train_loss`` 0.0725 to
    0.0732, flat across four backbones and 40x the MLP's, clearing the floor by
    2.7-3.5 deg against the MLP's 24.5-42.6 -- and its residual ordering does
    not reproduce the MLP's and nearly inverts the top. That is the control
    behind shipping a pose board with a nonlinear head, and the reason the
    decision is about what this corpus's numbers mean rather than a detail.
    """
    return nn.Sequential(nn.BatchNorm1d(2 * width), nn.Linear(2 * width, 7))


def pose_head(width: int) -> nn.Module:
    """``[B, 2D] -> [B, 7]``, BatchNorm then 512/256/128.

    Deliberately not `LinearHead`: this reproduces the protocol known to rank.
    Note the BatchNorm carries running statistics, which are fitted state that
    a probe's ``probe_state()`` has to save -- the `DetectionTask.grid_hw` trap.
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


def paired_features(
    cache: FeatureCache,
    backbone,
    dataset: NaviPoseDataset,
    batch_size: int,
) -> torch.Tensor:
    """``[P, 2D]`` -- the two frames of each pair, concatenated.

    The dataset presents unique frames and pairs them by index, so a frame used
    by eight pairs is extracted once and read eight times.

    **The pooling is resolved before it is asked for**, which is what
    `visbench.run` does and is not cosmetic: the cache keys on the string, so a
    script asking for ``"default"`` and a probe asking for the ``"cls"`` it
    resolves to write two sets of identical features under two keys. This
    script warmed 8,215 frames x four backbones under the wrong one, and the
    first probe run through `run()` re-extracted every frame -- a cache warmed
    by a script is not warm for the library unless it asks the same question.
    """
    pooled = cache.extract_dataset(
        backbone,
        dataset,
        batch_size=batch_size,
        keep="pooled",
        pooling=backbone.default_pooling(),
    )["pooled"]
    indices = dataset.labels().indices
    return torch.cat([pooled[indices[:, 0]], pooled[indices[:, 1]]], dim=1).float()


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
    """Train the head on one seed and return what it scored, plus its fit.

    ``train_loss`` is printed beside every score because the two failures this
    probe can have look identical from the score alone: a head that underfits
    (loss high, score at the floor) and one that memorised the split (loss ~0,
    score at the floor) are opposite conclusions about the representation.
    """
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
    scored = pose_metrics(predicted, val_y)
    scored["train_loss"] = float(loss.detach())
    return scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES))
    parser.add_argument("--max-angle", type=float, default=120.0)
    parser.add_argument("--stride", type=int, default=4, help="subsample anchors")
    parser.add_argument("--pair-seed", type=int, default=8)
    parser.add_argument(
        "--partners",
        type=int,
        default=1,
        help="distinct partners per TRAINING anchor; val stays at one",
    )
    parser.add_argument("--seeds", type=int, default=3, help="head seeds per backbone")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--head",
        choices=("mlp", "linear"),
        default="mlp",
        help="mlp reproduces probe3d's; linear is what VisBench ships elsewhere",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32, help="extraction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if not args.data.is_dir():
        print(f"NAVI not found at {args.data}; pass --data", file=sys.stderr)
        return 1

    shared = {
        "root": args.data,
        "max_angle": args.max_angle,
        "pair_seed": args.pair_seed,
        "stride": args.stride,
        "image_size": args.image_size,
    }
    train = NaviPoseDataset(split="train", partners=args.partners, **shared)
    val = NaviPoseDataset(split="val", **shared)
    train_y, val_y = train.labels().pose, val.labels().pose

    print("=== NAVI pairs ===")
    print(f"  {len(train_y)} train pairs over {len(train)} frames, {args.partners} partner(s)")
    print(f"  {len(val_y)} val pairs over {len(val)} frames, always 1 partner")
    angles = val.relative_rotation_deg()
    print(f"  val relative rotation: median {angles.median():.1f} deg, max {angles.max():.1f}")

    base = mean_pose_floor(train_y, val_y)
    print("\n=== the floor: predict the training mean, no features ===")
    print(
        f"  mean_pose             rot {base['floor_rotation_error_deg']:7.2f} deg"
        f"   trans {base['floor_translation_error']:.3f}"
    )

    cache = FeatureCache()
    results: dict[str, list[dict]] = {}

    for name in args.backbones.split(","):
        name = name.strip()
        if not name:
            continue
        backbone = get_backbone(name, device=args.device)
        train_x = paired_features(cache, backbone, train, args.batch_size)
        val_x = paired_features(cache, backbone, val, args.batch_size)
        runs = [
            fit_and_score(
                train_x,
                train_y,
                val_x,
                val_y,
                seed=seed,
                epochs=args.epochs,
                device=torch.device(args.device),
                head_kind=args.head,
            )
            for seed in range(args.seeds)
        ]
        results[name] = runs
        errors = [run["rotation_error_deg"] for run in runs]
        print(
            f"  {name:20s} rot {np.mean(errors):7.2f} deg  "
            f"(seed range {max(errors) - min(errors):.2f})  "
            f"trans {np.mean([run['translation_error'] for run in runs]):.3f}  "
            f"train_loss {np.mean([run['train_loss'] for run in runs]):.4f}"
        )

    print("\n=== does it separate them? ===")
    means = {n: float(np.mean([r["rotation_error_deg"] for r in v])) for n, v in results.items()}
    noise = max(
        (max(r["rotation_error_deg"] for r in v) - min(r["rotation_error_deg"] for r in v))
        for v in results.values()
    )
    spread = max(means.values()) - min(means.values())
    floor = base["floor_rotation_error_deg"]
    ordered = sorted(means.items(), key=lambda kv: kv[1])
    for position, (name, value) in enumerate(ordered):
        gap = (
            f"  gap {ordered[position + 1][1] - value:5.2f}" if position + 1 < len(ordered) else ""
        )
        print(f"  {name:20s} {value:7.2f} deg   {floor - value:+6.2f} vs the floor{gap}")
    if min(means.values()) >= floor:
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
    print(
        "\n  Quote the per-row margin over the floor and the adjacent gaps above.\n"
        "  spread / noise has misled in both directions here and is a summary of\n"
        "  neither: a backbone at the floor is at chance, not weak."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
