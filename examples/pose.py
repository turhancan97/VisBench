"""Relative camera pose: can the features say how the camera moved between two views?

Two photographs of one rigid object from different viewpoints, and the probe
regresses the rigid transform between the cameras — a wxyz quaternion and a
translation, seven numbers, under probe3d's pairwise protocol::

    python examples/pose.py --data /path/to/navi_v1
    python examples/pose.py --data ... --backbone mae_vitb16 --partners 8
    python examples/pose.py --data ... --hidden-dims ""     # the linear control

Expects the NAVI release (Jampani et al., NeurIPS 2023), one folder per object::

    <data>/<object_id>/multiview_00_<device>/annotations.json
    <data>/<object_id>/multiview_00_<device>/images/000.jpg

**Read the score against the floor this prints, never against zero.** Pairs are
drawn within 120 degrees of rotation, so their median relative rotation is
about 65 degrees and predicting the training set's mean pose — no features at
all — already scores about 67. A backbone landing there is *at chance*, which
is a different statement from weak, and quoting its 66 degrees as a result
states the wrong one. The floor travels in the metrics as ``floor_*``.

**This is the first board here whose head is not a linear map, deliberately.**
Every other VisBench board is quoted with the least expressive head that can
express the task, because then a gap between two backbones is a gap between two
representations — and every one of those heads has turned out to be an affine
layer or a 1x1 convolution. (`DPTHead` is nonlinear, and is a control rather
than a board.) A single affine map cannot express this one: it underfits, it
clears the floor by 2.7-3.5 degrees where the MLP clears it by 24.5-42.6, and
its ordering nearly inverts the MLP's at the top. Pass ``--hidden-dims ""`` to
see that for yourself — it is the control the board is published beside.

**The pair count is protocol, not a speed knob.** ``--partners`` draws that
many distinct partners per *training* anchor; validation always stays at one,
because it is the yardstick the floor is measured on. Rotation error keeps
falling as partners are added and nothing has converged at any count measured,
so two pose numbers are comparable only if they drew the same pairs — which is
why the training pair count is recorded in ``task_params``.

The first run extracts pooled features; later runs read the cache and the
backbone never executes. Pooled, so a pose run is cheap — no dense grid, no
streaming — and the features are shared with any other pooled probe over the
same frames.
"""

import argparse
import json
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data.navi import MAX_ANGLE, NaviPoseDataset
from visbench.heads.pose import POSE_HIDDEN_DIMS
from visbench.tasks.mid_level.pose import RelativePoseTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="NAVI root")
    parser.add_argument("--backbone", default="dinov2_vits14", help="visbench.list_backbones()")
    parser.add_argument(
        "--partners",
        type=int,
        default=8,
        help="distinct partners per TRAINING anchor; validation always uses one",
    )
    parser.add_argument("--max-angle", type=float, default=MAX_ANGLE)
    parser.add_argument("--pair-seed", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth anchor")
    parser.add_argument(
        "--hidden-dims",
        default=",".join(str(width) for width in POSE_HIDDEN_DIMS),
        help='MLP widths; pass "" for the single affine map, which is the control',
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=128, help="head training")
    parser.add_argument("--batch-size", type=int, default=32, help="feature extraction")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    parser.add_argument("--cache", type=Path, default=Path(".visbench_cache"))
    parser.add_argument("--results", type=Path, default=Path("results/visbench.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hidden_dims = tuple(int(w) for w in args.hidden_dims.split(",") if w.strip())

    shared = {
        "root": args.data,
        "max_angle": args.max_angle,
        "pair_seed": args.pair_seed,
        "stride": args.stride,
        "image_size": args.image_size,
    }
    train = NaviPoseDataset(split="train", partners=args.partners, **shared)
    val = NaviPoseDataset(split="val", **shared)

    print(f"{len(train.labels().pose)} training pairs over {len(train)} frames")
    print(f"{len(val.labels().pose)} validation pairs over {len(val)} frames")
    angles = val.relative_rotation_deg()
    print(f"  relative rotation: median {angles.median():.1f} deg, max {angles.max():.1f}")

    probe = RelativePoseTask(
        hidden_dims=hidden_dims,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.train_batch_size,
        device=args.device,
    )
    print(f"\nhead: {'MLP ' + str(hidden_dims) if hidden_dims else 'one affine map (control)'}")

    # The backbone by *name*: run() seeds before it constructs, so building one
    # here would put its random init outside the seeded window and make this
    # number irreproducible in a way no recorded field explains.
    result = visbench.run(
        args.backbone,
        probe,
        val,
        train_dataset=train,
        cache=FeatureCache(root=args.cache),
        results=args.results,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    floor = result.metrics["floor_rotation_error_deg"]
    error = result.metrics["rotation_error_deg"]
    print("\nrelative pose:")
    for name, value in result.metrics.items():
        if not name.startswith("floor_"):
            print(f"  {name:>22s}  {value:.4f}")
    print(f"\n  no-feature floor        {floor:.2f} deg")
    print(f"  margin over the floor   {floor - error:+.2f} deg   <- quote this, not the score")
    if error >= floor:
        print("\n  *** At or below the floor: this backbone is at chance here. ***")
        print("  Check train_loss before reading it as a weak representation — a loss")
        print("  near zero beside a score at the floor is the head memorising the")
        print("  split, and a loss that is large means it never fitted at all.")
    print(f"\n  train_loss {result.record.training['train_loss']:.4f}")

    print(f"\nrecord appended to {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
