"""Surface normal probing on a local folder, following probe3d's protocol.

Expects images and normal maps paired by filename stem::

    <data>/train/images/scene_0001.jpg
    <data>/train/normals/scene_0001.npy
    <data>/val/images/...
    <data>/val/normals/...

Normal maps may be ``.npy`` in either ``(3, H, W)`` or ``(H, W, 3)`` layout, or
8-bit RGB images under the usual ``2 * v / 255 - 1`` encoding. Either way they
are L2-normalised on load, and a pixel with no direction becomes ``(0, 0, 0)``
— which is what marks it invalid, the same role a 0 plays in a depth map.

Run it::

    python examples/normals.py --data /path/to/dataset --normal-source geonet
    python examples/normals.py --data ... --head dpt --layers 2 5 8 11

**Say where your normals came from.** NYU surface normals are *derived* — from
GeoNet's extraction, or Ladicky's, not from a sensor — and the sources disagree
enough to move every metric. ``--normal-source`` is recorded verbatim in the
result record; runs that leave it unset cannot be compared to each other with
any confidence.

**Report the linear head.** It is the only one under which a difference between
two backbones is a difference between two feature maps; a DPT head has enough
capacity to compensate for a weak representation and narrow the very gap the
probe exists to measure. Run both and say which is which.

The first run extracts features; later runs read the cache and the backbone
never executes, so sweeping probe settings is cheap. Features are shared with
``examples/depth.py`` when the images and ``--image-size`` match, so probing
both tasks on one dataset costs one extraction.

This is an example, not the CLI — that is still deferred until the Python API
for dense tasks has settled.
"""

import argparse
import json
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data import DenseFolderDataset, load_normal_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="root containing train/ and val/")
    parser.add_argument("--backbone", default="dinov2_vits14", help="see visbench.list_backbones()")
    parser.add_argument("--head", default="linear", help="linear | dpt; see visbench.heads")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="backbone depths, shallowest first; required by --head dpt",
    )
    parser.add_argument("--target-dir", default="normals", help="folder name under each split")
    parser.add_argument("--image-size", type=int, default=224, help="multiple of the patch size")
    parser.add_argument(
        "--normal-source",
        default=None,
        help="provenance of the ground truth (geonet | ladicky | sensor | a URL); recorded",
    )
    parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="plain angular loss instead of probe3d's uncertainty-aware default",
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="DPT width")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    parser.add_argument("--cache", type=Path, default=Path(".visbench_cache"))
    parser.add_argument("--results", type=Path, default=Path("results/visbench.jsonl"))
    parser.add_argument(
        "--limit", type=int, default=None, help="use at most N images per split (quick run)"
    )
    return parser.parse_args()


def load_split(root: Path, split: str, args: argparse.Namespace) -> DenseFolderDataset:
    dataset = DenseFolderDataset(
        root / split,
        split=split,
        target_dir=args.target_dir,
        image_size=args.image_size,
        target_loader=load_normal_map,
    )
    if args.limit is not None:
        dataset = dataset.subset(args.limit)
    return dataset


def main() -> None:
    args = parse_args()

    if args.head == "dpt" and not args.layers:
        raise SystemExit(
            "--head dpt is multiscale and needs several depths, e.g. --layers 2 5 8 11. "
            "It refuses a single layer rather than duplicating it, which would report a "
            "single-layer result as a DPT number."
        )
    if args.normal_source is None:
        print(
            "warning: --normal-source is unset. NYU normals are derived rather than "
            "sensed, and GeoNet's and Ladicky's disagree enough to move every metric "
            "below, so this record will not say what it was scored against.\n"
        )

    train = load_split(args.data, "train", args)
    test = load_split(args.data, "val", args)
    print(f"train: {len(train)} images at {args.image_size}px")
    print(f"val:   {len(test)} images")

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    probe = visbench.get_probe(
        "surface_normal",
        head=args.head,
        layers=args.layers,
        uncertainty_aware=not args.no_uncertainty,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        normal_source=args.normal_source,
        device=args.device,
    )

    print(f"\nextracting with {backbone.name} and fitting the {args.head} head...")
    result = visbench.run(
        backbone,
        probe,
        test,
        train_dataset=train,
        cache=cache,
        results=args.results,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    print("\nmetrics (probe3d protocol; angles in degrees, d-metrics are fractions):")
    for name, value in result.metrics.items():
        print(f"  {name:>8s}  {value:.4f}")
    # Not a result: it separates "this representation does not carry surface
    # orientation" from "this probe did not converge", which the metrics alone
    # cannot. In radians, unlike everything printed above it.
    print(f"\n  train loss {result.probe.train_loss:.4f}")
    print(f"\nrecord appended to {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
