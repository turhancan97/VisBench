"""Monocular depth probing on a local folder, following probe3d's protocol.

Expects images and depth maps paired by filename stem::

    <data>/train/images/scene_0001.jpg
    <data>/train/depths/scene_0001.npy
    <data>/val/images/...
    <data>/val/depths/...

Depth maps may be ``.npy`` in metres, or 16-bit PNG/TIFF in millimetres with
``--target-scale 1000``.

Run it::

    python examples/depth.py --data /path/to/dataset
    python examples/depth.py --data /path/to/dataset --head dpt --layers 2 5 8 11

**Report the linear head.** It is the only one under which a difference between
two backbones is a difference between two feature maps; a DPT head has enough
capacity to compensate for a weak representation and narrow the very gap the
probe exists to measure. Run both and say which is which::

    python examples/depth.py --data ... --head linear
    python examples/depth.py --data ... --head dpt --layers 2 5 8 11

The first run extracts features; later runs read the cache and the backbone
never executes, so sweeping probe settings is cheap.

This is an example, not the CLI — that is still deferred until the Python API
for dense tasks has settled.
"""

import argparse
import json
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data import DenseFolderDataset


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
    parser.add_argument("--image-size", type=int, default=224, help="multiple of the patch size")
    parser.add_argument("--target-scale", type=float, default=1.0, help="1000 for millimetre PNGs")
    parser.add_argument("--max-depth", type=float, default=10.0, help="beyond this is invalid")
    parser.add_argument("--n-bins", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512, help="DPT width")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--scale-invariant",
        action="store_true",
        help="fit a per-image scale and shift before scoring; changes what the number means",
    )
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
        image_size=args.image_size,
        target_scale=args.target_scale,
        max_target=args.max_depth,
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

    train = load_split(args.data, "train", args)
    test = load_split(args.data, "val", args)
    print(f"train: {len(train)} images at {args.image_size}px")
    print(f"val:   {len(test)} images")

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    probe = visbench.get_probe(
        "depth",
        head=args.head,
        layers=args.layers,
        max_depth=args.max_depth,
        n_bins=args.n_bins,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        scale_invariant=args.scale_invariant,
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

    print("\nmetrics (probe3d protocol):")
    for name, value in result.metrics.items():
        print(f"  {name:>8s}  {value:.4f}")
    # Not a result: it separates "this representation does not carry depth"
    # from "this probe did not converge", which the metrics alone cannot.
    print(f"\n  train loss {result.probe.train_loss:.4f}")
    print(f"\nrecord appended to {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
