"""Generic (binary) object segmentation on a local folder.

Figure-ground separation with no semantics: is this pixel *an object*, never
*which* object. Expects images and masks paired by filename stem::

    <data>/train/images/scene_0001.jpg
    <data>/train/masks/scene_0001.png
    <data>/val/images/...
    <data>/val/masks/...

Masks may be ``.npy`` or an image; either way **non-zero is foreground**, which
covers both the 0/1 and the 0/255 conventions without having to guess a scale.
They are never rescaled — a mask is a label, not a measurement.

Run it::

    python examples/segment.py --data /path/to/dataset
    python examples/segment.py --data ... --head dpt --layers 2 5 8 11
    python examples/segment.py --data ... --ignore-index 255

**Quote IoU, not accuracy.** Objects are a minority of most frames, so a probe
that predicts background everywhere already scores high pixel accuracy and zero
IoU. All three metrics are printed for exactly that reason.

**Report the linear head.** It is the only one under which a difference between
two backbones is a difference between two feature maps; a DPT head has enough
capacity to compensate for a weak representation and narrow the very gap the
probe exists to measure. Run both and say which is which.

**This protocol is not probe3d's** — that paper has no binary segmentation task.
Only the optimiser schedule is borrowed, so this number sits alongside the depth
and normal ones under one training budget. Records say
``protocol: "visbench_binary_seg"`` so the difference is never lost.

The first run extracts features; later runs read the cache and the backbone
never executes, so sweeping probe settings is cheap. Features are shared with
``examples/depth.py`` and ``examples/normals.py`` when the images and
``--image-size`` match, so probing all three tasks on one dataset costs one
extraction.

This is an example, not the CLI — that is still deferred until the Python API
for dense tasks has settled.
"""

import argparse
import functools
import json
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data import DenseFolderDataset, load_mask


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
    parser.add_argument("--target-dir", default="masks", help="folder name under each split")
    parser.add_argument("--image-size", type=int, default=224, help="multiple of the patch size")
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=None,
        help="raw mask value marking unlabelled pixels (255 for VOC-style masks); "
        "those pixels are excluded from both the loss and the metrics",
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
        target_loader=functools.partial(load_mask, ignore_index=args.ignore_index),
        # No max_target: it exists to mark out-of-range sensor readings invalid,
        # and against a label map it would erase the foreground class outright.
    )
    if args.limit is not None:
        # Every stem list stays in step: dropping one of the three would pair a
        # mask with the wrong image, which no later check would catch.
        dataset.stems = dataset.stems[: args.limit]
        dataset.image_paths = dataset.image_paths[: args.limit]
        dataset.target_paths = dataset.target_paths[: args.limit]
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

    foreground = train.targets().clamp(min=0).mean().item()
    print(f"\nforeground covers {foreground:.1%} of the training frames")
    print(
        f"  a probe predicting background everywhere would score {1 - foreground:.1%} pixel "
        "accuracy and 0 IoU"
    )

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    probe = visbench.get_probe(
        "generic_segmentation",
        head=args.head,
        layers=args.layers,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
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

    print("\nmetrics (foreground IoU is the one to quote):")
    for name, value in result.metrics.items():
        print(f"  {name:>10s}  {value:.4f}")
    # Not a result: it separates "this representation does not carry
    # figure-ground structure" from "this probe did not converge", which the
    # metrics alone cannot.
    print(f"\n  train loss {result.probe.train_loss:.4f}")
    print(f"\nrecord appended to {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
