"""Semantic (multi-class) segmentation — which category is each pixel.

The high-level counterpart to ``examples/segment.py``: same base class, same
schedule, same features. That one asks whether a pixel is *an object*; this one
asks *which* of N categories it is. Run both on one backbone and the difference
is a difference in what is being asked of the representation.

Two layouts are supported.

**Pascal VOC**, with ``--voc``, reads the devkit directly::

    python examples/segment_semantic.py --data /path/to/pascal_voc --voc

    <data>/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg
    <data>/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png
    <data>/VOCdevkit/VOC2012/ImageSets/Segmentation/{train,val}.txt

The split files matter: VOC ships 17k images beside 2.9k segmentation labels, so
membership comes from the official lists, not from whatever the folders happen
to contain.

**Folder pairs**, the default, matching ``examples/segment.py``::

    <data>/train/images/scene_0001.jpg
    <data>/train/labels/scene_0001.png

Label maps are read **without mode conversion**. VOC's PNGs are palette images
whose raw bytes are the class indices; ``convert("L")`` would apply the palette
and turn classes ``[0, 1, 15]`` into ``[0, 38, 147]``, which trains and scores
against labels that mean nothing. 255 becomes -1, the ignore value, and is
dropped from both the loss and the metrics.

**Two mIoUs are printed and they differ.** ``miou`` is the dataset-level
reduction — one confusion matrix over the split — which is what VOC and the
literature report. ``miou_per_image`` averages each image's own mIoU, which is
this codebase's convention elsewhere. Quote ``miou`` when comparing against
published numbers and say so.

**Quote the linear head.** It is the only one under which a difference between
two backbones is a difference between two feature maps.

**This protocol is not probe3d's** — that paper has no semantic segmentation
task, and only its optimiser schedule is borrowed. Records say
``protocol: "visbench_semantic_seg"``.

**Ten epochs is probe3d's budget, sized for NYUv2-scale data.** On a few hundred
images it underfits; ``train_loss`` is what separates "this representation lacks
the structure" from "this probe did not converge". Raise ``--epochs`` and
``--lr`` rather than reading a low mIoU as a verdict on the backbone.

This is an example, not the CLI, which is still deferred until the dense-task
Python API has settled.
"""

import argparse
import functools
import json
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data import DenseFolderDataset, load_label_map

#: VOC 2012: 20 object categories plus background. 255 marks the object
#: outlines, which are deliberately unlabelled.
VOC_CLASSES = 21
VOC_IGNORE = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="dataset root")
    parser.add_argument("--voc", action="store_true", help="read the Pascal VOC devkit layout")
    parser.add_argument("--backbone", default="dinov2_vits14", help="see visbench.list_backbones()")
    parser.add_argument("--head", default="linear", help="linear | dpt; see visbench.heads")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="backbone depths, shallowest first; required by --head dpt",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=f"including background; defaults to {VOC_CLASSES} with --voc, else required",
    )
    parser.add_argument("--image-dir", default="images", help="folder layout only")
    parser.add_argument("--target-dir", default="labels", help="folder layout only")
    parser.add_argument("--image-size", type=int, default=224, help="multiple of the patch size")
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=VOC_IGNORE,
        help="raw label value marking unlabelled pixels; those are excluded from "
        "both the loss and the metrics. Pass -1 to disable",
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


def load_split(args: argparse.Namespace, split: str) -> DenseFolderDataset:
    """Build one split, from either layout."""
    ignore = None if args.ignore_index < 0 else args.ignore_index
    loader = functools.partial(load_label_map, ignore_index=ignore)

    if args.voc:
        root = args.data / "VOCdevkit" / "VOC2012"
        listing = root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not listing.is_file():
            raise SystemExit(f"No VOC split list at {listing}")
        stems = listing.read_text().split()
        if args.limit is not None:
            stems = stems[: args.limit]
        return DenseFolderDataset(
            root,
            image_dir="JPEGImages",
            target_dir="SegmentationClass",
            split=split,
            image_size=args.image_size,
            target_loader=loader,
            stems=stems,
            # No max_target: it marks out-of-range *sensor* readings invalid,
            # and against class indices it would erase whole categories.
        )

    dataset = DenseFolderDataset(
        args.data / split,
        image_dir=args.image_dir,
        target_dir=args.target_dir,
        split=split,
        image_size=args.image_size,
        target_loader=loader,
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

    num_classes = args.num_classes or (VOC_CLASSES if args.voc else None)
    if num_classes is None:
        raise SystemExit(
            "--num-classes is required: it sizes the head's output, and a wrong value "
            "does not raise, it silently trains a head that cannot express some categories."
        )

    train = load_split(args, "train")
    test = load_split(args, "val")
    print(f"train: {len(train)} images at {args.image_size}px, {num_classes} classes")
    print(f"val:   {len(test)} images")

    targets = train.targets()
    labelled = (targets >= 0).float().mean().item()
    present = sorted({int(value) for value in targets.unique() if value >= 0})
    print(f"\nlabelled pixels: {labelled:.1%} (the rest are ignored, not background)")
    print(f"classes present in train: {len(present)} of {num_classes}")
    if present and present[-1] >= num_classes:
        raise SystemExit(
            f"Found class index {present[-1]} but --num-classes is {num_classes}. "
            "Out-of-range labels are dropped from the confusion matrix, so this would "
            "silently score against fewer pixels than the split contains."
        )

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    probe = visbench.get_probe(
        "semantic_segmentation",
        num_classes=num_classes,
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

    print("\nmetrics:")
    for name, value in result.metrics.items():
        print(f"  {name:>16s}  {value:.4f}")
    print("\n  miou is the dataset-level reduction, comparable to published VOC numbers.")
    print("  miou_per_image is this codebase's per-image convention. They differ; say which.")
    # Not a result: it separates "this representation lacks the structure" from
    # "this probe did not converge", which the metrics alone cannot.
    print(f"\n  train loss {result.probe.train_loss:.4f}")
    print(f"\nrecord appended to {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
