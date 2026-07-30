"""Object detection — an anchor-free single-scale probe over frozen features.

The last piece of step 6c, and the one built against a metric that was already
trusted: ``visbench/metrics/detection.py`` was cross-checked against a literal
``VOCevaldet.m`` transcription to zero difference before any head existed, so a
low mAP here is a statement about the head or the features, not about the
scorer.

**Pascal VOC 2012 Detection**, with ``--voc``, reads the devkit directly::

    python examples/detect.py --data /path/to/pascal_voc --voc

    <data>/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg
    <data>/VOCdevkit/VOC2012/Annotations/2007_000032.xml
    <data>/VOCdevkit/VOC2012/ImageSets/Main/{train,val}.txt

Note ``ImageSets/**Main**``, not ``ImageSets/Segmentation``: the detection split
is 5,717 train / 5,823 val, about four times the segmentation split the other
VOC examples use. A schedule sized on 1,464 images is not sized on these.

**Folder pairs**, the default::

    <data>/train/JPEGImages/scene_0001.jpg
    <data>/train/Annotations/scene_0001.xml

Four things about this probe that are not incidental:

**Read the mAP against another backbone, never against the literature.** A
single-scale linear head over a 16x16 patch grid has no feature pyramid, so
small objects fall between cells and are simply unrecoverable. That ceiling is
deliberate — VisBench measures what a frozen representation carries, and every
point an FPN would add is a point about the FPN. The record says
``protocol: "visbench_anchor_free_det"`` so the number cannot be mistaken for a
VOC detector's.

**``map_50`` is the VOC-comparable number; ``map_50_95`` is COCO-*style*.** It
averages COCO's ten IoU thresholds but integrates all recall points at each,
where COCO quantises recall to 101 points.

**The scored split keeps ``difficult`` objects and the training split drops
them, and that asymmetry is the protocol.** VOC *ignores* a detection matching a
difficult object rather than counting it wrong; dropping those boxes from the
ground truth instead scores 4.3 mAP lower on VOC val and looks like a weaker
detector. This example builds the two splits accordingly.

**``--image-size`` reaches the dataset and the probe from one flag.** Box
targets are absolute pixels in post-transform space, so two different values
would put every grid cell at the wrong coordinate — and the run would train,
score badly, and read as a weak backbone. Nothing raises on a mismatch except
the probe's own range check, which only catches one direction.
"""

import argparse
import json
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data import VOC_CLASSES, DetectionFolderDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="dataset root")
    parser.add_argument("--voc", action="store_true", help="read the Pascal VOC devkit layout")
    parser.add_argument("--backbone", default="dinov2_vits14", help="see visbench.list_backbones()")
    parser.add_argument("--head", default="detection", help="see visbench.heads.list_heads()")
    parser.add_argument("--image-dir", default="JPEGImages", help="folder layout only")
    parser.add_argument("--annotation-dir", default="Annotations", help="folder layout only")
    parser.add_argument("--image-size", type=int, default=224, help="multiple of the patch size")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=0,
        help="width of an optional 3x3 stem. 0 (the default) keeps the head linear, which "
        "is the only setting under which a difference between backbones is a difference "
        "between representations",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=1.0,
        help="drop boxes smaller than this after the centre crop; a centre crop genuinely "
        "removes objects, and scoring against one absent from the input measures nothing",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    parser.add_argument("--cache", type=Path, default=Path(".visbench_cache"))
    parser.add_argument("--results", type=Path, default=Path("results/visbench.jsonl"))
    parser.add_argument(
        "--limit", type=int, default=None, help="use at most N images per split (quick run)"
    )
    return parser.parse_args()


def load_split(args: argparse.Namespace, split: str, include_difficult: bool):
    """Build one split, from either layout.

    ``include_difficult`` is True for the split that gets *scored* and False for
    the one that trains — see the module docstring; it is worth 4.3 mAP and the
    wrong choice is silent.
    """
    if args.voc:
        root = args.data / "VOCdevkit" / "VOC2012"
        listing = root / "ImageSets" / "Main" / f"{split}.txt"
        if not listing.is_file():
            raise SystemExit(
                f"No VOC detection split list at {listing}. Note this is ImageSets/Main, "
                "not ImageSets/Segmentation — they are different splits."
            )
        stems = listing.read_text().split()
        if args.limit is not None:
            stems = stems[: args.limit]
        return DetectionFolderDataset(
            root,
            image_dir="JPEGImages",
            annotation_dir="Annotations",
            split=split,
            stems=stems,
            image_size=args.image_size,
            include_difficult=include_difficult,
            min_box_size=args.min_box_size,
        )

    dataset = DetectionFolderDataset(
        args.data / split,
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        split=split,
        image_size=args.image_size,
        include_difficult=include_difficult,
        min_box_size=args.min_box_size,
    )
    # subset() reindexes the three parallel lists together; slicing one by hand
    # would pair an image with another image's boxes and still train.
    return dataset if args.limit is None else dataset.subset(args.limit)


def main() -> None:
    args = parse_args()

    train = load_split(args, "train", include_difficult=False)
    test = load_split(args, "val", include_difficult=True)
    print(f"train: {len(train)} images at {args.image_size}px, {len(VOC_CLASSES)} classes")
    print(f"val:   {len(test)} images (difficult objects kept, as VOC's protocol needs)")

    boxes = sum(len(train.target(index)["boxes"]) for index in range(len(train)))
    dropped = sum(
        train.target(index)["num_original"] - len(train.target(index)["boxes"])
        for index in range(len(train))
    )
    print(f"\ntraining boxes: {boxes} ({dropped} dropped as difficult or outside the crop)")
    if boxes == 0:
        raise SystemExit(
            "No training boxes survived. At this --image-size the centre crop removed "
            "every object, so there is nothing to learn from."
        )

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    probe = visbench.get_probe(
        "detection",
        num_classes=len(VOC_CLASSES),
        # One value, from one flag. Box targets are absolute pixels, so a probe
        # and a dataset that disagree here misplace every cell centre.
        image_size=args.image_size,
        head=args.head,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold,
        nms_iou=args.nms_iou,
        max_detections=args.max_detections,
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
        print(f"  {name:>20s}  {value:.4f}")
    print("\n  map_50 is the VOC-comparable number; map_50_95 is COCO-style, not COCO.")
    print("  classes_scored is mAP's actual denominator — a class with no non-difficult")
    print("  objects in the split has undefined AP and is excluded, not scored 0.")
    print("  Compare these against another backbone, not against published VOC detectors:")
    print("  a single-scale linear head has no feature pyramid and cannot reach them.")
    # Not a result: it separates "this representation lacks the structure" from
    # "this probe did not converge", which the metrics alone cannot.
    print(f"\n  train loss {result.probe.train_loss:.4f}")
    print(f"\nrecord appended to {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
