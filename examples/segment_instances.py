"""Instance segmentation — detect-then-segment over frozen features.

Step 14a-3, and the first step in this line where a number about a *backbone*
appears. The two before it built the dataset and the metric, and both were
calibrated before any head existed: perfect predictions score exactly 1.0000
mask mAP, so a low number here is a statement about the head or the features
rather than about the scorer.

**Pascal VOC 2012**, the only layout this reads::

    python examples/segment_instances.py --data /path/to/pascal_voc

    <data>/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg
    <data>/VOCdevkit/VOC2012/SegmentationObject/2007_000032.png
    <data>/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png
    <data>/VOCdevkit/VOC2012/ImageSets/Segmentation/{train,val}.txt

``ImageSets/Segmentation``, so 1,464 train / 1,449 val — *the same images the
semantic segmentation probe scores*, which is the point: two boards over
identical pixels answering different questions.

**VOC and not COCO, and that was measured** (14a-1). A dense probe reads one
feature vector per patch, so instances have to survive the feature grid. On VOC
val at 224px on a 16x16 grid the median instance covers 16.92 patches and
**zero of 3,207 instance pairs share a grid cell**; on COCO the median is 2.27
patches. The ceiling a perfect per-patch predictor could reach here is mask
mAP@50 **0.6666**, against a connected-components floor of 0.1376 mean IoU.

Four things about this probe that are not incidental:

**Read the mask AP against another backbone, never against the segmentation
literature.** The mask branch is a *single 1x1 convolution* over RoI-aligned
features, where Mask R-CNN's is four 3x3 convolutions and a deconvolution on an
FPN. Every point those would add is a point about the branch rather than about
the frozen features, which is the same argument that keeps ``LinearHead`` the
head a dense VisBench number is quoted with. The record says
``protocol: "visbench_anchor_free_instance"`` so the number cannot be mistaken
for a segmenter's.

**``box_map_50`` is reported beside ``mask_map_50``, and reading them together
is the point.** Mask AP falls when either half fails. Poor masks inside good
boxes is a statement about whether the features carry an *outline*; boxes that
miss is a statement about localisation, and it is a different finding.

**The mask branch trains on ground-truth boxes and predicts on detected ones.**
Deliberate: the boxes come from the same head being trained, so early epochs
would hand the mask branch RoIs containing no object. Recorded as
``mask_train_boxes: "ground_truth"``.

**``--image-size`` reaches the dataset and the probe from one flag**, for the
reason ``examples/detect.py`` gives at more length: box targets are absolute
pixels in post-transform space, so two different values put every cell centre
and every pasted mask at the wrong coordinate — and the run trains, scores
badly, and reads as a weak backbone.

**This script and the published board differ in the third decimal, and the
board is the number to quote.** The probe is registered as of 14a-4, so
``visbench run instance_segmentation`` exists and is what built the corpus.
This script constructs the backbone itself — the house style for every
``examples/`` script here, because it prints things about the object — and
``run()`` seeds *before* it constructs one from a name, so the two paths fit
the head from different RNG states. Measured on DINOv2-S: **0.2641** here
against **0.2696** on the board, every recorded field identical. Neither is
wrong; they are not comparable to the last decimal, and the board is the one
with twelve backbones beside it.

The CLI also takes ``--data`` as the ``VOCdevkit/VOC2012`` directory itself,
matching ``detection`` and ``semantic_segmentation``, where this script takes
its parent and appends the rest.
"""

import argparse
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data import VOC_CLASSES, VOCInstanceDataset
from visbench.tasks.high_level.instance_segmentation import InstanceSegmentationTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="Pascal VOC root")
    parser.add_argument("--backbone", default="dinov2_vits14", help="see visbench.list_backbones()")
    parser.add_argument("--image-size", type=int, default=224, help="multiple of the patch size")
    parser.add_argument(
        "--mask-size",
        type=int,
        default=14,
        help="side of the RoI the mask is predicted at. 14 rather than Mask R-CNN's 28: the "
        "features are on a 16x16 grid, so a larger RoI asks RoIAlign to invent detail "
        "between patch centres that the backbone never produced",
    )
    parser.add_argument(
        "--mask-hidden-dim",
        type=int,
        default=0,
        help="width of an optional 3x3 stem before the mask convolution. 0 (the default) "
        "keeps the mask branch a linear readout, which is the only setting under which a "
        "difference between backbones is a difference between representations",
    )
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    parser.add_argument("--cache", type=Path, default=Path(".visbench_cache"))
    parser.add_argument("--results", type=Path, default=Path("results/visbench.jsonl"))
    parser.add_argument(
        "--limit", type=int, default=None, help="use at most N images per split (quick run)"
    )
    return parser.parse_args()


def load_split(args: argparse.Namespace, split: str) -> VOCInstanceDataset:
    """One official segmentation split of the VOC devkit."""
    root = args.data / "VOCdevkit" / "VOC2012"
    listing = root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not listing.is_file():
        raise SystemExit(
            f"No VOC segmentation split list at {listing}. Note this is "
            "ImageSets/Segmentation, not ImageSets/Main — the detection split is four "
            "times larger and a schedule sized on one is not sized on the other."
        )
    stems = listing.read_text().split()
    if args.limit is not None:
        stems = stems[: args.limit]
    return VOCInstanceDataset(root, split=split, stems=stems, image_size=args.image_size)


def main() -> None:
    args = parse_args()

    train = load_split(args, "train")
    test = load_split(args, "val")
    print(f"train: {len(train)} images at {args.image_size}px, {len(VOC_CLASSES)} classes")
    print(f"val:   {len(test)} images")

    instances = 0
    dropped = 0
    for index in range(len(train)):
        target = train.target(index)
        instances += target["masks"].shape[0]
        dropped += target["num_original"] - target["masks"].shape[0]
    print(f"\ntraining instances: {instances} ({dropped} removed by the centre crop)")
    if instances == 0:
        raise SystemExit(
            "No training instances survived. At this --image-size the centre crop removed "
            "every object, so there is nothing to learn from."
        )

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    # Constructed directly rather than by name: this probe is not registered
    # (see the module docstring). Built as an *object*, which is what run()
    # takes — `batch_size` means extraction there and training here, so passing
    # probe settings through run(**kwargs) is a TypeError by design.
    probe = InstanceSegmentationTask(
        num_classes=len(VOC_CLASSES),
        # One value, from one flag; a probe and a dataset that disagree here
        # misplace every cell centre and every pasted mask.
        image_size=args.image_size,
        mask_size=args.mask_size,
        mask_weight=args.mask_weight,
        mask_hidden_dim=args.mask_hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold,
        nms_iou=args.nms_iou,
        max_detections=args.max_detections,
        device=args.device,
    )

    print(f"\nextracting with {backbone.name} and fitting the instance head...")
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
        print(f"  {name:>22s}  {value:.4f}")

    summary = probe.training_summary()
    if summary:
        print("\ntraining (a diagnostic, never a score):")
        for name, value in summary.items():
            print(f"  {name:>22s}  {value:.4f}")
        print(
            "\n  A low mask AP with a converged mask loss says the features do not carry\n"
            "  the outline; a high one says the branch never fitted. Opposite conclusions."
        )

    print(
        "\n  mask_map_50 is the headline. Read it against another backbone, not against\n"
        "  the segmentation literature: the mask branch is one 1x1 convolution, and the\n"
        "  ceiling a perfect per-patch predictor reaches on this split is 0.6666."
    )


if __name__ == "__main__":
    main()
