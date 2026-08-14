"""Looking at what a probe saw, before and after training it::

    python examples/show_panels.py --data /path/to/depth/data --probe depth

    <data>/val/images/*.png
    <data>/val/depths/*.npy      (paired by filename stem)

The Python side of ``visbench show``. It draws a grid of image / target panels,
and — with ``--predict-from`` — a third column from a head saved by
``visbench run --save-probe``.

**This measures nothing.** It exists because a dense target that has drifted
from the image it belongs to fails *silently*: nothing raises, the probe trains,
and the number merely comes out mediocre — which reads as a hard task or a weak
representation, the two explanations VisBench exists to tell apart. Both the
correspondence misalignment that scored ``recall@1px = 0.003`` and VOC's palette
PNGs read through ``convert("L")`` were found by reading code, and both are
obvious in one frame.

**Nothing here resizes a panel.** ``DenseFolderDataset`` already yields a PIL
image at the working resolution, so the image panel is that image, pasted. A
viewer that applied its own geometry could make a misaligned pipeline look fine
and a correct one look broken, which is worse than no viewer at all: whether the
pair lines up *is* the evidence a panel carries.

**Magenta is a pixel with no ground truth**, per that probe's own convention.
There are four conventions across the nine drawable probes — ``0`` for depth,
the zero vector for normals, negative for both segmentations, ``NaN`` for
occlusion edges, and *nothing at all* for edge, keypoints2d and corner, where 0
is a real reading. None of them is visible in a tensor's shape, which is why
``visbench.viz.TARGET_STYLES`` is a listed table rather than a heuristic.

**A prediction is drawn against the target's range**, stated in each row's
label. Scaling each panel to its own extremes would make a prediction at half
the target's magnitude render identically to a correct one.
"""

import argparse
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data.dense import DenseFolderDataset
from visbench.viz import render_probe_panels, show_probes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="root holding <split>/")
    parser.add_argument(
        "--probe",
        default="depth",
        choices=[name for name in show_probes() if name != "detection"],
        help="which probe's target convention to draw by. Detection is omitted here "
        "because its dataset is a different class; `visbench show detection` covers it",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--target-dir", default="depths", help="must match --probe's target")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frames", type=int, default=4, help="rows to draw")
    parser.add_argument("--out", type=Path, default=Path("panels.png"))

    parser.add_argument(
        "--predict-from",
        type=Path,
        default=None,
        help="a probe saved by `visbench run --save-probe`; adds a prediction column. "
        "Without it this needs no backbone, no cache and no GPU",
    )
    parser.add_argument("--backbone", default="dinov2_vits14", help="only with --predict-from")
    parser.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    parser.add_argument("--cache", type=Path, default=Path(".visbench_cache"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = DenseFolderDataset(
        args.data / args.split,
        image_dir=args.image_dir,
        target_dir=args.target_dir,
        split=args.split,
        image_size=args.image_size,
    )
    indices = list(range(min(args.frames, len(dataset))))
    print(f"{args.probe}: {len(indices)} of {len(dataset)} frames from {args.data}")

    predictions = None
    if args.predict_from is not None:
        # load_probe checks the four identity fields, so a head fitted on a
        # different backbone -- or on the same one with different pooling, which
        # is shape-compatible and scores 0.9620 against 0.9820 -- is refused
        # rather than quietly drawn.
        from visbench.hub import load_probe

        backbone = visbench.get_backbone(args.backbone, device=args.device)
        probe = load_probe(args.predict_from, backbone=backbone)

        # subset() reindexes the parallel lists together, so features stay
        # paired with the frames they came from.
        frames = dataset.subset(indices)
        features = FeatureCache(root=args.cache).extract_dataset(
            backbone,
            frames,
            pooling=probe.pooling,
            layers=probe.layers,
            feature_mode=probe.feature_mode,
        )
        predictions = probe.predict(features, frames.labels())
        print(f"  predictions from {args.predict_from}")

    page = render_probe_panels(dataset, args.probe, indices, predictions)
    page.save(args.out)
    print(f"\nwrote {args.out} — magenta is a pixel with no ground truth.")
    print("Check that the target lines up with the image before trusting any score.")


if __name__ == "__main__":
    main()
