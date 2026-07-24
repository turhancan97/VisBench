"""Zero-shot geometric correspondence on a local image folder.

Nothing is trained and no annotation is needed: each image is warped by a known
random homography, so the true correspondence of every patch is exact and free.
Patch features are matched between the two views, and a match counts as correct
when it lands within N pixels of where the homography says it should.

    python examples/correspond.py --data /path/to/folder
    python examples/correspond.py --data /path/to/folder --max-warp 0.3

What this measures is how well the backbone's *dense* features survive a
viewpoint change. What it does not measure is 3D correspondence across genuine
parallax — a homography is a planar transform. probe3d's ScanNet/NAVI protocol
is the real thing and arrives in v0.2.

This is an example, not the CLI, which stays deferred to v0.2.
"""

import argparse
import json
import time
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data.pair_dataset import HomographyPairDataset
from visbench.results import ResultRecord, ResultWriter
from visbench.results.schema import utc_timestamp
from visbench.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="folder of images")
    parser.add_argument("--split", default="", help="subdirectory to use, if any")
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--max-warp", type=float, default=0.2, help="corner shift, 0-0.5")
    parser.add_argument("--num-corr", type=int, default=1000)
    parser.add_argument("--ratio", type=float, default=0.9, help="Lowe ratio threshold")
    parser.add_argument(
        "--units", default="patch", choices=("patch", "pixel"), help="threshold units"
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="default: 0.5 1 2 4 patch widths, or 1 2 5 10 pixels",
    )
    parser.add_argument("--limit", type=int, default=200, help="number of pairs")
    parser.add_argument("--image-size", type=int, default=224, help="must match the backbone")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache", type=Path, default=Path(".visbench_cache"))
    parser.add_argument("--results", type=Path, default=Path("results/visbench.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = set_seed(args.seed)

    root = args.data / args.split if args.split else args.data
    # image_size must match the backbone's, or the homography describes a
    # different plane from the one the patches sit on.
    dataset = HomographyPairDataset(
        root, max_warp=args.max_warp, seed=args.seed, image_size=args.image_size
    )
    if args.limit:
        dataset._source.paths = dataset._source.paths[: args.limit]
        dataset._source._labels = dataset._source._labels[: args.limit]
    print(f"{len(dataset)} pairs, max_warp={args.max_warp}")

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    probe = visbench.get_probe(
        "correspondence",
        num_corr=args.num_corr,
        ratio_threshold=args.ratio,
        thresholds=tuple(args.thresholds) if args.thresholds else None,
        threshold_units=args.units,
    )
    cache = FeatureCache(root=args.cache)
    started = time.perf_counter()

    print(f"\nextracting dense features with {backbone.name}...")
    pairs = []
    for index in range(len(dataset)):
        image_0, image_1, _ = dataset[index]
        # One image at a time: correspondence matches pairwise, and dense
        # features for a whole folder would not fit in memory anyway.
        views = []
        for image in (image_0, image_1):
            views.append(
                cache.extract_dataset(backbone, [image], batch_size=1, keep="dense", store="dense")
            )
        pairs.append(tuple(views))
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(dataset)} pairs")

    stats = cache.stats()
    print(f"cache: {stats['hits']} hits, {stats['misses']} misses")
    grid = pairs[0][0]["grid_hw"]
    print(f"dense grid: {grid[0]}x{grid[1]} patches")

    print("\nmatching (nearest neighbour + ratio test)...")
    geometries = dataset.labels()
    metrics = probe.fit(pairs).evaluate(pairs, geometries)
    ceiling = probe.evaluate_ceiling(pairs, geometries)
    duration = time.perf_counter() - started

    matches = int(metrics.pop("num_matches"))
    spacing = args.image_size / grid[0]
    print(f"  {'metric':<14}{'score':>9}{'ceiling':>10}")
    for name, value in sorted(metrics.items()):
        print(f"  {name:<14}{value:>9.4f}{ceiling[name]:>10.4f}")
    unit = "patch widths" if args.units == "patch" else "pixels"
    print(
        f"\n  thresholds are in {unit}; patches are {spacing:.0f}px apart.\n"
        f"  ceiling is what a perfect matcher scores: a target between two\n"
        f"  patch centres cannot be hit exactly."
    )
    print(
        f"\n  {matches} matches kept across {len(dataset)} pairs "
        f"({matches / len(dataset):.0f} per pair, of {grid[0] * grid[1]} patches)"
    )
    metrics["num_matches"] = float(matches)

    described = {**dataset.describe(), **probe.describe()}
    record = ResultRecord(
        backbone=backbone.name,
        backbone_key=backbone.cache_key(),
        task=described["task"],
        level=described["level"],
        dataset=root.name,
        split=described["split"],
        dataset_size=described["dataset_size"],
        dataset_fingerprint=described["dataset_fingerprint"],
        pooling=described["pooling"],
        feature_mode=described["feature_mode"],
        task_params={**described["task_params"], "max_warp": args.max_warp},
        metrics=metrics,
        timestamp=utc_timestamp(),
        visbench_version=visbench.__version__,
        seed=seed,
        duration_seconds=duration,
    )
    with ResultWriter(args.results) as writer:
        writer.write(record)
    print(f"\nwrote {args.results}")
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
