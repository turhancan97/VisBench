"""Zero-shot retrieval on a local image folder.

Nothing is trained. Every image queries every *other* image by cosine
similarity over pooled features, and a retrieval counts as correct when it
comes from the same class — so this measures whether the backbone's features
already group categories together.

Expects ``<data>/<split>/<class_name>/<image>``, or pass ``--split ""`` if
``<data>`` holds the class folders directly::

    python examples/retrieve.py --data /path/to/dataset --split val
    python examples/retrieve.py --data /path/to/dataset --split val --pooling mean

Features are shared with the classification example: both read the same cache,
so if you have run one, the other pays only for what it adds.

This is an example, not the CLI, which stays deferred to v0.2.
"""

import argparse
import json
import time
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data import ImageFolderDataset
from visbench.results import ResultRecord, ResultWriter
from visbench.results.schema import utc_timestamp
from visbench.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="val", help='subdirectory to use; "" for <data> itself')
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--pooling", default=None, help="cls | mean; default is the backbone's")
    parser.add_argument("--metric", default="cosine", choices=("cosine", "l2"))
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache", type=Path, default=Path(".visbench_cache"))
    parser.add_argument("--results", type=Path, default=Path("results/visbench.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="at most N images per class")
    return parser.parse_args()


def load(root: Path, split: str, limit: int = None) -> ImageFolderDataset:
    dataset = ImageFolderDataset(root / split if split else root, split=split or "all")
    if limit is None:
        return dataset

    kept_paths, kept_labels, seen = [], [], {}
    for path, label in zip(dataset.paths, dataset._labels):
        if seen.get(label, 0) >= limit:
            continue
        seen[label] = seen.get(label, 0) + 1
        kept_paths.append(path)
        kept_labels.append(label)
    dataset.paths, dataset._labels = kept_paths, kept_labels
    return dataset


def main() -> None:
    args = parse_args()
    seed = set_seed(args.seed)

    dataset = load(args.data, args.split, args.limit)
    print(f"{len(dataset)} images, {len(dataset.classes)} classes")

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    probe = visbench.get_probe("retrieval", topk=tuple(args.topk), metric=args.metric)
    pooling = args.pooling or probe.pooling
    if pooling == "default":
        pooling = backbone.default_pooling()

    cache = FeatureCache(root=args.cache)
    started = time.perf_counter()

    print(f"\nextracting with {backbone.name} (pooling={pooling})...")
    features = cache.extract_dataset(
        backbone, dataset, pooling=pooling, batch_size=args.batch_size, keep="pooled"
    )
    stats = cache.stats()
    print(f"cache: {stats['hits']} hits, {stats['misses']} misses")

    # Leave-one-out ranking is an N x N score matrix; at 50k images that is
    # ~10 GB, so warn rather than let it fail deep inside torch.
    pairs = len(dataset) ** 2
    if pairs > 50_000_000:
        print(
            f"  warning: {len(dataset)} images means a {len(dataset)}x{len(dataset)} "
            f"score matrix (~{pairs * 4 / 1e9:.1f} GB). Consider --limit."
        )

    print(f"\nranking ({args.metric}, leave-one-out)...")
    metrics = probe.fit(features).evaluate(features, dataset.labels())
    duration = time.perf_counter() - started

    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    chance = 1.0 / len(dataset.classes)
    print(f"\n  (chance recall@1 for {len(dataset.classes)} classes is ~{chance:.4f})")

    described = {**dataset.describe(), **probe.describe()}
    record = ResultRecord(
        backbone=backbone.name,
        backbone_key=backbone.cache_key(),
        task=described["task"],
        level=described["level"],
        dataset=args.data.name,
        split=described["split"],
        dataset_size=described["dataset_size"],
        dataset_fingerprint=described["dataset_fingerprint"],
        pooling=pooling,
        feature_mode=described["feature_mode"],
        task_params={"metric": args.metric, "topk": list(args.topk)},
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
