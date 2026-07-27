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
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data import ImageFolderDataset


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
    for path, label in zip(dataset.paths, dataset._labels, strict=True):
        if seen.get(label, 0) >= limit:
            continue
        seen[label] = seen.get(label, 0) + 1
        kept_paths.append(path)
        kept_labels.append(label)
    dataset.paths, dataset._labels = kept_paths, kept_labels
    return dataset


def main() -> None:
    args = parse_args()

    dataset = load(args.data, args.split, args.limit)
    print(f"{len(dataset)} images, {len(dataset.classes)} classes")

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    # Leave-one-out ranking is an N x N score matrix; at 50k images that is
    # ~10 GB, so warn rather than let it fail deep inside torch.
    pairs = len(dataset) ** 2
    if pairs > 50_000_000:
        print(
            f"  warning: {len(dataset)} images means a {len(dataset)}x{len(dataset)} "
            f"score matrix (~{pairs * 4 / 1e9:.1f} GB). Consider --limit."
        )

    print(f"\nextracting with {backbone.name} and ranking ({args.metric})...")
    result = visbench.run(
        backbone,
        "retrieval",
        dataset,
        cache=cache,
        results=args.results,
        batch_size=args.batch_size,
        seed=args.seed,
        topk=tuple(args.topk),
        metric=args.metric,
        pooling=args.pooling or "default",
    )

    stats = cache.stats()
    print(f"cache: {stats['hits']} hits, {stats['misses']} misses")
    for name, value in result.metrics.items():
        print(f"  {name}: {value:.4f}")
    chance = 1.0 / len(dataset.classes)
    print(f"\n  (chance recall@1 for {len(dataset.classes)} classes is ~{chance:.4f})")

    print(f"\nwrote {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
