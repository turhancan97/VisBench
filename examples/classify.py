"""Linear-probe classification on a local image folder.

Expects the standard layout::

    <data>/train/<class_name>/<image>
    <data>/val/<class_name>/<image>

Run it::

    python examples/classify.py --data /path/to/dataset
    python examples/classify.py --data /path/to/dataset --backbone dinov2_vitb14

The first run extracts features; every later run on the same data reads them
from the cache and the backbone never executes, so sweeping probe settings is
cheap::

    python examples/classify.py --data ... --epochs 500 --lr 0.05

This is an example, not the CLI. The packaged ``visbench`` command is
deliberately deferred to v0.2, once the Python API has settled (CLAUDE.md,
"v0.1 — explicitly deferred").
"""

import argparse
import json
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data import ImageFolderDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="root containing train/ and val/")
    parser.add_argument("--backbone", default="dinov2_vits14", help="see visbench.list_backbones()")
    parser.add_argument("--pooling", default=None, help="cls | mean; default is the backbone's")
    parser.add_argument("--batch-size", type=int, default=64, help="extraction batch size")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--standardize", action="store_true", help="normalise features first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    parser.add_argument("--cache", type=Path, default=Path(".visbench_cache"))
    parser.add_argument("--results", type=Path, default=Path("results/visbench.jsonl"))
    parser.add_argument(
        "--limit", type=int, default=None, help="use at most N images per class (for a quick run)"
    )
    return parser.parse_args()


def load_split(root: Path, split: str, limit: int = None) -> ImageFolderDataset:
    dataset = ImageFolderDataset(root / split, split=split)
    if limit is None:
        return dataset

    # Per *class*, not per split: slicing the first N paths overall would take
    # them all from class 0, and a single-class evaluation reports 1.0 while
    # measuring nothing.
    kept_paths, kept_labels, seen = [], [], {}
    for path, label in zip(dataset.paths, dataset._labels):
        if seen.get(label, 0) >= limit:
            continue
        seen[label] = seen.get(label, 0) + 1
        kept_paths.append(path)
        kept_labels.append(label)

    dataset.paths, dataset._labels = kept_paths, kept_labels
    # The fingerprint is derived from the surviving file list, so a limited run
    # can never be mistaken for a full one in the results.
    return dataset


def main() -> None:
    args = parse_args()

    train = load_split(args.data, "train", args.limit)
    test = load_split(args.data, "val", args.limit)
    print(f"train: {len(train)} images, {len(train.classes)} classes")
    print(f"val:   {len(test)} images")

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    # Configured up front and passed as an object: `device` means the backbone
    # in run()'s signature, so the probe's own device has to come this way.
    probe = visbench.get_probe(
        "classification",
        num_classes=len(train.classes),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        standardize=args.standardize,
        device=args.device,
        pooling=args.pooling or "default",
    )

    print(f"\nextracting with {backbone.name} and fitting the probe...")
    result = visbench.run(
        backbone,
        probe,
        test,
        train_dataset=train,
        cache=cache,
        results=args.results,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    stats = cache.stats()
    print(f"cache: {stats['hits']} hits, {stats['misses']} misses")
    print(f"\ntrain top1: {result.probe.train_top1:.4f}   (loss {result.probe.train_loss:.4f})")
    for name, value in result.metrics.items():
        print(f"val   {name}: {value:.4f}")
    if result.probe.train_top1 < 0.9:
        print(
            "\n  note: training accuracy is low, so the probe underfitted rather than\n"
            "  the backbone being weak. Try --epochs 500 --lr 0.05, or --standardize."
        )

    print(f"\nwrote {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
