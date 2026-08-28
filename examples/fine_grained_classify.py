"""Linear-probe fine-grained recognition on a folder of species photographs.

Same machinery as ``examples/classify.py`` — a linear layer on cached pooled
features — but a different question: which *species* of bird, not which basic-
level category. It is a distinct probe (``fine_grained_classification``) with
its own leaderboard board, because a backbone's ranking can move between the
two — and the object board is saturated where this one is not.

Expects the standard labelled layout, which is exactly how the CUB-200-2011
copy on this machine ships::

    <data>/train/<species>/<image>
    <data>/val/<species>/<image>

Run it::

    python examples/fine_grained_classify.py \
        --data /shared/sets/datasets/vision/CUB-200/images_train_test
    python examples/fine_grained_classify.py --data ... --backbone dinov2_vitb14 --limit 5

CUB's official split is 5994 train / 5794 val across 200 classes, and that
whole split is what the corpus board runs — small enough to need no cap. In
that copy ``val/`` *is* the official test set (``test/`` is a symlink to it).

``--limit N`` caps both splits to N images *per class*, which is the sane way
to run a first pass — a plain prefix would be entirely class 0, since the file
list is grouped by class, and would score 1.0 while measuring nothing. Note
that a limited run is a different dataset fingerprint and so lands in a
different comparability group; it is for checking the wiring, not for a number
to quote beside the board.

**The default schedule is enough here, which was checked rather than
assumed.** 200 classes over ~6k images looks like the case most likely to
underfit, and it is not: ``train top1`` is 1.0000 on all six backbones
measured, DINOv2-S/14 scoring 0.8642 / 0.9717 top-1/top-5 on val. So the gap is
generalisation, and a low score is the backbone rather than the probe. Read
``train top1`` anyway on a backbone that has not been measured — the note below
fires if it drops.

The first run extracts features; every later run on the same data reads them
from the cache and the backbone never executes.
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


def load_split(root: Path, split: str, limit: int | None = None) -> ImageFolderDataset:
    dataset = ImageFolderDataset(root / split, split=split)
    # Per *class*, not a prefix: the file list is grouped by class, so the first
    # N paths would all come from class 0 and a single-class evaluation reports
    # 1.0 while measuring nothing. The fingerprint follows the surviving files,
    # so a limited run cannot be mistaken for a full one in the results.
    return dataset if limit is None else dataset.balanced_subset(limit)


def main() -> None:
    args = parse_args()

    train = load_split(args.data, "train", args.limit)
    test = load_split(args.data, "val", args.limit)
    print(f"train: {len(train)} images, {len(train.classes)} species")
    print(f"val:   {len(test)} images")

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    probe = visbench.get_probe(
        "fine_grained_classification",
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
