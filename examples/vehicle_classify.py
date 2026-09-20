"""Linear-probe vehicle recognition on a folder of car photographs.

Same machinery as ``examples/fine_grained_classify.py`` — a linear layer on
cached pooled features — and the same *granularity*: subordinate categories
inside one basic-level class. What differs is the kind of object. Two bird
species share a body plan and differ in a wing bar; two car models share a
silhouette and differ in badge, grille and lamp geometry, and whether a
representation keeps one kind of detail says little about whether it keeps the
other. Measured over thirteen backbones, the two boards correlate +0.879 and
**change leader**: ``siglip_vitb16`` is first here and fifth on CUB, while
``supervised_vitb16`` falls from eleventh to last.

Expects the standard labelled layout::

    <data>/train/<model>/<image>
    <data>/val/<model>/<image>

**The split this probe's board uses is VisBench's own, and the script that
builds it is part of the protocol.** The Stanford Cars copy on this machine is
not the official 8,144/8,041 split — eleven train images are the same photograph
filed under two class directories, seven more pairs are in test, and one image
is blank. ``scripts/stage_cars_split.py`` drops them and stages **8,125 / 8,026**
as symlinks with a committed manifest, so:

    python scripts/stage_cars_split.py
    python examples/vehicle_classify.py --data data/cars_split

Numbers measured this way are **not comparable with published Stanford Cars
results**. Pointing ``--data`` at the raw copy runs fine and measures something
slightly different, which is why the board's flags live in
``scripts/build_corpus.sh`` rather than here.

``--limit N`` caps both splits to N images *per class* — a plain prefix would be
entirely class 0, since the file list is grouped by class, and would score 1.0
while measuring nothing. A limited run is a different fingerprint and so a
different comparability group; it is for checking the wiring.

**The default schedule is enough here, checked rather than assumed**: 196
classes over ~8k images reaches ``train top1`` 1.0000 on every backbone
measured, so the gap to the val score is generalisation rather than an
unconverged probe. Read ``train top1`` anyway on a backbone nobody has measured.

The first run extracts features; every later run reads them from the cache. A
staged symlink resolves to the same path as the original, so features extracted
from the raw copy are reused rather than recomputed.
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
    print(f"train: {len(train)} images, {len(train.classes)} car models")
    print(f"val:   {len(test)} images")

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    probe = visbench.get_probe(
        "vehicle_classification",
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
