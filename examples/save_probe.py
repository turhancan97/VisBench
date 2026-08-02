"""Train a probe, save it, load it back, and show what a mismatch costs.

    python examples/save_probe.py --data /path/to/imagenette2

Runs on real DINOv2 weights rather than the fake backbones in the test suite,
because the thing worth demonstrating is not that the bytes round-trip — the
tests cover that — but that a head loaded against the *wrong* features scores
without complaining. The last section does exactly that, on purpose.
"""

import argparse
from pathlib import Path

import torch

import visbench
from visbench.cache import FeatureCache
from visbench.data import ImageFolderDataset
from visbench.hub import IncompatibleProbe, load_probe, save_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="root of a labelled image folder")
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--limit", type=int, default=200, help="images per class")
    parser.add_argument("--out", default="checkpoints/classification.pt")
    parser.add_argument("--cache", default=".visbench_cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    # The split is a *subdirectory*, and `split=` only names it in the record --
    # the same convention visbench/cli/datasets.py uses.
    root = Path(args.data)
    train = ImageFolderDataset(root / "train", split="train").balanced_subset(args.limit)
    val = ImageFolderDataset(root / "val", split="val").balanced_subset(args.limit)

    probe = visbench.get_probe("classification", device=args.device)
    extract = dict(keep="pooled", pooling=probe.pooling)
    train_features = cache.extract_dataset(backbone, train, **extract)
    val_features = cache.extract_dataset(backbone, val, **extract)

    probe.fit(train_features, train.labels())
    before = probe.evaluate(val_features, val.labels())
    print(f"trained: top1={before['top1']:.4f}")

    path = save_probe(probe, args.out, backbone=backbone, notes="examples/save_probe.py")
    print(f"saved:   {path} ({path.stat().st_size / 1024:.1f} KB)")

    # A fresh probe, none of the training state -- everything comes off disk.
    reloaded = load_probe(path, backbone=backbone)
    after = reloaded.evaluate(val_features, val.labels())
    print(f"loaded:  top1={after['top1']:.4f}")
    assert abs(before["top1"] - after["top1"]) < 1e-9, "a reloaded probe must score identically"

    # The reason the identity is checked at all. Same backbone, same weights,
    # same width -- only the pooling differs, so the head consumes a
    # representation it was never fitted on. Nothing about the shapes says so.
    print("\nloading the same head against mean-pooled features instead of CLS:")
    other = visbench.get_probe("classification", device=args.device)
    other.pooling = "mean"
    try:
        load_probe(path, backbone=backbone, task=other)
    except IncompatibleProbe as error:
        print(f"  refused: {error}")

    mean_features = cache.extract_dataset(backbone, val, keep="pooled", pooling="mean")
    forced = load_probe(path, backbone=backbone, task=other, strict=False)
    wrong = forced.evaluate(mean_features, val.labels())
    print(f"\n  forced through anyway: top1={wrong['top1']:.4f}, against {before['top1']:.4f}")
    print("  It ran. It produced a number. Nothing but the check said it was wrong.")


if __name__ == "__main__":
    main()
