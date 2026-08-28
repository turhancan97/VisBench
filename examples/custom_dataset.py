"""Probing a dataset VisBench never shipped a loader for.

Two tiers already work without any of this: a folder layout needs no code, and
anything else is a ``BaseDataset`` subclass with two methods. What the bridges
add is the short path when the data already lives in a ``torch.utils.data``
dataset or a Hugging Face ``datasets.Dataset``::

    python examples/custom_dataset.py                 # torchvision.FakeData, no download
    python examples/custom_dataset.py --hf cifar10    # needs: pip install visbench[datasets]

**The one thing a bridge must get right is ``cache_identity``.** Return ``None``
there and every run silently re-decodes every image, forever. Both bridges
derive a real per-item token from the fact that the wrapped dataset is immutable
in index order — an HF ``_fingerprint`` plus row index, or a digest of the
torchvision repr plus index (or the file path, for the ``ImageFolder`` family).
This script prints the identity of item 0 so you can see it is stable and not
``None``.
"""

import argparse
from pathlib import Path

from torchvision.datasets import FakeData

import visbench
from visbench.cache import FeatureCache
from visbench.data import BaseDataset, HuggingFaceDataset, TorchvisionDataset


def torchvision_splits(n: int) -> tuple[TorchvisionDataset, TorchvisionDataset]:
    """``torchvision.datasets.FakeData`` — random images, three classes, no download.

    A deterministic ``random_offset`` per split so train and val differ, and
    ``transform=None`` so the dataset hands back PIL images the bridge can pass
    straight through.
    """
    train = FakeData(size=n, image_size=(3, 96, 96), num_classes=3, random_offset=0)
    val = FakeData(size=n, image_size=(3, 96, 96), num_classes=3, random_offset=10_000)
    return (
        TorchvisionDataset(train, split="train", name="fakedata"),
        TorchvisionDataset(val, split="val", name="fakedata"),
    )


def hf_splits(name: str) -> tuple[HuggingFaceDataset, HuggingFaceDataset]:
    from datasets import load_dataset

    train = load_dataset(name, split="train[:200]")
    val = load_dataset(name, split="test[:200]")
    return HuggingFaceDataset(train, split="train"), HuggingFaceDataset(val, split="val")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hf", default=None, help="a HuggingFace dataset name (e.g. cifar10)")
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--n", type=int, default=120, help="items per split (torchvision path)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train: BaseDataset
    val: BaseDataset
    if args.hf:
        train, val = hf_splits(args.hf)
    else:
        train, val = torchvision_splits(args.n)

    print(f"{train.name}: {len(train)} train, {len(val)} val")
    print(f"  cache_identity(0): {val.cache_identity(0)!r}  (must not be None)")
    print(f"  fingerprint: {val.fingerprint()}")
    print(f"  labels[:8]: {val.labels()[:8]}")

    probe = visbench.get_probe("classification", epochs=50, lr=1e-2, device=args.device)
    result = visbench.run(
        args.backbone,
        probe,
        val,
        train_dataset=train,
        cache=FeatureCache(Path(".visbench_cache")),
        device=args.device,
    )
    print("\nmetrics:")
    for name, value in result.metrics.items():
        print(f"  {name:>12s}  {value:.4f}")
    print(f"\n  dataset_params in the record: {result.record.dataset_params}")
    print("\n  Run it twice: the second run reads features from the cache and the")
    print("  backbone never executes — which is exactly what cache_identity buys.")


if __name__ == "__main__":
    main()
