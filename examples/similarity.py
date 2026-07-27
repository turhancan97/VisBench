"""Mid-level image similarity: does the backbone agree with human judgement?

Two-alternative forced choice. Each item is a reference and two candidates, and
a human said which candidate looks more like the reference. The probe compares
two cosine similarities in frozen feature space and picks a side::

    prefer_right = cos(ref, right) > cos(ref, left)

**Nothing is trained.** Like retrieval and correspondence, this is zero-shot —
there is no head and no training split involved. The protocol follows Chen,
Marks & Cheng (arXiv:2411.17474); note their README describes "training a
similarity estimator" while their code trains nothing, and the code is what is
followed here.

Expects the NIGHTS release (Fu et al., *DreamSim*, NeurIPS 2023)::

    <data>/data.csv
    <data>/ref/000/002.png
    <data>/distort/000/002_0.png

Run it::

    python examples/similarity.py --data /path/to/nights
    python examples/similarity.py --data ... --split test_no_imagenet
    python examples/similarity.py --data ... --backbone clip_vitb16

**This is not high-level retrieval.** The ground truth is perceptual — layout,
pose, structure — not category membership, which is exactly why the two are
separate tasks. A backbone can be excellent at one and ordinary at the other.

**Watch the two test subsets.** `--split test_imagenet` and `test_no_imagenet`
partition the test set by whether the reference came from ImageNet. A backbone
pretrained on ImageNet has seen those images, so a gap between the two is a
contamination signal rather than a similarity result. Run both before quoting a
single number.

**`min_votes` is part of the protocol, not a detail.** Triplets are kept only
where that many human votes agreed. Lowering it admits cases humans found
ambiguous, which drags every score down without saying anything about the
models. It is recorded in the result.

The first run extracts features; later runs read the cache and the backbone
never executes. Features are pooled (CLS on a ViT, mean on a CNN), so they are
small and shared with `examples/retrieve.py` when the images and pooling match.

This is an example, not the CLI, which is still deferred until the dense-task
Python API has settled.
"""

import argparse
import json
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data.triplet import NIGHTS_MIN_VOTES, TwoAFCDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="NIGHTS root")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "test_imagenet", "test_no_imagenet"],
        help="test_imagenet / test_no_imagenet split the test set by whether the "
        "reference image came from ImageNet",
    )
    parser.add_argument("--backbone", default="dinov2_vits14", help="see visbench.list_backbones()")
    parser.add_argument(
        "--pooling",
        default=None,
        help="cls | mean; default is the backbone's own (CLS for ViTs, mean for CNNs)",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=NIGHTS_MIN_VOTES,
        help="keep triplets with at least this many agreeing human votes",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    parser.add_argument("--cache", type=Path, default=Path(".visbench_cache"))
    parser.add_argument("--results", type=Path, default=Path("results/visbench.jsonl"))
    parser.add_argument(
        "--limit", type=int, default=None, help="use at most N triplets (quick run)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = TwoAFCDataset(
        args.data,
        split=args.split,
        min_votes=args.min_votes,
        max_triplets=args.limit,
    )
    triplets = dataset.labels()
    print(f"{len(triplets)} triplets over {len(dataset)} images, split={args.split}")

    # Chance is 50% by construction, but only if the votes are balanced; a
    # skewed split would let "always answer right" look like a result.
    right = triplets[:, 3].float().mean().item()
    print(f"  humans chose 'right' {right:.1%} of the time")
    print(f"  a probe answering 'right' always would score {max(right, 1 - right):.1%}")

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    probe = visbench.get_probe("similarity", min_votes=args.min_votes)
    if args.pooling:
        probe.pooling = args.pooling

    print(f"\nextracting pooled features with {backbone.name}...")
    result = visbench.run(
        backbone,
        probe,
        dataset,
        cache=FeatureCache(root=args.cache),
        results=args.results,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    print("\nagreement with human judgement:")
    for name, value in result.metrics.items():
        print(f"  {name:>10s}  {value:.4f}")
    print("\n  accuracy is the number to quote; f1/precision/recall treat 'right' as")
    print("  the positive class, which is a labelling convention, not a property.")
    if result.metrics.get("tie_rate", 0) > 0:
        print(f"\n  {result.metrics['tie_rate']:.1%} of triplets tied and were resolved")
        print("  arbitrarily — a large value means the features barely discriminate.")

    print(f"\nrecord appended to {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
