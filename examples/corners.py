"""Corner detection — a probe that brings its own target::

    python examples/corners.py --data /path/to/any/images --limit 600

    <data>/train/images/*.jpg
    <data>/val/images/*.jpg

That is the whole layout. There is **no target folder**, because the target is
computed from the images: Shi-Tomasi cornerness, the smaller eigenvalue of the
Gaussian-windowed structure tensor, compressed with ``log1p``. Any folder of
photographs will do, which makes this the one dense probe in VisBench that needs
no download at all.

**The generator is part of the protocol, and every flag here changes the
number.** A stored target is identified by the dataset it came from; a derived
one is identified only by the code that computed it. So ``--sigma``,
``--transform`` and ``--scale`` all land in ``dataset_params``, which puts two
settings of the same operator into two comparability groups automatically. Two
records both saying "corners" are not thereby comparable — check the fields.

**Not Harris's** ``R``. Harris is signed, so using it as a magnitude needs a
clip (which throws the edge information away and leaves the sparsest target
tried) or an absolute value (which conflates corners with edges). Shi-Tomasi's
lambda_min is non-negative by construction and carries no ``k`` parameter, which
removes one of the free choices that makes "Harris corners" a family rather than
a definition.

**``log1p`` is not cosmetic.** The raw response holds 27% of its mass in its
strongest 1% of pixels — closer to ``edge_occlusion``'s 0.46, which scored 0.088
and could not separate two backbones, than to the ~0.10 of the probes that work.
At the default scale the compressed target lands at 0.089, and its frame mean at
0.593, which is also what an L1 loss needs: ``sign()`` gradients do not shrink
to match a small target.

**Quote ``corner_correlation``.** As with every magnitude probe here, RMSE
rewards a probe that predicts the split's mean everywhere; correlation scores
that probe 0 and asks only whether the representation knows *where* the corners
are.

**This target overlaps with the edge target — 0.52 per-image correlation on
Taskonomy frames, against 0.147 between the two Taskonomy probes.** A corner
score and an edge score are therefore not independent evidence about a backbone.
They do rank differently, which is why the probe exists: measured over six
backbones, CLIP-B/16 comes first on edges and third on corners. See
``visbench/tasks/low_level/README.md``.
"""

import argparse
import json
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data.derived import DerivedTargetDataset, ShiTomasiResponse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="root holding <split>/images/")
    parser.add_argument("--backbone", default="dinov2_vits14", help="see visbench.list_backbones()")
    parser.add_argument("--head", default="linear", help="linear | dpt; see visbench.heads")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="backbone depths, shallowest first; required by --head dpt",
    )
    parser.add_argument("--image-dir", default="images", help="image folder inside a split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--split", default="val", help="the split that is scored")
    parser.add_argument("--image-size", type=int, default=224, help="multiple of the patch size")

    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian window of the structure tensor, in pixels. This is the scale the "
        "operator sees, not a smoothing nicety: 1.0 and 2.0 give different targets",
    )
    parser.add_argument(
        "--transform",
        default="log1p",
        choices=("log1p", "none"),
        help="compression of the raw response. 'none' is offered so the concentration "
        "argument above can be checked rather than taken on trust",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1e4,
        help="multiplier applied before the compression; sets both the tail and the "
        "target's magnitude",
    )

    parser.add_argument("--hidden-dim", type=int, default=512, help="DPT width")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--finetune-blocks",
        type=int,
        default=0,
        help="unfreeze this many trailing backbone blocks (v0.3). Default 0, a frozen "
        "probe -- which is what every published VisBench number is",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="learning rate for the unfrozen blocks; default --lr / 100",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    parser.add_argument("--cache", type=Path, default=Path(".visbench_cache"))
    parser.add_argument("--results", type=Path, default=Path("results/visbench.jsonl"))
    parser.add_argument("--limit", type=int, default=600, help="images per split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generator = ShiTomasiResponse(sigma=args.sigma, transform=args.transform, scale=args.scale)

    def load(split: str) -> DerivedTargetDataset:
        return DerivedTargetDataset(
            root=args.data / split,
            split=split,
            image_dir=args.image_dir,
            image_size=args.image_size,
            generator=generator,
            max_images=args.limit,
        )

    train, test = load(args.train_split), load(args.split)
    print(f"{len(train)} training images, {len(test)} scored")
    print(f"target: {generator.describe()}")

    probe = visbench.get_probe(
        "corner",
        head=args.head,
        layers=args.layers,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        finetune_blocks=args.finetune_blocks,
        backbone_lr=args.backbone_lr,
    )
    cache = FeatureCache(args.cache)

    result = visbench.run(
        args.backbone,
        probe,
        test,
        train_dataset=train,
        cache=cache,
        results=args.results,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    print("\nmetrics:")
    for name, value in result.metrics.items():
        print(f"  {name:>20s}  {value:.4f}")
    print("\n  corner_correlation is the number to quote: it is invariant to scale and")
    print("  offset, so a probe predicting the split's mean everywhere scores 0 on it")
    print("  while still achieving a small rmse. Report both, or say which.")
    print("\n  This number is not independent of an edge number on the same frames —")
    print("  the two targets correlate at 0.52. They do rank backbones differently.")
    print(f"\n  train loss {result.probe.train_loss:.4f}")
    print(f"\nrecord appended to {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
