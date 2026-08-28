"""Gradient orientation — a derived target that is a direction, not a magnitude::

    python examples/orientation.py --data /path/to/any/images --limit 600

    <data>/train/images/*.jpg
    <data>/val/images/*.jpg

No target folder: the target is computed from the images, like ``corners.py``.
What differs is *what* is computed — the local orientation structure runs, read
as ``2*theta = atan2(2*Ixy, Ixx - Iyy)`` from the Gaussian-windowed structure
tensor. The angle is defined modulo pi (an edge and its reverse run the same
way), so the target is the unit vector ``(cos 2t, sin 2t)``, single-valued under
that wrap, with its length set to the **coherence**
``(lambda_max - lambda_min) / (lambda_max + lambda_min)``.

**It measures phase, which no other probe here does.** Per-image ``|r|`` with
the ``edge_texture`` target is 0.07 and with ``corner`` 0.08, where ``corner``
and ``edge`` themselves sit at 0.53 — so an orientation score is close to
independent evidence about a backbone. That is exactly the gap a DoG-blob probe
could not fill: its target correlated 0.51 with ``corner``.

**Coherence is a weight, not a mask.** The loss and the metric both weight by
the target's per-pixel length, so a flat isotropic patch contributes ~0 rather
than being dropped by a threshold. On Taskonomy tiny val only 1.4% of pixels
fall below coherence 0.1.

**Quote ``orientation_error``** — the coherence-weighted mean angular error in
degrees, halved into ``[0, 90]`` so 45 is chance. ``d1``/``d2`` are the
fractions within 11.25 and 22.5 degrees.

**No compression, and only one operator setting.** An angle has no heavy tail,
so unlike ``corners.py`` there is no ``--transform`` and no ``--scale``. The one
knob is ``--sigma``, which lands in ``dataset_params`` and splits the
comparability groups on its own — two records both saying "orientation" are not
thereby comparable.
"""

import argparse
import json
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data.derived import DerivedTargetDataset, OrientationResponse


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
        help="Gaussian window of the structure tensor, in pixels — the scale the "
        "operator sees. Dominant orientation moves under 3 degrees between 1.5 and 3.0",
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

    generator = OrientationResponse(sigma=args.sigma)

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
        "orientation",
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
    print("\n  orientation_error is the number to quote: coherence-weighted mean angular")
    print("  error in degrees, halved so 45 is chance. A backbone can be strong here")
    print("  and weak on corners or edges — the targets are near-independent.")
    print(f"\n  train loss {result.probe.train_loss:.4f}")
    print(f"\nrecord appended to {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
