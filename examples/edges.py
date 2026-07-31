"""Edge detection — does this representation still know where the edges are.

The first **low-level** probe in VisBench, on Taskonomy's ``edge_texture``
maps::

    python examples/edges.py --data /path/to/taskonomy --limit 600

    <data>/rgb/allensville/point_0_view_0_domain_rgb.png
    <data>/edge_texture/allensville/point_0_view_0_domain_edge_texture.png
    <data>/splits/tiny_train.csv   (columns: building, point, view)

**The splits are disjoint by building**, not by frame — 25 buildings train, 4
validate, 5 test — so a val number is measured in rooms the probe has never
seen. That is stronger than a random frame split over shared rooms, and it is
why the official lists are used rather than the directory listing.

**Quote ``edge_correlation``, not ``rmse``.** Edge magnitude is concentrated
near zero: a frame's mean is about 0.011 of the 16-bit container's range while
its peak is around 0.13. A probe that ignores its input and predicts that
constant everywhere therefore gets a *small* RMSE while having learned nothing.
Pearson correlation is invariant to scale and offset, so it asks only whether
the representation knows **where** the edges are, and scores that constant probe
at 0. ``rmse`` and ``mae`` are printed alongside because correlation is blind to
the opposite failure — right shape, wrong magnitude.

**Nothing is masked.** Depth has holes and normals have zero-length vectors,
both meaning "no ground truth". An edge map has neither: 0 means *no edge*, a
real reading covering most of most frames. Masking it away would score the probe
only where an edge already is, which is the one place the answer is easy.

**This protocol is not BSDS500's.** BSDS is the canonical edge benchmark, but
its ODS/OIS/AP matches predicted edge pixels to several annotators' by bipartite
correspondence after non-maximum suppression, swept over thresholds — not a
per-pixel metric. Borrowing a protocol is only worth it if borrowed exactly, so
records say ``protocol: "visbench_edge_regression"`` and these numbers must not
be put in a table beside published BSDS ones.

**Ten epochs is probe3d's budget**, borrowed for its optimiser schedule alone so
that an edge number sits alongside a backbone's depth and segmentation numbers
under one training budget. On a few hundred frames it underfits; ``train_loss``
is what separates "this representation lacks the structure" from "this probe did
not converge".
"""

import argparse
import json
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data import TaskonomyDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="Taskonomy root")
    parser.add_argument("--backbone", default="dinov2_vits14", help="see visbench.list_backbones()")
    parser.add_argument("--head", default="linear", help="linear | dpt; see visbench.heads")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="backbone depths, shallowest first; required by --head dpt",
    )
    parser.add_argument("--domain", default="edge_texture", help="Taskonomy target domain")
    parser.add_argument(
        "--partition",
        default="tiny",
        help="download size naming splits/<partition>_<split>.csv (tiny, medium, fullplus)",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--split", default="val", help="the split that is scored")
    parser.add_argument("--image-size", type=int, default=224, help="multiple of the patch size")
    parser.add_argument("--hidden-dim", type=int, default=512, help="DPT width")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--finetune-blocks",
        type=int,
        default=0,
        help="unfreeze this many trailing backbone blocks and train them with the head "
        "(v0.3). Default 0, a frozen probe -- which is what every published VisBench "
        "number is. The two are not comparable, and the record says which is which",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=600,
        help="frames per split. Defaults to 600 rather than None: the tiny train list is "
        "272,296 rows, so an unlimited run is a very different proposition from one on "
        "the other examples' datasets",
    )
    return parser.parse_args()


def load_split(args: argparse.Namespace, split: str) -> TaskonomyDataset:
    """One split, from the official building-disjoint list.

    ``--limit`` goes to the constructor rather than to ``subset()`` afterwards,
    so 272k paths are never built to be discarded. The rows are already
    interleaved across buildings, so a prefix is not one room.
    """
    return TaskonomyDataset(
        args.data,
        domain=args.domain,
        split=split,
        partition=args.partition,
        image_size=args.image_size,
        max_images=args.limit,
    )


def main() -> None:
    args = parse_args()

    train = load_split(args, args.train_split)
    test = load_split(args, args.split)
    print(f"{len(train)} training frames, {len(test)} scored, domain {args.domain}")

    probe = visbench.get_probe(
        "edge",
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
        print(f"  {name:>18s}  {value:.4f}")
    print("\n  edge_correlation is the number to quote: it is invariant to scale and")
    print("  offset, so a probe predicting the split's mean everywhere scores 0 on it")
    print("  while still achieving a small rmse. Report both, or say which.")
    # Not a result: it separates "this representation lacks the structure" from
    # "this probe did not converge", which the metrics alone cannot.
    print(f"\n  train loss {result.probe.train_loss:.4f}")
    if result.record.finetune:
        unfrozen = result.record.finetune
        print(
            f"  fine-tuned {unfrozen['blocks']} block(s), "
            f"{unfrozen['trainable_params'] / 1e6:.2f}M backbone parameters trained."
        )
        print("  Compare only against another fine-tuned run, never a frozen one.")
    print(f"\nrecord appended to {args.results}")
    print(json.dumps(result.record.to_dict(), indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
