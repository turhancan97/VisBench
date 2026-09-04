#!/usr/bin/env python
"""Run the relative-depth control on one backbone, appending one record.

    scripts/run_relative_depth.py --backbone dinov2_vits14 --data <nyuv2_new>

Driven by ``scripts/build_relative_depth_control.sh``, which holds the flag set
so it cannot drift. That script's docstring says what the control is and why the
task it runs is deliberately **not a registered probe**.

Why this exists rather than `visbench run relative_depth`: the CLI resolves a
probe name through the registry, and this task is absent from it on purpose --
so a registered probe's obligations (a corpus board, a CLI row, a
``TARGET_STYLES`` entry, a committed gallery figure) cannot be acquired by
accident by a readout that has not earned a board. Constructing the task and
calling :func:`visbench.run` is the same path ``examples/custom_backbone.py``
documents for an unregistered backbone.

**The backbone is passed by NAME, not as an object.** ``run()`` seeds before it
constructs, so handing it a pre-built backbone would fit the head from a
different RNG state with every recorded field identical -- the bug that shipped
in ``--push-to``. The object comes back off ``RunResult.backbone`` if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import visbench
from visbench.cache import FeatureCache
from visbench.data.dense import DenseFolderDataset, load_depth_map
from visbench.tasks.mid_level.relative_depth import RelativeDepthTask

#: Equal to ``probe_depth``'s in ``scripts/build_corpus.sh``, and it must stay
#: that way: only the readout may differ, or the control cannot say which of the
#: two moved a rank. ``target_scale`` is 1.0 because these are ``.npy`` files
#: already in metres -- 1000 would divide a 3-metre reading to 3 millimetres.
TARGET_SCALE = 1.0
MAX_DEPTH = 10.0


def split(root: Path, name: str) -> DenseFolderDataset:
    return DenseFolderDataset(
        root / name,
        image_dir="images",
        target_dir="depths",
        target_loader=load_depth_map,
        target_scale=TARGET_SCALE,
        max_target=MAX_DEPTH,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--data", type=Path, required=True, help="the nyuv2_new root")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--split", default="test", help="evaluated split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--pairs-per-image", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evaluate = split(args.data, args.split)
    train = split(args.data, args.train_split)
    print(f"relative_depth on {args.backbone}")
    print(f"  scoring:  {len(evaluate)} items from {args.data}")
    print(f"  training: {len(train)} items")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    # `results=` rather than a ResultWriter here: run() writes the record it
    # built, so there is no second place a field could be filled in by hand.
    result = visbench.run(
        args.backbone,
        RelativeDepthTask(pairs_per_image=args.pairs_per_image),
        evaluate,
        train_dataset=train,
        seed=args.seed,
        results=args.results,
        **({"cache": FeatureCache(root=args.cache)} if args.cache else {}),
    )
    for name in sorted(result.record.metrics):
        print(f"  {name:26s} {result.record.metrics[name]:.4f}")
    print(f"\nappended to {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
