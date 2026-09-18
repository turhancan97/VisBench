#!/usr/bin/env python
"""How much does a pose number move, and what moves it? (19b)

    python scripts/measure_pose_noise.py --part all

`relative_pose` shipped with a reproducibility claim resting on **one**
comparison: the `mae_vitb16` cell scored 22.6984 on the cluster and 21.7695
from a second extraction of the same frames, so the board is quoted to whole
degrees and five adjacent pairs are called ties. One observation is thin for a
rule that decides which rows a reader may separate, and the standing
instruction in `CLAUDE.md` is to repeat a measurement before concluding from
it -- `duration_seconds` is the example that cost three files and a merged PR.

This measures the three things that claim actually rests on.

**Part 1, the observation, at n=2 rather than n=1.** Score `mae_vitb16` and
`clip_vitb16` from each of the two caches that exist on this machine, same seed
and same pairs, and report the movement. Only those two are comparable: the
local cache holds no entry for `dino_vitb16` or `sam_vitb16` under the pooling
the probe resolves, so a third and fourth row would need an hour of decode each
and would still be the same experiment.

**Part 2, the mechanism, which was inferred and is now measured.** The write-up
attributed the feature difference to extraction batch size -- the proof run used
64 where the corpus script defaults to 32 -- on the evidence that a *later*
probe whose two caches were built at the same batch size came out bit-identical.
That is consistent with the hypothesis and does not test it. This extracts the
same frames at both batch sizes into fresh roots and measures the difference
directly.

**Part 3, the amplification, which is the number the rule should rest on.**
Inject noise of a known size into the cached features, refit, and see what the
score does. That asks the question the cross-cache comparison only gestures at:
given a perturbation of the size two extractions produce, how far does this
probe's number move? Several magnitudes and several draws, so the answer is a
curve with a spread rather than a single difference -- and the seed-to-seed
spread of the unperturbed probe is printed beside it, because a movement
smaller than that is not a finding.

Nothing here is a corpus record. The runs perturb features, which no flag
expresses and no `ResultRecord` could honestly describe; the numbers land in
`results/controls/pose_noise.json` with the config that produced them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visbench import get_backbone  # noqa: E402
from visbench.cache import FeatureCache  # noqa: E402
from visbench.data import NaviPoseDataset  # noqa: E402
from visbench.tasks.mid_level.pose import RelativePoseTask  # noqa: E402
from visbench.utils import set_seed  # noqa: E402

NAVI = Path("/shared/sets/datasets/vision/probing_3D/navi_v1")
LOCAL_CACHE = Path(".visbench_cache")
CLUSTER_CACHE = Path("/shared/results/common/kargin/visbench_cache")

#: The two backbones with comparable entries in both cache roots.
COMPARABLE = ("mae_vitb16", "clip_vitb16")

#: Perturbation sizes, spanning the measured cross-cache difference (~1e-5).
MAGNITUDES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)


def splits(image_size: int = 224) -> tuple[NaviPoseDataset, NaviPoseDataset]:
    """The pinned protocol: eight partners per training anchor, val at one."""
    shared = {"root": NAVI, "image_size": image_size}
    return (
        NaviPoseDataset(split="train", partners=8, **shared),
        NaviPoseDataset(split="val", **shared),
    )


def pooled(cache: FeatureCache, backbone, dataset, batch_size: int = 32) -> torch.Tensor:
    resolved = backbone.default_pooling()
    return cache.extract_dataset(
        backbone, dataset, batch_size=batch_size, keep="pooled", pooling=resolved
    )["pooled"]


def score(train_feats, train, val_feats, val, seed: int, device: str) -> dict:
    """One fit and one evaluation, with the caller's seed governing both."""
    set_seed(seed)
    probe = RelativePoseTask(device=device)
    probe.fit({"pooled": train_feats}, train.labels())
    metrics = probe.evaluate({"pooled": val_feats}, val.labels())
    metrics["train_loss"] = probe.train_loss
    return metrics


def part_one(args, train, val) -> list[dict]:
    """The observation, at n=2: the same cell from two caches."""
    print("\n=== part 1: the same cell scored from two caches ===")
    rows = []
    for name in args.backbones.split(","):
        name = name.strip()
        if not name:
            continue
        backbone = get_backbone(name, device=args.device)
        by_root = {}
        for label, root in (("local", LOCAL_CACHE), ("cluster", CLUSTER_CACHE)):
            cache = FeatureCache(root=root)
            t0 = time.time()
            tr, va = pooled(cache, backbone, train), pooled(cache, backbone, val)
            got = score(tr, train, va, val, seed=args.seed, device=args.device)
            by_root[label] = got
            print(
                f"  {name:14s} {label:8s} rot {got['rotation_error_deg']:7.4f} deg"
                f"  train_loss {got['train_loss']:.5f}  ({time.time() - t0:.0f}s)"
            )
        delta = by_root["local"]["rotation_error_deg"] - by_root["cluster"]["rotation_error_deg"]
        print(f"  {name:14s} {'delta':8s} {delta:+7.4f} deg")
        rows.append(
            {
                "backbone": name,
                "local": by_root["local"],
                "cluster": by_root["cluster"],
                "delta_deg": delta,
            }
        )
    return rows


def part_two(args, train) -> dict:
    """The mechanism: does extraction batch size change the features?"""
    print("\n=== part 2: the same frames extracted at two batch sizes ===")
    subset = NaviPoseDataset(root=NAVI, split="val", stride=args.stride, image_size=224)
    backbone = get_backbone(args.mechanism_backbone, device=args.device)
    scratch = Path(args.scratch)
    feats = {}
    for batch_size in (32, 64):
        root = scratch / f"batch{batch_size}"
        t0 = time.time()
        feats[batch_size] = pooled(FeatureCache(root=root), backbone, subset, batch_size=batch_size)
        print(f"  batch {batch_size:3d}: {len(subset)} frames in {time.time() - t0:.0f}s")
    a, b = feats[32], feats[64]
    per_frame = (a - b).abs().amax(dim=1)
    identical = int((per_frame == 0).sum())
    summary = {
        "frames": len(subset),
        "backbone": args.mechanism_backbone,
        "max_abs_difference": float(per_frame.max()),
        "mean_abs_difference": float((a - b).abs().mean()),
        "identical_frames": identical,
        "identical_share": identical / len(subset),
    }
    print(
        f"  max |difference| {summary['max_abs_difference']:.2e}"
        f"   mean {summary['mean_abs_difference']:.2e}"
    )
    print(f"  bit-identical frames: {identical}/{len(subset)} ({summary['identical_share']:.0%})")
    return summary


def amplification(args, name, train, val) -> dict:
    """One backbone's curve: a known perturbation in, a score movement out."""
    backbone = get_backbone(name, device=args.device)
    cache = FeatureCache(root=CLUSTER_CACHE)
    train_feats, val_feats = pooled(cache, backbone, train), pooled(cache, backbone, val)
    print(f"\n  {name}: mean |feature| {float(train_feats.abs().mean()):.4f}, so the")
    print("  perturbations below are absolute, to compare against the ~1e-5 measured above")

    baseline = [
        score(train_feats, train, val_feats, val, seed=s, device=args.device)["rotation_error_deg"]
        for s in range(args.seeds)
    ]
    spread = max(baseline) - min(baseline)
    print(
        f"  seed-to-seed spread, unperturbed: {spread:.4f} deg over {args.seeds} seeds"
        f" ({min(baseline):.4f} to {max(baseline):.4f})"
    )

    rows = []
    for magnitude in MAGNITUDES:
        if magnitude == 0.0:
            continue
        moves = []
        for draw in range(args.draws):
            generator = torch.Generator().manual_seed(1000 + draw)
            noise = (torch.rand(train_feats.shape, generator=generator) * 2 - 1) * magnitude
            val_noise = (torch.rand(val_feats.shape, generator=generator) * 2 - 1) * magnitude
            got = score(
                train_feats + noise,
                train,
                val_feats + val_noise,
                val,
                seed=args.seed,
                device=args.device,
            )
            moves.append(got["rotation_error_deg"] - baseline[args.seed])
        rows.append(
            {
                "magnitude": magnitude,
                "moves_deg": moves,
                "mean_abs_move": statistics.mean(abs(m) for m in moves),
                "max_abs_move": max(abs(m) for m in moves),
            }
        )
        print(
            f"  +/-{magnitude:.0e}: moves {', '.join(f'{m:+.4f}' for m in moves)}"
            f"   mean |move| {rows[-1]['mean_abs_move']:.4f} deg"
        )
    return {
        "backbone": name,
        "baseline_seeds": baseline,
        "seed_spread": spread,
        "curve": rows,
    }


def part_three(args, train, val) -> list[dict]:
    """The amplification, over more than one backbone.

    One row would repeat the mistake this study is correcting, and the DPT
    control's n=2 claims are the standing example of what that costs.
    """
    print("\n=== part 3: a known perturbation in, a score movement out ===")
    return [
        amplification(args, name.strip(), train, val)
        for name in args.noise_backbones.split(",")
        if name.strip()
    ]


def summarise(path: Path) -> int:
    """Reprint a finished study, so a quoted number comes from the file."""
    if not path.is_file():
        print(f"no study at {path} -- run --part all first", file=sys.stderr)
        return 1
    study = json.loads(path.read_text())

    print("=== 1. the same cell scored from two caches ===")
    for row in study.get("two_caches", []):
        print(
            f"  {row['backbone']:14s} local {row['local']['rotation_error_deg']:8.4f}"
            f"  cluster {row['cluster']['rotation_error_deg']:8.4f}"
            f"  delta {row['delta_deg']:+7.4f}"
        )

    batch = study.get("batch_size")
    if batch:
        print("\n=== 2. batch 32 against batch 64, same frames, same GPU ===")
        print(
            f"  {batch['backbone']}, {batch['frames']} frames:"
            f" max |difference| {batch['max_abs_difference']:.2e},"
            f" mean {batch['mean_abs_difference']:.2e}"
        )
        print(
            f"  bit-identical: {batch['identical_frames']}/{batch['frames']}"
            f" ({batch['identical_share']:.0%})"
        )

    print("\n=== 3. a known perturbation in, a score movement out ===")
    for curve in study.get("amplification", []):
        base = curve["baseline_seeds"]
        low, high = min(base), max(base)
        print(
            f"\n  {curve['backbone']}: unperturbed {low:.4f} to {high:.4f}"
            f" over {len(base)} seeds (range {curve['seed_spread']:.4f})"
        )
        for row in curve["curve"]:
            scores = [base[0] + move for move in row["moves_deg"]]
            inside = sum(low <= score <= high for score in scores)
            print(
                f"    +/-{row['magnitude']:.0e}: mean |move| {row['mean_abs_move']:.4f} deg,"
                f" {inside}/{len(scores)} of the perturbed scores inside the seed range"
            )
        every = [move for row in curve["curve"] for move in row["moves_deg"]]
        print(
            f"    over every magnitude and draw: mean |move| "
            f"{statistics.mean(abs(move) for move in every):.4f},"
            f" max {max(abs(move) for move in every):.4f}, n={len(every)}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", default="all", choices=("all", "1", "2", "3"))
    parser.add_argument(
        "--summarise",
        action="store_true",
        help="reprint the committed study instead of measuring it again",
    )
    parser.add_argument("--backbones", default=",".join(COMPARABLE))
    parser.add_argument("--noise-backbones", default=",".join(COMPARABLE))
    parser.add_argument("--mechanism-backbone", default="mae_vitb16")
    parser.add_argument("--stride", type=int, default=8, help="part 2 subsamples the val anchors")
    parser.add_argument("--seeds", type=int, default=3, help="baseline seeds in part 3")
    parser.add_argument("--draws", type=int, default=3, help="noise draws per magnitude")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scratch", default="/tmp/visbench-pose-noise")
    parser.add_argument("--out", type=Path, default=Path("results/controls/pose_noise.json"))
    args = parser.parse_args(argv)

    if args.summarise:
        return summarise(args.out)

    if not NAVI.is_dir():
        print(f"NAVI not found at {NAVI}", file=sys.stderr)
        return 1

    t0 = time.time()
    train, val = splits()
    print(
        f"{len(train.labels().pose)} train / {len(val.labels().pose)} val pairs"
        f" ({time.time() - t0:.0f}s)"
    )

    study: dict = {"pairs": {"train": len(train.labels().pose), "val": len(val.labels().pose)}}
    if args.part in ("all", "1"):
        study["two_caches"] = part_one(args, train, val)
    if args.part in ("all", "2"):
        study["batch_size"] = part_two(args, train)
    if args.part in ("all", "3"):
        study["amplification"] = part_three(args, train, val)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(study, indent=2) + "\n")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
