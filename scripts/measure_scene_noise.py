#!/usr/bin/env python
"""Why does `scene_classification` alone fail to reproduce its board? (20d)

    python scripts/measure_scene_noise.py
    python scripts/measure_scene_noise.py --summarise

20c swept every trained board at five seeds. Thirteen reproduced their published
cells; **`scene_classification` did not** -- eleven of its thirteen seed-0 rows
miss, worst **−0.0102** on `resnet50`, *and* carry a different `train_loss`,
which is the signature of a different configuration rather than a moved metric.
Its sweep is held out at `results/controls/scene_classification_seeds.jsonl` and
that board has no separability verdict.

Three causes were ruled out by measurement (see `results/controls/README.md`):
the dataset fingerprints match, two further repeats agree bit for bit, and the
only cache writes since are NAVI's 8,217 frames. A fourth -- the **silicon** --
was left open, and this study is why it is no longer worth an A100 slot:

* five boards were published in the *same* 2026-09-10 batch and all five
  reproduce today on a V100, so the machine underneath cannot be what moved; and
* **no dense board reproduces exactly either.** They sit at 1.7e-07 to 2.9e-05,
  the float32 reduction-order floor. The same tiny disturbance is present
  everywhere; what differs is how far a board amplifies it.

**So the question is not "what changed" but "how much does this board amplify a
change", and that is measurable on hardware we can get.** This injects noise of
a known size into the cached features, refits, and reports the movement --
19b's part 3, asked of a different board.

**The prediction that makes it a test rather than a demonstration.** If
amplification is the story, the backbones that fail to reproduce should be the
ones whose fit is closest to interpolating, and `mae_vitb16` -- the least
interpolating row, `train_top1` 0.916 -- should be one of the two that
reproduce. It is. So the curve is measured for both a sensitive row
(`resnet50`, `train_top1` 0.9998, published-vs-today −0.0102) and a stable one
(`mae_vitb16`, reproduces exactly), and the two should disagree.

**The gate.** This script assembles the dataset itself, which is how a
pre-measurement stops predicting the probe (16a-1's EXIF lesson). So the
unperturbed baseline must reproduce the committed sweep's seed-0 row for the
same backbone, exactly -- today's runs are deterministic. It is checked first
and everything else is void without it.

Nothing here is a corpus record: the runs perturb features, which no flag
expresses and no `ResultRecord` could honestly describe. The numbers land in
`results/controls/scene_noise.json` with the config that produced them, exactly
as `pose_noise.json` does.
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
from visbench.cli.datasets import _folder_split  # noqa: E402
from visbench.results.writer import iter_records  # noqa: E402
from visbench.tasks.high_level.scene_classification import SceneClassificationTask  # noqa: E402
from visbench.utils.seed import set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PLACES365 = Path("/shared/sets/datasets/vision/places365_standard")
CLUSTER_CACHE = Path("/shared/results/common/kargin/visbench_cache")
SWEEP = ROOT / "results" / "controls" / "scene_classification_seeds.jsonl"
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"

#: The board's flags, from `probe_scene_classification` in build_corpus.sh.
#: `--limit 100` pins which files are read, so it is part of the identity of the
#: thing being measured rather than a way to make this cheaper.
LIMIT = 100

#: The probe's own schedule, `CLASSIFICATION_SCHEDULE_DEFAULTS` -- 200 epochs at
#: 1e-2, not the dense probes' 10 at 5e-4. Quoted rather than imported from the
#: CLI table so this script says what it ran; the gate below is what checks it.
EPOCHS, LR, BATCH = 200, 1e-2, 256

#: A thousand-fold range, as 19b used. Absolute, so they are comparable against
#: the ~1e-7 float32 floor every board sits at.
MAGNITUDES = (1e-6, 1e-5, 1e-4, 1e-3)

#: One row that fails to reproduce and one that does. The contrast is the test.
SENSITIVE, STABLE = "resnet50", "mae_vitb16"


def splits():
    """Places365-standard as the board reads it: val scored, train fitted.

    Built by **calling the CLI's own `_folder_split`** rather than constructing
    the datasets here. A copy is what this script first did, and it drifted on
    the first run -- `ImageFolderDataset` takes the split *directory* as its
    root and the split *name* beside it, so a hand-rolled version found no
    images at all. It could as easily have found the wrong ones, which is
    16a-1's lesson: a pre-measurement that assembles its own data stops
    predicting the probe. The gate below is the backstop; this is the fix.
    """
    args = argparse.Namespace(data=PLACES365, limit=LIMIT, dataset=None)
    return _folder_split(args, "train"), _folder_split(args, "val")


def pooled(cache: FeatureCache, backbone, dataset, batch_size: int = 32) -> torch.Tensor:
    resolved = backbone.default_pooling()
    return cache.extract_dataset(
        backbone, dataset, batch_size=batch_size, keep="pooled", pooling=resolved
    )["pooled"]


def score(name: str, train_feats, train, val_feats, val, seed: int, device: str) -> dict:
    """One fit and one evaluation, **in `run()`'s order**: seed, construct, fit.

    `run()` calls `set_seed` and *then* builds the backbone
    (`visbench/runner.py:171-175`), and constructing one draws from the global
    RNG — so a script that builds the backbone first and seeds afterwards
    initialises the head from a different RNG state, with every recorded field
    identical and the number different. That is `CLAUDE.md`'s standing rule, and
    this script's first version broke it: its unperturbed baseline missed the
    sweep's seed-0 row by **6e-3** on `resnet50` and 5e-5 on `mae_vitb16` — the
    same shape as the disagreement being investigated, which is exactly why the
    gate exists and why nothing below it may be read until the gate passes.

    The constructed backbone is discarded: the features are already extracted
    and cached, so it is here only to consume the RNG draws `run()` makes.
    """
    set_seed(seed)
    get_backbone(name, device=device)
    probe = SceneClassificationTask(epochs=EPOCHS, lr=LR, batch_size=BATCH, device=device)
    probe.fit({"pooled": train_feats}, train.labels())
    metrics = probe.evaluate({"pooled": val_feats}, val.labels())
    metrics.update(probe.training_summary() or {})
    return metrics


def reference(name: str) -> dict:
    """What this backbone reads in the committed sweep and in the corpus."""
    sweep = [r for r in iter_records(SWEEP) if r.backbone == name]
    published = [
        r for r in iter_records(CORPUS) if r.task == "scene_classification" and r.backbone == name
    ]
    by_seed = {r.seed: r.metrics["top1"] for r in sweep}
    latest = max(published, key=lambda r: r.timestamp)
    values = list(by_seed.values())
    return {
        "seed_zero": by_seed.get(0),
        "seed_spread": max(values) - min(values) if len(values) > 1 else None,
        "seed_sd": statistics.stdev(values) if len(values) > 1 else None,
        "published": latest.metrics["top1"],
        "published_minus_today": latest.metrics["top1"] - by_seed.get(0, float("nan")),
        "published_train_top1": (latest.training or {}).get("train_top1"),
    }


def curve(args, name: str, train, val) -> dict:
    """One backbone's curve: a known perturbation in, a score movement out."""
    ref = reference(name)
    print(f"\n=== {name} ===")
    print(
        f"  published {ref['published']:.6f}, sweep seed 0 {ref['seed_zero']:.6f}"
        f"  (published - today {ref['published_minus_today']:+.2e})"
    )
    print(
        f"  train_top1 {ref['published_train_top1']:.6f}"
        f" | seed sd {ref['seed_sd']:.5f}, spread {ref['seed_spread']:.5f} over 5 seeds"
    )

    backbone = get_backbone(name, device=args.device)
    cache = FeatureCache(root=Path(args.cache))
    train_feats, val_feats = pooled(cache, backbone, train), pooled(cache, backbone, val)
    scale = float(train_feats.abs().mean())
    print(f"  mean |feature| {scale:.4f}, so the perturbations below are absolute")

    started = time.time()
    baseline = score(name, train_feats, train, val_feats, val, seed=0, device=args.device)
    base = baseline["top1"]
    delta = base - ref["seed_zero"]
    ok = delta == 0.0
    print(
        f"  GATE: unperturbed seed 0 reads {base:.6f} against the sweep's "
        f"{ref['seed_zero']:.6f} (delta {delta:+.2e}) -- "
        f"{'measures the same thing' if ok else 'DOES NOT MATCH, everything below is void'}"
        f"  [{time.time() - started:.0f}s]"
    )

    if args.gate_only:
        return {
            "backbone": name,
            "reference": ref,
            "baseline_top1": base,
            "gate_delta": delta,
            "gate_ok": ok,
            "curve": [],
        }

    rows = []
    for magnitude in MAGNITUDES:
        moves = []
        for draw in range(args.draws):
            generator = torch.Generator().manual_seed(1000 + draw)
            noise = (torch.rand(train_feats.shape, generator=generator) * 2 - 1) * magnitude
            val_noise = (torch.rand(val_feats.shape, generator=generator) * 2 - 1) * magnitude
            got = score(
                name,
                train_feats + noise,
                train,
                val_feats + val_noise,
                val,
                seed=0,
                device=args.device,
            )
            moves.append(got["top1"] - base)
        rows.append(
            {
                "magnitude": magnitude,
                "moves": moves,
                "mean_abs_move": statistics.mean(abs(m) for m in moves),
                "max_abs_move": max(abs(m) for m in moves),
            }
        )
        print(
            f"  +/-{magnitude:.0e}: moves {', '.join(f'{m:+.5f}' for m in moves)}"
            f"  (mean |move| {rows[-1]['mean_abs_move']:.5f})"
        )
    return {
        "backbone": name,
        "reference": ref,
        "mean_abs_feature": scale,
        "baseline_top1": base,
        "gate_delta": delta,
        "gate_ok": ok,
        "curve": rows,
    }


def summarise(path: Path) -> int:
    if not path.is_file():
        print(f"no study at {path}", file=sys.stderr)
        return 1
    study = json.loads(path.read_text())
    print("scene_classification: how far does a known perturbation move the score?\n")
    for row in study["backbones"]:
        ref = row["reference"]
        print(f"{row['backbone']}  (train_top1 {ref['published_train_top1']:.4f})")
        print(
            f"  published - today {ref['published_minus_today']:+.2e}"
            f" | seed sd {ref['seed_sd']:.5f}"
        )
        for entry in row["curve"]:
            print(f"    +/-{entry['magnitude']:.0e}  mean |move| {entry['mean_abs_move']:.5f}")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", default=str(CLUSTER_CACHE))
    parser.add_argument("--draws", type=int, default=3, help="noise draws per magnitude")
    parser.add_argument("--backbones", default=f"{SENSITIVE},{STABLE}")
    parser.add_argument("--out", type=Path, default=ROOT / "results/controls/scene_noise.json")
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="run the unperturbed baseline and stop; 30 s against the curve's 45 minutes",
    )
    args = parser.parse_args(argv)

    if args.summarise:
        return summarise(args.out)

    train, val = splits()
    print(f"places365: {len(train)} train / {len(val)} val at --limit {LIMIT}")
    rows = [curve(args, name, train, val) for name in args.backbones.split(",")]

    if args.gate_only:
        return 0 if all(row["gate_ok"] for row in rows) else 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "probe": "scene_classification",
                "limit": LIMIT,
                "schedule": {"epochs": EPOCHS, "lr": LR, "batch_size": BATCH},
                "magnitudes": list(MAGNITUDES),
                "draws": args.draws,
                "backbones": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")
    return 0 if all(row["gate_ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
