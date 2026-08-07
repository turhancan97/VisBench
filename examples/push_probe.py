"""Share a trained probe through the Hugging Face Hub — push, card, and pull.

    # train a probe and show exactly what would be uploaded, without uploading
    python examples/push_probe.py --data /path/to/imagenette2 --repo-id you/imagenette-probe

    # actually upload it (private by default), once the dry run looks right
    python examples/push_probe.py --data /path/to/imagenette2 --repo-id you/imagenette-probe --push

    # fetch someone's probe and load it against your own backbone
    python examples/push_probe.py --pull you/imagenette-probe --backbone dinov2_vits14

`examples/save_probe.py` covers the local half — what an artifact holds and what
loading one against the wrong features costs. This is the transport half, and
the thing worth seeing here is how little it adds: `push_probe` is `save_probe`
plus an upload, and `load_probe_from_hub` is `load_probe` plus a download. The
identity checks and `weights_only=True` are in the local functions, so a
downloaded probe cannot skip them.

**It does not push unless you pass `--push`.** Publishing under someone's
account is not something an example does as a side effect, and a push is not
reversible the way a local write is: a repository that was public for a minute
may already have been fetched. Needs `pip install visbench[hub]`.
"""

import argparse
from pathlib import Path

import torch

import visbench
from visbench.cache import FeatureCache
from visbench.data import ImageFolderDataset
from visbench.hub import probe_card, probe_metadata, save_probe
from visbench.utils import set_seed


def _train(args: argparse.Namespace) -> tuple[object, object, dict[str, float]]:
    """Fit a classification probe, and return it with the scores for its card."""
    set_seed(0)

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    cache = FeatureCache(root=args.cache)

    root = Path(args.data)
    train = ImageFolderDataset(root / "train", split="train").balanced_subset(args.limit)
    val = ImageFolderDataset(root / "val", split="val").balanced_subset(args.limit)

    probe = visbench.get_probe("classification", device=args.device)
    extract = dict(keep="pooled", pooling=probe.pooling)

    probe.fit(cache.extract_dataset(backbone, train, **extract), train.labels())
    metrics = probe.evaluate(cache.extract_dataset(backbone, val, **extract), val.labels())
    print(f"trained: top1={metrics['top1']:.4f}")
    return probe, backbone, metrics


def _push(args: argparse.Namespace) -> None:
    probe, backbone, metrics = _train(args)

    # The card and the artifact come from one source -- probe_metadata -- so the
    # page a visitor reads cannot disagree with the file they download.
    meta = probe_metadata(probe, backbone)
    print("\nthe identity that travels with the weights:")
    for field in ("backbone", "backbone_key", "task", "pooling", "feature_mode", "layers"):
        print(f"  {field:<15} {meta[field]}")
    print("\n  The first four are checked on load. A head fitted on these features")
    print("  and loaded against any others is refused, not silently scored.")

    if not args.push:
        staged = Path(args.out)
        save_probe(probe, staged, backbone=backbone, notes="examples/push_probe.py")
        size = staged.stat().st_size / 1024
        print(f"\nDRY RUN — nothing was uploaded. Wrote {staged} ({size:.1f} KB) instead.")
        print(f"  would create : https://huggingface.co/{args.repo_id}")
        print(f"  visibility   : {'PUBLIC' if args.public else 'private'}")
        print("  files        : probe.pt, README.md")
        print("\nthe card that would go beside the weights:")
        print("=" * 70)
        print(probe_card(probe, backbone, args.repo_id, metrics=metrics))
        print("=" * 70)
        print("Re-run with --push to upload. You will need a token: either")
        print("`huggingface-cli login`, HF_TOKEN in the environment, or --token.")
        return

    from visbench.hub import push_probe

    if args.public:
        # private=True is the default in push_probe for a reason; making it
        # public should be a sentence someone typed, not a flag they inherited.
        print("\n--public: this repository will be readable by anyone.")

    url = push_probe(
        probe,
        args.repo_id,
        backbone=backbone,
        metrics=metrics,
        private=not args.public,
        token=args.token,
    )
    print(f"\npushed: {url}")
    print("  The push refuses an unfitted or zero-shot probe *before* creating")
    print("  the repository, so a rejected push leaves nothing behind.")


def _pull(args: argparse.Namespace) -> None:
    from visbench.hub import load_probe_from_hub

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    probe = load_probe_from_hub(
        args.pull,
        backbone=backbone,
        revision=args.revision,
        token=args.token,
    )
    print(f"loaded {probe.name!r} from {args.pull} onto {args.backbone}")
    print("  Same load_probe a local file gets: weights_only=True on the read,")
    print("  and the four identity fields checked against this backbone.")

    if args.revision is None:
        print("\nNo --revision passed. A Hub repository is mutable, so `main` today")
        print("is not promised to be `main` next month — pin a commit for anything")
        print("whose number you intend to quote.")

    if args.data:
        cache = FeatureCache(root=args.cache)
        val = ImageFolderDataset(Path(args.data) / "val", split="val").balanced_subset(args.limit)
        features = cache.extract_dataset(backbone, val, keep="pooled", pooling=probe.pooling)
        print(f"\nscored on your data: top1={probe.evaluate(features, val.labels())['top1']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", metavar="REPO_ID", help="download and load instead of pushing")
    parser.add_argument("--repo-id", help="target repository, e.g. you/dinov2s-imagenette")
    parser.add_argument("--data", help="root of a labelled image folder (required to push)")
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--limit", type=int, default=200, help="images per class")
    parser.add_argument("--push", action="store_true", help="actually upload; off by default")
    parser.add_argument("--public", action="store_true", help="publish readable by anyone")
    parser.add_argument("--revision", help="pin a commit when pulling")
    parser.add_argument("--token", help="Hub token; falls back to HF_TOKEN or a cached login")
    parser.add_argument("--out", default="checkpoints/hub_probe.pt", help="dry-run artifact path")
    parser.add_argument("--cache", default=".visbench_cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.pull:
        _pull(args)
        return
    if not args.repo_id or not args.data:
        parser.error("pushing needs --repo-id and --data; pulling needs --pull")
    _push(args)


if __name__ == "__main__":
    main()
