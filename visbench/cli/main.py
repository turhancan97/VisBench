"""Argument parsing and command dispatch for ``visbench``.

The parser is built in :func:`build_parser` and returned rather than parsed
in place, so the whole surface is testable without a subprocess — every flag,
default and error message in this module is checked by ``tests/cli/``.

:func:`main` returns an exit code instead of calling ``sys.exit``, for the same
reason.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import visbench
from visbench import registry
from visbench.cache import DEFAULT_CACHE_DIR, FeatureCache
from visbench.cli.datasets import showable_probes, spec_for, supported_probes

__all__ = ["main", "build_parser"]

_LIST_CHOICES = ("all", "backbones", "probes", "heads")


def _add_run_common(parser: argparse.ArgumentParser) -> None:
    """Flags every ``run`` subcommand shares, whatever the probe."""
    parser.add_argument("--data", type=Path, required=True, help="dataset root")
    parser.add_argument("--backbone", default="dinov2_vits14", help="see `visbench list backbones`")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="shorten the split; per class for labelled folders, by triplet for similarity",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="feature extraction batch size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE_DIR, help="feature cache directory"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="extract without reading or writing the cache; for timing a cold run",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/visbench.jsonl"),
        help="JSONL to append the result record to; 'none' to skip writing",
    )
    parser.add_argument("--notes", default=None, help="free text stored in the record")
    parser.add_argument(
        "--push-to",
        metavar="REPO_ID",
        default=None,
        help="upload the trained probe to this Hub repository; needs visbench[hub]",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="make the pushed repository readable by anyone (default: private)",
    )
    parser.add_argument(
        "--save-probe",
        type=Path,
        metavar="PATH",
        default=None,
        help="write the trained head here, with the backbone identity beside it, so "
        "`visbench show --predict-from` can draw what it predicts. Needs no Hub account",
    )
    parser.add_argument("--json", action="store_true", help="print the full record as JSON")
    parser.add_argument("--quiet", action="store_true", help="print metrics only")
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="re-raise on error instead of printing a one-line message",
    )


def _add_show_common(parser: argparse.ArgumentParser) -> None:
    """Flags every ``show`` subcommand shares, whatever the probe."""
    parser.add_argument("--data", type=Path, required=True, help="dataset root")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="PNG to write; default is visbench-show-<probe>.png in the working directory",
    )
    parser.add_argument("--frames", type=int, default=4, help="how many rows to draw (default: 4)")
    parser.add_argument(
        "--start", type=int, default=0, help="index of the first frame to draw (default: 0)"
    )
    parser.add_argument(
        "--predict-from",
        type=Path,
        metavar="PATH",
        default=None,
        help="a probe saved by `visbench run --save-probe`; adds a prediction column. "
        "The backbone must be the one it was fitted on, which is checked, not assumed",
    )
    parser.add_argument(
        "--backbone",
        default="dinov2_vits14",
        help="only used with --predict-from; see `visbench list backbones`",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="feature extraction batch size")
    parser.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE_DIR, help="feature cache directory"
    )
    parser.add_argument("--no-cache", action="store_true", help="extract without using the cache")
    parser.add_argument("--seed", type=int, default=0)
    # Correspondence only. Kept on the shared block rather than on that one
    # subcommand because its `show_arguments` is its `add_arguments` -- nothing
    # about a match is a training setting, so there is no second half to put
    # these in without giving `run correspondence` two flags it would ignore.
    parser.add_argument(
        "--matches",
        type=int,
        default=40,
        help="correspondence only: how many matches to draw per pair, sampled evenly "
        "across the kept set rather than taking the most confident (default: 40)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="correspondence only: pixels within which a match is drawn green. "
        "Defaults to 5, the probe's headline recall@5px",
    )
    parser.add_argument("--quiet", action="store_true", help="print the output path only")
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="re-raise instead of printing a one-line error",
    )


def build_parser() -> argparse.ArgumentParser:
    """The full ``visbench`` command line."""
    parser = argparse.ArgumentParser(
        prog="visbench",
        description="Probe vision backbones across high-, mid- and low-level tasks.",
    )
    parser.add_argument("--version", action="version", version=f"visbench {visbench.__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    listing = commands.add_parser("list", help="show registered backbones, probes and heads")
    listing.add_argument("what", nargs="?", default="all", choices=_LIST_CHOICES)

    run = commands.add_parser(
        "run",
        help="run one probe on one backbone over one dataset",
        description=(
            "Each probe is its own subcommand, because they do not take the same data: "
            "`visbench run depth --help` shows the folder layout depth expects."
        ),
    )
    probes = run.add_subparsers(dest="probe", metavar="<probe>", required=True)
    for name in supported_probes():
        spec = spec_for(name)
        subparser = probes.add_parser(
            name,
            help=spec.summary,
            description=f"{spec.summary}\n\nExpected layout:\n  {spec.layout}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        _add_run_common(subparser)
        spec.add_arguments(subparser)

    show = commands.add_parser(
        "show",
        help="draw a probe's images beside their targets — and its predictions",
        description=(
            "Writes a grid of image / target / prediction panels to a file.\n\n"
            "Measures nothing and records nothing. It exists because a dense "
            "target drifting from the image it belongs to fails silently -- the "
            "probe trains, and the number merely comes out mediocre, which reads "
            "as a hard task rather than as a bug. Two of this project's most "
            "expensive failures were exactly that, and both are obvious in one "
            "frame.\n\n"
            "Without --predict-from it needs no backbone, no cache and no GPU, "
            "which is when you actually want to look: before spending a "
            "training budget on the data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    show_probes_parser = show.add_subparsers(dest="probe", metavar="<probe>", required=True)
    for name in showable_probes():
        spec = spec_for(name)
        assert spec.show_arguments is not None  # showable_probes() filtered on it
        subparser = show_probes_parser.add_parser(
            name,
            help=spec.summary,
            description=f"{spec.summary}\n\nExpected layout:\n  {spec.layout}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        _add_show_common(subparser)
        spec.show_arguments(subparser)

    cache = commands.add_parser("cache", help="inspect or drop extracted features")
    cache_commands = cache.add_subparsers(dest="cache_command", metavar="<action>", required=True)
    stats = cache_commands.add_parser("stats", help="entry count and size on disk")
    stats.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)
    clear = cache_commands.add_parser("clear", help="delete cached features")
    clear.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)
    clear.add_argument(
        "--backbone",
        default=None,
        help="drop only this backbone's entries; default drops the whole cache",
    )
    clear.add_argument("--yes", action="store_true", help="do not ask for confirmation")

    demo = commands.add_parser(
        "demo",
        help="run a probe on generated data — no dataset, no setup",
        description=(
            "Draws a small labelled set of geometric shapes, wraps torchvision's "
            "ResNet-18 (~45 MB, a core dependency) and runs a real probe over "
            "them through the ordinary visbench.run path.\n\n"
            "Colour carries no information and the shapes are rotated, scaled "
            "and noised, so the score means something: ~0.81 top1 against a "
            "0.25 chance baseline. Raise --noise to watch it fall toward chance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    demo.add_argument(
        "--probe",
        default="classification",
        choices=("classification", "retrieval"),
        help="default: classification",
    )
    demo.add_argument("--images", type=int, default=20, help="per class per split")
    demo.add_argument("--noise", type=float, default=45.0, help="pixel noise; raise to harden")
    demo.add_argument(
        "--data",
        type=Path,
        default=None,
        help="where to write the images; default is a temporary directory",
    )
    demo.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)
    demo.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    demo.add_argument("--seed", type=int, default=0)

    return parser


# -- commands ----------------------------------------------------------------


def _command_list(args: argparse.Namespace, out: Any) -> int:
    from visbench.heads import list_heads

    sections = {
        "backbones": visbench.list_backbones,
        "probes": visbench.list_probes,
        "heads": list_heads,
    }
    wanted = list(sections) if args.what == "all" else [args.what]
    needed: set[str] = set()
    for name in wanted:
        print(f"{name}:", file=out)
        for entry in sections[name]():
            note = ""
            if name == "probes" and entry not in supported_probes():
                # A registered probe the CLI cannot build data for is still real
                # and still reachable from Python; saying so beats omitting it.
                note = "   (Python API only — the CLI cannot guess its data layout)"
            if name == "backbones":
                # Listed either way: a missing extra does not unregister a
                # backbone, it only makes constructing one raise. Marking it here
                # is the difference between "you need an extra" and discovering
                # that at the end of a long command.
                extra = registry.missing_extra(entry)
                if extra is not None:
                    needed.add(extra)
                    note = f"   (needs the '{extra}' extra)"
            print(f"  {entry}{note}", file=out)
    if needed:
        print(f"\n  pip install 'visbench[{','.join(sorted(needed))}]'", file=out)
    return 0


def _command_cache(args: argparse.Namespace, out: Any) -> int:
    cache = FeatureCache(root=args.cache)
    if args.cache_command == "stats":
        stats = cache.stats()
        print(f"root:    {stats['root']}", file=out)
        print(f"entries: {stats['entries']}", file=out)
        print(f"size:    {stats['bytes'] / 1e9:.3f} GB", file=out)
        return 0

    target = f"backbone {args.backbone!r}" if args.backbone else "every backbone"
    if not args.yes:
        # Deleting a full dense cache costs hours of GPU time to rebuild, and
        # the flag that scopes it is easy to forget.
        print(f"This deletes cached features for {target} under {args.cache}.", file=out)
        print("Re-run with --yes to confirm.", file=out)
        return 1

    key = None
    if args.backbone is not None:
        # The directory is named after the cache key, not the registered name —
        # resolving it here means `--backbone resnet50` does what it looks like.
        key = visbench.get_backbone(args.backbone).cache_key()
    removed = cache.clear(backbone_key=key)
    print(f"removed {removed} entries for {target}", file=out)
    return 0


def _command_run(args: argparse.Namespace, out: Any) -> int:
    spec = spec_for(args.probe)
    splits = spec.build(args)
    probe = visbench.get_probe(args.probe, **spec.probe_kwargs(args))

    # Refused here rather than by save_probe, which would raise the same
    # ValueError -- after training. A zero-shot probe fits nothing, so there
    # are no weights to share and the backbone alone reproduces it; finding
    # that out at the end of a run costs the whole run.
    if (args.push_to or args.save_probe) and getattr(probe, "zero_shot", False):
        destination = "push" if args.push_to else "save"
        raise ValueError(
            f"{args.probe!r} is zero-shot: it trains nothing, so there is no head to "
            f"{destination}. Anyone with the backbone reproduces this number -- share "
            "the command, not a file."
        )

    if not args.quiet:
        print(f"{args.probe} on {args.backbone}", file=out)
        print(f"  scoring: {len(splits.evaluate)} items from {args.data}", file=out)
        if splits.train is not None:
            print(f"  training: {len(splits.train)} items", file=out)

    results = None if str(args.results).lower() == "none" else args.results
    # The probe is built here and passed as an object rather than by name with
    # kwargs. `run()` owns `batch_size` (extraction) and `device` (the
    # backbone's), and several probes take constructor arguments of the same
    # names meaning different things — a dense probe's `batch_size` is its
    # training batch. Forwarding them as task kwargs is a TypeError, and the
    # examples reached for the same resolution before the CLI existed.
    # The backbone goes in **by name**, whether or not this run pushes, and
    # push_probe takes the object back out of the result. Constructing it here
    # instead would build it before run()'s set_seed() rather than after, and a
    # backbone's random init draws from the global RNG -- so the head would be
    # seeded from a different state and every trained probe would score
    # differently, with the record showing the same seed. --push-to must not be
    # able to move a number.
    result = visbench.run(
        args.backbone,
        probe,
        splits.evaluate,
        train_dataset=splits.train,
        cache=FeatureCache(root=args.cache, enabled=not args.no_cache),
        results=results,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        notes=args.notes,
        use_prefix_cache=not getattr(args, "no_prefix_cache", False),
    )

    width = max(len(name) for name in result.metrics)
    for name, value in sorted(result.metrics.items()):
        print(f"  {name:<{width}}  {value:.4f}", file=out)

    if args.save_probe:
        # No optional extra: save_probe is pure torch, so a core install can
        # write a head and draw its predictions without a Hub account. Runs
        # after the probe is fitted, unlike the zero-shot refusal above, because
        # this one cannot leave anything behind on failure.
        from visbench.hub import save_probe

        written = save_probe(result.probe, args.save_probe, backbone=result.backbone)
        print(f"\nsaved the trained head to {written}", file=out)

    if args.push_to:
        # Imported here, not at module scope: huggingface_hub is an optional
        # extra, and `visbench run` without --push-to must work in a core
        # install. The same reason visbench.hub.remote defers its own import.
        from visbench.hub import push_probe

        url = push_probe(
            result.probe,
            args.push_to,
            backbone=result.backbone,
            metrics=result.metrics,
            private=not args.public,
        )
        visibility = "public" if args.public else "private"
        print(f"\npushed to {url} ({visibility})", file=out)

    if args.json:
        print(json.dumps(result.record.to_dict(), indent=2, sort_keys=True), file=out)
    elif not args.quiet and results is not None:
        print(f"\nappended to {results}", file=out)
    return 0


def _command_show(args: argparse.Namespace, out: Any) -> int:
    """Draw what the probe saw, and optionally what a saved head predicts.

    Nothing here re-reads an image or resizes a panel: the dataset already
    yields a PIL image at the working resolution, and that image is pasted
    unchanged. A viewer that applied its own geometry could make a misaligned
    pipeline look fine and a correct one look broken, which would make it worse
    than no viewer at all.
    """
    from visbench.viz import render_probe_panels

    spec = spec_for(args.probe)
    if args.frames < 1:
        raise ValueError(f"--frames must be at least 1, got {args.frames}")

    # Both halves are built by every `build`, and one of them is never drawn.
    # Mirroring --stems means a viewer can be pointed at an official split list
    # without also naming a training list it will not read.
    if getattr(args, "stems", None) is not None and getattr(args, "train_stems", None) is None:
        args.train_stems = args.stems
    # Shortening happens where each probe already shortens: `--limit` reaches
    # Taskonomy and corner as the constructor's `max_images` and the folder
    # probes as `subset()`, so asking for it by that name keeps one code path.
    args.limit = args.start + args.frames

    dataset = spec.build(args).evaluate
    if len(dataset) <= args.start:
        raise ValueError(
            f"--start {args.start} is past the end of the split, which holds {len(dataset)} items."
        )
    indices = list(range(args.start, min(len(dataset), args.start + args.frames)))

    probe = visbench.get_probe(args.probe, **spec.probe_kwargs(args))
    if getattr(probe, "uses_pairs", False):
        # Correspondence draws two views and the relation between them, not a
        # target beside an image, so it has its own renderer. It also *always*
        # needs a backbone: the matches are the thing being looked at, and they
        # do not exist until features do.
        if args.predict_from is not None:
            raise ValueError(
                f"{args.probe!r} is zero-shot: it trains nothing, so there is no saved "
                "head to draw from. Drop --predict-from; the backbone alone produces "
                "the matches."
            )
        page = _match_page(args, probe, dataset, indices, out)
        predictions = None
    else:
        predictions = None
        if args.predict_from is not None:
            predictions = _predictions(args, spec, dataset, indices, out)
        class_names = getattr(dataset, "classes", None)
        page = render_probe_panels(dataset, args.probe, indices, predictions, class_names)

    destination = args.out if args.out is not None else Path(f"visbench-show-{args.probe}.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    page.save(destination)

    if not args.quiet:
        columns = "image, target, prediction" if predictions is not None else "image, target"
        if getattr(probe, "uses_pairs", False):
            columns = f"view 0 | view 1, matched within {args.threshold:g}px"
        print(f"{args.probe}: {len(indices)} frames from {args.data}", file=out)
        print(f"  columns: {columns}", file=out)
    print(f"{destination}", file=out)
    return 0


def _match_page(
    args: argparse.Namespace, probe: Any, dataset: Any, indices: list[int], out: Any
) -> Any:
    """Extract both views of the chosen pairs and draw the matches between them.

    Uses the same flatten-and-regroup route ``run()`` does — ``PairViewDataset``
    presents a pair's two views as ``2N`` single images, so the cache, its
    identity memo and the batching all stay unchanged. Reimplementing the
    pairing here would also lose ``view_identity``, without which a cached run
    still decodes and warps every frame.
    """
    from visbench.data.pair_dataset import PairViewDataset
    from visbench.viz import render_match_panels

    if not args.quiet:
        print(f"loading {args.backbone} and matching {len(indices)} pairs...", file=out)

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    pairs = dataset.subset(indices)
    views = FeatureCache(root=args.cache, enabled=not args.no_cache).materialise(
        backbone,
        PairViewDataset(pairs),
        pooling=probe.pooling,
        layers=probe.layers,
        feature_mode=probe.feature_mode,
        batch_size=args.batch_size,
    )
    return render_match_panels(
        pairs,
        probe,
        PairViewDataset.regroup(views),
        range(len(indices)),
        threshold=args.threshold,
        max_matches=args.matches,
    )


def _predictions(
    args: argparse.Namespace, spec: Any, dataset: Any, indices: list[int], out: Any
) -> Any:
    """Run a saved head over exactly ``indices``, in that order.

    The probe is constructed from the same ``probe_kwargs`` ``run`` uses, then
    :func:`visbench.hub.load_probe` refuses a backbone the head was not fitted
    on — a head fed the wrong pooling scores 0.9620 against 0.9820 without
    raising, so the check is the reason this path goes through the artifact
    module rather than ``torch.load``.
    """
    from visbench.hub import load_probe

    if not args.quiet:
        print(f"loading {args.backbone} and the head from {args.predict_from}...", file=out)

    backbone = visbench.get_backbone(args.backbone, device=args.device)
    probe = visbench.get_probe(args.probe, **spec.probe_kwargs(args))
    probe = load_probe(args.predict_from, backbone=backbone, task=probe)

    # subset() reindexes every parallel list together, so the features stay
    # paired with the frames they came from. Slicing an attribute by hand is
    # what pairs a target with the wrong image, silently.
    frames = dataset.subset(indices)
    features = FeatureCache(root=args.cache, enabled=not args.no_cache).extract_dataset(
        backbone,
        frames,
        pooling=probe.pooling,
        layers=probe.layers,
        feature_mode=probe.feature_mode,
        batch_size=args.batch_size,
    )
    # Labels go in even though a dense probe's predict() ignores them: detection
    # pairs each image's annotation with its features and refuses None, and one
    # call that works for both beats a branch on the probe's kind.
    return probe.predict(features, frames.labels())


def _command_demo(args: argparse.Namespace, out: Any) -> int:
    """A full probe run on generated data, through the ordinary code path.

    Nothing here is special-cased: the same ``visbench.run``, the same cache and
    the same result record as any other run. A demo that took a shortcut would
    prove the shortcut works.
    """
    import tempfile

    from visbench.data import ImageFolderDataset
    from visbench.demo import SHAPES, demo_backbone, synthesise

    with tempfile.TemporaryDirectory() as scratch:
        root = args.data if args.data is not None else Path(scratch) / "shapes"
        print(f"drawing {args.images} images per class for {len(SHAPES)} shapes...", file=out)
        synthesise(root, per_class=args.images, seed=args.seed, noise=args.noise)
        print(f"  {root}", file=out)

        print("loading resnet18 (torchvision, ~45 MB on first run)...", file=out)
        backbone = demo_backbone(device=args.device)

        train = ImageFolderDataset(root / "train", split="train")
        val = ImageFolderDataset(root / "val", split="val")

        kwargs: dict = {}
        if args.probe == "classification":
            kwargs["train_dataset"] = train

        print(f"running the {args.probe} probe...\n", file=out)
        result = visbench.run(
            backbone,
            args.probe,
            val,
            cache=FeatureCache(root=args.cache),
            device=args.device,
            seed=args.seed,
            **kwargs,
        )

        for name, value in result.metrics.items():
            print(f"  {name:12s} {value:.4f}", file=out)
        chance = 1.0 / len(SHAPES)
        print(f"\n  chance is {chance:.2f} — the shapes differ in outline only.", file=out)
        print("  Colour, size, position and rotation are randomised, so a", file=out)
        print("  backbone that has not learned geometry scores about chance.", file=out)
        print(f"\nRecord: {result.record.backbone} / {result.record.task}, ", end="", file=out)
        print(f"pooling={result.record.pooling}, seed={result.record.seed}", file=out)
        print("\nNext: `visbench run --help`, or point a probe at your own folder.", file=out)

    return 0


_COMMANDS = {
    "list": _command_list,
    "run": _command_run,
    "show": _command_show,
    "cache": _command_cache,
    "demo": _command_demo,
}

#: Failures that mean "your inputs were wrong", reported as one line rather
#: than a traceback. Deliberately not a bare ``Exception``: a bug inside a
#: training loop should look like a bug, and ``--traceback`` recovers the full
#: stack when one of these turns out to be one.
_USER_ERRORS = (KeyError, ValueError, TypeError, OSError)


def main(argv: list[str] | None = None, out: Any = None) -> int:
    """Parse ``argv`` and run the command. Returns an exit code."""
    out = out if out is not None else sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(file=out)
        return 1

    try:
        return _COMMANDS[args.command](args, out)
    except _USER_ERRORS as error:
        if getattr(args, "traceback", False):
            raise
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
