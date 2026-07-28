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
from visbench.cli.datasets import spec_for, supported_probes

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
    parser.add_argument("--json", action="store_true", help="print the full record as JSON")
    parser.add_argument("--quiet", action="store_true", help="print metrics only")
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="re-raise on error instead of printing a one-line message",
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
    result = visbench.run(
        args.backbone,
        visbench.get_probe(args.probe, **spec.probe_kwargs(args)),
        splits.evaluate,
        train_dataset=splits.train,
        cache=FeatureCache(root=args.cache, enabled=not args.no_cache),
        results=results,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        notes=args.notes,
    )

    width = max(len(name) for name in result.metrics)
    for name, value in sorted(result.metrics.items()):
        print(f"  {name:<{width}}  {value:.4f}", file=out)

    if args.json:
        print(json.dumps(result.record.to_dict(), indent=2, sort_keys=True), file=out)
    elif not args.quiet and results is not None:
        print(f"\nappended to {results}", file=out)
    return 0


_COMMANDS = {"list": _command_list, "run": _command_run, "cache": _command_cache}

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
