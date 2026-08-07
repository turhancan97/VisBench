#!/usr/bin/env python3
"""Gather published probe repositories into one Hugging Face collection.

    # see exactly what would be created and added, touching nothing
    python scripts/publish_collection.py --owner your-hf-name

    # do it
    python scripts/publish_collection.py --owner your-hf-name --create

The repositories themselves come from ``visbench run --push-to`` — one per
(probe, backbone) pair, because a head fitted on one backbone is refused
against any other. This script only groups them, so a visitor sees a benchmark
rather than twenty unrelated files.

**It does not create anything without --create.** A collection is outward-facing
and a public one may be indexed the moment it exists.

**The repositories must be public for a visitor to use them.** A collection of
private repositories renders empty to everyone but you, which looks like a
broken page rather than a permissions choice — push with ``PUSH_PUBLIC=1`` (or
``visbench run --push-to ... --public``) for anything you intend to share.

Needs ``pip install visbench[hub]`` and a token: ``huggingface-cli login``,
``HF_TOKEN`` in the environment, or ``--token``.
"""

from __future__ import annotations

import argparse
import sys

#: The probes that fit a head. The other three — retrieval, correspondence and
#: mid-level similarity — are zero-shot: they train nothing, so there is no
#: artifact to publish and the backbone alone reproduces their numbers.
TRAINED_PROBES = (
    "classification",
    "semantic_segmentation",
    "generic_segmentation",
    "detection",
    "depth",
    "surface_normal",
    "occlusion_edge",
    "edge",
    "keypoints2d",
    "corner",
)

#: The Hub caps this at 150 characters, so it carries only the thing a visitor
#: has to know before the weights mean anything: a head belongs to exactly one
#: backbone. The rest of the argument is on each model card.
DESCRIPTION = (
    "Heads fitted on frozen features, one per (task, backbone). Evaluate "
    "without training. Each head is valid only against its named backbone."
)
assert len(DESCRIPTION) < 150, f"the Hub rejects this: {len(DESCRIPTION)} chars"


def _repo_ids(owner: str, probes: list[str], backbones: list[str]) -> list[tuple[str, str]]:
    """``(repo_id, note)`` pairs, in the order they should appear."""
    return [
        (f"{owner}/visbench-{probe}-{backbone}", f"{probe} probe for {backbone}")
        for probe in probes
        for backbone in backbones
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--owner", required=True, help="Hub user or organisation")
    parser.add_argument("--title", default="VisBench probes", help="collection title")
    parser.add_argument("--probes", nargs="*", default=list(TRAINED_PROBES))
    parser.add_argument(
        "--backbones",
        nargs="*",
        default=["dinov2_vits14", "dinov2_vitb14"],
        help="one repository per probe per backbone",
    )
    parser.add_argument("--create", action="store_true", help="actually create; off by default")
    parser.add_argument("--private", action="store_true", help="hide the collection")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    unknown = set(args.probes) - set(TRAINED_PROBES)
    if unknown:
        # Named rather than silently skipped: a zero-shot probe in this list is
        # a misunderstanding of what is being published, not a typo to absorb.
        parser.error(
            f"no head is published for {sorted(unknown)}. Trained probes are: "
            f"{', '.join(TRAINED_PROBES)}"
        )

    items = _repo_ids(args.owner, args.probes, args.backbones)

    if not args.create:
        print(f"DRY RUN — nothing was created. Would build under {args.owner}:\n")
        print(f"  title       {args.title}")
        print(f"  visibility  {'private' if args.private else 'public'}")
        print(f"  items       {len(items)}\n")
        for repo_id, note in items:
            print(f"    {repo_id:<52} {note}")
        print("\nRe-run with --create. Every repository above must already exist")
        print("and be public, or the collection will render empty to visitors:")
        print(f"  PUSH_TO={args.owner} PUSH_PUBLIC=1 scripts/build_corpus.sh")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("needs huggingface_hub: pip install visbench[hub]", file=sys.stderr)
        return 1

    api = HfApi(token=args.token)
    collection = api.create_collection(
        title=args.title,
        namespace=args.owner,
        description=DESCRIPTION,
        private=args.private,
        exists_ok=True,
    )
    print(f"collection: https://huggingface.co/collections/{collection.slug}")

    for repo_id, note in items:
        try:
            # exists_ok so re-running after adding a backbone is not an error;
            # this script should be safe to run again as the set grows.
            api.add_collection_item(
                collection_slug=collection.slug,
                item_id=repo_id,
                item_type="model",
                note=note,
                exists_ok=True,
            )
            print(f"  added {repo_id}")
        except Exception as error:  # noqa: BLE001 - one missing repo must not stop the rest
            print(f"  !! {repo_id}: {error}", file=sys.stderr)

    print("\nPaste the collection URL above into README.md and docs/hub.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
