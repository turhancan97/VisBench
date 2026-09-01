#!/usr/bin/env python3
"""Fetch BSDS500's images and human annotations, and nothing else.

BSDS500 is the canonical boundary-detection benchmark: 500 natural images, each
segmented by several people, scored by ODS/OIS/AP. VisBench's existing `edge`
probe is *not* it — that is dense magnitude regression on Taskonomy's
`edge_texture` with a Pearson metric, and it says so in its `protocol` field.
This is the dataset half of adding the real thing.

**Berkeley is unreachable from this machine** (`www2.eecs.berkeley.edu` times
out; the older host 403s) while the network is otherwise fine, so this reads the
`BIDS/BSDS500` mirror on GitHub — "a mirror of the January 2013 update", which
carries the complete BSR package.

**It takes `BSDS500/data/` and deliberately leaves the rest.** The package also
ships `bench/` (the MATLAB evaluation suite) and `grouping/` (the gPb detector).
Neither the mirror nor the `.m` files carry a licence, so this repository may
not vendor them — the same position `NOTICE` already records for probe3d's
CC BY-NC correspondence code. The metric is implemented from the published
description (Arbelaez, Maire, Fowlkes & Malik, TPAMI 2011), never copied, and
extracting the code here would make that claim harder to believe rather than
easier.

The images and annotations themselves are research data used in place, exactly
as Taskonomy, VOC and NYUv2 are: read locally to produce numbers, never
redistributed. `/data/` is gitignored, so nothing fetched here is committed.

Usage::

    python scripts/fetch_bsds500.py
    python scripts/fetch_bsds500.py --dest data/bsds500 --keep-archive

Idempotent: an existing complete tree is left alone, so re-running costs a stat
per split and changes nothing.
"""

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: The mirror, pinned to a **commit** rather than a branch. A branch would let
#: the bytes under a published number change without the number changing, which
#: is the same reason `HUB_REF` is pinned for DINOv2 — and there the pin is what
#: keeps every cached feature valid.
#:
#: This one is the mirror's tip as of 2026-09-01, itself dated 2016-04-01: the
#: repository has not moved in a decade, which is what a mirror of a 2013
#: package should look like. `--ref` overrides it for anyone who needs to.
MIRROR = "BIDS/BSDS500"
COMMIT = "a04b7c6c3a9f0ace74bf205c72a43d32e1c72722"


#: Only these two, and only under `BSDS500/data/`. See the module docstring for
#: why `bench/` and `grouping/` are left in the archive.
WANTED = ("images", "groundTruth")

#: The 2013 package ships a Windows `Thumbs.db` in each image directory. Filter
#: on extension rather than blacklisting that one name: a stray file in an image
#: folder is exactly what `list_files` would later pick up as an image.
KEEP_SUFFIXES = {".jpg": "images", ".mat": "groundTruth"}

#: The official split sizes. Checked rather than assumed: a partial extraction
#: reads as a working dataset that silently scores on fewer images than it
#: claims, which is the failure `DenseFolderDataset`'s stem check exists for.
EXPECTED = {"train": 200, "val": 100, "test": 200}


def archive_url(ref: str) -> str:
    """Tarball URL for a mirror ref. Split out so `--ref` and the pin share it."""
    return f"https://codeload.github.com/{MIRROR}/tar.gz/{ref}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dest",
        default=str(REPO / "data" / "bsds500"),
        help="Where to write images/ and groundTruth/. Default data/bsds500.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded tarball, so a re-extract needs no second download.",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="Use an already-downloaded tarball instead of fetching one.",
    )
    parser.add_argument(
        "--ref",
        default=COMMIT,
        help=f"Mirror ref to fetch. Defaults to the pinned commit {COMMIT[:12]}.",
    )
    return parser.parse_args()


def already_complete(dest: Path) -> bool:
    """True when every split holds its full complement of images and annotations."""
    for split, count in EXPECTED.items():
        images = dest / "images" / split
        truth = dest / "groundTruth" / split
        if not images.is_dir() or not truth.is_dir():
            return False
        if len(list(images.glob("*.jpg"))) != count or len(list(truth.glob("*.mat"))) != count:
            return False
    return True


def download(url: str, target: Path) -> None:
    print(f"fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)
    print(f"  {target.stat().st_size / 1e6:.1f} MB")


def extract(archive: Path, dest: Path) -> int:
    """Unpack `BSDS500/data/{images,groundTruth}` into ``dest``. Returns files written."""
    written = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            # <repo>-<ref>/BSDS500/data/<kind>/<split>/<file>
            if len(parts) < 6 or parts[1] != "BSDS500" or parts[2] != "data":
                continue
            kind, split = parts[3], parts[4]
            if kind not in WANTED or split not in EXPECTED:
                continue
            if KEEP_SUFFIXES.get(Path(parts[5]).suffix.lower()) != kind:
                continue
            out = dest / kind / split / parts[5]
            out.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, out.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            written += 1
    return written


def verify(dest: Path) -> list[str]:
    """Every split at its official size, and every image paired with an annotation."""
    problems = []
    for split, count in EXPECTED.items():
        images = {p.stem for p in (dest / "images" / split).glob("*.jpg")}
        truth = {p.stem for p in (dest / "groundTruth" / split).glob("*.mat")}
        if len(images) != count:
            problems.append(f"{split}: {len(images)} images, expected {count}")
        if images != truth:
            missing = sorted(images ^ truth)[:5]
            problems.append(
                f"{split}: {len(images ^ truth)} stem(s) unpaired between images and "
                f"groundTruth, e.g. {', '.join(missing)}"
            )
    return problems


def main() -> int:
    args = parse_args()
    dest = Path(args.dest)

    if already_complete(dest):
        print(f"{dest} already holds all {sum(EXPECTED.values())} images and annotations")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(args.archive) if args.archive else Path(scratch) / "bsds500.tar.gz"
        if not archive.exists():
            download(archive_url(args.ref), archive)
        written = extract(archive, dest)
        print(f"extracted {written} files to {dest}")
        if args.keep_archive and not args.archive:
            shutil.copy2(archive, dest / "bsds500.tar.gz")

    problems = verify(dest)
    if problems:
        print("\nIncomplete:")
        for line in problems:
            print(f"  {line}")
        return 1

    print(f"\n{sum(EXPECTED.values())} images, each with human annotations:")
    for split, count in EXPECTED.items():
        print(f"  {split:5s} {count}")
    print("\nbench/ and grouping/ were deliberately not extracted -- neither carries a")
    print("licence, and the metric is implemented from the paper rather than copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
