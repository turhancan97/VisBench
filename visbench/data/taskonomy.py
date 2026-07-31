"""Taskonomy — the low-level dense targets, v0.3 (step 6d-1).

Taskonomy (Zamir et al., CVPR 2018, `<https://arxiv.org/abs/1804.08328>`_) ships
many per-pixel annotations for the same indoor frames, several of which are
low-level in the sense this library's task taxonomy means: recoverable from the
signal without naming an object. ``edge_texture`` is the first one VisBench
probes.

**Why this and not BSDS500.** BSDS500 is the canonical edge benchmark, but its
protocol is not a per-pixel metric — ODS/OIS/AP match predicted to annotated
edge pixels by bipartite correspondence after non-maximum suppression, swept
over thresholds, against several annotators per image. Borrowing a protocol is
only worth it if it is borrowed exactly (see ``NOTICE`` and the depth probe's
256-bin expectation, which a from-memory reconstruction would have got wrong),
and that is a step of its own. Taskonomy's edge maps are dense and continuous,
so they measure the same underlying capability under a protocol this codebase
can state honestly and in full: see
:class:`~visbench.tasks.low_level.edge.EdgeTask`.

**Layout.** Unlike every other dataset here, the two halves of a pair do not
share a filename — the domain is *in* the filename — and the frames are nested
one directory per building::

    root/
      rgb/          allensville/point_0_view_0_domain_rgb.png ...
      edge_texture/ allensville/point_0_view_0_domain_edge_texture.png ...
      splits/       tiny_train.csv  tiny_val.csv  tiny_test.csv

So :class:`~visbench.data.dense.DenseFolderDataset`'s stem pairing cannot apply,
and this subclasses it for the *geometry* alone — the resize, the crop and the
nearest-neighbour target resampling — while indexing from the official split
lists instead.
"""

import csv
import functools
import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from visbench.data.dense import DenseFolderDataset, load_edge_map

__all__ = ["TaskonomyDataset", "load_taskonomy_split", "TASKONOMY_DOMAINS"]

#: Target domains this loader knows how to name. Membership says the filename
#: convention is understood, not that the files are present — a Taskonomy
#: download is per-domain and most are partial.
TASKONOMY_DOMAINS = (
    "depth_zbuffer",
    "edge_occlusion",
    "edge_texture",
    "keypoints2d",
    "keypoints3d",
    "normal",
    "principal_curvature",
    "reshading",
)

#: Frames whose target is derived from the 3D reconstruction rather than from
#: the RGB frame, and which therefore have invalid regions recorded separately
#: in ``mask_valid/``. ``edge_texture`` is deliberately not among them: it is
#: computed from the image itself, so every pixel is a real measurement and
#: there is nothing to mask. Probing one of these without wiring that mask up
#: would score a probe against reprojection holes.
_NEEDS_VALID_MASK = frozenset(
    {"depth_zbuffer", "edge_occlusion", "normal", "principal_curvature", "reshading"}
)


def load_taskonomy_split(root: str | Path, split: str, partition: str = "tiny") -> list[tuple]:
    """Read ``splits/<partition>_<split>.csv`` as ``(building, point, view)`` rows.

    Taskonomy's splits are **disjoint by building**, not by frame: the tiny
    partition puts 25 buildings in train, 4 in val and 5 in test with no overlap,
    so a val number is measured on rooms the probe has never seen. That is a
    stronger guarantee than a random frame split over the same rooms would give,
    and it is the reason to use the official lists rather than slicing the
    directory.
    """
    path = Path(root) / "splits" / f"{partition}_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"No split list at {path}. Taskonomy ships them under splits/ as "
            f"<partition>_<split>.csv; partition is the download size (tiny, medium, "
            "fullplus) and is part of what a number means, so it is not guessed."
        )

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    missing = {"building", "point", "view"} - set(rows[0] if rows else {})
    if missing:
        raise ValueError(
            f"{path.name} has no {sorted(missing)} column(s). Columns are read by name "
            "rather than position, so a reordered file cannot silently pair the wrong "
            "frames."
        )
    return [(row["building"], row["point"], row["view"]) for row in rows]


class TaskonomyDataset(DenseFolderDataset):
    """Taskonomy RGB frames and one per-pixel target domain.

    Inherits :class:`~visbench.data.dense.DenseFolderDataset`'s geometry
    unchanged — image and target take the same resize and centre crop, and the
    target resamples nearest-neighbour — and replaces only the indexing.
    """

    #: Reindexed together by :meth:`~visbench.data.base.BaseDataset.subset`, as
    #: for any dense dataset: slicing one alone would pair a target with the
    #: wrong image and every later step would still see equal lengths.
    _parallel_attrs = ("stems", "image_paths", "target_paths")

    def __init__(
        self,
        root: str | Path,
        domain: str = "edge_texture",
        split: str = "train",
        partition: str = "tiny",
        image_size: int = 224,
        target_scale: float = 1000.0,
        target_loader: Callable[[Path], torch.Tensor] | None = None,
        max_images: int | None = None,
        buildings: Sequence[str] | None = None,
    ) -> None:
        """Index one domain against the RGB frames named by an official split.

        Parameters
        ----------
        domain:
            Target domain directory, e.g. ``"edge_texture"``. Must be one of
            :data:`TASKONOMY_DOMAINS`, and the ones derived from the 3D
            reconstruction are refused: they have invalid regions recorded in
            ``mask_valid/`` which this does not read, and scoring a probe
            against reprojection holes would understate every backbone equally
            and silently.
        target_scale:
            Divisor for the integer target, passed to
            :func:`~visbench.data.dense.load_edge_map`. **Defaults to 1000, not
            to the container's 65535, and that was measured rather than chosen
            for tidiness.**

            The probe's loss is L1, whose gradient is ``sign(pred - target)`` —
            magnitude 1 regardless of how big the target is. So the optimiser's
            step size does not shrink with the target, and against Taskonomy's
            raw scale (a frame's mean is 0.011 of the container range) it simply
            oscillates around the answer. On 600 train / 600 val frames with
            DINOv2-S/14 and probe3d's schedule:

            ==============  ============  ====================
            ``target_scale``  frame mean    ``edge_correlation``
            ==============  ============  ====================
            65535           0.011         0.047
            6553.5          0.109         0.285
            **1000**        **0.717**     **0.456**
            100             7.165         0.467
            ==============  ============  ====================

            It plateaus once the target is order 1 — a further 7x buys 0.011 —
            so 1000 sits at the knee. Scaling the *target* rather than raising
            the learning rate is deliberate: the scale is arbitrary (65535 is
            just "uint16 max") and the headline metric is invariant to it, while
            the learning rate is what keeps this number under the same training
            budget as every other dense probe here. Given a free parameter that
            means nothing and one that means something, move the first. Raising
            the rate instead was tried and is worse anyway: lr 5e-3 reaches
            0.348, and 5e-2 collapses back to 0.066.
        max_images:
            Keep only the first ``n`` rows of the split. Worth having on the
            constructor rather than relying on
            :meth:`~visbench.data.base.BaseDataset.subset` afterwards: the tiny
            train list is 272,296 rows, and building 272k paths to discard all
            but 600 of them is pure waste. Rows are in the file's own order,
            which is shuffled across buildings — so unlike a labelled image
            folder, a prefix here is not one class.
        buildings:
            Restrict to these buildings. The split is already disjoint by
            building, so this is for cutting a split down further, not for
            making one.

        Notes
        -----
        **Existence is not checked at construction, deliberately.** The split
        lists name up to 272k frames and confirming them would be one stat per
        file on a network mount — precisely the cost step 6d-0 measured at 5.69 s
        per 2,913 files and removed. A frame named by the split but absent from
        disk raises when it is read, naming the stem; that is late, but it is
        not silent, which is the property that matters.
        """
        if domain not in TASKONOMY_DOMAINS:
            raise ValueError(
                f"Unknown Taskonomy domain {domain!r}. Known: {', '.join(TASKONOMY_DOMAINS)}"
            )
        if domain in _NEEDS_VALID_MASK:
            raise NotImplementedError(
                f"{domain!r} is derived from Taskonomy's 3D reconstruction and has invalid "
                f"regions listed in mask_valid/, which this dataset does not read. Scoring "
                f"against those pixels would depress every backbone's number without "
                f"saying so. Only image-derived domains are supported so far: "
                f"{', '.join(sorted(set(TASKONOMY_DOMAINS) - _NEEDS_VALID_MASK))}."
            )
        if max_images is not None and max_images < 1:
            raise ValueError(f"max_images must be >= 1, got {max_images}")

        self._init_geometry(
            root=root,
            split=split,
            image_size=image_size,
            target_scale=target_scale,
            # Marks values invalid, and for an edge map 0 is a real reading
            # ("no edge") while the largest values are the strongest edges.
            # There is nothing here it could correctly remove.
            max_target=None,
            # Bound here, not left to DenseFolderDataset.target(): that applies
            # `target_scale` only on its *default* depth path, so a custom
            # loader silently ignores it. Passing the bare function would make
            # `target_scale` a parameter that is recorded in describe(), folded
            # into the fingerprint, and does nothing — which is how it was first
            # written, and a scale sweep that returned four identical numbers is
            # what caught it.
            target_loader=(
                target_loader
                if target_loader is not None
                else functools.partial(load_edge_map, scale=target_scale)
            ),
        )
        self.name = f"taskonomy_{domain}"
        self.domain = domain
        self.partition = partition

        rows = load_taskonomy_split(self.root, split, partition)
        if buildings is not None:
            allowed = set(buildings)
            unknown = allowed - {building for building, _, _ in rows}
            if unknown:
                raise ValueError(
                    f"buildings={sorted(unknown)} do not appear in the {partition}_{split} "
                    "split. Taskonomy's splits are disjoint by building, so a building in "
                    "one split is absent from the others by construction."
                )
            rows = [row for row in rows if row[0] in allowed]
        if max_images is not None:
            rows = rows[:max_images]
        if not rows:
            raise ValueError(
                f"No frames selected from {partition}_{split}. "
                f"{'buildings= matched nothing' if buildings else 'the split list is empty'}."
            )

        image_root = self.root / "rgb"
        target_root = self.root / domain
        for label, directory in (("rgb", image_root), (domain, target_root)):
            if not directory.is_dir():
                raise NotADirectoryError(f"{label}/ is not a directory under {self.root}")

        # The building is part of the stem because point/view numbering restarts
        # in every building: "point_0_view_0" names 36 different frames, and a
        # stem list that collided would make the fingerprint claim two distinct
        # splits were the same one.
        self.stems = [f"{building}/point_{point}_view_{view}" for building, point, view in rows]
        self.image_paths = [
            image_root / building / f"point_{point}_view_{view}_domain_rgb.png"
            for building, point, view in rows
        ]
        self.target_paths = [
            target_root / building / f"point_{point}_view_{view}_domain_{domain}.png"
            for building, point, view in rows
        ]

    def fingerprint(self) -> str:
        """Short hash over the frame list, the domain and the geometry.

        Overrides the inherited version, which stats every image and every
        target to fold their sizes in. Two reasons that is the wrong trade here.
        A Taskonomy partition is a **fixed published release**, so the
        ``(partition, split, building, point, view)`` tuple already identifies
        the bytes as precisely as a size would. And the split lists are large:
        stat-ing 2N files on a network mount is the cost step 6d-0 exists to
        have removed, and paying it once per run to re-derive something the
        release already pins would be reintroducing it by the back door.

        What this gives up is catching a target file edited in place, which the
        size-based version would catch only if the edit changed its length
        anyway. Edited *images* are still caught downstream regardless, because
        the feature cache keys on decoded pixel content.
        """
        digest = hashlib.sha256()
        digest.update(
            f"{self.name}|{self.partition}|{self.split}|{self.domain}|{len(self.stems)}|"
            f"{self.image_size}|{self.target_scale}".encode()
        )
        for stem in self.stems:
            digest.update(stem.encode())
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        info = super().describe()
        info["domain"] = self.domain
        info["partition"] = self.partition
        info["target_scale"] = self.target_scale
        return info
