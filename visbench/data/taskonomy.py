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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from visbench.data.dense import (
    DenseFolderDataset,
    load_depth_map,
    load_edge_map,
    load_normal_map,
)

__all__ = [
    "TaskonomyDataset",
    "load_taskonomy_split",
    "load_valid_mask",
    "TASKONOMY_DOMAINS",
    "TASKONOMY_SUPPORTED_DOMAINS",
]

#: Taskonomy's depth is stored as uint16 in units of 1/512 m, so this divisor is
#: what puts :func:`~visbench.data.dense.load_depth_map` into metres — the unit
#: :func:`~visbench.metrics.dense.depth_metrics` reports RMSE in and NYUv2
#: numbers are quoted in. Verified rather than assumed: over 10 buildings the
#: valid readings run to 6,633, i.e. 13.0 m, which is an indoor room; at 1000 it
#: would be 6.6 m and every scene would be implausibly shallow.
TASKONOMY_DEPTH_SCALE = 512.0

#: Value ``depth_zbuffer`` stores where the reconstruction has no surface. It is
#: the uint16 maximum, i.e. "infinitely far", and at 1/512 m it decodes to 128 m
#: — a number that would sail past any indoor ``max_target`` a caller thought to
#: set and land in the loss as a real reading.
_DEPTH_SENTINEL = 65535


@dataclass(frozen=True)
class _DomainSpec:
    """How one Taskonomy domain is read, and how it says a pixel is invalid.

    ``invalid`` names the convention, and it is the field worth reading twice —
    a probe's loss and metric both key off it, and getting it wrong is silent.
    """

    #: ``target_scale -> loader``; the loader takes the target path.
    loader: Callable[[float], Callable[[Path], torch.Tensor]]
    #: Divisor for an integer target. See :attr:`TaskonomyDataset.target_scale`.
    target_scale: float
    #: ``"none"``, ``"zero"``, ``"zero_vector"`` or ``"nan"``.
    invalid: str
    #: Whether reading this domain must consult ``mask_valid/``.
    needs_mask: bool
    #: Apply ``log1p`` after scaling. Set per domain **from measurement**, never
    #: by analogy — see :data:`_DOMAIN_SPECS` for the numbers.
    log_transform: bool = False
    #: Non-empty means the domain is refused, and says why.
    unsupported: str = ""


def _log1p(load: Callable, path: Path, scale: float) -> torch.Tensor:
    """``load`` then ``log1p``. A module-level function so it stays picklable."""
    return torch.log1p(load(path, scale=scale))


def _scalar_loader(
    load: Callable, log_transform: bool = False
) -> Callable[[float], Callable[[Path], torch.Tensor]]:
    """Bind ``target_scale`` — and optionally ``log1p`` — into a loader."""
    if log_transform:
        return lambda scale: functools.partial(_log1p, load, scale=scale)
    return lambda scale: functools.partial(load, scale=scale)


#: Every domain name whose filename convention is understood. Membership says
#: the naming is known, not that the files are present or that the domain can be
#: probed — see :data:`TASKONOMY_SUPPORTED_DOMAINS` for the latter.
#:
#: **Which domains need ``mask_valid/`` was measured, not inferred from whether
#: the domain is geometric.** Over 150 frames across 10 buildings:
#:
#: * ``depth_zbuffer`` — ``mask_valid == 0`` is *exactly* ``depth == 65535`` on
#:   every frame, so the sentinel already carries the mask's information and
#:   zeroing it is enough. The mask file is read anyway, because "exactly on 150
#:   frames" is evidence and not a guarantee, and disagreeing with it is free to
#:   check.
#: * ``normal`` — the invalid region is stored as ``(128, 128, 128)``, which
#:   :func:`~visbench.data.dense.load_normal_map` already decodes to a length of
#:   ~0.007 and zeroes by its existing threshold. Its zeroed pixels matched the
#:   mask on every frame, to the pixel. So the surface-normal probe needed
#:   *nothing* to run on Taskonomy; the convention it has carried since v0.2
#:   turns out to be Taskonomy's convention too.
#: * ``edge_occlusion`` and ``keypoints3d`` — genuinely need the file. Their
#:   invalid regions hold plain magnitudes (mostly 0, and for ``keypoints3d``
#:   sometimes not), which is indistinguishable from a real "no edge here".
#:
#: **``log_transform`` is set on exactly one domain, and measurement is the only
#: reason.** The magnitude protocol pairs an L1 loss, chosen so a handful of
#: strong pixels cannot dominate, with a Pearson correlation, which is dominated
#: by exactly those pixels. On a moderately heavy tail the two coexist; on a
#: severe one they pull in opposite directions and the probe measures nothing.
#: Share of total target mass held by the strongest 1% of pixels, over 40 val
#: frames:
#:
#: =================  =============  ================================
#: domain             mass in top 1%  ``correlation``, DINOv2-S vs -B
#: =================  =============  ================================
#: ``edge_texture``   0.10           0.4558 / 0.4481
#: ``keypoints2d``    0.11           (see below)
#: ``edge_occlusion`` **0.46**       **0.0877 / 0.0842** — does not rank
#: ``edge_occlusion``, ``log1p``  0.09  **0.2924 / 0.3167** — ranks
#: =================  =============  ================================
#:
#: The linear-target occlusion probe was flat under everything tried: four
#: target scales spanning 30x (0.077-0.082), 4x the training budget (+0.01), and
#: a 10x learning rate. Its S-vs-B gap was 0.0035, i.e. noise, so it could not
#: have ranked two backbones. Under ``log1p`` the tail matches the other domains
#: and the gap becomes 0.024 in DINOv2-B's favour. **Do not extend the transform
#: to the other domains on the strength of this**: their tails are already mild,
#: ``edge_texture``'s published v0.4.0 number is a linear-target number, and a
#: log-space correlation and a linear-space one are not the same measurement.
#: The transform is recorded in ``dataset_params`` so the two cannot be pooled.
_DOMAIN_SPECS: dict[str, _DomainSpec] = {
    "depth_zbuffer": _DomainSpec(
        loader=_scalar_loader(load_depth_map),
        target_scale=TASKONOMY_DEPTH_SCALE,
        invalid="zero",
        needs_mask=True,
    ),
    "normal": _DomainSpec(
        loader=lambda _scale: load_normal_map,
        target_scale=1.0,
        invalid="zero_vector",
        needs_mask=True,
    ),
    "edge_texture": _DomainSpec(
        loader=_scalar_loader(load_edge_map),
        target_scale=1000.0,
        invalid="none",
        needs_mask=False,
    ),
    "keypoints2d": _DomainSpec(
        loader=_scalar_loader(load_edge_map),
        target_scale=30.0,
        invalid="none",
        needs_mask=False,
    ),
    "edge_occlusion": _DomainSpec(
        loader=_scalar_loader(load_edge_map, log_transform=True),
        target_scale=50.0,
        invalid="nan",
        needs_mask=True,
        log_transform=True,
    ),
    "keypoints3d": _DomainSpec(
        loader=_scalar_loader(load_edge_map),
        target_scale=5000.0,
        invalid="nan",
        needs_mask=True,
    ),
    "principal_curvature": _DomainSpec(
        loader=_scalar_loader(load_edge_map),
        target_scale=1.0,
        invalid="nan",
        needs_mask=True,
        unsupported=(
            "its three channels encode two principal curvatures plus an unused one, and "
            "which is which, and in what units, is not established here. Reading it as a "
            "magnitude map would train and score against a number this library cannot name"
        ),
    ),
    "reshading": _DomainSpec(
        loader=_scalar_loader(load_edge_map),
        target_scale=1.0,
        invalid="nan",
        needs_mask=True,
        unsupported=(
            "it is a re-rendered shading image, so its target is an RGB frame rather than a "
            "per-pixel measurement, and no probe here consumes one. Adding it is a task "
            "decision, not a loader decision"
        ),
    ),
}

TASKONOMY_DOMAINS = tuple(_DOMAIN_SPECS)

#: The domains that can actually be probed. The rest are named so the error can
#: say what is wrong rather than "unknown domain".
TASKONOMY_SUPPORTED_DOMAINS = tuple(
    name for name, spec in _DOMAIN_SPECS.items() if not spec.unsupported
)


def load_valid_mask(path: str | Path) -> torch.Tensor:
    """Read a ``mask_valid/`` file as a ``(H, W)`` bool tensor, ``True`` where valid.

    The files are 8-bit greyscale holding 0 and 255 only, and the rule is simply
    **non-zero is valid** — the same "it is a label, not a measurement" reading
    :func:`~visbench.data.dense.load_mask` applies, and for the same reason: a
    mask divided by 255 would make every valid pixel 1/255.

    Notes
    -----
    **The filenames say ``depth_zbuffer``, and they are not depth maps.** Every
    file under ``mask_valid/`` is named ``..._domain_depth_zbuffer.png``
    whatever it masks, because Taskonomy derived one mask per frame from the
    depth render and never renamed it. Reading it as depth would produce a map
    of 0 and 255 that loads, trains and scores; :class:`TaskonomyDataset`
    therefore builds mask paths from the ``mask_valid`` directory with the
    depth suffix hard-coded, rather than from the requested domain.
    """
    path = Path(path)
    with Image.open(path) as handle:
        array = np.array(handle.convert("L"))
    if array.ndim != 2:
        raise ValueError(f"{path.name} holds a {array.ndim}D array {array.shape}; a mask is 2D")
    return torch.from_numpy(array) != 0


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
    #: ``mask_paths`` joined them in 6d-2 and *must* stay here — a mask sliced
    #: out of step with its target would invalidate the wrong pixels, which
    #: trains and scores without raising.
    _parallel_attrs = ("stems", "image_paths", "target_paths", "mask_paths")

    def __init__(
        self,
        root: str | Path,
        domain: str = "edge_texture",
        split: str = "train",
        partition: str = "tiny",
        image_size: int = 224,
        target_scale: float | None = None,
        target_loader: Callable[[Path], torch.Tensor] | None = None,
        max_images: int | None = None,
        buildings: Sequence[str] | None = None,
        max_target: float | None = None,
    ) -> None:
        """Index one domain against the RGB frames named by an official split.

        Parameters
        ----------
        domain:
            Target domain directory, e.g. ``"edge_texture"``. Must be one of
            :data:`TASKONOMY_SUPPORTED_DOMAINS`. Each carries its own loader,
            default scale and validity convention; see :data:`_DOMAIN_SPECS`.
            ``principal_curvature`` and ``reshading`` are named but refused,
            with the reason, rather than reported as unknown.
        target_scale:
            Divisor for an integer target. ``None`` takes the domain's default,
            which is what should normally happen: **the right scale differs per
            domain and one of them is not free to choose.**

            ``depth_zbuffer`` is fixed at 512, because Taskonomy stores depth in
            units of 1/512 m and that divisor is what puts it in **metres** —
            the unit ``depth_metrics`` reports RMSE in and every published NYUv2
            number is quoted in. Changing it would not rescale a free parameter,
            it would silently change what the number means.

            Every other domain's default puts a frame's mean near 1, for the
            reason measured below. Those scales *are* free — the correlation
            metric is invariant to them.

            **Defaults to 1000 for edge_texture, not to the container's 65535,
            and that was measured rather than chosen for tidiness.**

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
        max_target:
            Passed through to :class:`~visbench.data.dense.DenseFolderDataset`,
            which marks larger values *invalid* rather than clamping them.
            Meaningful for ``depth_zbuffer`` and for nothing else here — against
            a magnitude map it would delete exactly the strongest edges, and
            against a normal map it raises.

        Notes
        -----
        **Existence is not checked at construction, deliberately.** The split
        lists name up to 272k frames and confirming them would be one stat per
        file on a network mount — precisely the cost step 6d-0 measured at 5.69 s
        per 2,913 files and removed. A frame named by the split but absent from
        disk raises when it is read, naming the stem; that is late, but it is
        not silent, which is the property that matters.
        """
        spec = _DOMAIN_SPECS.get(domain)
        if spec is None:
            raise ValueError(
                f"Unknown Taskonomy domain {domain!r}. Known: {', '.join(TASKONOMY_DOMAINS)}"
            )
        if spec.unsupported:
            raise NotImplementedError(
                f"{domain!r} is not supported: {spec.unsupported}. Supported domains: "
                f"{', '.join(TASKONOMY_SUPPORTED_DOMAINS)}."
            )
        if max_images is not None and max_images < 1:
            raise ValueError(f"max_images must be >= 1, got {max_images}")
        if target_scale is None:
            target_scale = spec.target_scale
        elif domain == "depth_zbuffer" and target_scale != TASKONOMY_DEPTH_SCALE:
            # Not a style objection. `depth_metrics` reports RMSE and the delta
            # thresholds in whatever unit the target is in, and every number they
            # are compared against is in metres. A caller who "scaled the target
            # to order 1" here, which is the right move for every other domain in
            # this table, would produce a depth record whose RMSE silently means
            # something else.
            raise ValueError(
                f"target_scale={target_scale} would take depth_zbuffer out of metres. "
                f"Taskonomy stores depth in units of 1/512 m, so the divisor is fixed at "
                f"{TASKONOMY_DEPTH_SCALE}: depth metrics are reported in the target's unit "
                "and are only comparable to published ones in metres. Pass max_target= to "
                "cap the range instead."
            )

        self._init_geometry(
            root=root,
            split=split,
            image_size=image_size,
            target_scale=target_scale,
            max_target=max_target,
            # Bound here, not left to DenseFolderDataset.target(): that applies
            # `target_scale` only on its *default* depth path, so a custom
            # loader silently ignores it. Passing the bare function would make
            # `target_scale` a parameter that is recorded in describe(), folded
            # into the fingerprint, and does nothing — which is how it was first
            # written, and a scale sweep that returned four identical numbers is
            # what caught it.
            target_loader=(
                target_loader if target_loader is not None else spec.loader(target_scale)
            ),
        )
        self.name = f"taskonomy_{domain}"
        self.domain = domain
        self.partition = partition
        self._spec = spec

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
        mask_root = self.root / "mask_valid"
        required = [("rgb", image_root), (domain, target_root)]
        if spec.needs_mask:
            required.append(("mask_valid", mask_root))
        for label, directory in required:
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
        # Always built, even for an unmasked domain, so `_parallel_attrs` can
        # name it unconditionally and subset() cannot leave a half-populated
        # dataset behind. `needs_mask` decides whether they are ever *read*.
        # Note the hard-coded suffix: every mask file is named after
        # depth_zbuffer whatever it masks. See load_valid_mask.
        self.mask_paths = [
            mask_root / building / f"point_{point}_view_{view}_domain_depth_zbuffer.png"
            for building, point, view in rows
        ]

    # -- reading ---------------------------------------------------------------

    def _load_raw_target(self, index: int) -> torch.Tensor:
        """The target with its invalid region marked, before any geometry.

        Overrides the base hook because Taskonomy's validity lives in a *second*
        file for four of its six supported domains, which ``target_loader``
        cannot reach — it is one callable taking one path.

        The marker is whatever the target's own type already means by "no ground
        truth", so nothing downstream needs to know a mask was involved:

        ==================  ==================  =====================================
        ``invalid``         marker              read as invalid by
        ==================  ==================  =====================================
        ``"none"``          (nothing masked)    — every pixel is a measurement
        ``"zero"``          0                   every depth metric and loss here
        ``"zero_vector"``   ``(0, 0, 0)``       every surface-normal metric and loss
        ``"nan"``           ``NaN``             :func:`~visbench.metrics.dense.magnitude_metrics`
        ==================  ==================  =====================================

        The first three are in-band sentinels this codebase already had, which is
        why ``depth_zbuffer`` and ``normal`` reach the *existing* depth and
        surface-normal probes with no change to either. ``NaN`` is the fourth
        convention and exists because a magnitude map has no impossible value;
        see :func:`~visbench.metrics.dense.magnitude_metrics`.
        """
        target = super()._load_raw_target(index)
        if self._spec.invalid == "none":
            return target

        if self._spec.invalid == "zero":
            # depth_zbuffer only. The sentinel is checked as well as the mask,
            # not instead of it: they agreed on every one of 150 frames sampled
            # across 10 buildings, and an `or` of two agreeing tests costs
            # nothing while a disagreement on some frame nobody sampled would
            # otherwise put 128 m into the loss as a real reading.
            target = torch.where(
                target * self.target_scale >= _DEPTH_SENTINEL, torch.zeros_like(target), target
            )

        valid = load_valid_mask(self.mask_paths[index])
        if valid.shape != target.shape[-2:]:
            raise ValueError(
                f"mask_valid {self.mask_paths[index].name} is {tuple(valid.shape)} but the "
                f"{self.domain} target is {tuple(target.shape[-2:])}. They must be the same "
                "frame at the same resolution; masking with a mismatched shape would "
                "invalidate the wrong pixels."
            )

        if self._spec.invalid == "nan":
            return torch.where(valid, target, torch.full_like(target, float("nan")))
        # "zero" and "zero_vector" are the same operation: a (C, H, W) target
        # broadcasts against the (H, W) mask and every channel goes to 0, which
        # is exactly the zero-length vector load_normal_map already emits.
        return torch.where(valid, target, torch.zeros_like(target))

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
            f"{self.image_size}|{self.target_scale}|{self.max_target}|"
            # Both shape every target pixel, so two runs differing only in one
            # of them are two different sets of supervision.
            f"{self._spec.invalid}|{self._spec.log_transform}".encode()
        )
        for stem in self.stems:
            digest.update(stem.encode())
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        info = super().describe()
        info["domain"] = self.domain
        info["partition"] = self.partition
        info["target_scale"] = self.target_scale
        # Both land in the record's dataset_params. `invalid` is there because a
        # masked number and an unmasked one over the same frames are different
        # measurements — the mask removes 3.1% of pixels on average and 12% on
        # the worst frames sampled — and the record is what keeps them apart.
        info["invalid"] = self._spec.invalid
        info["masked"] = self._spec.needs_mask
        # A log-space correlation and a linear-space one are different
        # measurements of the same frames; this is what keeps them apart.
        info["target_transform"] = "log1p" if self._spec.log_transform else "none"
        if self.max_target is not None:
            info["max_target"] = self.max_target
        return info
