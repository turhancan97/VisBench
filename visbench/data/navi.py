"""NAVI — multi-view captures of one rigid object, with per-image camera pose (16a-1).

The dataset half of relative camera pose: two views of the same object and the
rigid transform between the cameras that took them. NAVI (Jampani et al.,
NeurIPS 2023) ships, per multi-view scene, a folder of photographs and an
``annotations.json`` giving each one's camera quaternion, translation and focal
length::

    root/
      <object_id>/
        multiview_00_<device>/
          annotations.json          camera.q (wxyz), camera.t (mm), split, ...
          images/000.jpg ...

**A pair task is a flat image dataset plus indices, not a widened cache.** This
class presents the *unique frames* its pairs refer to and puts the pairing in
:meth:`labels` as indices into itself, which is the move
:class:`~visbench.data.triplet.TwoAFCDataset` already makes for triplets. That
matters more here than there: at eight partners per anchor the split holds
50,519 pairs over 8,217 frames, so pairing by presentation would extract every
frame eight times and stack 100,000 feature vectors to hold 8,217 distinct
ones.

Three things about this data are protocol rather than detail, and the pairing
one has already been measured to move a board by degrees:

* **The pair count is part of the protocol**, like ``corner``'s pinned frame
  set. Error keeps falling as partners are added and the ordering only settles
  from two partners on, so two people's pose numbers are comparable *only* if
  they drew the same pairs. :meth:`describe` reports every sampler parameter
  for that reason, and :meth:`fingerprint` folds them in so two pair counts
  cannot land in one comparability group.
* **Validation stays at one partner however high ``partners`` goes.** It is
  the yardstick — the no-feature floor is measured on it — and growing both
  halves at once would change the measurement and the thing measured together,
  leaving no way to tell a real improvement from an easier split.
* **Translation is millimetres on disk**, and :data:`TRANSLATION_SCALE` is what
  makes it metres. See that constant: omitting it does not look like a units
  bug, it looks like every backbone being worse than a constant.
* **327 of the 8,217 frames carry an EXIF orientation tag, and the camera poses
  describe the image with that tag applied.** All 327 are tag 3, a half turn,
  and they sit in eight ``ipad_5`` scenes, seven of which are entirely tagged.
  Reading them as stored would hand the backbone an upside-down image beside a
  pose that says otherwise — 4.0% of the release supervised against its own
  negation, silently, since the frame still loads and the pair still scores.
  :func:`~visbench.utils.image.load_image` applies the tag, which is why this
  class uses it rather than opening the file directly.

  **That was checked rather than assumed**, because the opposite convention is
  just as plausible and reads the same: in the one scene where tagged and
  untagged frames sit together, the tagged cameras' up vectors are 0.7 to 22.8
  degrees from the untagged mean, inside the 6 to 50 degree spread of the
  untagged frames themselves, where the other convention would put them near
  180. ``scripts/premeasure_pose.py`` opened these files without the tag, so
  the numbers in ``visbench/tasks/low_level/README.md`` were measured on 4.0%
  different pixels; how much that moves them is 16a-2's to report.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

import torch
from PIL import Image

from visbench.data.base import BaseDataset
from visbench.metrics.pose import (
    pose_vector,
    quaternion_to_matrix,
    relative_pose,
    rotation_angle_deg,
)
from visbench.utils.image import load_image

__all__ = ["NaviPoseDataset", "PosePairs", "camera_matrix", "TRANSLATION_SCALE", "MAX_ANGLE"]

#: NAVI stores translation in millimetres, and the reference implementation
#: divides by this. **It is not cosmetic.** The head regresses the 7-vector
#: ``[quat, trans]`` under an MSE loss, so a translation of magnitude ~200
#: outweighs a unit quaternion by about four orders of magnitude: the head
#: optimises translation alone and rotation is never learned.
#:
#: Omitting it is what the first pre-measurement run did, and the failure is
#: worth keeping because it does not read as a units bug. Every backbone landed
#: at 107-109 degrees rotation error against a no-feature floor of 65.4 — worse
#: than predicting a constant — with ``train_loss`` 466-1001. **A trained head
#: cannot lose to a constant unless the loss is not optimising the scored
#: term**, which is the tell, and it is the same failure as 6d-1's
#: ``target_scale``: a gradient does not rescale itself to match its target.
TRANSLATION_SCALE = 1000.0

#: Largest relative rotation a pair may span, in degrees. probe3d's protocol as
#: the reference implementation writes it. Widening it does not merely add
#: harder pairs — it moves the no-feature floor, since the floor is the mean
#: relative pose of whatever was drawn, so a board at 150 degrees cannot be
#: read against one at 120.
MAX_ANGLE = 120.0


class PosePairs(NamedTuple):
    """What :meth:`NaviPoseDataset.labels` returns: the pairing and its target."""

    #: ``(P, 2)`` long, indexing the dataset's own frames — column 0 is the
    #: anchor, column 1 the partner. Indices rather than paths because features
    #: arrive in dataset order and a target that travelled any other way would
    #: drift from them the moment a loader shuffled. Named ``indices`` rather
    #: than ``index`` because a ``NamedTuple`` is a tuple, and ``index`` is one
    #: of the two methods a tuple already has.
    indices: torch.Tensor

    #: ``(P, 7)`` float, the relative pose of the partner *with respect to* the
    #: anchor, laid out as :data:`~visbench.metrics.pose.POSE_COLUMNS`,
    #: translation in metres.
    pose: torch.Tensor


def camera_matrix(annotation: dict) -> torch.Tensor:
    """A NAVI annotation's 4x4 world-to-camera matrix, translation in metres.

    Parameters
    ----------
    annotation : dict
        One record of an ``annotations.json``, carrying ``camera.q`` (wxyz) and
        ``camera.t`` (millimetres).

    Returns
    -------
    torch.Tensor
        ``(4, 4)`` in float64 — the composition and inversion that follow are
        where a relative pose loses digits, not the storage.
    """
    camera = annotation["camera"]
    q = torch.tensor(camera["q"], dtype=torch.float64)
    t = torch.tensor(camera["t"], dtype=torch.float64) / TRANSLATION_SCALE
    rt = torch.eye(4, dtype=torch.float64)
    rt[:3, :3] = quaternion_to_matrix(q)
    rt[:3, 3] = t
    return rt


class NaviPoseDataset(BaseDataset):
    """NAVI multi-view frames plus the pairs a relative-pose probe is scored on.

    Yields ``(pil_image, None)`` like any other image dataset; the pairs are in
    :meth:`labels`.

    **Pairs are built over the whole release and then filtered to one split**,
    not built per split. The sampler draws one partner for every anchor in
    scene order from a seeded generator, so the pairs a ``"val"`` dataset holds
    are bit-identical to the ones it would hold if the training split had never
    been asked for — and every backbone sees the same pairs, which is the only
    reason a pose board compares backbones rather than samplers.

    Parameters
    ----------
    root : str or pathlib.Path
        The NAVI release directory, holding one folder per object.
    split : str
        ``"train"`` or ``"val"``, read from each annotation's own ``split``
        field. NAVI's split is by *image*, not by object, so both splits see
        every scene.
    partners : int
        Distinct partners drawn per **training** anchor. One partner per anchor
        exhausts the pairing rule long before NAVI runs out of eligible views,
        and at that size the head overfits at ``n ~ d`` and two of four
        backbones read as being at chance — a candidate measured where the head
        cannot generalise looks exactly like one that does not rank. Validation
        is always one partner; see the module docstring.
    max_angle : float
        Largest relative rotation a pair may span, in degrees. See
        :data:`MAX_ANGLE`.
    pair_seed : int
        Seeds the partner draw. Part of the protocol, not a nuisance parameter.
    stride : int
        Keep every ``stride``-th anchor, for a cheap run. Applied before the
        extra partners are drawn, so ``stride`` and ``partners`` compose the
        way the pre-measurement ran them.
    image_size : int
        Short side, then a centre crop to this square.
    max_pairs : int or None
        Keep the first this many pairs. This is why :meth:`subset` is refused:
        the pairs index into the frame list, so the two have to be shortened
        together, at construction.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        partners: int = 1,
        max_angle: float = MAX_ANGLE,
        pair_seed: int = 8,
        stride: int = 1,
        image_size: int = 224,
        max_pairs: int | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root does not exist: {self.root}")
        if split not in ("train", "val"):
            raise ValueError(f"Unknown split {split!r}. NAVI's annotations carry train and val.")
        if partners < 1:
            raise ValueError(f"partners must be >= 1, got {partners}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")
        if not 0 < max_angle <= 180:
            raise ValueError(f"max_angle must be in (0, 180], got {max_angle}")

        self.name = self.root.name
        self.split = split
        self.max_angle = float(max_angle)
        self.pair_seed = pair_seed
        self.stride = stride
        self.image_size = image_size
        self.max_pairs = max_pairs

        #: What the caller asked for, kept beside what this split actually got,
        #: the way a record keeps ``pooling_requested`` beside the resolved
        #: pooling: a validation split reporting ``partners: 8`` would name a
        #: training budget it does not hold.
        self.partners_requested = partners
        self.partners = partners if split == "train" else 1

        pairs = self._build_pairs(partners)
        if max_pairs is not None:
            if max_pairs < 1:
                raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
            pairs = pairs[:max_pairs]
        if not pairs:
            raise ValueError(
                f"No pairs in split {split!r} of {self.root}: no anchor has a partner "
                f"within {max_angle} degrees. Check the root holds multiview_*/ scenes "
                "with an annotations.json each."
            )

        # One entry per distinct file, so a frame used by many pairs is
        # extracted once. Insertion-ordered, so the index a pair stores is
        # stable across runs and machines.
        index_of: dict[Path, int] = {}
        self.paths: list[Path] = []
        indices = []
        for pair in pairs:
            row = []
            for path in (pair["a"], pair["b"]):
                if path not in index_of:
                    index_of[path] = len(self.paths)
                    self.paths.append(path)
                row.append(index_of[path])
            indices.append(row)

        self._index = torch.tensor(indices, dtype=torch.long)
        self._pose = torch.stack([pair["pose"] for pair in pairs])

    def _build_pairs(self, partners: int) -> list[dict]:
        """Every pair of the release, filtered to this split.

        Walks the scenes in sorted order drawing one partner per anchor from a
        generator seeded by ``pair_seed``, applies ``stride``, then draws the
        extra training partners from a *second* generator. The second stream is
        separate so that the first partner of every anchor is drawn from an
        untouched sequence: a one-partner run and the first partner of an
        eight-partner run are then the same pairs, which is what makes the
        pair-count curve a curve through one experiment rather than five.
        """
        generator = torch.Generator().manual_seed(self.pair_seed)
        first: list[dict] = []
        contexts: list[dict] = []

        for annotation_path in sorted(self.root.glob("*/multiview_*/annotations.json")):
            scene = annotation_path.parent
            records = json.loads(annotation_path.read_text())
            usable = [r for r in records if (scene / "images" / r["filename"]).is_file()]
            if len(usable) < 2:
                continue

            rts = torch.stack([camera_matrix(record) for record in usable])

            for position, anchor in enumerate(usable):
                eligible = self._eligible(rts, position)
                if not bool(eligible.any()):
                    continue
                partner = int(torch.multinomial(eligible.double(), 1, generator=generator).item())
                first.append(
                    {
                        "a": scene / "images" / anchor["filename"],
                        "b": scene / "images" / usable[partner]["filename"],
                        "pose": self._pose_between(rts, position, partner),
                        "split": anchor.get("split", "train"),
                    }
                )
                contexts.append(
                    {
                        "scene": scene,
                        "usable": usable,
                        "rts": rts,
                        "position": position,
                        "taken": {partner},
                    }
                )

        if self.stride > 1:
            first, contexts = first[:: self.stride], contexts[:: self.stride]

        pairs = [pair for pair in first if pair["split"] == self.split]
        if partners <= 1 or self.split != "train":
            return pairs

        extra_generator = torch.Generator().manual_seed(self.pair_seed + 1)
        extras: list[dict] = []
        for pair, context in zip(first, contexts, strict=True):
            # Drawn for every training anchor in order, so the stream does not
            # depend on which anchors happened to be eligible for extras.
            if pair["split"] != "train":
                continue
            rts, position = context["rts"], context["position"]
            eligible = self._eligible(rts, position)
            for used in context["taken"]:
                eligible[used] = False
            wanted = min(partners - 1, int(eligible.sum()))
            if wanted <= 0:
                continue
            drawn = torch.multinomial(
                eligible.double(), wanted, replacement=False, generator=extra_generator
            )
            for partner in drawn.tolist():
                extras.append(
                    {
                        "a": pair["a"],
                        "b": context["scene"] / "images" / context["usable"][partner]["filename"],
                        "pose": self._pose_between(rts, position, partner),
                        "split": "train",
                    }
                )
        return pairs + extras

    def _eligible(self, rts: torch.Tensor, position: int) -> torch.Tensor:
        """Which views of a scene may partner the anchor at ``position``.

        Rotation only: a pair is admitted on how far the camera turned, never
        on how far it moved, which is probe3d's rule and is why a pose number
        is quoted against a floor measured on the same draw.
        """
        rotations = rts[:, :3, :3]
        relative = rotations[position].unsqueeze(0) @ rotations.transpose(-1, -2)
        trace = relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2]
        angle = torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)))
        eligible = (angle > 0) & (angle <= self.max_angle)
        eligible[position] = False
        return eligible

    @staticmethod
    def _pose_between(rts: torch.Tensor, anchor: int, partner: int) -> torch.Tensor:
        """The 7-vector taking the anchor camera's frame to the partner's."""
        return pose_vector(relative_pose(rts[anchor], rts[partner]))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, None]:
        """``(pil_image, None)`` — a single frame has no pose of its own to score."""
        return self._crop_image(load_image(self.paths[index])), None

    def _crop_image(self, image: Image.Image) -> Image.Image:
        """Resize the short side to ``image_size``, then centre-crop.

        Byte-identical to :meth:`~visbench.data.dense.DenseFolderDataset._crop_image`,
        copied rather than inherited because this class has no target folder to
        extend that one with, and pinned pixel for pixel by a test — the same
        guarantee the detection, instance and derived datasets give.
        """
        width, height = image.size
        scale = self.image_size / min(width, height)
        resized = image.resize(
            (
                max(self.image_size, round(width * scale)),
                max(self.image_size, round(height * scale)),
            ),
            Image.Resampling.BICUBIC,
        )
        left = (resized.width - self.image_size) // 2
        top = (resized.height - self.image_size) // 2
        return resized.crop((left, top, left + self.image_size, top + self.image_size))

    @property
    def pairs(self) -> PosePairs:
        """The pairs and their relative poses — what scoring this task needs."""
        return PosePairs(indices=self._index, pose=self._pose)

    def labels(self) -> PosePairs:
        """:attr:`pairs`."""
        return self.pairs

    def relative_rotation_deg(self) -> torch.Tensor:
        """``(P,)`` how far each pair's camera turned, in degrees.

        The difficulty of the draw, and the context a mean error is read
        against: pairs admitted to 120 degrees have a median near 65, which is
        why predicting a constant already scores about 67.
        """
        return rotation_angle_deg(self._pose[:, :4])

    def cache_identity(self, index: int) -> str | None:
        path = self.paths[index]
        try:
            stat = path.stat()
        except OSError:
            return None
        return f"{path}|{stat.st_size}|{stat.st_mtime_ns}"

    def fingerprint(self) -> str | None:
        """Covers the frames *and* every sampler parameter.

        Two pair counts over the same frames are two different measurements —
        error falls by degrees between them — so a fingerprint that saw only
        the file list would let them land in one comparability group and be
        ranked against each other.
        """
        digest = hashlib.sha256()
        digest.update(
            f"{self.name}|{self.split}|{self.partners}|{self.max_angle}|{self.pair_seed}|"
            f"{self.stride}|{self.image_size}|{len(self._index)}".encode()
        )
        for path in self.paths:
            # object / scene / file rather than the absolute path, so the same
            # release fingerprints the same on two machines.
            scene = path.parent.parent
            digest.update(f"{scene.parent.name}/{scene.name}/{path.name}|".encode())
        digest.update(self._index.numpy().tobytes())
        digest.update(self._pose.numpy().tobytes())
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        """Adds the pair count and every sampler parameter.

        ``dataset_size`` counts *frames* here, as it does for triplets, so the
        pair count has to be stated separately — and the sampler parameters
        with it, since they are the protocol a second run has to match.
        """
        described = super().describe()
        described["num_pairs"] = len(self._index)
        described["partners"] = self.partners
        described["partners_requested"] = self.partners_requested
        described["max_angle"] = self.max_angle
        described["pair_seed"] = self.pair_seed
        described["stride"] = self.stride
        described["image_size"] = self.image_size
        return described

    def subset(self, indices: int | Any) -> "NaviPoseDataset":
        """Not supported: frames and pairs cannot be sliced independently.

        Subsetting the frame list would leave pairs pointing at indices that
        have moved or gone, and every one of them would still *look* valid.
        Pass ``max_pairs=`` to the constructor for a shorter run — and note that
        shortening the *training* split changes the protocol rather than the
        run's cost, since the score has not converged in the pair count.
        """
        raise NotImplementedError(
            "NaviPoseDataset cannot be subset by frame: the pairs index into the frame "
            "list, so slicing it would silently repoint them. Pass max_pairs= to the "
            "constructor for a shorter run."
        )
