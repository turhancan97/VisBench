"""Two-alternative-forced-choice triplet datasets — v0.2.

Mid-level image similarity is scored on triplets, not pairs: a reference and two
candidates, with a human judgement of which candidate looks more like the
reference. That judgement is *perceptual* — layout, pose, structure — rather
than categorical, which is exactly what separates this from high-level
retrieval, and why the two are deliberately distinct tasks (CLAUDE.md).

The awkward part is that a triplet is three images while the feature cache is
built around one image at a time. Rather than widen the cache, this dataset
presents itself as a **flat collection of unique images**, and puts the triplet
structure in :meth:`labels` as *indices into itself*. So the cache, the
fingerprint and :func:`visbench.run` all work unchanged, features are extracted
once per image however many triplets use it, and the pairing travels by index —
never by iteration order, which is the failure mode CLAUDE.md warns about.
"""

import csv
import hashlib
from pathlib import Path
from typing import Any

import torch

from visbench.data.base import BaseDataset
from visbench.utils.image import load_image

__all__ = ["TwoAFCDataset", "NIGHTS_MIN_VOTES"]

#: Triplets with fewer agreeing votes than this are dropped. Six of eight is the
#: filter the reference implementation applies, and it is a real protocol
#: parameter rather than a detail: relaxing it admits triplets humans disagreed
#: about, which lowers every model's score without telling you anything about
#: the models. Recorded in ``task_params`` for that reason.
NIGHTS_MIN_VOTES = 6

#: Column order of the triplet tensor returned by :meth:`TwoAFCDataset.labels`.
_TRIPLET_COLUMNS = ("ref", "left", "right", "vote")


class TwoAFCDataset(BaseDataset):
    """NIGHTS-style triplets: ``ref``, two candidates, and a human vote.

    Expects the layout the NIGHTS release ships (Fu et al., *DreamSim*,
    NeurIPS 2023)::

        root/
          data.csv                 id, left_vote, right_vote, votes,
                                   ref_path, left_path, right_path, split, ...
          ref/000/002.png
          distort/000/002_0.png
          distort/000/002_1.png

    Yields ``(pil_image, None)`` like any other image dataset; the triplets are
    in :meth:`labels`.

    Columns are read **by name**. The reference implementation indexes them
    positionally (``iloc[idx, 2]`` for the vote, ``4``/``5``/``6`` for the
    paths), which silently reads the wrong column if anyone ever reorders the
    CSV — and the failure would look like a mediocre score, not an error.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        min_votes: int = NIGHTS_MIN_VOTES,
        max_triplets: int | None = None,
        annotations: str = "data.csv",
    ) -> None:
        """Index the CSV and collect the images its triplets refer to.

        Parameters
        ----------
        split:
            ``train``, ``val``, ``test``, or the two documented test subsets
            ``test_imagenet`` / ``test_no_imagenet``. The last two exist because
            a backbone pretrained on ImageNet has seen the reference images in
            one of them, so a gap between the two is a contamination signal
            rather than a similarity result.
        min_votes:
            Keep only triplets where at least this many human votes agreed.
        max_triplets:
            Keep the first N triplets, for a quick run. This is why
            :meth:`subset` is refused: the triplets index into the image list,
            so the two have to be shortened together, at construction.
        """
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root does not exist: {self.root}")

        table = self.root / annotations
        if not table.is_file():
            raise FileNotFoundError(
                f"No annotation table at {table}. A 2AFC dataset needs the CSV naming "
                "each triplet; images alone do not say which candidate was preferred."
            )

        self.name = self.root.name
        self.split = split
        self.min_votes = min_votes
        self.max_triplets = max_triplets

        rows = self._select(table, split, min_votes)
        if max_triplets is not None:
            if max_triplets < 1:
                raise ValueError(f"max_triplets must be >= 1, got {max_triplets}")
            rows = rows[:max_triplets]
        if not rows:
            raise ValueError(
                f"No triplets in split {split!r} with at least {min_votes} agreeing votes"
            )

        # One entry per distinct file, so a reference used by several triplets
        # is extracted once. Insertion-ordered, so the index a triplet stores is
        # stable across runs and machines.
        index_of: dict[str, int] = {}
        self.paths: list[Path] = []
        triplets = []
        for row in rows:
            indices = []
            for column in ("ref_path", "left_path", "right_path"):
                relative = row[column]
                if relative not in index_of:
                    index_of[relative] = len(self.paths)
                    self.paths.append(self.root / relative)
                indices.append(index_of[relative])
            triplets.append([*indices, int(row["right_vote"])])

        self._triplets = torch.tensor(triplets, dtype=torch.long)

    @staticmethod
    def _select(table: Path, split: str, min_votes: int) -> list[dict]:
        """Rows for ``split`` that clear ``min_votes``."""
        with table.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        missing = {"right_vote", "votes", "ref_path", "left_path", "right_path", "split"} - set(
            rows[0] if rows else {}
        )
        if missing:
            raise ValueError(f"{table.name} is missing column(s): {sorted(missing)}")

        rows = [row for row in rows if int(row["votes"]) >= min_votes]

        if split in ("train", "val", "test"):
            return [row for row in rows if row["split"] == split]
        if split in ("test_imagenet", "test_no_imagenet"):
            wanted = str(split == "test_imagenet").upper()
            return [
                row
                for row in rows
                if row["split"] == "test" and str(row["is_imagenet"]).upper() == wanted
            ]
        raise ValueError(
            f"Unknown split {split!r}. Expected train, val, test, test_imagenet or "
            "test_no_imagenet."
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, None]:
        """``(pil_image, None)`` — the label of a single image is meaningless here."""
        return load_image(self.paths[index]), None

    @property
    def triplets(self) -> torch.Tensor:
        """``(T, 4)`` of ``(ref, left, right, vote)``; the first three index ``self``.

        ``vote`` is 1 when humans judged the **right** candidate more similar,
        matching the reference implementation's use of the ``right_vote``
        column, so a prediction of "left" is 0.
        """
        return self._triplets

    def labels(self) -> torch.Tensor:
        """The triplets, which is what scoring this task needs."""
        return self._triplets

    def cache_identity(self, index: int) -> str | None:
        path = self.paths[index]
        try:
            stat = path.stat()
        except OSError:
            return None
        return f"{path}|{stat.st_size}|{stat.st_mtime_ns}"

    def fingerprint(self) -> str | None:
        """Covers the images *and* the triplet structure and vote filter.

        Two splits can share images and differ entirely in what they ask, so
        hashing the file list alone would let one split's record look like
        another's.
        """
        digest = hashlib.sha256()
        digest.update(f"{self.name}|{self.split}|{self.min_votes}|{len(self._triplets)}".encode())
        for path in self.paths:
            digest.update(f"{path.name}|".encode())
        digest.update(self._triplets.numpy().tobytes())
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        """Adds the triplet count, since ``dataset_size`` counts images here."""
        described = super().describe()
        described["num_triplets"] = len(self._triplets)
        return described

    def subset(self, indices: int | Any) -> "TwoAFCDataset":
        """Not supported: images and triplets cannot be sliced independently.

        Subsetting the image list would leave triplets pointing at indices that
        have moved or gone, and every one of them would still *look* valid.
        Build a smaller dataset with ``max_triplets`` instead.
        """
        raise NotImplementedError(
            "TwoAFCDataset cannot be subset by image: the triplets index into the "
            "image list, so slicing it would silently repoint them. Pass "
            "max_triplets= to the constructor for a shorter run."
        )
