"""Image-pair dataset for correspondence — v0.1.

Correspondence is the one v0.1 task that consumes pairs plus geometry rather
than single images and labels, so it gets its own interface instead of being
forced into :class:`ImageFolderDataset`.
"""

from typing import Any

from visbench.data.base import BaseDataset

__all__ = ["PairDataset"]


class PairDataset(BaseDataset):
    """Yields ``(image_0, image_1, geometry)`` for correspondence evaluation.

    ``geometry`` carries whatever the evaluation protocol needs to verify a
    match — for a depth-and-pose dataset that is depth maps, relative pose and
    intrinsics; for a keypoint dataset, annotated point pairs. Kept as an
    opaque dict here so adding a dataset does not change this interface.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[Any, Any, dict]:
        """Return ``(pil_image_0, pil_image_1, geometry_dict)``."""
        raise NotImplementedError
