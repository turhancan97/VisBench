"""Datasets.

v0.1 ships a local image-folder loader and a correspondence pair interface
only — no dataset downloading, no benchmark-specific loaders. Those arrive
alongside the v0.2 tasks that need them.
"""

from visbench.data.base import BaseDataset
from visbench.data.image_folder import ImageFolderDataset
from visbench.data.pair_dataset import PairDataset

__all__ = ["BaseDataset", "ImageFolderDataset", "PairDataset"]
