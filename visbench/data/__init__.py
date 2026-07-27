"""Datasets.

v0.1 shipped a local image-folder loader and a correspondence pair interface
only — no dataset downloading, no benchmark-specific loaders. v0.2 adds the
dense-target loader the dense tasks need.
"""

from visbench.data.base import BaseDataset
from visbench.data.dense import (
    DenseFolderDataset,
    load_depth_map,
    load_label_map,
    load_mask,
    load_normal_map,
)
from visbench.data.image_folder import ImageFolderDataset
from visbench.data.pair_dataset import PairDataset
from visbench.data.triplet import TwoAFCDataset

__all__ = [
    "BaseDataset",
    "ImageFolderDataset",
    "PairDataset",
    "DenseFolderDataset",
    "load_depth_map",
    "load_normal_map",
    "load_mask",
    "load_label_map",
    "TwoAFCDataset",
]
