"""Datasets.

v0.1 shipped a local image-folder loader and a correspondence pair interface
only — no dataset downloading, no benchmark-specific loaders. v0.2 adds the
dense-target loader the dense tasks need.
"""

from visbench.data.base import BaseDataset
from visbench.data.bridges import HuggingFaceDataset, TorchvisionDataset
from visbench.data.bsds import BSDS500Dataset
from visbench.data.dense import (
    DenseFolderDataset,
    load_depth_map,
    load_edge_map,
    load_label_map,
    load_mask,
    load_normal_map,
)
from visbench.data.detection import VOC_CLASSES, DetectionFolderDataset, load_voc_boxes
from visbench.data.image_folder import ImageFolderDataset
from visbench.data.pair_dataset import HomographyPairDataset, PairDataset, PairViewDataset
from visbench.data.taskonomy import (
    TASKONOMY_DOMAINS,
    TASKONOMY_SUPPORTED_DOMAINS,
    TaskonomyDataset,
    load_taskonomy_split,
    load_valid_mask,
)
from visbench.data.triplet import TwoAFCDataset

__all__ = [
    "BaseDataset",
    "ImageFolderDataset",
    "TorchvisionDataset",
    "HuggingFaceDataset",
    "PairDataset",
    "HomographyPairDataset",
    "PairViewDataset",
    "DenseFolderDataset",
    "BSDS500Dataset",
    "load_depth_map",
    "load_normal_map",
    "load_mask",
    "load_label_map",
    "load_edge_map",
    "TwoAFCDataset",
    "DetectionFolderDataset",
    "load_voc_boxes",
    "VOC_CLASSES",
    "TaskonomyDataset",
    "load_taskonomy_split",
    "TASKONOMY_DOMAINS",
    "TASKONOMY_SUPPORTED_DOMAINS",
    "load_valid_mask",
]
