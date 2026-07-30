"""Shared dataset utilities and image preprocessing."""

import os
from pathlib import Path

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandFlipd,
    Resized,
    ToTensord,
)
from torch.utils.data import Dataset


def read_split(root, split):
    """Read sample identifiers from ``<root>/<split>.txt``."""
    split_path = Path(root) / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Missing {split} split file: {split_path}. "
            "Each line must begin with a sample identifier."
        )
    identifiers = [
        line.split()[0] for line in split_path.read_text().splitlines() if line.split()
    ]
    if not identifiers:
        raise ValueError(f"Split file contains no samples: {split_path}")
    return identifiers


def require_files(paths, description):
    """Raise a focused error for the first missing prepared input."""
    missing = next((Path(path) for path in paths if not Path(path).is_file()), None)
    if missing is not None:
        raise FileNotFoundError(f"Missing {description}: {missing}")


def load_embedding(path):
    embedding = np.load(path)
    if embedding.ndim < 2:
        raise ValueError(
            f"Expected a token embedding array, got shape {embedding.shape}: {path}"
        )
    return embedding.astype(np.float32, copy=False)


class BaseImageDataset(Dataset):
    """Base class for paired 2D images, masks, and text embeddings."""

    def __init__(
        self,
        root,
        embedding_root,
        split,
        spatial_size=(384, 384),
        do_augmentation=False,
        size_rate=1.0,
        resize_label=True,
    ):
        super().__init__()
        self.root = Path(root)
        self.embedding_root = Path(embedding_root)
        self.split = split
        self.spatial_size = (
            (spatial_size, spatial_size)
            if isinstance(spatial_size, int)
            else tuple(spatial_size)
        )
        self.do_augmentation = do_augmentation
        self.size_rate = size_rate
        self.resize_label = resize_label
        self.img_norm_cfg = {
            "mean": np.array([123.675, 116.28, 103.53]),
            "std": np.array([58.395, 57.12, 57.375]),
        }

        identifiers = read_split(self.root, split)
        sample_count = max(1, int(len(identifiers) * size_rate))
        self.sample_list = identifiers[:sample_count]
        self.images = []
        self.labels = []
        self.embeddings = []
        self.transform = self.get_transform()

    def validate_prepared_inputs(self):
        require_files(self.images, "image")
        require_files(self.labels, "ground-truth mask")
        require_files(self.embeddings, "text embedding")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        data = self._transform_custom(self.transform(self._fetch_data(idx)))
        return {
            "idx": idx,
            "subject_id": self.sample_list[idx],
            "path": os.path.basename(self.images[idx]),
        } | data

    def get_transform(self):
        resize_keys = ["image", "label"] if self.resize_label else ["image"]
        resize_mode = ["bilinear", "nearest"] if self.resize_label else ["bilinear"]
        transforms = [
            EnsureChannelFirstd(keys=["image"], channel_dim=2),
            EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
            NormalizeIntensityd(
                keys=["image"],
                subtrahend=self.img_norm_cfg["mean"],
                divisor=self.img_norm_cfg["std"],
                channel_wise=True,
            ),
            Resized(keys=resize_keys, spatial_size=self.spatial_size, mode=resize_mode),
        ]
        if self.split == "train" and self.do_augmentation:
            transforms.extend(
                [
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                ]
            )
        transforms.append(ToTensord(keys=["image", "label"]))
        return Compose(transforms)

    def _transform_custom(self, data):
        return data

    def _fetch_data(self, idx):
        raise NotImplementedError
