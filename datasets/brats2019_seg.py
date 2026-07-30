"""BraTS 2019 image and dual-pseudo-mask datasets."""

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
from PIL import Image

from .base_dataset import BaseImageDataset, load_embedding, require_files


class BraTS2019Dataset(BaseImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images = [
            self.root / "images" / f"{name}.png" for name in self.sample_list
        ]
        self.labels = [self.root / "masks" / f"{name}.png" for name in self.sample_list]
        self.embeddings = [
            self.embedding_root / f"{name}.npy" for name in self.sample_list
        ]
        self.validate_prepared_inputs()

    def _fetch_data(self, idx):
        return {
            "image": np.asarray(
                Image.open(self.images[idx]).convert("L"), dtype=np.float32
            ),
            "label": np.asarray(
                Image.open(self.labels[idx]).convert("L"), dtype=np.int16
            ),
            "embedding": load_embedding(self.embeddings[idx]),
        }

    def _transform_custom(self, data):
        data["label"] = (data["label"].float() / 255.0).long()
        return data

    def get_transform(self):
        resize_keys = ["image", "label"] if self.resize_label else ["image"]
        resize_mode = ["bilinear", "nearest"] if self.resize_label else ["bilinear"]
        transforms = [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            NormalizeIntensityd(keys=["image"]),
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


class BraTS2019GazeDataset(BraTS2019Dataset):
    """Training dataset with high- and low-confidence AutoGazeSeg targets."""

    def __init__(self, pseudo_high_root, pseudo_low_root, *args, **kwargs):
        self.num_levels = 2
        super().__init__(*args, **kwargs)
        self.labelshigh = [
            Path(pseudo_high_root) / f"{name}.png" for name in self.sample_list
        ]
        self.labelslow = [
            Path(pseudo_low_root) / f"{name}.png" for name in self.sample_list
        ]
        require_files(self.labelshigh, "high-confidence pseudo mask")
        require_files(self.labelslow, "low-confidence pseudo mask")

    def get_transform(self):
        pseudo_keys = [f"pseudo_label_{index + 1}" for index in range(self.num_levels)]
        resize_keys = (
            ["image", "label", *pseudo_keys] if self.resize_label else ["image"]
        )
        resize_mode = (
            ["bilinear", "nearest", *("bilinear" for _ in pseudo_keys)]
            if self.resize_label
            else ["bilinear"]
        )
        transforms = [
            EnsureChannelFirstd(
                keys=["image", "label", *pseudo_keys], channel_dim="no_channel"
            ),
            NormalizeIntensityd(keys=["image"]),
            Resized(keys=resize_keys, spatial_size=self.spatial_size, mode=resize_mode),
        ]
        if self.split == "train" and self.do_augmentation:
            transforms.extend(
                [
                    RandFlipd(
                        keys=["image", "label", *pseudo_keys], prob=0.5, spatial_axis=0
                    ),
                    RandFlipd(
                        keys=["image", "label", *pseudo_keys], prob=0.5, spatial_axis=1
                    ),
                ]
            )
        transforms.append(ToTensord(keys=["image", "label", *pseudo_keys]))
        return Compose(transforms)

    def _fetch_data(self, idx):
        data = super()._fetch_data(idx)
        high = (
            np.asarray(Image.open(self.labelshigh[idx]).convert("L"), dtype=np.uint8)
            > 0
        )
        low = (
            np.asarray(Image.open(self.labelslow[idx]).convert("L"), dtype=np.uint8) > 0
        )
        data["pseudo_label_1"] = np.logical_and(high, low).astype(np.uint8)
        data["pseudo_label_2"] = np.logical_or(high, low).astype(np.uint8)
        return data
