"""Dataset factory for the four AutoGazeSeg benchmarks."""

from torch.utils.data import DataLoader

from .brats2019_seg import BraTS2019Dataset, BraTS2019GazeDataset
from .isic_seg import ISICDataset, ISICGazeDataset
from .kvasir_seg import KvasirDataset, KvasirGazeDataset
from .nci_isbi import NCIISBIDataset, NCIISBIGazeDataset

EVALUATION_DATASETS = {
    "kvasir": KvasirDataset,
    "nci": NCIISBIDataset,
    "isic": ISICDataset,
    "brats2019": BraTS2019Dataset,
}
TRAINING_DATASETS = {
    "kvasir": KvasirGazeDataset,
    "nci": NCIISBIGazeDataset,
    "isic": ISICGazeDataset,
    "brats2019": BraTS2019GazeDataset,
}


def get_dataloader(args, split):
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    common = {
        "root": args.root,
        "embedding_root": args.embedding_root,
        "split": split,
        "spatial_size": args.spatial_size,
        "do_augmentation": split == "train",
        "resize_label": split == "train",
        "size_rate": args.data_size_rate,
    }
    if split == "train":
        dataset = TRAINING_DATASETS[args.data](
            pseudo_high_root=args.pseudo_high_root,
            pseudo_low_root=args.pseudo_low_root,
            **common,
        )
    else:
        dataset = EVALUATION_DATASETS[args.data](**common)

    return DataLoader(
        dataset,
        batch_size=args.batch_size if split == "train" else 1,
        shuffle=split == "train",
        num_workers=args.num_worker,
    )
