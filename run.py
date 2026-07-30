"""Main entry point for AutoGazeSeg training and evaluation."""

import os
import random
from pathlib import Path

import numpy as np

from parse_args import args_parser

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIRECTORIES = {
    "kvasir": "Kvasir-SEG",
    "nci": "NCI-ISBI",
    "isic": "ISIC",
    "brats2019": "BraTS2019",
}


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def init_arguments(args):
    """Resolve portable paths and derive the run identifier."""
    if args.root is None:
        args.root = PROJECT_ROOT / "data" / DATASET_DIRECTORIES[args.data]
    else:
        args.root = _project_path(args.root)

    args.embedding_root = (
        _project_path(args.embedding_root)
        if args.embedding_root is not None
        else args.root / "embeddings"
    )
    args.pseudo_high_root = (
        _project_path(args.pseudo_high_root)
        if args.pseudo_high_root is not None
        else args.root / "pseudo_masks" / "labelshigh"
    )
    args.pseudo_low_root = (
        _project_path(args.pseudo_low_root)
        if args.pseudo_low_root is not None
        else args.root / "pseudo_masks" / "labelslow"
    )
    args.exp_result_path = _project_path(args.exp_result_path)
    args.log_path = _project_path(args.log_path)
    args.params_path = _project_path(args.params_path)
    if args.ckpt_path is not None:
        args.ckpt_path = _project_path(args.ckpt_path)

    if not args.root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {args.root}. "
            "Set --root to the prepared dataset directory."
        )
    if (args.test or args.resume) and not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {args.ckpt_path}")

    for path in (args.exp_result_path, args.log_path, args.params_path):
        path.mkdir(parents=True, exist_ok=True)

    args.experiment_name = f"AutoGazeSeg_{args.data}_seed{args.seed}"
    args.run_id = args.experiment_name

    # Downstream components accept path-like strings on all supported Python versions.
    for name in (
        "root",
        "embedding_root",
        "pseudo_high_root",
        "pseudo_low_root",
        "exp_result_path",
        "log_path",
        "params_path",
        "ckpt_path",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, str(value))
    return args


def _seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv=None):
    args = init_arguments(args_parser(argv))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Import CUDA-aware modules only after selecting visible devices.
    import torch

    from datasets import get_dataloader
    from models import get_model_opt
    from trainers import get_trainer_class
    from utils import get_criterion, setup_logger

    logger = setup_logger(
        "autogazeseg",
        args.log_path,
        args.experiment_name,
        screen=True,
        tofile=not args.test,
    )
    logger.info("Configuration: %s", args)
    _seed_everything(args.seed, torch)

    test_loader = get_dataloader(args, split="test")
    train_loader = None if args.test else get_dataloader(args, split="train")

    model, optimizer = get_model_opt(args)
    trainer = get_trainer_class(args)(
        args=args,
        logger=logger,
        model=model,
        optimizer=optimizer,
        criterion=get_criterion(args),
        train_dataloader=train_loader,
        test_dataloader=test_loader,
    )

    if args.test:
        trainer.load_for_eval(args.ckpt_path)
        performance = trainer.validate_save(
            test_loader,
            save_root=os.path.join(args.exp_result_path, args.save_name),
        )
        trainer.report_progress_result(performance, mode="test", use_wandb=args.wandb)
        return performance

    trainer.run()
    return None


if __name__ == "__main__":
    main()
