"""Command-line interface for AutoGazeSeg training and evaluation."""

import argparse

DATASET_CHOICES = ("kvasir", "nci", "isic", "brats2019")


def args_parser(argv=None):
    parser = argparse.ArgumentParser(
        description="Train or evaluate AutoGazeSeg on one of the four paper datasets."
    )

    # Only the proposed method is exposed in the anonymous release.
    parser.add_argument(
        "-m",
        "--method",
        choices=("autogazeseg",),
        default="autogazeseg",
    )
    parser.add_argument(
        "--model",
        choices=("autogazeseg",),
        default="autogazeseg",
    )
    parser.add_argument("--data", choices=DATASET_CHOICES, required=True)

    parser.add_argument(
        "--root",
        "--data_root",
        dest="root",
        default=None,
        help="Dataset root. Defaults to data/<dataset> inside the repository.",
    )
    parser.add_argument(
        "--embedding_root",
        default=None,
        help="Text-embedding directory. Defaults to <root>/embeddings.",
    )
    parser.add_argument(
        "--pseudo_high_root",
        default=None,
        help="High-confidence pseudo-mask directory. Defaults to <root>/pseudo_masks/labelshigh.",
    )
    parser.add_argument(
        "--pseudo_low_root",
        default=None,
        help="Low-confidence pseudo-mask directory. Defaults to <root>/pseudo_masks/labelslow.",
    )
    parser.add_argument(
        "--exp_result_path",
        default="outputs/results",
        help="Directory for predictions and evaluation outputs.",
    )
    parser.add_argument(
        "--log_path",
        default="outputs/logs",
        help="Directory for training logs.",
    )
    parser.add_argument(
        "--params_path",
        default="outputs/checkpoints",
        help="Directory for trained checkpoints.",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Checkpoint file or directory used by --test or --resume.",
    )
    parser.add_argument(
        "--save_name",
        default="predictions",
        help="Subdirectory used when saving test predictions.",
    )

    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--spatial_size", type=int, default=224)
    parser.add_argument("--opt", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", choices=("cos",), default="cos")
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    parser.add_argument("--data_size_rate", type=float, default=1.0)
    parser.add_argument("--max_ite", type=int, default=15000)
    parser.add_argument("--warmup_ite", type=int, default=0)
    parser.add_argument("-bs", "--batch_size", type=int, default=8)
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--val_step", type=int, default=100)
    parser.add_argument(
        "--ppc_weight",
        type=float,
        default=1.0,
        help="Weight lambda_p for prototype-guided pseudo-mask calibration.",
    )
    parser.add_argument(
        "--feature_distillation_weight",
        type=float,
        default=0.5,
        help="Weight lambda_c for region-aware feature distillation.",
    )
    parser.add_argument(
        "--image_proto_blend",
        type=float,
        default=0.25,
        help="Image-appearance blending weight alpha used by PPC.",
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--device", default="0")

    args = parser.parse_args(argv)
    if not 0 < args.data_size_rate <= 1:
        parser.error("--data_size_rate must be in the interval (0, 1].")
    if args.max_ite < 1:
        parser.error("--max_ite must be at least 1.")
    if args.ppc_weight < 0:
        parser.error("--ppc_weight must be non-negative.")
    if args.feature_distillation_weight < 0:
        parser.error("--feature_distillation_weight must be non-negative.")
    if not 0 <= args.image_proto_blend <= 1:
        parser.error("--image_proto_blend must be in the interval [0, 1].")
    if args.test and not args.ckpt_path:
        parser.error("--ckpt_path is required when --test is set.")
    if args.resume and not args.ckpt_path:
        parser.error("--ckpt_path is required when --resume is set.")
    return args
