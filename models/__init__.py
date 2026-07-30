import torch

from .autogazeseg import AutoGazeSeg


def _make_optimizer(args, model):
    if model is None:
        return None

    if args.opt == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.opt == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    raise NotImplementedError(f"Optimizer {args.opt} not implemented")


def _build_autogazeseg_pair(args):
    def make_one():
        return AutoGazeSeg(
            in_channels=args.in_channels,
            out_channels=2,
            feat_dim=128,
            text_embedding_dim=2560,
            adapter_channels=256,
            beta=2.35,
            gate_init=-2.0,
            fusion_dropout=0.0,
            max_text_tokens=128,
        )

    return [make_one(), make_one()]


def get_model_opt(args):
    if args.method != "autogazeseg":
        raise NotImplementedError(f"Unknown method: {args.method}")
    if args.model != "autogazeseg":
        raise NotImplementedError(f"Unknown model: {args.model}")

    models = _build_autogazeseg_pair(args)
    optimizers = [_make_optimizer(args, model) for model in models]
    return models, optimizers


__all__ = ["AutoGazeSeg", "get_model_opt"]
