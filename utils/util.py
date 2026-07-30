"""Filesystem, logging, and optimization helpers."""

import logging
import math
import os


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def mkdirs(paths):
    if isinstance(paths, (str, os.PathLike)):
        mkdir(paths)
        return
    for path in paths:
        mkdir(path)


def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    mkdir(root)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    if tofile:
        file_handler = logging.FileHandler(os.path.join(root, f"{phase}.log"), mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if screen:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def adjust_learning_rate(optimizer, epoch, total_epoch, base_lr, min_lr, warmup_ite):
    """Apply linear warmup followed by half-cycle cosine decay."""
    if warmup_ite > 0 and epoch < warmup_ite:
        learning_rate = base_lr * epoch / warmup_ite
    else:
        progress = (epoch - warmup_ite) / max(1, total_epoch - warmup_ite)
        learning_rate = min_lr + (base_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
    for current_optimizer in optimizers:
        for parameter_group in current_optimizer.param_groups:
            parameter_group["lr"] = learning_rate
    return learning_rate
