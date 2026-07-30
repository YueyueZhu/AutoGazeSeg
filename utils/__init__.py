"""Public utility interface used by AutoGazeSeg."""

from .losses import BCEWithLogitsMaskLoss
from .util import adjust_learning_rate, mkdir, mkdirs, setup_logger

__all__ = [
    "BCEWithLogitsMaskLoss",
    "adjust_learning_rate",
    "get_criterion",
    "mkdir",
    "mkdirs",
    "setup_logger",
]


def get_criterion(args):
    del args
    return BCEWithLogitsMaskLoss()
