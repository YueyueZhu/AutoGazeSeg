from .autogazeseg_trainer import AutoGazeSegTrainer


def get_trainer_class(_args):
    """Return the AutoGazeSeg trainer included in this anonymous release."""
    return AutoGazeSegTrainer
