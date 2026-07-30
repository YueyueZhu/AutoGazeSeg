import csv
import os

import torch
import numpy as np
from PIL import Image
try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

from .base import BaseTrainer
from utils import mkdirs
from utils.util import adjust_learning_rate


TEST_METRIC_NAMES = ("dice", "iou", "hd95", "asd")
TEST_METRIC_ALIASES = {
    "dice": "mdice",
    "iou": "miou",
    "hd95": "hd95",
    "asd": "asd",
}
TEST_METRIC_DISPLAY = {
    "dice": "Dice",
    "iou": "IoU",
    "hd95": "HD95",
    "asd": "ASD",
}


def _case_name(subject_id):
    return os.path.splitext(os.path.basename(str(subject_id)))[0]


def _as_batch_channel_numpy(tensor):
    array = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    if array.ndim == 3:
        array = array[:, None, :, :]
    if array.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] or [B, H, W], got shape={array.shape}")
    return array


def _empty_surface_penalty(shape, spacing):
    spacing = np.asarray(spacing, dtype=np.float64)
    extent = (np.asarray(shape, dtype=np.float64) - 1.0) * spacing
    return float(max(np.sqrt(np.sum(extent * extent)), 1.0))


def compute_surface_metrics(pred_bin, label_bin, spacing=None):
    if ndi is None:
        raise ImportError("scipy is required for HD95/ASD in validate_save.")

    pred_bin = np.asarray(pred_bin).astype(bool)
    label_bin = np.asarray(label_bin).astype(bool)
    if pred_bin.shape != label_bin.shape:
        raise ValueError(f"Prediction/label shape mismatch: {pred_bin.shape} vs {label_bin.shape}")

    if spacing is None:
        spacing = (1.0,) * pred_bin.ndim

    pred_empty = not np.any(pred_bin)
    label_empty = not np.any(label_bin)
    if pred_empty and label_empty:
        return 0.0, 0.0
    if pred_empty or label_empty:
        penalty = _empty_surface_penalty(pred_bin.shape, spacing)
        return penalty, penalty

    structure = np.ones((3,) * pred_bin.ndim, dtype=bool)
    pred_surface = pred_bin ^ ndi.binary_erosion(pred_bin, structure=structure, border_value=0)
    label_surface = label_bin ^ ndi.binary_erosion(label_bin, structure=structure, border_value=0)

    pred_to_label = ndi.distance_transform_edt(~label_surface, sampling=spacing)[pred_surface]
    label_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)[label_surface]
    distances = np.concatenate([pred_to_label, label_to_pred]).astype(np.float64)
    if distances.size == 0:
        return 0.0, 0.0

    return float(np.percentile(distances, 95)), float(np.mean(distances))


def compute_test_case_metrics(prob, label, valid_mask=None, threshold=0.5, spacing=None):
    pred_bin = np.asarray(prob) > threshold
    label_bin = np.asarray(label) > 0.5

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask).astype(bool)
        pred_bin = pred_bin & valid_mask
        label_bin = label_bin & valid_mask

    intersection = float(np.logical_and(pred_bin, label_bin).sum())
    pred_sum = float(pred_bin.sum())
    label_sum = float(label_bin.sum())
    union = float(np.logical_or(pred_bin, label_bin).sum())

    dice = 1.0 if pred_sum + label_sum == 0 else 2.0 * intersection / (pred_sum + label_sum + 1e-7)
    iou = 1.0 if union == 0 else intersection / (union + 1e-7)
    hd95, asd = compute_surface_metrics(pred_bin, label_bin, spacing=spacing)

    metrics = {
        "dice": float(dice),
        "iou": float(iou),
        "hd95": float(hd95),
        "asd": float(asd),
    }
    return metrics, pred_bin.astype(np.uint8), label_bin.astype(np.uint8)


def save_test_case_prediction(save_root, case_name, pred_bin, label_bin=None):
    pred_dir = os.path.join(save_root, "pred_png")
    mkdirs([pred_dir])

    pred_path = os.path.join(pred_dir, f"{case_name}.png")
    Image.fromarray((pred_bin.astype(np.uint8) * 255)).save(pred_path)

    label_path = ""
    if label_bin is not None:
        label_dir = os.path.join(save_root, "label_png")
        mkdirs(label_dir)
        label_path = os.path.join(label_dir, f"{case_name}.png")
        Image.fromarray((label_bin.astype(np.uint8) * 255)).save(label_path)

    return {
        "pred_path": pred_path,
        "label_path": label_path,
    }


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def _format_mean_std(mean, std):
    return f"{mean:.4f} +/- {std:.4f}"


def summarize_test_metrics(rows, suffix=""):
    summary = {}
    for metric_name in TEST_METRIC_NAMES:
        mean, std = _mean_std([row[metric_name] for row in rows])
        alias = TEST_METRIC_ALIASES[metric_name]
        display = TEST_METRIC_DISPLAY[metric_name]

        summary[f"{alias}{suffix}"] = mean
        summary[f"{alias}_std{suffix}"] = std
        summary[f"{display}_mean_std{suffix}"] = _format_mean_std(mean, std)
    return summary


def write_test_metric_tables(save_root, rows, summary):
    mkdirs(save_root)

    per_case_path = os.path.join(save_root, "per_case_metrics.csv")
    fieldnames = ["case", *TEST_METRIC_NAMES, "pred_path", "label_path"]
    with open(per_case_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    summary_path = os.path.join(save_root, "summary_metrics.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "mean_std"])
        writer.writeheader()
        for metric_name in TEST_METRIC_NAMES:
            alias = TEST_METRIC_ALIASES[metric_name]
            display = TEST_METRIC_DISPLAY[metric_name]
            writer.writerow(
                {
                    "metric": display,
                    "mean": summary[alias],
                    "std": summary[f"{alias}_std"],
                    "mean_std": summary[f"{display}_mean_std"],
                }
            )

    return per_case_path, summary_path


def print_test_metric_summary(summary, title="test"):
    metric_text = []
    for metric_name in TEST_METRIC_NAMES:
        display = TEST_METRIC_DISPLAY[metric_name]
        key = f"{display}_mean_std"
        if key in summary:
            metric_text.append(f"{display}: {summary[key]}")

    print(f"[{title}] " + ", ".join(metric_text), flush=True)
    if "summary_metrics" in summary:
        print(f"[{title}] summary_metrics: {summary['summary_metrics']}", flush=True)
    if "per_case_metrics" in summary:
        print(f"[{title}] per_case_metrics: {summary['per_case_metrics']}", flush=True)
    if "save_root" in summary:
        print(f"[{title}] save_root: {summary['save_root']}", flush=True)


class SegTrainer(BaseTrainer):
    def __init__(self, *argv, **kargs):
        super().__init__(*argv, **kargs)

        self.main_metric = "mdice"

    def save_pred_batch(self, pred, save_root, save_filenames):
        if torch.is_tensor(pred):
            pred = torch.sigmoid(pred).detach().cpu().numpy()
        for i_b in range(len(save_filenames)):
            save_pred = pred[i_b, 0].astype(np.float32)
            save_path = os.path.join(save_root, f"{os.path.splitext(save_filenames[i_b])[0]}.npy")
            mkdirs(os.path.dirname(save_path))
            np.save(save_path, save_pred)

    def _epoch_begin_hook(self):
        if self.args.lr_scheduler is not None:
            if self.args.lr_scheduler == "cos":
                adjust_learning_rate(
                    self.optimizer, self.epoch, self.total_epoch, self.args.lr, self.args.lr_min, self.args.warmup_ite
                )
            else:
                raise NotImplementedError

        return super()._epoch_begin_hook()
