"""Foreground segmentation metrics."""

import torch


def compute_iou(input, target, mask=None, include_background=False, do_threshold=False):
    input = torch.sigmoid(input)
    if do_threshold:
        input = (input > 0.5).float()
    target = target.float()

    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.int()

    assert input.shape == target.shape

    reduce_axis = list(range(2, len(input.shape)))

    intersection = torch.sum(input * target * mask, dim=reduce_axis)
    input_o = torch.sum(input * mask, dim=reduce_axis)
    target_o = torch.sum(target * mask, dim=reduce_axis)

    union = input_o + target_o - intersection

    if include_background and input.shape[1] > 1:
        intersection = intersection[:, 1:]
        union = union[:, 1:]

    iou = torch.mean(intersection / (union + 1e-7), dim=1)

    return iou


def compute_acc(input, target, mask=None, include_background=False):
    input = (torch.sigmoid(input) > 0.5).float()
    target = target.float()

    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.int()

    assert input.shape == target.shape

    reduce_axis = list(range(2, len(input.shape)))

    corrects = (input == target).float()
    corrects = torch.sum(corrects * mask, dim=reduce_axis)

    divison = torch.sum(mask, dim=reduce_axis)

    if include_background and input.shape[1] > 1:
        corrects = corrects[:, 1:]
        divison = divison[:, 1:]

    iou = torch.mean(corrects / (divison + 1e-7), dim=1)

    return iou


def compute_dice(
    input, target, mask=None, include_background=False, do_threshold=False
):
    input = torch.sigmoid(input)
    if do_threshold:
        input = (input > 0.5).float()
    target = target.float()

    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.int()

    assert input.shape == target.shape

    reduce_axis = list(range(2, len(input.shape)))

    intersection = torch.sum(input * target * mask, dim=reduce_axis)
    input_o = torch.sum(input * mask, dim=reduce_axis)
    target_o = torch.sum(target * mask, dim=reduce_axis)

    if include_background and input.shape[1] > 1:
        intersection = intersection[:, 1:]
        input_o = input_o[:, 1:]
        target_o = target_o[:, 1:]

    dice = torch.mean(2 * intersection / (input_o + target_o + 1e-7), dim=1)

    return dice
