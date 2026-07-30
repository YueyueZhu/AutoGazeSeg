"""Masked binary segmentation objective."""

import torch
import torch.nn as nn


class BCEWithLogitsMaskLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.reduction = reduction

    def forward(self, input, target, mask=None):
        loss = self.bce(input, target.float())
        reduce_axis = list(range(2, len(input.shape)))

        if mask is not None:
            mask = mask.float()
            loss = torch.sum(loss * mask, dim=reduce_axis) / (
                torch.sum(mask, dim=reduce_axis) + 1e-8
            )
        else:
            loss = loss.mean(reduce_axis)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unsupported reduction: {self.reduction}")
