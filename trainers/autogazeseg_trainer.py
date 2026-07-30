import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .seg import (
    SegTrainer,
    _as_batch_channel_numpy,
    _case_name,
    compute_test_case_metrics,
    print_test_metric_summary,
    save_test_case_prediction,
    summarize_test_metrics,
    write_test_metric_tables,
)
from utils.metric import compute_dice, compute_iou
from .pseudo_mask_calibration import PrototypeGuidedPseudoMaskCalibration


def _arg(args, name, default):
    return getattr(args, name, default)


class WeightedBCEDiceLoss(nn.Module):
    """
    Binary segmentation loss with a spatial weight map.

    logits: [B, 1, H, W]
    target: [B, 1, H, W]
    weight: [B, 1, H, W]
    """

    def __init__(self, dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits, target, weight):
        target = target.float()
        weight = weight.float()

        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
        if weight.shape[-2:] != logits.shape[-2:]:
            weight = F.interpolate(weight, size=logits.shape[-2:], mode="nearest")

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * weight).sum() / (weight.sum() + self.eps)

        prob = torch.sigmoid(logits)
        inter = (prob * target * weight).sum(dim=(1, 2, 3))
        denom = ((prob + target) * weight).sum(dim=(1, 2, 3))
        dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        dice = dice.mean()

        return bce + self.dice_weight * dice


class SigmoidRampupWeight:
    def __init__(self, final_w=1.0, iter_per_epoch=100, rampup_epochs=40):
        self.final_w = final_w
        self.iter_per_epoch = iter_per_epoch
        self.start_iter = 0
        self.rampup_length = rampup_epochs * iter_per_epoch

    def __call__(self, current_idx):
        if current_idx <= self.start_iter:
            return 0.0
        return self.final_w * self.sigmoid(current_idx - self.start_iter, self.rampup_length)

    @staticmethod
    def sigmoid(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class AutoGazeSegTrainer(SegTrainer):
    """
    Optimize the dual AutoGazeSeg branches with direct supervision,
    prototype-guided pseudo-mask calibration, and L_COS feature distillation.
    """

    def __init__(self, *argv, **kargs):
        super().__init__(*argv, **kargs)
        self.main_metric = "mdice"
        self.iter_num = 0

        iter_per_epoch = _arg(self.args, "iter_per_epoch", 100)
        rampup_epochs = _arg(self.args, "consistency_rampup_epochs", 40)
        consistency_final_w = _arg(self.args, "consistency_weight", 1.0)
        self.weight_scheduler = SigmoidRampupWeight(
            final_w=consistency_final_w,
            iter_per_epoch=iter_per_epoch,
            rampup_epochs=rampup_epochs,
        )

        dice_weight = _arg(self.args, "dice_weight", 1.0)
        self.direct_supervision_loss = WeightedBCEDiceLoss(dice_weight=dice_weight)

        self.pseudo_mask_calibration = PrototypeGuidedPseudoMaskCalibration(
            ppc_weight=_arg(self.args, "ppc_weight", 1.0),
            feature_distillation_weight=_arg(self.args, "feature_distillation_weight", 0.5),
            prototype_uncertain_weight=_arg(self.args, "prototype_uncertain_weight", 0.35),
            calibration_supervision_weight=1.0,
            prototype_metric_weight=_arg(self.args, "prototype_metric_weight", 0.25),
            feature_uncertain_weight=_arg(self.args, "feature_uncertain_weight", 0.55),
            prototype_temperature=_arg(self.args, "prototype_temperature", 0.15),
            image_temperature=_arg(self.args, "image_temperature", 0.5),
            image_proto_blend=_arg(self.args, "image_proto_blend", 0.25),
            use_softmax_prob=_arg(self.args, "use_softmax_prob", False),
        )

        self.fix_swapped_pseudo = _arg(self.args, "fix_swapped_pseudo", True)

    @staticmethod
    def _ensure_binary_mask(mask):
        if mask is None:
            return None
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        return (mask > 0.5).float()

    def _build_pseudo_regions(self, pseudo_label_1, pseudo_label_2):
        """
        Split dual pseudo masks into the three regions used by PPC.

        R+ = M_high, R- = 1 - M_low, and U = M_low - M_high.
        """
        high = self._ensure_binary_mask(pseudo_label_1)
        low = self._ensure_binary_mask(pseudo_label_2)

        if high is None or low is None:
            return None

        high, low = self._fix_pseudo_order(high, low)
        low = torch.maximum(low, high)
        uncertain = ((low > 0.5) & (high < 0.5)).float()
        return {
            "high": high,
            "low": low,
            "uncertain": uncertain,
        }

    def _fix_pseudo_order(self, high, low):
        if not self.fix_swapped_pseudo:
            return high, low

        b = high.shape[0]
        swap = (high.view(b, -1).sum(dim=1) > low.view(b, -1).sum(dim=1)).view(b, 1, 1, 1)
        return torch.where(swap, low, high), torch.where(swap, high, low)

    def _get_feature(self, feature, volume):
        if feature is None:
            return volume
        if isinstance(feature, torch.Tensor):
            return feature
        if isinstance(feature, dict):
            for key in ["bottleneck", "x4", "x3", "x2", "x1"]:
                if key in feature:
                    return feature[key]
            raise KeyError(f"feature dict has no usable key, got keys={feature.keys()}")
        raise TypeError(f"Unsupported feature type: {type(feature)}")

    @staticmethod
    def _haar_low_frequency(x):
        if x.shape[-2] % 2 != 0:
            x = x[:, :, :-1, :]
        if x.shape[-1] % 2 != 0:
            x = x[:, :, :, :-1]

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        return (x00 + x01 + x10 + x11) * 0.5

    def _build_mix_mask(
        self,
        volume,
        feature,
        rand_index,
        patch_size=4,
        max_partner_weight=0.7,
        eps=1e-6,
    ):
        _, _, h, w = volume.shape
        feat = self._get_feature(feature, volume).detach()
        feat_1 = feat
        feat_2 = feat[rand_index]

        ll_1 = self._haar_low_frequency(feat_1)
        ll_2 = self._haar_low_frequency(feat_2)

        _, _, hf, wf = ll_1.shape
        p = min(patch_size, hf, wf)
        p = max(p, 1)

        pad_h = (p - hf % p) % p
        pad_w = (p - wf % p) % p
        if pad_h > 0 or pad_w > 0:
            ll_1 = F.pad(ll_1, (0, pad_w, 0, pad_h), mode="replicate")
            ll_2 = F.pad(ll_2, (0, pad_w, 0, pad_h), mode="replicate")

        token_1 = F.avg_pool2d(ll_1, kernel_size=p, stride=p)
        token_2 = F.avg_pool2d(ll_2, kernel_size=p, stride=p)
        sim = F.cosine_similarity(token_1, token_2, dim=1, eps=eps)
        sim = torch.clamp(sim, min=0.0, max=1.0)
        partner_weight = sim.unsqueeze(1) * max_partner_weight
        partner_weight = F.interpolate(partner_weight, size=(h, w), mode="bilinear", align_corners=False)
        return torch.clamp(partner_weight, min=0.0, max=max_partner_weight)

    def mix(self, volume=None, mask=None, feature=None, spatial_weight=None):
        assert volume is not None, "volume cannot be None"
        assert mask is not None, "mask cannot be None"
        assert volume.dim() == 4, f"mix only supports [B, C, H, W], got {volume.shape}"

        b, _, _, _ = volume.shape
        rand_index = torch.randperm(b, device=volume.device)
        partner_weight = self._build_mix_mask(
            volume=volume,
            feature=feature,
            rand_index=rand_index,
            patch_size=4,
            max_partner_weight=0.7,
        )
        self_weight = 1.0 - partner_weight

        mix_volume = volume * self_weight + volume[rand_index] * partner_weight

        if mask.dim() == 4:
            mix_target = mask.float() * self_weight + mask[rand_index].float() * partner_weight
        elif mask.dim() == 3:
            mix_target = mask.float() * self_weight.squeeze(1) + mask[rand_index].float() * partner_weight.squeeze(1)
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        if spatial_weight is None:
            return mix_volume, mix_target

        if spatial_weight.dim() == 3:
            spatial_weight = spatial_weight.unsqueeze(1)
        if spatial_weight.shape[-2:] != volume.shape[-2:]:
            spatial_weight = F.interpolate(spatial_weight, size=volume.shape[-2:], mode="nearest")

        mix_weight = spatial_weight.float() * self_weight + spatial_weight[rand_index].float() * partner_weight
        return mix_volume, mix_target, mix_weight

    def _pseudo_labels_to_cuda(self, minibatch):
        pseudo_label_1 = minibatch.get("pseudo_label_1", None)
        pseudo_label_2 = minibatch.get("pseudo_label_2", None)
        if pseudo_label_1 is not None:
            pseudo_label_1 = pseudo_label_1.cuda(non_blocking=True)
        if pseudo_label_2 is not None:
            pseudo_label_2 = pseudo_label_2.cuda(non_blocking=True)
        return pseudo_label_1, pseudo_label_2

    def _direct_supervision_losses(self, pred_dict1, pred_dict2, pseudo_regions, image):
        if pseudo_regions is None:
            zero = image.new_tensor(0.0)
            return zero, zero, zero

        weight = torch.ones_like(pseudo_regions["high"])
        loss_high = self.direct_supervision_loss(
            pred_dict1["logits"][:, 1:2],
            pseudo_regions["high"],
            weight,
        )
        loss_low = self.direct_supervision_loss(
            pred_dict2["logits"][:, 1:2],
            pseudo_regions["low"],
            weight,
        )
        return loss_high, loss_low, pseudo_regions["uncertain"].mean()

    @staticmethod
    def _record_losses(loss_dict, **losses):
        for name, value in losses.items():
            loss_dict[name] = float(value.detach() if torch.is_tensor(value) else value)

    def _update(self, minibatch):
        self.iter_num += 1

        image = minibatch["image"].cuda(non_blocking=True)
        embedding = minibatch["embedding"].cuda(non_blocking=True)
        branch_1, branch_2 = self.model
        optimizer_1, optimizer_2 = self.optimizer

        loss_dict = {"lr": optimizer_1.param_groups[0]["lr"]}
        branch_1.train()
        branch_2.train()

        pseudo_label_1, pseudo_label_2 = self._pseudo_labels_to_cuda(minibatch)
        pseudo_regions = self._build_pseudo_regions(pseudo_label_1, pseudo_label_2)

        with torch.autocast(device_type="cuda", enabled=self.args.fp16):
            pred_dict1 = branch_1(image, embedding)
            pred_dict2 = branch_2(image, embedding)

            loss_high, loss_low, uncertain_ratio = self._direct_supervision_losses(
                pred_dict1,
                pred_dict2,
                pseudo_regions,
                image,
            )
            consistency_weight = self.weight_scheduler(self.iter_num)

            calibration_dict = self.pseudo_mask_calibration(
                pred_dict1,
                pred_dict2,
                pseudo_regions,
                consistency_weight,
                image=image,
                branch_1=branch_1,
                branch_2=branch_2,
                embedding=embedding,
                mix_fn=self.mix,
            )
            loss_direct_supervision = loss_high + loss_low
            loss = loss_direct_supervision + calibration_dict["loss_objective"]

            self._record_losses(
                loss_dict,
                loss_direct_supervision=loss_direct_supervision,
                loss_direct_high=loss_high,
                loss_direct_low=loss_low,
                loss_ppc=calibration_dict["loss_ppc"],
                loss_ppc_1=calibration_dict["loss_ppc_1"],
                loss_ppc_2=calibration_dict["loss_ppc_2"],
                loss_l_cos=calibration_dict["loss_l_cos"],
                loss_prototype_regularization=calibration_dict[
                    "loss_prototype_regularization"
                ],
                loss_prototype_1=calibration_dict["loss_prototype_1"],
                loss_prototype_2=calibration_dict["loss_prototype_2"],
                consistency_weight=consistency_weight,
                uncertain_ratio=uncertain_ratio,
                loss=loss,
            )

        optimizer_1.zero_grad(set_to_none=True)
        optimizer_2.zero_grad(set_to_none=True)

        if self.args.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer_1)
            self.scaler.step(optimizer_2)
            self.scaler.update()
        else:
            loss.backward()
            optimizer_1.step()
            optimizer_2.step()

        return loss_dict

    def validate(self, dataloader, model=None, save_pred=False, save_root=None):
        branch_1, branch_2 = self.model
        branch_1.eval()
        branch_2.eval()

        pred_count = 2
        iou_sub_l = [[] for _ in range(pred_count)]
        dice_sub_l = [[] for _ in range(pred_count)]
        iou_l, dice_l = [], []

        with torch.no_grad():
            for minibatch in dataloader:
                image = minibatch["image"].cuda(non_blocking=True)
                label = minibatch["label"].cuda(non_blocking=True)
                embedding = minibatch["embedding"].cuda(non_blocking=True)
                mask = ~minibatch["trimap"].cuda(non_blocking=True) if "trimap" in minibatch.keys() else None

                with torch.autocast(device_type="cuda", enabled=self.args.fp16):
                    pred_dict1 = branch_1(image, embedding)
                    pred_dict2 = branch_2(image, embedding)
                    logits_1 = pred_dict1["logits"]
                    logits_2 = pred_dict2["logits"]
                    pred_sub_l = [
                        F.interpolate(logits_1, size=label.shape[2:], mode="bilinear", align_corners=False),
                        F.interpolate(logits_2, size=label.shape[2:], mode="bilinear", align_corners=False),
                    ]
                    pred = torch.stack(pred_sub_l, dim=0).mean(dim=0)

                if save_pred and save_root is not None:
                    self.save_pred_batch(
                        pred.clone(),
                        save_root=save_root,
                        save_filenames=minibatch["subject_id"],
                    )
                    for i, pred_i in enumerate(pred_sub_l):
                        save_root_i = os.path.join(save_root, f"pred_level_{i + 1}")
                        self.save_pred_batch(
                            pred_i.clone(),
                            save_root=save_root_i,
                            save_filenames=minibatch["subject_id"],
                        )

                for i in range(pred_count):
                    iou_sub_l[i].append(
                        compute_iou(pred_sub_l[i][:, 1:2, :, :], label, mask=mask, do_threshold=True).cpu().numpy()
                    )
                    dice_sub_l[i].append(
                        compute_dice(pred_sub_l[i][:, 1:2, :, :], label, mask=mask, do_threshold=True).cpu().numpy()
                    )

                iou_l.append(compute_iou(pred[:, 1:2, :, :], label, mask=mask, do_threshold=True).cpu().numpy())
                dice_l.append(compute_dice(pred[:, 1:2, :, :], label, mask=mask, do_threshold=True).cpu().numpy())

        iou_l = np.concatenate(iou_l, axis=0)
        dice_l = np.concatenate(dice_l, axis=0)

        performance_dict = {
            "miou": np.mean(iou_l),
            "miou_std": np.std(iou_l),
            "mdice": np.mean(dice_l),
            "mdice_std": np.std(dice_l),
        }

        for i in range(pred_count):
            iou_sub = np.concatenate(iou_sub_l[i], axis=0)
            dice_sub = np.concatenate(dice_sub_l[i], axis=0)
            performance_dict[f"miou_{i + 1}"] = np.mean(iou_sub)
            performance_dict[f"miou_std_{i + 1}"] = np.std(iou_sub)
            performance_dict[f"mdice_{i + 1}"] = np.mean(dice_sub)
            performance_dict[f"mdice_std_{i + 1}"] = np.std(dice_sub)

        return performance_dict

    def validate_save(self, dataloader, model=None, save_root=None, threshold=0.5):
        branch_1, branch_2 = self.model
        branch_1.eval()
        branch_2.eval()

        if save_root is None:
            save_root = os.path.join(self.args.exp_result_path, f"{self.args.experiment_name}_test")

        rows = []
        case_idx = 0

        with torch.no_grad():
            for minibatch in dataloader:
                image = minibatch["image"].cuda(non_blocking=True)
                label = minibatch["label"].cuda(non_blocking=True)
                embedding = minibatch["embedding"].cuda(non_blocking=True)
                mask = ~minibatch["trimap"].cuda(non_blocking=True) if "trimap" in minibatch.keys() else None

                with torch.autocast(device_type="cuda", enabled=self.args.fp16):
                    pred_dict1 = branch_1(image, embedding)
                    pred_dict2 = branch_2(image, embedding)
                    logits_1 = pred_dict1["logits"]
                    logits_2 = pred_dict2["logits"]
                    pred_sub_l = [
                        F.interpolate(logits_1, size=label.shape[2:], mode="bilinear", align_corners=False),
                        F.interpolate(logits_2, size=label.shape[2:], mode="bilinear", align_corners=False),
                    ]
                    pred = torch.stack(pred_sub_l, dim=0).mean(dim=0)

                prob_np = _as_batch_channel_numpy(torch.sigmoid(pred[:, 1:2]))[:, 0]
                label_np = _as_batch_channel_numpy(label)[:, 0]
                mask_np = _as_batch_channel_numpy(mask)[:, 0] if mask is not None else None
                subject_ids = minibatch.get("subject_id", None)

                for i_b in range(prob_np.shape[0]):
                    subject_id = subject_ids[i_b] if subject_ids is not None else f"case_{case_idx:04d}"
                    case_name = _case_name(subject_id)
                    valid_mask = mask_np[i_b] if mask_np is not None else None

                    metrics, pred_bin, label_bin = compute_test_case_metrics(
                        prob_np[i_b],
                        label_np[i_b],
                        valid_mask=valid_mask,
                        threshold=threshold,
                    )
                    paths = save_test_case_prediction(save_root, case_name, pred_bin, label_bin)
                    rows.append({"case": case_name, **metrics, **paths})

                    case_idx += 1

        performance_dict = summarize_test_metrics(rows)
        per_case_path, summary_path = write_test_metric_tables(save_root, rows, performance_dict)
        performance_dict["num_cases"] = len(rows)
        performance_dict["save_root"] = save_root
        performance_dict["per_case_metrics"] = per_case_path
        performance_dict["summary_metrics"] = summary_path

        print_test_metric_summary(performance_dict, title="test")
        return performance_dict
