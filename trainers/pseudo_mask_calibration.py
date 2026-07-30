import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_KEYS = ("x1", "x2", "x3", "x4", "bottleneck")


def as_mask(mask: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError(f"Expected [B, 1, H, W] or [B, H, W], got {mask.shape}")
    mask = mask.float()
    if size is not None and mask.shape[-2:] != size:
        mask = F.interpolate(mask, size=size, mode="nearest")
    return mask


def resize_prob(prob: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    if prob.shape[-2:] != size:
        prob = F.interpolate(prob.float(), size=size, mode="bilinear", align_corners=False)
    return prob.float()


def foreground_logits(logits: torch.Tensor, use_softmax_prob: bool = False) -> torch.Tensor:
    if logits.shape[1] == 1:
        return logits
    if use_softmax_prob:
        return logits[:, 1:2] - logits[:, 0:1]
    return logits[:, 1:2]


def safe_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(0.0)


def weighted_average(features: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    numerator = (features * weight).sum(dim=(2, 3), keepdim=True)
    denominator = weight.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return numerator / denominator


def weighted_mean(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return (values * weight.float()).sum() / (weight.float().sum() + eps)


def valid_region(weight: torch.Tensor, min_pixels: float) -> torch.Tensor:
    return (weight.sum(dim=(2, 3), keepdim=True) >= min_pixels).float()


def split_pseudo_regions(
    high: torch.Tensor,
    low: torch.Tensor,
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    high = (as_mask(high, size) > 0.5).float()
    low = torch.maximum((as_mask(low, size) > 0.5).float(), high)
    bg = 1.0 - low
    uncertain = (low - high).clamp_min(0.0)
    return high, low, bg, uncertain


def feature_dict(pred_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
    for name in ("decoder_features", "encoder_features"):
        feats = pred_dict.get(name)
        if isinstance(feats, dict):
            return feats
    return None


def main_feature(pred_dict: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Aggregate the two deepest decoder features used to form prototypes."""
    feats = feature_dict(pred_dict)
    feature_list = []

    if feats is not None:
        for key in ("x3", "x4"):
            if key in feats:
                feature_list.append(feats[key])

    if not feature_list:
        return None

    target_size = feature_list[0].shape[-2:]

    resized_features = []
    for feat in feature_list:
        if feat.shape[-2:] != target_size:
            feat = F.interpolate(
                feat,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        resized_features.append(feat)

    return torch.cat(resized_features, dim=1)


def collect_features(pred_dict: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    out = []
    feats = feature_dict(pred_dict)
    if feats is not None:
        out.extend(feats[key] for key in FEATURE_KEYS if key in feats)
    if "feature" in pred_dict:
        out.append(pred_dict["feature"])
    return out


def weighted_logits_bce_dice_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    target = target.float().clamp(0.0, 1.0)
    weight = weight.float()

    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    bce = (bce * weight).sum() / (weight.sum() + eps)

    pred_prob = torch.sigmoid(pred_logits).clamp(eps, 1.0 - eps)
    inter = (pred_prob * target * weight).sum(dim=(1, 2, 3))
    denom = ((pred_prob + target) * weight).sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return bce + dice.mean()


class PrototypeGuidedPseudoMaskCalibrator(nn.Module):
    """Calibrate the uncertain region with feature and image-appearance prototypes."""

    def __init__(
        self,
        eps: float = 1e-6,
        temperature: float = 0.15,
        image_temperature: float = 0.50,
        image_proto_blend: float = 0.25,
        uncertain_weight: float = 0.35,
        metric_weight: float = 0.25,
        separation_margin: float = 0.20,
        min_region_pixels: float = 4.0,
    ):
        super().__init__()
        self.eps = eps
        self.temperature = temperature
        self.image_temperature = image_temperature
        self.image_proto_blend = image_proto_blend
        self.uncertain_weight = uncertain_weight
        self.metric_weight = metric_weight
        self.separation_margin = separation_margin
        self.min_region_pixels = min_region_pixels

    def forward(
        self,
        features: torch.Tensor,
        pred_logits: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        size = features.shape[-2:]

        if pred_logits.shape[-2:] != size:
            pred_logits = F.interpolate(pred_logits, size=size, mode="bilinear", align_corners=False)

        high, _, bg, uncertain = split_pseudo_regions(high, low, size)
        valid_pair = valid_region(high, self.min_region_pixels) * valid_region(bg, self.min_region_pixels)

        feat_norm = F.normalize(features, dim=1, eps=self.eps)
        fg_proto = F.normalize(weighted_average(feat_norm, high, self.eps), dim=1, eps=self.eps)
        bg_proto = F.normalize(weighted_average(feat_norm, bg, self.eps), dim=1, eps=self.eps)

        sim_fg = (feat_norm * fg_proto).sum(dim=1, keepdim=True)
        sim_bg = (feat_norm * bg_proto).sum(dim=1, keepdim=True)

        calibrated_fg_prob = self._calibrated_foreground_probability(
            sim_fg,
            sim_bg,
            high,
            bg,
            image,
            size,
        )
        calibration_confidence = (2.0 * (calibrated_fg_prob - 0.5).abs()).clamp(0.0, 1.0)

        calibrated_mask = (high + uncertain * calibrated_fg_prob).clamp(0.0, 1.0)
        loss_weight = (
            high
            + bg
            + self.uncertain_weight * uncertain * calibration_confidence
        ) * valid_pair

        loss_metric, loss_calibration_pull = self._metric_loss(
            sim_fg=sim_fg,
            sim_bg=sim_bg,
            fg_proto=fg_proto,
            bg_proto=bg_proto,
            high=high,
            bg=bg,
            uncertain=uncertain,
            calibrated_fg_prob=calibrated_fg_prob,
            calibration_confidence=calibration_confidence,
            valid_pair=valid_pair,
        )
        loss = self.metric_weight * loss_metric

        return {
            "loss": loss,
            "loss_metric": loss_metric.detach(),
            "loss_calibration_pull": loss_calibration_pull.detach(),
            "calibrated_mask": calibrated_mask,
            "calibrated_fg_prob": calibrated_fg_prob.detach(),
            "calibration_confidence": calibration_confidence.detach(),
            "fg_proto": fg_proto.detach(),
            "bg_proto": bg_proto.detach(),
            "valid_pair": valid_pair.detach(),
            "loss_weight": loss_weight,
            "pred_logits": pred_logits,
        }

    def _calibrated_foreground_probability(
        self,
        sim_fg: torch.Tensor,
        sim_bg: torch.Tensor,
        high: torch.Tensor,
        bg: torch.Tensor,
        image: torch.Tensor | None,
        size: tuple[int, int],
    ) -> torch.Tensor:
        proto_logits = torch.cat([sim_fg, sim_bg], dim=1) / max(self.temperature, self.eps)
        feat_prob = torch.softmax(proto_logits, dim=1)[:, 0:1]

        if image is None or self.image_proto_blend <= 0.0:
            return feat_prob.clamp(0.0, 1.0)

        image = self._normalize_image(image, size)
        img_fg = weighted_average(image, high, self.eps)
        img_bg = weighted_average(image, bg, self.eps)
        dist_fg = ((image - img_fg) ** 2).mean(dim=1, keepdim=True)
        dist_bg = ((image - img_bg) ** 2).mean(dim=1, keepdim=True)
        img_logits = torch.cat([-dist_fg, -dist_bg], dim=1) / max(self.image_temperature, self.eps)
        img_prob = torch.softmax(img_logits, dim=1)[:, 0:1]

        blend = min(max(self.image_proto_blend, 0.0), 1.0)
        return ((1.0 - blend) * feat_prob + blend * img_prob).clamp(0.0, 1.0)

    def _normalize_image(self, image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        image = image.float()
        if image.shape[-2:] != size:
            image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        mean = image.mean(dim=(2, 3), keepdim=True)
        std = image.std(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return (image - mean) / std

    def _metric_loss(
        self,
        sim_fg: torch.Tensor,
        sim_bg: torch.Tensor,
        fg_proto: torch.Tensor,
        bg_proto: torch.Tensor,
        high: torch.Tensor,
        bg: torch.Tensor,
        uncertain: torch.Tensor,
        calibrated_fg_prob: torch.Tensor,
        calibration_confidence: torch.Tensor,
        valid_pair: torch.Tensor,
    ) -> torch.Tensor:
        loss_fg = weighted_mean(1.0 - sim_fg, high * valid_pair, self.eps)
        loss_bg = weighted_mean(1.0 - sim_bg, bg * valid_pair, self.eps)

        u_prob = calibrated_fg_prob.detach()
        u_weight = uncertain * calibration_confidence.detach() * valid_pair
        loss_uncertain = weighted_mean(
            u_prob * (1.0 - sim_fg) + (1.0 - u_prob) * (1.0 - sim_bg),
            u_weight,
            self.eps,
        )

        proto_sim = (fg_proto * bg_proto).sum(dim=1, keepdim=True)
        loss_separation = weighted_mean(F.relu(proto_sim - self.separation_margin), valid_pair, self.eps)
        loss_metric = loss_fg + loss_bg + self.uncertain_weight * loss_uncertain + loss_separation
        return loss_metric, loss_uncertain


class RegionAwareFeatureDistillationLoss(nn.Module):
    """Apply pixel- and prototype-level cosine alignment over four regions."""

    def __init__(
        self,
        beta: float = 0.5,
        pixel_weight: float = 1.0,
        prototype_weight: float = 1.0,
        uncertain_weight: float = 0.35,
        eps: float = 1e-6,
        min_region_pixels: float = 4.0,
    ):
        super().__init__()
        self.beta = beta
        self.pixel_weight = pixel_weight
        self.prototype_weight = prototype_weight
        self.uncertain_weight = uncertain_weight
        self.eps = eps
        self.min_region_pixels = min_region_pixels

    def forward(
        self,
        feats1: list[torch.Tensor],
        feats2: list[torch.Tensor],
        high: torch.Tensor,
        low: torch.Tensor,
        calibrated_fg_prob: torch.Tensor | None = None,
        calibration_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not feats1 or not feats2:
            return safe_zero(high)

        total = None
        valid_layers = 0
        for f1, f2 in zip(feats1, feats2):
            if f1.shape[1] != f2.shape[1]:
                continue
            layer_loss = self._layer_loss(
                f1,
                f2,
                high,
                low,
                calibrated_fg_prob,
                calibration_confidence,
            )
            total = layer_loss if total is None else total + layer_loss
            valid_layers += 1

        if valid_layers == 0:
            return safe_zero(feats1[0])
        return self.beta * total / valid_layers

    def _layer_loss(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        calibrated_fg_prob: torch.Tensor | None,
        calibration_confidence: torch.Tensor | None,
    ) -> torch.Tensor:
        if f2.shape[-2:] != f1.shape[-2:]:
            f2 = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        high, _, bg, uncertain = split_pseudo_regions(high, low, f1.shape[-2:])
        u_prob = self._region_prob(calibrated_fg_prob, high)
        u_conf = self._region_conf(calibration_confidence, u_prob, high)

        align_weight = high + bg + self.uncertain_weight * uncertain * u_conf
        loss_pixel = self._pixel_distill(f1, f2, align_weight)

        region_weights = (
            high,
            bg,
            uncertain * u_prob.detach() * u_conf.detach(),
            uncertain * (1.0 - u_prob.detach()) * u_conf.detach(),
        )
        loss_proto = torch.stack([self._prototype_distill(f1, f2, w) for w in region_weights]).mean()
        return self.pixel_weight * loss_pixel + self.prototype_weight * loss_proto

    @staticmethod
    def _region_prob(prob: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
        if prob is None:
            return 0.5 * torch.ones_like(reference)
        return resize_prob(prob, reference.shape[-2:]).clamp(0.0, 1.0)

    @staticmethod
    def _region_conf(conf: torch.Tensor | None, prob: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if conf is None:
            return (2.0 * (prob - 0.5).abs()).clamp(0.0, 1.0)
        return resize_prob(conf, reference.shape[-2:]).clamp(0.0, 1.0)

    def _pixel_distill(self, f1: torch.Tensor, f2: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        n1 = F.normalize(f1, dim=1, eps=self.eps)
        n2 = F.normalize(f2, dim=1, eps=self.eps)
        loss_12 = 1.0 - F.cosine_similarity(n1, n2.detach(), dim=1, eps=self.eps).unsqueeze(1)
        loss_21 = 1.0 - F.cosine_similarity(n2, n1.detach(), dim=1, eps=self.eps).unsqueeze(1)
        return 0.5 * (weighted_mean(loss_12, weight, self.eps) + weighted_mean(loss_21, weight, self.eps))

    def _prototype_distill(self, f1: torch.Tensor, f2: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        valid = valid_region(weight, self.min_region_pixels)
        if not torch.any(valid > 0):
            return safe_zero(f1)

        n1 = F.normalize(f1, dim=1, eps=self.eps)
        n2 = F.normalize(f2, dim=1, eps=self.eps)
        p1 = F.normalize(weighted_average(n1, weight, self.eps), dim=1, eps=self.eps)
        p2 = F.normalize(weighted_average(n2, weight, self.eps), dim=1, eps=self.eps)
        return weighted_mean(1.0 - (p1 * p2).sum(dim=1, keepdim=True), valid, self.eps)


class PrototypeGuidedPseudoMaskCalibration(nn.Module):
    """Build calibrated soft pseudo masks and exchange them across the two branches."""

    def __init__(
        self,
        ppc_weight: float = 1.0,
        feature_distillation_weight: float = 0.5,
        use_consistency_ramp_for_ppc: bool = True,
        use_consistency_ramp_for_feature_distillation: bool = True,
        prototype_uncertain_weight: float = 0.35,
        calibration_supervision_weight: float = 1.0,
        prototype_metric_weight: float = 0.25,
        feature_uncertain_weight: float = 0.55,
        prototype_temperature: float = 0.15,
        image_temperature: float = 0.50,
        image_proto_blend: float = 0.25,
        use_softmax_prob: bool = False,
    ):
        super().__init__()
        self.pseudo_mask_calibrator = PrototypeGuidedPseudoMaskCalibrator(
            temperature=prototype_temperature,
            image_temperature=image_temperature,
            image_proto_blend=image_proto_blend,
            uncertain_weight=prototype_uncertain_weight,
            metric_weight=prototype_metric_weight,
        )
        self.feature_distillation = RegionAwareFeatureDistillationLoss(
            beta=feature_distillation_weight,
            uncertain_weight=feature_uncertain_weight,
        )

        self.ppc_weight = ppc_weight
        self.use_consistency_ramp_for_ppc = use_consistency_ramp_for_ppc
        self.use_consistency_ramp_for_feature_distillation = use_consistency_ramp_for_feature_distillation
        self.use_softmax_prob = use_softmax_prob
        self.calibration_supervision_weight = calibration_supervision_weight

    def forward(
        self,
        pred_dict1: dict[str, torch.Tensor],
        pred_dict2: dict[str, torch.Tensor],
        pseudo_regions: dict[str, torch.Tensor] | None,
        consistency_weight: float | torch.Tensor = 1.0,
        image: torch.Tensor | None = None,
        branch_1: nn.Module | None = None,
        branch_2: nn.Module | None = None,
        embedding: torch.Tensor | None = None,
        mix_fn=None,
    ) -> dict[str, torch.Tensor]:
        zero = pred_dict1["logits"].new_tensor(0.0)
        if pseudo_regions is None:
            return self._loss_dict(zero)

        high, low = pseudo_regions["high"], pseudo_regions["low"]
        logits_1 = foreground_logits(pred_dict1["logits"], self.use_softmax_prob)
        logits_2 = foreground_logits(pred_dict2["logits"], self.use_softmax_prob)
        prob_1 = torch.sigmoid(logits_1)

        feat_1, feat_2 = main_feature(pred_dict1), main_feature(pred_dict2)
        if feat_1 is None or feat_2 is None:
            return self._loss_dict(zero)

        calibration_1 = self.pseudo_mask_calibrator(feat_1, logits_1, high, low, image=image)
        calibration_2 = self.pseudo_mask_calibrator(feat_2, logits_2, high, low, image=image)

        loss_ppc_1 = self._cross_branch_calibration_loss(
            target_branch=branch_1,
            image=image,
            embedding=embedding,
            target=calibration_2["calibrated_mask"],
            weight=calibration_2["loss_weight"],
            feature=feat_2,
            mix_fn=mix_fn,
        )
        loss_ppc_2 = self._cross_branch_calibration_loss(
            target_branch=branch_2,
            image=image,
            embedding=embedding,
            target=calibration_1["calibrated_mask"],
            weight=calibration_1["loss_weight"],
            feature=feat_1,
            mix_fn=mix_fn,
        )
        loss_ppc = loss_ppc_1 + loss_ppc_2
        loss_prototype_1, loss_prototype_2 = calibration_1["loss"], calibration_2["loss"]
        loss_prototype_regularization = loss_prototype_1 + loss_prototype_2
        loss_calibration_objective = (
            loss_prototype_regularization
            + self.calibration_supervision_weight * loss_ppc
        )

        calibrated_prob, calibration_confidence = self._merge_calibrated_assignments(
            calibration_1,
            calibration_2,
        )
        loss_l_cos = self.feature_distillation(
            collect_features(pred_dict1),
            collect_features(pred_dict2),
            high=high,
            low=low,
            calibrated_fg_prob=calibrated_prob,
            calibration_confidence=calibration_confidence,
        )

        consistency_weight = self._as_loss_weight(consistency_weight, prob_1)
        ppc_scale = (
            consistency_weight if self.use_consistency_ramp_for_ppc else prob_1.new_tensor(1.0)
        )
        feature_scale = (
            consistency_weight
            if self.use_consistency_ramp_for_feature_distillation
            else prob_1.new_tensor(1.0)
        )
        loss_objective = (
            self.ppc_weight * ppc_scale * loss_calibration_objective
            + feature_scale * loss_l_cos
        )

        return self._loss_dict(
            zero,
            loss_objective=loss_objective,
            loss_ppc=loss_ppc.detach(),
            loss_ppc_1=loss_ppc_1.detach(),
            loss_ppc_2=loss_ppc_2.detach(),
            loss_l_cos=loss_l_cos.detach(),
            loss_prototype_regularization=loss_prototype_regularization.detach(),
            loss_prototype_1=loss_prototype_1.detach(),
            loss_prototype_2=loss_prototype_2.detach(),
            loss_prototype_metric_1=calibration_1["loss_metric"],
            loss_prototype_metric_2=calibration_2["loss_metric"],
            calibrated_pseudo_mask_1=calibration_1["calibrated_mask"],
            calibrated_pseudo_mask_2=calibration_2["calibrated_mask"],
            calibrated_fg_prob=calibrated_prob.detach(),
            calibration_confidence=calibration_confidence.detach(),
        )

    def _cross_branch_calibration_loss(
        self,
        target_branch: nn.Module | None,
        image: torch.Tensor | None,
        embedding: torch.Tensor | None,
        target: torch.Tensor,
        weight: torch.Tensor,
        feature: torch.Tensor,
        mix_fn,
    ) -> torch.Tensor:
        if target_branch is None or image is None or embedding is None or mix_fn is None:
            return safe_zero(target)

        target = target.detach()
        weight = weight.detach()
        size = image.shape[-2:]
        if target.shape[-2:] != size:
            target = F.interpolate(target.float(), size=size, mode="bilinear", align_corners=False)
        if weight.shape[-2:] != size:
            weight = F.interpolate(weight.float(), size=size, mode="bilinear", align_corners=False)

        noisy_image = image + torch.zeros_like(image).uniform_(-0.2, 0.2)
        mixed_image, mixed_target, mixed_weight = mix_fn(
            volume=noisy_image,
            mask=target.clamp(0.0, 1.0),
            feature=feature,
            spatial_weight=weight.clamp(0.0, 1.0),
        )

        branch_prediction = target_branch(mixed_image, embedding)
        pred_logits = foreground_logits(branch_prediction["logits"], self.use_softmax_prob)
        if pred_logits.shape[-2:] != mixed_target.shape[-2:]:
            pred_logits = F.interpolate(
                pred_logits,
                size=mixed_target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return weighted_logits_bce_dice_loss(
            pred_logits.float(),
            mixed_target.detach(),
            mixed_weight.detach(),
            self.pseudo_mask_calibrator.eps,
        )

    @staticmethod
    def _as_loss_weight(weight: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(weight):
            weight = reference.new_tensor(float(weight))
        return weight.to(reference.device, dtype=reference.dtype)

    @staticmethod
    def _merge_calibrated_assignments(
        calibration_1: dict[str, torch.Tensor],
        calibration_2: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prob_1 = calibration_1["calibrated_fg_prob"]
        confidence_1 = calibration_1["calibration_confidence"]
        prob_2 = calibration_2["calibrated_fg_prob"]
        confidence_2 = calibration_2["calibration_confidence"]

        if prob_2.shape[-2:] != prob_1.shape[-2:]:
            prob_2 = F.interpolate(prob_2, size=prob_1.shape[-2:], mode="bilinear", align_corners=False)
            confidence_2 = F.interpolate(
                confidence_2,
                size=prob_1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return (
            (0.5 * (prob_1 + prob_2)).clamp(0.0, 1.0),
            (0.5 * (confidence_1 + confidence_2)).clamp(0.0, 1.0),
        )

    @staticmethod
    def _loss_dict(zero: torch.Tensor, **updates: torch.Tensor) -> dict[str, torch.Tensor]:
        keys = (
            "loss_objective",
            "loss_ppc",
            "loss_ppc_1",
            "loss_ppc_2",
            "loss_l_cos",
            "loss_prototype_regularization",
            "loss_prototype_1",
            "loss_prototype_2",
            "loss_prototype_metric_1",
            "loss_prototype_metric_2",
            "calibrated_pseudo_mask_1",
            "calibrated_pseudo_mask_2",
            "calibrated_fg_prob",
            "calibration_confidence",
        )
        out = {key: zero for key in keys}
        out.update(updates)
        return out
