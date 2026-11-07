from typing import Dict

import torch
import torch.nn.functional as F


def signature_refitting_loss(
    predicted_exposures: torch.Tensor,
    target_exposures: torch.Tensor,
    predicted_confidence: torch.Tensor,
    reconstruction: torch.Tensor,
    target_profile: torch.Tensor,
    confidence_scale: float = 10.0,
    composition_weight: float = 1.0,
    confidence_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    pool_size = predicted_exposures.shape[1]
    composition_mse = F.mse_loss(predicted_exposures, target_exposures)
    confidence_target = torch.exp(-confidence_scale * torch.abs(predicted_exposures.detach() - target_exposures))
    confidence_mse = F.mse_loss(predicted_confidence, confidence_target)
    reconstruction_mse = F.mse_loss(reconstruction, target_profile)
    total = composition_weight * composition_mse * pool_size + confidence_weight * confidence_mse
    return {
        "loss": total,
        "composition_mse": composition_mse,
        "confidence_mse": confidence_mse,
        "reconstruction_mse": reconstruction_mse,
    }


def reconstruction_refitting_loss(
    reconstruction: torch.Tensor,
    target_profile: torch.Tensor,
    predicted_exposures: torch.Tensor = None,
    predicted_active_logits: torch.Tensor = None,
    target_exposures: torch.Tensor = None,
    target_exposure_valid_mask: torch.Tensor = None,
    target_depths: torch.Tensor = None,
    residual_prediction: torch.Tensor = None,
    target_residual: torch.Tensor = None,
    loss_name: str = "kl",
    kl_weight: float = 1.0,
    mse_weight: float = 1.0,
    residual_weight: float = 0.0,
    cosine_weight: float = 0.0,
    exposure_entropy_weight: float = 0.0,
    exposure_topk_tail_weight: float = 0.0,
    exposure_topk_tail_k: int = 3,
    exposure_soft_active_weight: float = 0.0,
    exposure_soft_active_threshold: float = 0.01,
    exposure_soft_active_temperature: float = 0.002,
    exposure_soft_active_target: float = 4.0,
    exposure_supervision_weight: float = 0.0,
    exposure_supervision_loss: str = "mse",
    exposure_effective_supervision_weight: float = 0.0,
    exposure_top3_supervision_weight: float = 0.0,
    exposure_top3_supervision_k: int = 3,
    active_supervision_weight: float = 0.0,
    active_supervision_mode: str = "exposure",
    active_supervision_threshold: float = 0.01,
    active_supervision_min_expected_mutations: float = 10.0,
    active_supervision_temperature: float = 0.002,
    active_supervision_positive_weight: float = 1.0,
    active_supervision_loss: str = "bce",
    active_supervision_focal_gamma: float = 2.0,
    active_supervision_burden_weighted: bool = False,
    candidate_mask: torch.Tensor = None,
    candidate_presence_weight: float = 0.0,
    candidate_mass_weight: float = 0.0,
    candidate_mixture_weight: float = 0.0,
    cosmic_preservation_weight: float = 0.0,
    candidate_positive_weight: float = 1.0,
    candidate_presence_threshold: float = 0.01,
    candidate_presence_temperature: float = 0.01,
) -> Dict[str, torch.Tensor]:
    reconstruction = torch.clamp(reconstruction, min=1e-8)
    target_profile = torch.clamp(target_profile, min=1e-8)
    reconstruction_kl = F.kl_div(reconstruction.log(), target_profile, reduction="batchmean")
    reconstruction_mse = F.mse_loss(reconstruction, target_profile)
    reconstruction_mae = F.l1_loss(reconstruction, target_profile)
    reconstruction_cosine = F.cosine_similarity(reconstruction, target_profile, dim=1, eps=1e-8).mean()
    cosine_loss = 1.0 - reconstruction_cosine
    residual_mse = reconstruction_mse.new_tensor(0.0)
    if residual_prediction is not None and target_residual is not None:
        residual_mse = F.mse_loss(residual_prediction, target_residual)
    exposure_entropy = reconstruction_mse.new_tensor(0.0)
    exposure_entropy_loss = reconstruction_mse.new_tensor(0.0)
    exposure_topk_tail_loss = reconstruction_mse.new_tensor(0.0)
    exposure_soft_active_count = reconstruction_mse.new_tensor(0.0)
    exposure_soft_active_loss = reconstruction_mse.new_tensor(0.0)
    exposure_supervision = reconstruction_mse.new_tensor(0.0)
    exposure_effective_supervision = reconstruction_mse.new_tensor(0.0)
    exposure_top3_supervision = reconstruction_mse.new_tensor(0.0)
    active_supervision_bce = reconstruction_mse.new_tensor(0.0)
    candidate_presence_bce = reconstruction_mse.new_tensor(0.0)
    candidate_mass_loss = reconstruction_mse.new_tensor(0.0)
    candidate_mixture_loss = reconstruction_mse.new_tensor(0.0)
    cosmic_preservation_loss = reconstruction_mse.new_tensor(0.0)
    candidate_target_mass_mean = reconstruction_mse.new_tensor(0.0)
    candidate_predicted_mass_mean = reconstruction_mse.new_tensor(0.0)
    candidate_predicted_mass_positive_mean = reconstruction_mse.new_tensor(0.0)
    candidate_predicted_mass_negative_mean = reconstruction_mse.new_tensor(0.0)
    if predicted_exposures is not None:
        exposures = torch.clamp(predicted_exposures, min=1e-8)
        supervised_exposures = exposures
        supervised_targets = target_exposures
        supervised_active_logits = predicted_active_logits
        supervised_depths = target_depths
        supervised_candidate_mask = candidate_mask
        if target_exposures is not None and target_exposure_valid_mask is not None:
            valid_rows = target_exposure_valid_mask.to(
                device=exposures.device, dtype=torch.bool
            ).reshape(-1)
            if valid_rows.shape[0] != exposures.shape[0]:
                raise ValueError(
                    "target_exposure_valid_mask length {0} does not match batch size {1}.".format(
                        valid_rows.shape[0], exposures.shape[0]
                    )
                )
            if bool(valid_rows.any().item()):
                supervised_exposures = exposures[valid_rows]
                supervised_targets = target_exposures[valid_rows]
                if predicted_active_logits is not None:
                    supervised_active_logits = predicted_active_logits[valid_rows]
                if target_depths is not None:
                    supervised_depths = target_depths[valid_rows]
                if candidate_mask is not None and candidate_mask.ndim > 1:
                    supervised_candidate_mask = candidate_mask[valid_rows]
            else:
                supervised_targets = None
        exposure_entropy = -(exposures * exposures.log()).sum(dim=1).mean()
        exposure_entropy_loss = exposure_entropy
        if exposure_topk_tail_weight > 0.0:
            topk = max(1, min(int(exposure_topk_tail_k), exposures.shape[1]))
            topk_mass = torch.topk(exposures, k=topk, dim=1).values.sum(dim=1)
            exposure_topk_tail_loss = (1.0 - topk_mass).mean()
        if exposure_soft_active_weight > 0.0:
            temperature = max(float(exposure_soft_active_temperature), 1e-8)
            threshold = float(exposure_soft_active_threshold)
            soft_active = torch.sigmoid((exposures - threshold) / temperature).sum(dim=1)
            exposure_soft_active_count = soft_active.mean()
            target = exposure_soft_active_count.new_tensor(float(exposure_soft_active_target))
            exposure_soft_active_loss = F.relu(exposure_soft_active_count - target)
        if supervised_targets is not None and exposure_supervision_weight > 0.0:
            target = torch.clamp(supervised_targets, min=1e-8)
            supervision_loss = str(exposure_supervision_loss).strip().lower()
            if supervision_loss == "mse":
                exposure_supervision = F.mse_loss(supervised_exposures, target)
            elif supervision_loss in {"mae", "l1"}:
                exposure_supervision = F.l1_loss(supervised_exposures, target)
            elif supervision_loss == "kl":
                exposure_supervision = F.kl_div(supervised_exposures.log(), target, reduction="batchmean")
            elif supervision_loss == "cosine":
                exposure_supervision = 1.0 - F.cosine_similarity(supervised_exposures, target, dim=1, eps=1e-8).mean()
            else:
                raise ValueError(
                    "Unsupported exposure_supervision_loss '{0}'. Expected mse, mae, kl, or cosine.".format(
                        exposure_supervision_loss
                    )
                )
        if supervised_targets is not None and exposure_effective_supervision_weight > 0.0:
            target = torch.clamp(supervised_targets, min=1e-8)
            target = target / torch.clamp(target.sum(dim=1, keepdim=True), min=1e-8)
            predicted_effective = torch.exp(-(supervised_exposures * supervised_exposures.log()).sum(dim=1))
            target_effective = torch.exp(-(target * target.log()).sum(dim=1))
            exposure_effective_supervision = F.l1_loss(predicted_effective, target_effective)
        if supervised_targets is not None and exposure_top3_supervision_weight > 0.0:
            target = torch.clamp(supervised_targets, min=1e-8)
            target = target / torch.clamp(target.sum(dim=1, keepdim=True), min=1e-8)
            topk = max(1, min(int(exposure_top3_supervision_k), supervised_exposures.shape[1]))
            predicted_topk_mass = torch.topk(supervised_exposures, k=topk, dim=1).values.sum(dim=1)
            target_topk_mass = torch.topk(target, k=topk, dim=1).values.sum(dim=1)
            exposure_top3_supervision = F.l1_loss(predicted_topk_mass, target_topk_mass)
        if supervised_targets is not None and active_supervision_weight > 0.0:
            temperature = max(float(active_supervision_temperature), 1e-8)
            threshold = float(active_supervision_threshold)
            active_mode = str(active_supervision_mode).strip().lower()
            active_target = (supervised_targets >= threshold)
            if active_mode == "burden_aware":
                if supervised_depths is None:
                    raise ValueError("active_supervision_mode='burden_aware' requires target_depths.")
                depth = supervised_depths
                if depth.ndim == 1:
                    depth = depth.unsqueeze(1)
                min_expected = float(active_supervision_min_expected_mutations)
                active_target = active_target & ((supervised_targets * depth) >= min_expected)
            elif active_mode != "exposure":
                raise ValueError("Unsupported active_supervision_mode '{0}'. Expected exposure or burden_aware.".format(active_supervision_mode))
            active_target = active_target.float()
            if supervised_active_logits is not None:
                active_logits = supervised_active_logits
                active_prob = torch.sigmoid(active_logits)
            else:
                active_logits = (supervised_exposures - threshold) / temperature
                active_prob = torch.sigmoid(active_logits)
            positive_weight = max(float(active_supervision_positive_weight), 1.0)
            bce_weight = 1.0 + (positive_weight - 1.0) * active_target
            if active_supervision_burden_weighted and supervised_depths is not None:
                depth = supervised_depths
                if depth.ndim == 1:
                    depth = depth.unsqueeze(1)
                burden_fraction = torch.clamp(
                    supervised_targets * depth / max(float(active_supervision_min_expected_mutations), 1e-8),
                    min=0.0,
                    max=1.0,
                )
                bce_weight = torch.where(
                    active_target > 0.0,
                    1.0 + (positive_weight - 1.0) * burden_fraction,
                    bce_weight,
                )
            active_loss_name = str(active_supervision_loss).strip().lower()
            if active_loss_name == "bce":
                active_supervision_bce = F.binary_cross_entropy(
                    torch.clamp(active_prob, min=1e-6, max=1.0 - 1e-6),
                    active_target,
                    weight=bce_weight,
                )
            elif active_loss_name == "focal":
                bce = F.binary_cross_entropy_with_logits(active_logits, active_target, reduction="none")
                pt = torch.where(active_target > 0.0, active_prob, 1.0 - active_prob)
                focal_weight = torch.pow(torch.clamp(1.0 - pt, min=0.0, max=1.0), float(active_supervision_focal_gamma))
                active_supervision_bce = (bce * focal_weight * bce_weight).mean()
            else:
                raise ValueError("Unsupported active_supervision_loss '{0}'. Expected bce or focal.".format(active_supervision_loss))

        candidate_loss_enabled = any(
            float(weight) > 0.0
            for weight in (
                candidate_presence_weight,
                candidate_mass_weight,
                candidate_mixture_weight,
                cosmic_preservation_weight,
            )
        )
        if supervised_targets is not None and candidate_loss_enabled:
            if supervised_candidate_mask is None:
                raise ValueError("Candidate-specific supervision requires candidate_mask.")
            candidate_mask = supervised_candidate_mask.to(device=supervised_exposures.device, dtype=torch.bool)
            if candidate_mask.ndim == 1:
                candidate_mask = candidate_mask.unsqueeze(0).expand_as(supervised_exposures)
            if candidate_mask.shape != supervised_exposures.shape:
                raise ValueError(
                    "candidate_mask shape {0} does not match exposure shape {1}.".format(
                        tuple(candidate_mask.shape), tuple(supervised_exposures.shape)
                    )
                )
            if not bool(candidate_mask.any().item()):
                raise ValueError("Candidate-specific supervision is enabled but candidate_mask is empty.")

            target = torch.clamp(supervised_targets, min=0.0)
            mask_float = candidate_mask.to(dtype=supervised_exposures.dtype)
            candidate_prediction = supervised_exposures * mask_float
            candidate_target = target * mask_float
            predicted_mass = candidate_prediction.sum(dim=1)
            target_mass = candidate_target.sum(dim=1)
            target_positive = target_mass > 1e-8
            positive_weight = max(float(candidate_positive_weight), 1.0)
            sample_weight = 1.0 + (positive_weight - 1.0) * target_positive.to(supervised_exposures.dtype)

            candidate_target_mass_mean = target_mass.mean()
            candidate_predicted_mass_mean = predicted_mass.mean()
            if bool(target_positive.any().item()):
                candidate_predicted_mass_positive_mean = predicted_mass[target_positive].mean()
            if bool((~target_positive).any().item()):
                candidate_predicted_mass_negative_mean = predicted_mass[~target_positive].mean()

            presence_temperature = max(float(candidate_presence_temperature), 1e-8)
            presence_logits = (predicted_mass - float(candidate_presence_threshold)) / presence_temperature
            presence_bce = F.binary_cross_entropy_with_logits(
                presence_logits,
                target_positive.to(supervised_exposures.dtype),
                reduction="none",
            )
            candidate_presence_bce = (presence_bce * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)

            mass_error = torch.abs(predicted_mass - target_mass)
            candidate_mass_loss = (mass_error * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)

            if bool(target_positive.any().item()):
                positive_prediction = candidate_prediction[target_positive]
                positive_target = candidate_target[target_positive]
                predicted_mixture = positive_prediction / predicted_mass[target_positive].unsqueeze(1).clamp_min(1e-8)
                target_mixture = positive_target / target_mass[target_positive].unsqueeze(1).clamp_min(1e-8)
                candidate_mixture_loss = 1.0 - F.cosine_similarity(
                    predicted_mixture,
                    target_mixture,
                    dim=1,
                    eps=1e-8,
                ).mean()

            cosmic_mask = (~candidate_mask).to(dtype=supervised_exposures.dtype)
            cosmic_prediction = supervised_exposures * cosmic_mask
            cosmic_target = target * cosmic_mask
            predicted_cosmic_mass = cosmic_prediction.sum(dim=1, keepdim=True)
            target_cosmic_mass = cosmic_target.sum(dim=1, keepdim=True)
            predicted_cosmic_mixture = cosmic_prediction / predicted_cosmic_mass.clamp_min(1e-8)
            target_cosmic_mixture = cosmic_target / target_cosmic_mass.clamp_min(1e-8)
            cosmic_preservation_loss = 1.0 - F.cosine_similarity(
                predicted_cosmic_mixture,
                target_cosmic_mixture,
                dim=1,
                eps=1e-8,
            ).mean()

    if loss_name == "kl":
        total = reconstruction_kl
    elif loss_name == "mse":
        total = reconstruction_mse
    elif loss_name == "mae":
        total = reconstruction_mae
    elif loss_name == "kl_mse":
        total = kl_weight * reconstruction_kl + mse_weight * reconstruction_mse
    else:
        raise ValueError("Unsupported real-data loss '{0}'. Expected one of: kl, mse, mae, kl_mse.".format(loss_name))

    if cosine_weight > 0.0:
        total = total + cosine_weight * cosine_loss
    if residual_weight > 0.0 and residual_prediction is not None and target_residual is not None:
        total = total + residual_weight * residual_mse
    if exposure_entropy_weight > 0.0 and predicted_exposures is not None:
        total = total + exposure_entropy_weight * exposure_entropy_loss
    if exposure_topk_tail_weight > 0.0 and predicted_exposures is not None:
        total = total + exposure_topk_tail_weight * exposure_topk_tail_loss
    if exposure_soft_active_weight > 0.0 and predicted_exposures is not None:
        total = total + exposure_soft_active_weight * exposure_soft_active_loss
    if exposure_supervision_weight > 0.0 and predicted_exposures is not None and target_exposures is not None:
        total = total + exposure_supervision_weight * exposure_supervision
    if exposure_effective_supervision_weight > 0.0 and predicted_exposures is not None and target_exposures is not None:
        total = total + exposure_effective_supervision_weight * exposure_effective_supervision
    if exposure_top3_supervision_weight > 0.0 and predicted_exposures is not None and target_exposures is not None:
        total = total + exposure_top3_supervision_weight * exposure_top3_supervision
    if active_supervision_weight > 0.0 and predicted_exposures is not None and target_exposures is not None:
        total = total + active_supervision_weight * active_supervision_bce
    if candidate_presence_weight > 0.0:
        total = total + candidate_presence_weight * candidate_presence_bce
    if candidate_mass_weight > 0.0:
        total = total + candidate_mass_weight * candidate_mass_loss
    if candidate_mixture_weight > 0.0:
        total = total + candidate_mixture_weight * candidate_mixture_loss
    if cosmic_preservation_weight > 0.0:
        total = total + cosmic_preservation_weight * cosmic_preservation_loss

    return {
        "loss": total,
        "composition_mse": total.new_tensor(0.0),
        "confidence_mse": total.new_tensor(0.0),
        "reconstruction_mse": reconstruction_mse,
        "reconstruction_kl": reconstruction_kl,
        "reconstruction_mae": reconstruction_mae,
        "reconstruction_cosine_loss": cosine_loss,
        "residual_mse": residual_mse,
        "exposure_entropy": exposure_entropy,
        "exposure_entropy_loss": exposure_entropy_loss,
        "exposure_topk_tail_loss": exposure_topk_tail_loss,
        "exposure_soft_active_count": exposure_soft_active_count,
        "exposure_soft_active_loss": exposure_soft_active_loss,
        "exposure_supervision_loss": exposure_supervision,
        "exposure_effective_supervision_loss": exposure_effective_supervision,
        "exposure_top3_supervision_loss": exposure_top3_supervision,
        "active_supervision_bce": active_supervision_bce,
        "candidate_presence_bce": candidate_presence_bce,
        "candidate_mass_loss": candidate_mass_loss,
        "candidate_mixture_loss": candidate_mixture_loss,
        "cosmic_preservation_loss": cosmic_preservation_loss,
        "candidate_target_mass_mean": candidate_target_mass_mean,
        "candidate_predicted_mass_mean": candidate_predicted_mass_mean,
        "candidate_predicted_mass_positive_mean": candidate_predicted_mass_positive_mean,
        "candidate_predicted_mass_negative_mean": candidate_predicted_mass_negative_mean,
    }
