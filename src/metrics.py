from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def refitting_metrics(reconstruction: torch.Tensor, target_profile: torch.Tensor) -> Dict[str, float]:
    mse = F.mse_loss(reconstruction, target_profile).item()
    cosine = F.cosine_similarity(reconstruction, target_profile, dim=1).mean().item()
    mae = (reconstruction - target_profile).abs().mean().item()
    reconstruction_np = reconstruction.detach().cpu().numpy().astype(np.float64)
    target_np = target_profile.detach().cpu().numpy().astype(np.float64)
    residual_np = target_np - reconstruction_np
    eps = 1e-12
    p = np.clip(target_np, a_min=0.0, a_max=None)
    q = np.clip(reconstruction_np, a_min=0.0, a_max=None)
    p = p / np.clip(p.sum(axis=1, keepdims=True), eps, None)
    q = q / np.clip(q.sum(axis=1, keepdims=True), eps, None)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = 0.5 * (p + q)
    centered = target_np - target_np.mean(axis=1, keepdims=True)
    sst = np.sum(centered**2, axis=1)
    sse = np.sum((target_np - reconstruction_np) ** 2, axis=1)
    r2 = np.where(sst > eps, 1.0 - (sse / np.clip(sst, eps, None)), np.nan)
    positive_residual = np.clip(residual_np, a_min=0.0, a_max=None)
    residual_mass = positive_residual / np.clip(positive_residual.sum(axis=1, keepdims=True), eps, None)
    residual_entropy = -np.sum(np.where(residual_mass > 0, residual_mass * np.log(residual_mass + eps), 0.0), axis=1)
    return {
        "cosine_similarity": float(cosine),
        "mse": float(mse),
        "mae": float(mae),
        "histogram_intersection_similarity": float(np.mean(np.minimum(p, q).sum(axis=1))),
        "kl_divergence": float(np.mean(np.sum(p * np.log(p / q), axis=1))),
        "js_divergence": float(np.mean(0.5 * np.sum(p * np.log(p / m), axis=1) + 0.5 * np.sum(q * np.log(q / m), axis=1))),
        "explained_variance_r2": float(np.nanmean(r2)),
        "positive_residual_sum": float(np.mean(positive_residual.sum(axis=1))),
        "absolute_residual_sum": float(np.mean(np.abs(residual_np).sum(axis=1))),
        "signed_residual_sum": float(np.mean(residual_np.sum(axis=1))),
        "residual_entropy": float(np.mean(residual_entropy)),
        "residual_l2_norm": float(np.mean(np.sqrt(np.sum(residual_np**2, axis=1)))),
    }


@torch.no_grad()
def exposure_sparsity_metrics(
    exposures: torch.Tensor,
    active_thresholds: Iterable[float] = (0.0, 0.001, 0.01, 0.05),
) -> Dict[str, float]:
    exposures_np = exposures.detach().cpu().numpy().astype(np.float64)
    exposures_np = np.clip(exposures_np, a_min=0.0, a_max=None)
    eps = 1e-12
    row_sums = exposures_np.sum(axis=1, keepdims=True)
    normalized = np.divide(exposures_np, row_sums, out=np.zeros_like(exposures_np), where=row_sums > eps)
    entropy = -(normalized * np.log(np.clip(normalized, eps, 1.0))).sum(axis=1)
    effective_signatures = np.exp(entropy)

    metrics = {
        "exposure_entropy_eval": float(np.mean(entropy)),
        "effective_signatures": float(np.mean(effective_signatures)),
        "top1_mass": float(np.mean(np.max(normalized, axis=1))),
    }
    for k in (3, 5):
        topk = min(k, normalized.shape[1])
        if topk == 0:
            metrics["top{0}_mass".format(k)] = 0.0
            continue
        partitioned = np.partition(normalized, kth=normalized.shape[1] - topk, axis=1)
        metrics["top{0}_mass".format(k)] = float(np.mean(partitioned[:, -topk:].sum(axis=1)))
    for threshold in active_thresholds:
        metrics["active_gt_{0:g}".format(threshold)] = float(np.mean((normalized > threshold).sum(axis=1)))
    return metrics
