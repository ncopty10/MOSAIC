"""Truth-aware metrics for generated synthetic mutational-signature datasets."""

from __future__ import annotations

import math

from .activity import active_mask


def safe_normalize(values: list[float]) -> list[float]:
    total = float(sum(values))
    if total <= 0.0:
        if not values:
            return []
        return [1.0 / len(values)] * len(values)
    return [float(value) / total for value in values]


def cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(lhs, rhs))
    lhs_norm = math.sqrt(sum(a * a for a in lhs))
    rhs_norm = math.sqrt(sum(b * b for b in rhs))
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return 0.0
    return numerator / (lhs_norm * rhs_norm)


def mse(lhs: list[float], rhs: list[float]) -> float:
    if not lhs:
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(lhs, rhs)) / len(lhs)


def mae(lhs: list[float], rhs: list[float]) -> float:
    if not lhs:
        return 0.0
    return sum(abs(a - b) for a, b in zip(lhs, rhs)) / len(lhs)


def entropy(values: list[float]) -> float:
    probs = safe_normalize(values)
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def effective_signatures(values: list[float]) -> float:
    """Return exp(entropy), a smooth effective-number-of-signatures metric."""

    return math.exp(entropy(values))


def top_k_mass(values: list[float], k: int = 3) -> float:
    if k <= 0:
        return 0.0
    probs = sorted(safe_normalize(values), reverse=True)
    return sum(probs[:k])


def active_detection_metrics(
    true_exposures: list[float],
    predicted_exposures: list[float],
    depth: int | float,
    exposure_threshold: float = 0.01,
    min_expected_mutations: float = 10.0,
) -> dict[str, float]:
    true_active = active_mask(
        true_exposures,
        depth,
        exposure_threshold=exposure_threshold,
        min_expected_mutations=min_expected_mutations,
    )
    pred_active = active_mask(
        predicted_exposures,
        depth,
        exposure_threshold=exposure_threshold,
        min_expected_mutations=min_expected_mutations,
    )
    tp = sum(1 for truth, pred in zip(true_active, pred_active) if truth and pred)
    fp = sum(1 for truth, pred in zip(true_active, pred_active) if not truth and pred)
    fn = sum(1 for truth, pred in zip(true_active, pred_active) if truth and not pred)
    if tp == 0 and fp == 0 and fn == 0:
        return {
            "active_precision": 1.0,
            "active_recall": 1.0,
            "active_f1": 1.0,
            "true_active_count": 0.0,
            "pred_active_count": 0.0,
        }
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "active_precision": precision,
        "active_recall": recall,
        "active_f1": f1,
        "true_active_count": float(sum(true_active)),
        "pred_active_count": float(sum(pred_active)),
    }


def exposure_recovery_metrics(
    true_exposures: list[float],
    predicted_exposures: list[float],
    depth: int | float,
) -> dict[str, float]:
    predicted = safe_normalize(predicted_exposures)
    truth = safe_normalize(true_exposures)
    return {
        "exposure_mae": mae(truth, predicted),
        "exposure_mse": mse(truth, predicted),
        "exposure_cosine": cosine_similarity(truth, predicted),
        "true_effective_signatures": effective_signatures(truth),
        "pred_effective_signatures": effective_signatures(predicted),
        "true_top3_mass": top_k_mass(truth, 3),
        "pred_top3_mass": top_k_mass(predicted, 3),
        **active_detection_metrics(truth, predicted, depth),
    }
