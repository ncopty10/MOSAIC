from typing import Dict, Tuple

import numpy as np
import torch


def _sample_dirichlet(rng: np.random.Generator, alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha, a_min=1e-6, a_max=None)
    return rng.dirichlet(alpha)


def build_augmented_signature_bank(
    signature_bank: np.ndarray,
    num_mock_signatures: int,
    max_cosine_similarity: float,
    seed: int,
) -> np.ndarray:
    if num_mock_signatures <= 0:
        return signature_bank.astype(np.float32, copy=True)

    rng = np.random.default_rng(seed)
    num_channels = signature_bank.shape[1]
    normalized_bank = signature_bank / np.clip(signature_bank.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    all_signatures = [row.astype(np.float32, copy=True) for row in normalized_bank]
    all_norms = [
        row / np.clip(np.linalg.norm(row), a_min=1e-12, a_max=None)
        for row in normalized_bank
    ]

    while len(all_signatures) < normalized_bank.shape[0] + num_mock_signatures:
        alpha = np.full(num_channels, rng.choice([0.1, 0.3, 0.7, 1.0, 3.0, 10.0], replace=True), dtype=np.float64)
        candidate = _sample_dirichlet(rng, alpha).astype(np.float32)
        candidate_norm = candidate / np.clip(np.linalg.norm(candidate), a_min=1e-12, a_max=None)
        cosine_scores = np.asarray([float(candidate_norm @ existing_norm) for existing_norm in all_norms], dtype=np.float32)
        if float(cosine_scores.max()) <= max_cosine_similarity:
            all_signatures.append(candidate)
            all_norms.append(candidate_norm)

    return np.stack(all_signatures, axis=0).astype(np.float32)


def _choose_reference_subset(
    rng: np.random.Generator,
    full_bank: np.ndarray,
    num_cosmic_signatures: int,
    min_pool_size: int,
    max_pool_size: int,
    min_cosmic_fraction: float,
) -> np.ndarray:
    pool_size = int(rng.integers(min_pool_size, max_pool_size + 1))
    min_cosmic = min(num_cosmic_signatures, max(1, int(round(pool_size * min_cosmic_fraction))))
    cosmic_count = int(rng.integers(min_cosmic, min(pool_size, num_cosmic_signatures) + 1))
    mock_count = pool_size - cosmic_count

    cosmic_indices = rng.choice(num_cosmic_signatures, size=cosmic_count, replace=False)
    if mock_count > 0:
        mock_indices = rng.choice(
            np.arange(num_cosmic_signatures, full_bank.shape[0]),
            size=mock_count,
            replace=False,
        )
        chosen = np.concatenate([cosmic_indices, mock_indices])
    else:
        chosen = cosmic_indices
    rng.shuffle(chosen)
    return full_bank[chosen].astype(np.float32)


def sample_synthetic_batch(
    reference_bank: np.ndarray,
    num_cosmic_signatures: int,
    batch_size: int,
    num_channels: int,
    rng: np.random.Generator,
    device: torch.device,
    simulation_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    reference_subset = _choose_reference_subset(
        rng=rng,
        full_bank=reference_bank,
        num_cosmic_signatures=num_cosmic_signatures,
        min_pool_size=int(simulation_cfg.get("min_reference_pool_size", 10)),
        max_pool_size=int(simulation_cfg.get("max_reference_pool_size", 90)),
        min_cosmic_fraction=float(simulation_cfg.get("min_cosmic_fraction", 0.5)),
    )
    pool_size = reference_subset.shape[0]

    depth_ranges = simulation_cfg.get(
        "depth_ranges",
        {"low": [100, 300], "medium": [1000, 3000], "high": [10000, 30000]},
    )
    max_active_by_regime = simulation_cfg.get("max_active_signatures", {"low": 4, "medium": 10, "high": 16})
    regime_names = list(depth_ranges.keys())
    alpha_mix_values = np.asarray(simulation_cfg.get("mix_alpha_values", [0.1, 1.0, 10.0]), dtype=np.float64)
    alpha_noise_values = np.asarray(simulation_cfg.get("noise_alpha_values", [40.0, 120.0, 280.0, 2300.0]), dtype=np.float64)
    norm_frac = float(simulation_cfg.get("norm_frac", 0.25))

    inputs = []
    normalized_profiles = []
    true_exposures = []
    depths = []

    for _ in range(batch_size):
        regime = regime_names[int(rng.integers(0, len(regime_names)))]
        depth_low, depth_high = depth_ranges[regime]
        depth = int(rng.integers(int(depth_low), int(depth_high) + 1))
        max_active = min(int(max_active_by_regime.get(regime, pool_size)), pool_size)
        active_count = int(rng.integers(1, max_active + 1))
        active_indices = rng.choice(pool_size, size=active_count, replace=False)

        exposure_alpha = float(alpha_mix_values[int(rng.integers(0, len(alpha_mix_values)))])
        active_exposures = _sample_dirichlet(rng, np.full(active_count, exposure_alpha, dtype=np.float64)).astype(np.float32)
        exposure_vector = np.zeros(pool_size, dtype=np.float32)
        exposure_vector[active_indices] = active_exposures

        clean_profile = exposure_vector @ reference_subset
        noise_alpha = float(alpha_noise_values[int(rng.integers(0, len(alpha_noise_values)))])
        noisy_profile = _sample_dirichlet(rng, noise_alpha * np.clip(clean_profile, a_min=1e-6, a_max=None)).astype(np.float32)
        counts = rng.multinomial(depth, noisy_profile).astype(np.float32)
        normalized_noisy_profile = counts / np.clip(counts.sum(), a_min=1.0, a_max=None)

        if float(rng.random()) < norm_frac:
            model_input = normalized_noisy_profile
        else:
            model_input = counts

        inputs.append(model_input.astype(np.float32))
        normalized_profiles.append(normalized_noisy_profile.astype(np.float32))
        true_exposures.append(exposure_vector.astype(np.float32))
        depths.append(float(depth))

    batch_reference = np.broadcast_to(reference_subset, (batch_size, pool_size, num_channels)).copy()
    return {
        "input_profile": torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32, device=device),
        "target_profile": torch.tensor(np.stack(normalized_profiles, axis=0), dtype=torch.float32, device=device),
        "true_exposures": torch.tensor(np.stack(true_exposures, axis=0), dtype=torch.float32, device=device),
        "reference_signatures": torch.tensor(batch_reference, dtype=torch.float32, device=device),
        "depths": torch.tensor(np.asarray(depths, dtype=np.float32), dtype=torch.float32, device=device),
    }


def simulation_reference_bank_from_payload(payload: Dict, config: Dict) -> Tuple[np.ndarray, int]:
    simulation_cfg = config["training"].get("simulation", {})
    base_bank = np.asarray(payload["signature_bank"], dtype=np.float32)
    augmented_bank = build_augmented_signature_bank(
        signature_bank=base_bank,
        num_mock_signatures=int(simulation_cfg.get("num_mock_signatures", 500)),
        max_cosine_similarity=float(simulation_cfg.get("max_mock_cosine_similarity", 0.6)),
        seed=int(config["experiment"]["seed"]),
    )
    return augmented_bank, base_bank.shape[0]
