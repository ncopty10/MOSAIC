from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import nnls
import torch

from src.config import load_config
from src.data import create_datasets
from src.metrics import refitting_metrics
from src.nmf_discovery import discover_signatures_from_config
from src.utils import ensure_dir, save_json, seed_everything


def _resolve_split_indices(payload: Dict, split: str):
    if split == "train":
        return payload["train_idx"]
    if split == "val":
        return payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    if split == "test":
        return payload["test_idx"] if len(payload["test_idx"]) > 0 else payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    return np.arange(len(payload["sample_ids"]), dtype=np.int64)


def _split_output_path(base_path: Path, split: str) -> Path:
    return base_path.with_name("{0}_{1}{2}".format(base_path.stem, split, base_path.suffix))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(matrix.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)


def _fit_single_nnls(
    target_profile: np.ndarray,
    signature_bank: np.ndarray,
) -> np.ndarray:
    exposures, _ = nnls(signature_bank.T, target_profile.astype(np.float64))
    exposures = exposures.astype(np.float32)
    row_sum = float(exposures.sum())
    if row_sum > 0:
        exposures = exposures / row_sum
    return exposures


def _build_reference_pool_indices(payload: Dict, strategy: str) -> np.ndarray:
    if strategy == "matched_pool":
        return payload["reference_pool_indices"]
    num_samples = len(payload["sample_ids"])
    num_signatures = payload["signature_bank"].shape[0]
    return np.tile(np.arange(num_signatures, dtype=np.int64), (num_samples, 1))


def _save_split_outputs(
    payload: Dict,
    indices,
    split: str,
    target: Path,
    reference_pool_indices: np.ndarray,
) -> Dict[str, str]:
    exposure_rows = []
    reconstruction_rows = []
    residual_rows = []
    signature_names = payload["signature_names"]
    mutation_types = payload["mutation_types"]
    full_signature_bank = payload["signature_bank"]

    for row_idx in indices:
        row_idx = int(row_idx)
        target_profile = payload["normalized_profiles"][row_idx].astype(np.float32)
        profile_counts = payload["count_matrix"][row_idx].astype(np.float32)
        pool = reference_pool_indices[row_idx]
        signature_bank = full_signature_bank[pool]
        exposures = _fit_single_nnls(target_profile, signature_bank)
        reconstruction = exposures @ signature_bank
        reconstruction = reconstruction / np.clip(reconstruction.sum(), a_min=1e-12, a_max=None)
        residual = target_profile - reconstruction
        metrics = refitting_metrics(
            torch.tensor(reconstruction, dtype=torch.float32).unsqueeze(0),
            torch.tensor(target_profile, dtype=torch.float32).unsqueeze(0),
        )

        exposure_row = {
            "sample_id": payload["sample_ids"][row_idx],
            "depth": float(payload["depths"][row_idx]),
            "split": split,
        }
        for exposure, sig_idx in zip(exposures, pool):
            exposure_row[signature_names[int(sig_idx)]] = float(exposure)
        exposure_rows.append(exposure_row)

        reconstruction_row = {
            "sample_id": payload["sample_ids"][row_idx],
            "depth": float(payload["depths"][row_idx]),
            "split": split,
            "cosine_similarity": float(metrics["cosine_similarity"]),
            "mse": float(metrics["mse"]),
            "mae": float(metrics["mae"]),
        }
        residual_row = {
            "sample_id": payload["sample_ids"][row_idx],
            "depth": float(payload["depths"][row_idx]),
            "split": split,
        }
        for channel_name, observed_value, reconstructed_value, residual_value in zip(
            mutation_types,
            target_profile,
            reconstruction,
            residual,
        ):
            reconstruction_row["observed::{0}".format(channel_name)] = float(observed_value)
            reconstruction_row["reconstructed::{0}".format(channel_name)] = float(reconstructed_value)
            residual_row["residual::{0}".format(channel_name)] = float(residual_value)
        reconstruction_rows.append(reconstruction_row)
        residual_rows.append(residual_row)

    exposure_df = pd.DataFrame(exposure_rows).fillna(0.0)
    target.parent.mkdir(parents=True, exist_ok=True)
    exposure_df.to_csv(target, index=False)

    reconstruction_path = target.with_name("{0}_reconstructions{1}".format(target.stem, target.suffix))
    pd.DataFrame(reconstruction_rows).to_csv(reconstruction_path, index=False)

    residual_path = target.with_name("{0}_residuals{1}".format(target.stem, target.suffix))
    pd.DataFrame(residual_rows).to_csv(residual_path, index=False)

    return {
        "split": split,
        "exposures_path": str(target),
        "reconstructions_path": str(reconstruction_path),
        "residuals_path": str(residual_path),
        "num_samples": int(len(indices)),
    }


def run_nnls_baseline_from_config(config: Dict) -> Dict[str, object]:
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    _, _, _, payload = create_datasets(config)
    baseline_cfg = config.get("nnls_baseline", {})
    split = str(baseline_cfg.get("split", config.get("inference", {}).get("split", "all_splits")))
    output_path = str(
        baseline_cfg.get(
            "output_path",
            "outputs/predictions/nnls_signature_exposures_{0}.csv".format(config["run_timestamp"]),
        )
    )
    reference_pool_strategy = str(baseline_cfg.get("reference_pool_strategy", "all")).strip().lower()
    if reference_pool_strategy not in {"all", "matched_pool"}:
        raise ValueError(
            "Unsupported nnls_baseline.reference_pool_strategy '{0}'. Use 'all' or 'matched_pool'.".format(
                reference_pool_strategy
            )
        )

    reference_pool_indices = _build_reference_pool_indices(payload, reference_pool_strategy)
    target = Path(output_path)
    ensure_dir(str(target.parent))

    split_names = ["train", "val", "test"] if split == "all_splits" else [split]
    output_records = []
    for split_name in split_names:
        split_target = _split_output_path(target, split_name) if split == "all_splits" else target
        split_indices = _resolve_split_indices(payload, split_name)
        output_records.append(
            _save_split_outputs(
                payload=payload,
                indices=split_indices,
                split=split_name,
                target=split_target,
                reference_pool_indices=reference_pool_indices,
            )
        )

    summary = {
        "config_path": config.get("config_path"),
        "run_timestamp": config.get("run_timestamp"),
        "reference_pool_strategy": reference_pool_strategy,
        "output_records": output_records,
    }
    summary_path = target.with_name("{0}_summary.json".format(target.stem))
    save_json(summary, str(summary_path))
    summary["summary_path"] = str(summary_path)
    return summary


def run_nnls_residual_discovery_from_config(config: Dict) -> Dict[str, object]:
    nnls_outputs = run_nnls_baseline_from_config(config)
    discovery_cfg = deepcopy(config)
    nnls_discovery_cfg = discovery_cfg.setdefault("nnls_residual_discovery", {})
    desired_split = str(nnls_discovery_cfg.get("split", "val"))
    selected_record = None
    for record in nnls_outputs["output_records"]:
        if record["split"] == desired_split:
            selected_record = record
            break
    if selected_record is None:
        raise ValueError("Could not find NNLS residual outputs for split '{0}'.".format(desired_split))

    nmf_cfg = discovery_cfg.setdefault("nmf_discovery", {})
    nmf_cfg["residual_path"] = selected_record["residuals_path"]
    nmf_cfg["split"] = desired_split
    if nnls_discovery_cfg.get("output_dir"):
        nmf_cfg["output_dir"] = nnls_discovery_cfg["output_dir"]
    if "refit" in nnls_discovery_cfg:
        nmf_cfg["refit"] = deepcopy(nnls_discovery_cfg["refit"])

    discovery_outputs = discover_signatures_from_config(discovery_cfg)
    return {
        "nnls_outputs": nnls_outputs,
        "discovery_outputs": discovery_outputs,
    }


def load_config_and_run_nnls_baseline(config_path: str) -> Dict[str, object]:
    return run_nnls_baseline_from_config(load_config(config_path))


def load_config_and_run_nnls_residual_discovery(config_path: str) -> Dict[str, object]:
    return run_nnls_residual_discovery_from_config(load_config(config_path))
