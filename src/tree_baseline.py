from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor
import torch

from src.config import load_config
from src.data import create_datasets
from src.metrics import refitting_metrics
from src.utils import ensure_dir, save_json, seed_everything


def _resolve_split_indices(payload: Dict, split: str):
    if split == "train":
        return payload["train_idx"]
    if split == "val":
        return payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    if split == "test":
        return payload["test_idx"] if len(payload["test_idx"]) > 0 else payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    return np.arange(len(payload["sample_ids"]), dtype=np.int64)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(matrix.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)


def _fit_nnls_targets(profiles: np.ndarray, signature_bank: np.ndarray) -> np.ndarray:
    target_rows: List[np.ndarray] = []
    for row in profiles:
        exposures, _ = nnls(signature_bank.T, row.astype(np.float64))
        exposures = exposures.astype(np.float32)
        if exposures.sum() > 0:
            exposures = exposures / exposures.sum()
        target_rows.append(exposures)
    return np.asarray(target_rows, dtype=np.float32)


def _build_feature_matrix(
    payload: Dict,
    include_rna: bool,
    include_cnv: bool,
    include_methylation: bool,
    include_demographics: bool,
) -> np.ndarray:
    feature_parts = [payload["normalized_profiles"].astype(np.float32)]
    if include_rna:
        if "rna_embeddings" not in payload:
            raise ValueError("Tree baseline requested RNA features, but no RNA embeddings are available in the dataset payload.")
        feature_parts.append(payload["rna_embeddings"].astype(np.float32))
    if include_cnv:
        if "cnv_embeddings" not in payload:
            raise ValueError("Tree baseline requested CNV features, but no CNV embeddings are available in the dataset payload.")
        feature_parts.append(payload["cnv_embeddings"].astype(np.float32))
    if include_methylation:
        if "methylation_embeddings" not in payload:
            raise ValueError(
                "Tree baseline requested methylation features, but no methylation embeddings are available in the dataset payload."
            )
        feature_parts.append(payload["methylation_embeddings"].astype(np.float32))
    if include_demographics:
        if "demographic_vectors" not in payload:
            raise ValueError(
                "Tree baseline requested demographic features, but no demographic vectors are available in the dataset payload."
            )
        feature_parts.append(payload["demographic_vectors"].astype(np.float32))
    return np.concatenate(feature_parts, axis=1).astype(np.float32)


def _save_outputs(
    payload: Dict,
    indices: np.ndarray,
    split: str,
    output_path: Path,
    predicted_exposures: np.ndarray,
    signature_bank: np.ndarray,
) -> Dict[str, str]:
    exposure_rows = []
    reconstruction_rows = []
    residual_rows = []
    mutation_types = payload["mutation_types"]
    signature_names = payload["signature_names"]

    for local_idx, row_idx in enumerate(indices):
        row_idx = int(row_idx)
        target_profile = payload["normalized_profiles"][row_idx].astype(np.float32)
        exposures = predicted_exposures[local_idx].astype(np.float32)
        exposures = np.clip(exposures, a_min=0.0, a_max=None)
        if exposures.sum() > 0:
            exposures = exposures / exposures.sum()
        reconstruction = exposures @ signature_bank
        reconstruction = reconstruction / np.clip(reconstruction.sum(), a_min=1e-12, a_max=None)
        residual = target_profile - reconstruction
        metrics = refitting_metrics(
            torch.tensor(np.asarray(reconstruction, dtype=np.float32)[None, :]),
            torch.tensor(np.asarray(target_profile, dtype=np.float32)[None, :]),
        )

        exposure_row = {
            "sample_id": payload["sample_ids"][row_idx],
            "depth": float(payload["depths"][row_idx]),
            "split": split,
        }
        for sig_name, exposure_value in zip(signature_names, exposures):
            exposure_row[sig_name] = float(exposure_value)
        exposure_rows.append(exposure_row)

        reconstruction_row = {
            "sample_id": payload["sample_ids"][row_idx],
            "depth": float(payload["depths"][row_idx]),
            "split": split,
        }
        for metric_name, metric_value in metrics.items():
            reconstruction_row[metric_name] = float(metric_value)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exposure_rows).fillna(0.0).to_csv(output_path, index=False)
    reconstruction_path = output_path.with_name("{0}_reconstructions{1}".format(output_path.stem, output_path.suffix))
    pd.DataFrame(reconstruction_rows).to_csv(reconstruction_path, index=False)
    residual_path = output_path.with_name("{0}_residuals{1}".format(output_path.stem, output_path.suffix))
    pd.DataFrame(residual_rows).to_csv(residual_path, index=False)
    return {
        "split": split,
        "exposures_path": str(output_path),
        "reconstructions_path": str(reconstruction_path),
        "residuals_path": str(residual_path),
        "num_samples": int(len(indices)),
    }


def run_tree_baseline_from_config(config: Dict) -> Dict[str, object]:
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    _, _, _, payload = create_datasets(config)
    baseline_cfg = config.get("tree_baseline", {})
    include_rna = bool(baseline_cfg.get("include_rna", False))
    include_cnv = bool(baseline_cfg.get("include_cnv", False))
    include_methylation = bool(baseline_cfg.get("include_methylation", False))
    include_demographics = bool(baseline_cfg.get("include_demographics", False))
    split = str(baseline_cfg.get("split", "all_splits"))
    output_path = Path(
        str(
            baseline_cfg.get(
                "output_path",
                "outputs/predictions/tree_baseline_signature_exposures_{0}.csv".format(config["run_timestamp"]),
            )
        )
    )
    ensure_dir(str(output_path.parent))

    features = _build_feature_matrix(
        payload,
        include_rna=include_rna,
        include_cnv=include_cnv,
        include_methylation=include_methylation,
        include_demographics=include_demographics,
    )
    signature_bank = payload["signature_bank"].astype(np.float32)
    pseudo_targets = _fit_nnls_targets(payload["normalized_profiles"].astype(np.float32), signature_bank)

    train_indices = np.asarray(payload["train_idx"], dtype=np.int64)
    regressor = RandomForestRegressor(
        n_estimators=int(baseline_cfg.get("n_estimators", 300)),
        max_depth=baseline_cfg.get("max_depth"),
        min_samples_leaf=int(baseline_cfg.get("min_samples_leaf", 1)),
        random_state=int(config.get("experiment", {}).get("seed", 42)),
        n_jobs=int(baseline_cfg.get("n_jobs", -1)),
    )
    regressor.fit(features[train_indices], pseudo_targets[train_indices])

    split_names = ["train", "val", "test"] if split == "all_splits" else [split]
    output_records = []
    for split_name in split_names:
        split_indices = np.asarray(_resolve_split_indices(payload, split_name), dtype=np.int64)
        split_output = output_path.with_name("{0}_{1}{2}".format(output_path.stem, split_name, output_path.suffix)) if split == "all_splits" else output_path
        predictions = regressor.predict(features[split_indices]).astype(np.float32)
        output_records.append(
            _save_outputs(
                payload=payload,
                indices=split_indices,
                split=split_name,
                output_path=split_output,
                predicted_exposures=predictions,
                signature_bank=signature_bank,
            )
        )

    summary = {
        "config_path": config.get("config_path"),
        "run_timestamp": config.get("run_timestamp"),
        "include_rna": include_rna,
        "include_cnv": include_cnv,
        "include_methylation": include_methylation,
        "include_demographics": include_demographics,
        "target_exposure_source": "nnls_all_signatures",
        "output_records": output_records,
    }
    summary_path = output_path.with_name("{0}_summary.json".format(output_path.stem))
    save_json(summary, str(summary_path))
    summary["summary_path"] = str(summary_path)
    return summary


def load_config_and_run_tree_baseline(config_path: str) -> Dict[str, object]:
    return run_tree_baseline_from_config(load_config(config_path))
