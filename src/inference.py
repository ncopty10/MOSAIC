from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch import nn

from src.data import create_datasets
from src.metrics import refitting_metrics
from src.model import build_model_from_config
from src.utils import ensure_dir, infer_device, seed_everything


def _gather_reference_signatures(signature_bank: torch.Tensor, pool_indices: torch.Tensor) -> torch.Tensor:
    flat_indices = pool_indices.reshape(-1)
    gathered = torch.index_select(signature_bank, 0, flat_indices)
    return gathered.reshape(pool_indices.shape[0], pool_indices.shape[1], signature_bank.shape[1])


def _load_model_bundle(checkpoint_path: str) -> Dict:
    # Checkpoints are trusted local artifacts from this project and may contain
    # non-tensor metadata, which requires weights_only=False on PyTorch 2.6+.
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _resolve_split_indices(payload: Dict, split: str):
    if split == "train":
        return payload["train_idx"]
    if split == "val":
        return payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    if split == "test":
        return payload["test_idx"] if len(payload["test_idx"]) > 0 else payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    return torch.arange(len(payload["sample_ids"])).numpy()


def _split_output_path(base_path: Path, split: str) -> Path:
    return base_path.with_name("{0}_{1}{2}".format(base_path.stem, split, base_path.suffix))


def _save_split_outputs(
    bundle: Dict,
    payload: Dict,
    indices,
    split: str,
    target: Path,
    model: nn.Module,
    signature_bank: torch.Tensor,
    device: torch.device,
) -> Dict[str, str]:
    exposure_rows = []
    active_probability_rows = []
    reconstruction_rows = []
    residual_rows = []
    config = bundle.get("config", {})

    with torch.no_grad():
        for row_idx in indices:
            profile = torch.tensor(payload["count_matrix"][row_idx], dtype=torch.float32, device=device).unsqueeze(0)
            pool_indices = torch.tensor(payload["reference_pool_indices"][row_idx], dtype=torch.long, device=device).unsqueeze(0)
            rna_embedding = None
            cnv_embedding = None
            methylation_embedding = None
            demographic_vector = None
            known_exposures = None
            if model.rna_dim > 0 and "rna_embeddings" in payload:
                rna_embedding = torch.tensor(payload["rna_embeddings"][row_idx], dtype=torch.float32, device=device).unsqueeze(0)
            if model.cnv_dim > 0 and "cnv_embeddings" in payload:
                cnv_embedding = torch.tensor(payload["cnv_embeddings"][row_idx], dtype=torch.float32, device=device).unsqueeze(0)
            if model.methylation_dim > 0 and "methylation_embeddings" in payload:
                methylation_embedding = torch.tensor(
                    payload["methylation_embeddings"][row_idx], dtype=torch.float32, device=device
                ).unsqueeze(0)
            if model.demographic_dim > 0 and "demographic_vectors" in payload:
                demographic_vector = torch.tensor(payload["demographic_vectors"][row_idx], dtype=torch.float32, device=device).unsqueeze(0)
            feed_known_exposures = bool(
                config.get("inference", {}).get(
                    "feed_known_exposures_to_model",
                    config.get("training", {}).get("feed_known_exposures_to_model", True),
                )
            )
            if feed_known_exposures and "known_exposures" in payload:
                known_exposures = torch.tensor(payload["known_exposures"][row_idx], dtype=torch.float32, device=device).unsqueeze(0)
            ref_signatures = _gather_reference_signatures(signature_bank, pool_indices)
            outputs = model(
                profile,
                ref_signatures,
                rna_embedding=rna_embedding,
                cnv_embedding=cnv_embedding,
                methylation_embedding=methylation_embedding,
                demographic_vector=demographic_vector,
                known_exposures=known_exposures,
            )
            exposures = outputs["exposures"].squeeze(0).cpu().numpy()
            active_probabilities = (
                outputs["active_probabilities"].squeeze(0).cpu().numpy()
                if "active_probabilities" in outputs
                else None
            )
            reconstruction = outputs["reconstruction"].squeeze(0).cpu().numpy()
            target_profile = payload["normalized_profiles"][row_idx]
            residual = target_profile - reconstruction
            pool = payload["reference_pool_indices"][row_idx]

            exposure_row = {
                "sample_id": payload["sample_ids"][row_idx],
                "depth": float(payload["depths"][row_idx]),
                "split": split,
            }
            for exposure, sig_idx in zip(exposures, pool):
                exposure_row[bundle["signature_names"][int(sig_idx)]] = float(exposure)
            exposure_rows.append(exposure_row)
            if active_probabilities is not None:
                active_probability_row = {
                    "sample_id": payload["sample_ids"][row_idx],
                    "depth": float(payload["depths"][row_idx]),
                    "split": split,
                }
                for probability, sig_idx in zip(active_probabilities, pool):
                    active_probability_row[bundle["signature_names"][int(sig_idx)]] = float(probability)
                active_probability_rows.append(active_probability_row)

            metrics = refitting_metrics(
                torch.tensor(reconstruction, dtype=torch.float32).unsqueeze(0),
                torch.tensor(target_profile, dtype=torch.float32).unsqueeze(0),
            )
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
                bundle["mutation_types"],
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
    active_probability_path = target.with_name("{0}_active_probabilities{1}".format(target.stem, target.suffix))
    if active_probability_rows:
        active_probability_df = pd.DataFrame(active_probability_rows).fillna(0.0)
        active_probability_df.to_csv(active_probability_path, index=False)

    reconstruction_path = target.with_name("{0}_reconstructions{1}".format(target.stem, target.suffix))
    reconstruction_df = pd.DataFrame(reconstruction_rows)
    reconstruction_df.to_csv(reconstruction_path, index=False)

    residual_path = target.with_name("{0}_residuals{1}".format(target.stem, target.suffix))
    residual_df = pd.DataFrame(residual_rows)
    residual_df.to_csv(residual_path, index=False)

    print("Saved {0} signature exposures to {1}".format(split, target))
    if active_probability_rows:
        print("Saved {0} active probabilities to {1}".format(split, active_probability_path))
    print("Saved {0} reconstructions to {1}".format(split, reconstruction_path))
    print("Saved {0} residuals to {1}".format(split, residual_path))
    return {
        "split": split,
        "exposures_path": str(target),
        "active_probabilities_path": str(active_probability_path) if active_probability_rows else "",
        "reconstructions_path": str(reconstruction_path),
        "residuals_path": str(residual_path),
    }


def infer_from_checkpoint(
    checkpoint_path: str,
    output_path: str,
    split: str = "all_splits",
    seed: int = 42,
) -> List[Dict[str, str]]:
    seed_everything(seed)
    bundle = _load_model_bundle(checkpoint_path)
    config = bundle["config"]
    device = infer_device(config["experiment"].get("device", "auto"))

    _, _, _, payload = create_datasets(config)
    bundle_rna_dim = int(bundle.get("rna_dim", 0))
    rna_dim = bundle_rna_dim if bundle_rna_dim > 0 else 0
    bundle_cnv_dim = int(bundle.get("cnv_dim", 0))
    cnv_dim = bundle_cnv_dim if bundle_cnv_dim > 0 else 0
    bundle_methylation_dim = int(bundle.get("methylation_dim", 0))
    methylation_dim = bundle_methylation_dim if bundle_methylation_dim > 0 else 0
    bundle_demographic_dim = int(bundle.get("demographic_dim", 0))
    demographic_dim = bundle_demographic_dim if bundle_demographic_dim > 0 else 0

    model = build_model_from_config(
        config=config,
        num_channels=len(bundle["mutation_types"]),
        rna_dim=rna_dim,
        cnv_dim=cnv_dim,
        methylation_dim=methylation_dim,
        demographic_dim=demographic_dim,
    ).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()

    signature_bank = torch.tensor(payload["signature_bank"], dtype=torch.float32, device=device)
    target = Path(output_path)
    ensure_dir(str(target.parent))

    split_names = ["train", "val", "test"] if split == "all_splits" else [split]
    output_records = []
    for split_name in split_names:
        split_target = _split_output_path(target, split_name) if split == "all_splits" else target
        split_indices = _resolve_split_indices(payload, split_name)
        output_records.append(
            _save_split_outputs(
                bundle=bundle,
                payload=payload,
                indices=split_indices,
                split=split_name,
                target=split_target,
                model=model,
                signature_bank=signature_bank,
                device=device,
            )
        )

    return output_records
