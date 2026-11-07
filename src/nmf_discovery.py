import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-{0}".format(os.environ.get("USER", "codex")))
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

from src.data import create_datasets
from src.model import build_model_from_config
from src.utils import ensure_dir, infer_device, save_json, seed_everything


def _read_table(path: str, delimiter: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt", ".all"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=delimiter)


def load_residual_matrix(residual_csv_path: str) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, str]:
    df = pd.read_csv(residual_csv_path)
    residual_columns = [column for column in df.columns if column.startswith("residual::")]
    if not residual_columns:
        raise ValueError("Expected residual columns prefixed with 'residual::' in {0}".format(residual_csv_path))

    sample_ids = df["sample_id"].astype(str).tolist()
    depths = df["depth"].to_numpy(dtype=np.float32) if "depth" in df.columns else np.zeros(len(df), dtype=np.float32)
    channels = [column.split("residual::", 1)[1] for column in residual_columns]
    matrix = df[residual_columns].to_numpy(dtype=np.float32)
    split = str(df["split"].iloc[0]) if "split" in df.columns and len(df) > 0 else "unknown"
    return sample_ids, depths, channels, matrix, split


def make_positive_residual_matrix(residual_matrix: np.ndarray) -> np.ndarray:
    return np.maximum(residual_matrix, 0.0).astype(np.float32)


def filter_residual_samples(
    sample_ids: List[str],
    depths: np.ndarray,
    residual_matrix: np.ndarray,
    min_residual_sum: float,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, float]]:
    row_sums = residual_matrix.sum(axis=1)
    keep_mask = row_sums >= float(min_residual_sum)
    filtered_sample_ids = [sample_id for sample_id, keep in zip(sample_ids, keep_mask) if keep]
    filtered_depths = depths[keep_mask]
    filtered_matrix = residual_matrix[keep_mask]
    summary = {
        "num_input_samples": int(len(sample_ids)),
        "num_kept_samples": int(keep_mask.sum()),
        "num_dropped_samples": int((~keep_mask).sum()),
        "min_residual_sum": float(min_residual_sum),
        "mean_positive_residual_sum_before_filter": float(row_sums.mean()) if len(row_sums) > 0 else 0.0,
        "mean_positive_residual_sum_after_filter": float(filtered_matrix.sum(axis=1).mean()) if len(filtered_matrix) > 0 else 0.0,
    }
    return filtered_sample_ids, filtered_depths, filtered_matrix, summary


def run_single_nmf(
    residual_matrix: np.ndarray,
    rank: int,
    seed: int,
    max_iter: int,
    init: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model = NMF(
        n_components=rank,
        init=init,
        random_state=seed,
        max_iter=max_iter,
        solver="cd",
        beta_loss="frobenius",
    )
    exposures = model.fit_transform(residual_matrix)
    signatures = model.components_
    scale = np.clip(signatures.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    signatures = signatures / scale
    exposures = exposures * scale.squeeze(1)
    reconstruction_error = float(model.reconstruction_err_)
    return exposures.astype(np.float32), signatures.astype(np.float32), reconstruction_error


def _run_single_nmf_task(task: Dict) -> Dict:
    row_indices = np.asarray(task["row_indices"], dtype=np.int64)
    residual_matrix = np.asarray(task["residual_matrix"], dtype=np.float32)
    sample_ids = task["sample_ids"]
    fit_matrix = residual_matrix[row_indices]
    fit_sample_ids = [sample_ids[int(index)] for index in row_indices]
    exposures, signatures, reconstruction_error = run_single_nmf(
        residual_matrix=fit_matrix,
        rank=int(task["rank"]),
        seed=int(task["seed"]),
        max_iter=int(task["max_iter"]),
        init=str(task["init"]),
    )
    return {
        "rank": int(task["rank"]),
        "run_idx": int(task["run_idx"]),
        "seed": int(task["seed"]),
        "row_indices": row_indices,
        "sample_ids": fit_sample_ids,
        "exposures": exposures,
        "signatures": signatures,
        "reconstruction_error": reconstruction_error,
        "num_fit_samples": int(fit_matrix.shape[0]),
    }


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs and n_jobs > 0:
        return int(n_jobs)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def run_repeated_nmf(
    residual_matrix: np.ndarray,
    sample_ids: List[str],
    rank: int,
    n_runs: int,
    bootstrap_rows: bool,
    max_iter: int,
    init: str,
    base_seed: int,
    n_jobs: int = 1,
) -> List[Dict]:
    rng = np.random.default_rng(base_seed + rank * 1000)
    num_samples = residual_matrix.shape[0]
    tasks = []

    for run_idx in range(n_runs):
        seed = int(rng.integers(0, 1_000_000_000))
        if bootstrap_rows:
            row_indices = rng.integers(0, num_samples, size=num_samples)
        else:
            row_indices = np.arange(num_samples, dtype=np.int64)
        tasks.append(
            {
                "residual_matrix": residual_matrix,
                "sample_ids": sample_ids,
                "rank": int(rank),
                "run_idx": int(run_idx),
                "seed": seed,
                "row_indices": row_indices.astype(np.int64),
                "max_iter": int(max_iter),
                "init": str(init),
            }
        )

    resolved_n_jobs = max(1, min(_resolve_n_jobs(n_jobs), n_runs))
    if resolved_n_jobs == 1:
        run_results = [_run_single_nmf_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=resolved_n_jobs) as executor:
            run_results = list(executor.map(_run_single_nmf_task, tasks))
    run_results.sort(key=lambda item: item["run_idx"])
    return run_results


def summarize_rank_results(run_results: List[Dict]) -> pd.DataFrame:
    rows = []
    for rank in sorted({result["rank"] for result in run_results}):
        rank_results = [result for result in run_results if result["rank"] == rank]
        errors = np.asarray([result["reconstruction_error"] for result in rank_results], dtype=np.float64)
        rows.append(
            {
                "rank": int(rank),
                "num_runs": int(len(rank_results)),
                "mean_reconstruction_error": float(errors.mean()),
                "std_reconstruction_error": float(errors.std(ddof=0)),
                "min_reconstruction_error": float(errors.min()),
                "max_reconstruction_error": float(errors.max()),
            }
        )
    return pd.DataFrame(rows)


def load_cosmic_bank(
    signature_path: str,
    signature_delimiter: str,
    signature_type_col: str,
    expected_channels: List[str],
) -> Tuple[List[str], np.ndarray]:
    sig_df = _read_table(signature_path, signature_delimiter)
    if signature_type_col not in sig_df.columns:
        raise ValueError("Signature file must include a '{0}' column.".format(signature_type_col))
    sig_df = sig_df.set_index(signature_type_col)
    missing = [channel for channel in expected_channels if channel not in sig_df.index]
    if missing:
        raise ValueError("Signature bank is missing expected channels: {0}".format(", ".join(missing[:10])))

    sig_df = sig_df.loc[expected_channels]
    signature_names = sig_df.columns.astype(str).tolist()
    signature_bank = sig_df.transpose().to_numpy(dtype=np.float32)
    signature_bank = signature_bank / np.clip(signature_bank.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return signature_names, signature_bank


def compare_candidates_to_cosmic(
    candidate_signatures: np.ndarray,
    cosmic_names: List[str],
    cosmic_bank: np.ndarray,
    novelty_cosine_threshold: float,
    known_like_cosine_threshold: float,
) -> pd.DataFrame:
    cosine_matrix = cosine_similarity(candidate_signatures, cosmic_bank)
    best_indices = cosine_matrix.argmax(axis=1)
    best_scores = cosine_matrix[np.arange(cosine_matrix.shape[0]), best_indices]
    labels = []
    for score in best_scores:
        if score >= known_like_cosine_threshold:
            labels.append("known_like")
        elif score >= novelty_cosine_threshold:
            labels.append("related")
        else:
            labels.append("distinct_candidate")

    rows = []
    for candidate_idx, (best_idx, best_score, label) in enumerate(zip(best_indices, best_scores, labels)):
        rows.append(
            {
                "candidate_index": int(candidate_idx),
                "best_cosmic_match": cosmic_names[int(best_idx)],
                "max_cosine_to_cosmic": float(best_score),
                "novelty_label": label,
            }
        )
    return pd.DataFrame(rows)


def _get_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError:
        return None


def _gather_reference_signatures(signature_bank: torch.Tensor, pool_indices: torch.Tensor) -> torch.Tensor:
    flat_indices = pool_indices.reshape(-1)
    gathered = torch.index_select(signature_bank, 0, flat_indices)
    return gathered.reshape(pool_indices.shape[0], pool_indices.shape[1], signature_bank.shape[1])


def _load_model_bundle(checkpoint_path: str) -> Dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _resolve_split_indices(payload: Dict, split: str):
    if split == "train":
        return payload["train_idx"]
    if split == "val":
        return payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    if split == "test":
        return payload["test_idx"] if len(payload["test_idx"]) > 0 else payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    return np.arange(len(payload["sample_ids"]), dtype=np.int64)


def _normalize_signature(signature: np.ndarray) -> np.ndarray:
    signature = np.asarray(signature, dtype=np.float32)
    return signature / np.clip(signature.sum(), a_min=1e-12, a_max=None)


def _extract_component_records(run_results: List[Dict]) -> List[Dict]:
    component_records = []
    raw_candidate_rows = []
    raw_exposure_rows = []
    flat_signatures = []

    for result in run_results:
        for component_idx, signature in enumerate(result["signatures"]):
            candidate_index = len(component_records)
            component_exposures = result["exposures"][:, component_idx]
            component_records.append(
                {
                    "candidate_index": int(candidate_index),
                    "rank": int(result["rank"]),
                    "run_idx": int(result["run_idx"]),
                    "seed": int(result["seed"]),
                    "component_idx": int(component_idx),
                    "signature": signature.astype(np.float32),
                    "sample_ids": list(result["sample_ids"]),
                    "exposures": component_exposures.astype(np.float32),
                    "reconstruction_error": float(result["reconstruction_error"]),
                }
            )
            raw_candidate_rows.append(
                {
                    "candidate_index": int(candidate_index),
                    "rank": int(result["rank"]),
                    "run_idx": int(result["run_idx"]),
                    "seed": int(result["seed"]),
                    "component_idx": int(component_idx),
                    "reconstruction_error": float(result["reconstruction_error"]),
                }
            )
            flat_signatures.append(signature.astype(np.float32))

        for sample_position, sample_id in enumerate(result["sample_ids"]):
            row = {
                "rank": int(result["rank"]),
                "run_idx": int(result["run_idx"]),
                "seed": int(result["seed"]),
                "sample_id": sample_id,
            }
            for component_idx, value in enumerate(result["exposures"][sample_position]):
                row["component_{0}".format(component_idx)] = float(value)
            raw_exposure_rows.append(row)

    return component_records, raw_candidate_rows, raw_exposure_rows, np.asarray(flat_signatures, dtype=np.float32)


def _cluster_rank_components(component_records: List[Dict], cosine_threshold: float) -> List[Dict]:
    clusters: List[Dict] = []
    sorted_records = sorted(component_records, key=lambda item: (item["rank"], item["run_idx"], item["component_idx"]))

    for record in sorted_records:
        signature = record["signature"]
        best_cluster_idx = -1
        best_score = -1.0
        for cluster_idx, cluster in enumerate(clusters):
            score = float(cosine_similarity(signature.reshape(1, -1), cluster["centroid"].reshape(1, -1))[0, 0])
            if score > best_score:
                best_score = score
                best_cluster_idx = cluster_idx

        if best_cluster_idx >= 0 and best_score >= cosine_threshold:
            cluster = clusters[best_cluster_idx]
            cluster["members"].append(record)
            cluster["centroid"] = _normalize_signature(np.mean([member["signature"] for member in cluster["members"]], axis=0))
        else:
            clusters.append(
                {
                    "rank": int(record["rank"]),
                    "members": [record],
                    "centroid": signature.copy(),
                }
            )
    return clusters


def build_consensus_components(
    run_results: List[Dict],
    channels: List[str],
    filtered_depths_by_sample: Dict[str, float],
    cosmic_names: List[str],
    cosmic_bank: np.ndarray,
    consensus_cosine_threshold: float,
    novelty_cosine_threshold: float,
    known_like_cosine_threshold: float,
    top_samples_per_component: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, List[Dict]]]:
    component_records_by_rank: Dict[int, List[Dict]] = {}
    for result in run_results:
        for component_idx, signature in enumerate(result["signatures"]):
            component_records_by_rank.setdefault(int(result["rank"]), []).append(
                {
                    "rank": int(result["rank"]),
                    "run_idx": int(result["run_idx"]),
                    "seed": int(result["seed"]),
                    "component_idx": int(component_idx),
                    "signature": signature.astype(np.float32),
                    "sample_ids": list(result["sample_ids"]),
                    "exposures": result["exposures"][:, component_idx].astype(np.float32),
                    "reconstruction_error": float(result["reconstruction_error"]),
                }
            )

    consensus_rows = []
    sample_rows = []
    top_sample_rows = []
    clusters_by_rank: Dict[int, List[Dict]] = {}

    for rank in sorted(component_records_by_rank):
        rank_records = component_records_by_rank[rank]
        clusters = _cluster_rank_components(rank_records, consensus_cosine_threshold)
        clusters.sort(
            key=lambda cluster: (
                -len({member["run_idx"] for member in cluster["members"]}),
                -len(cluster["members"]),
            )
        )

        num_runs_for_rank = len({member["run_idx"] for member in rank_records})
        finalized_clusters = []
        for cluster_idx, cluster in enumerate(clusters):
            centroid = _normalize_signature(np.mean([member["signature"] for member in cluster["members"]], axis=0))
            centroid_scores = cosine_similarity(
                np.asarray([member["signature"] for member in cluster["members"]], dtype=np.float32),
                centroid.reshape(1, -1),
            ).reshape(-1)
            run_fraction = len({member["run_idx"] for member in cluster["members"]}) / max(num_runs_for_rank, 1)

            cosmic_info = compare_candidates_to_cosmic(
                candidate_signatures=centroid.reshape(1, -1),
                cosmic_names=cosmic_names,
                cosmic_bank=cosmic_bank,
                novelty_cosine_threshold=novelty_cosine_threshold,
                known_like_cosine_threshold=known_like_cosine_threshold,
            ).iloc[0]

            exposure_by_sample: Dict[str, List[float]] = {}
            for member in cluster["members"]:
                for sample_id, exposure in zip(member["sample_ids"], member["exposures"]):
                    exposure_by_sample.setdefault(sample_id, []).append(float(exposure))

            consensus_id = "rank{0}_component{1}".format(rank, cluster_idx)
            consensus_rows.append(
                {
                    "rank": int(rank),
                    "consensus_component": consensus_id,
                    "cluster_idx": int(cluster_idx),
                    "num_member_components": int(len(cluster["members"])),
                    "num_runs_represented": int(len({member["run_idx"] for member in cluster["members"]})),
                    "run_fraction": float(run_fraction),
                    "mean_member_cosine_to_centroid": float(centroid_scores.mean()),
                    "min_member_cosine_to_centroid": float(centroid_scores.min()),
                    "best_cosmic_match": str(cosmic_info["best_cosmic_match"]),
                    "max_cosine_to_cosmic": float(cosmic_info["max_cosine_to_cosmic"]),
                    "novelty_label": str(cosmic_info["novelty_label"]),
                    **{"channel::{0}".format(channel): float(value) for channel, value in zip(channels, centroid)},
                }
            )

            exposure_summary_rows = []
            for sample_id, values in exposure_by_sample.items():
                values_arr = np.asarray(values, dtype=np.float32)
                exposure_summary_rows.append(
                    {
                        "rank": int(rank),
                        "consensus_component": consensus_id,
                        "sample_id": sample_id,
                        "depth": float(filtered_depths_by_sample.get(sample_id, 0.0)),
                        "mean_exposure": float(values_arr.mean()),
                        "max_exposure": float(values_arr.max()),
                        "num_observations": int(values_arr.size),
                    }
                )
            exposure_summary_rows.sort(key=lambda row: row["mean_exposure"], reverse=True)
            sample_rows.extend(exposure_summary_rows)
            top_sample_rows.extend(exposure_summary_rows[:top_samples_per_component])

            finalized_clusters.append(
                {
                    "rank": int(rank),
                    "consensus_component": consensus_id,
                    "centroid": centroid,
                    "members": cluster["members"],
                    "sample_exposures": exposure_summary_rows,
                }
            )

        clusters_by_rank[rank] = finalized_clusters

    return (
        pd.DataFrame(consensus_rows),
        pd.DataFrame(sample_rows),
        pd.DataFrame(top_sample_rows),
        clusters_by_rank,
    )


def _plot_rank_reconstruction(rank_summary_df: pd.DataFrame, output_path: Path, plt) -> None:
    if rank_summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        rank_summary_df["rank"],
        rank_summary_df["mean_reconstruction_error"],
        yerr=rank_summary_df["std_reconstruction_error"],
        marker="o",
        linewidth=1.5,
        capsize=4,
    )
    ax.set_xlabel("NMF rank")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Residual Reconstruction Error by Rank")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_consensus_component(signature: np.ndarray, channels: List[str], title: str, output_path: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(np.arange(len(channels)), signature, color="#1f77b4", width=0.9)
    ax.set_title(title)
    ax.set_ylabel("Contribution")
    ax.set_xlim(-0.5, len(channels) - 0.5)
    ax.set_xticks(np.arange(len(channels)))
    ax.set_xticklabels(channels, rotation=90, fontsize=6)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_exposure_histogram(exposures: List[float], title: str, output_path: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.asarray(exposures, dtype=np.float32), bins=30, color="#ff7f0e", edgecolor="black", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel("Mean sample exposure")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_consensus_plots(
    consensus_df: pd.DataFrame,
    consensus_sample_df: pd.DataFrame,
    channels: List[str],
    rank_summary_df: pd.DataFrame,
    output_dir: Path,
) -> List[str]:
    plt = _get_pyplot()
    if plt is None:
        print("Skipping NMF discovery plots because matplotlib is not installed.")
        return []

    plot_dir = ensure_dir(str(output_dir / "plots"))
    saved_paths = []
    rank_plot_path = plot_dir / "rank_reconstruction_error.png"
    _plot_rank_reconstruction(rank_summary_df, rank_plot_path, plt)
    saved_paths.append(str(rank_plot_path))

    if consensus_df.empty:
        return saved_paths

    channel_columns = ["channel::{0}".format(channel) for channel in channels]
    for _, row in consensus_df.iterrows():
        signature = row[channel_columns].to_numpy(dtype=np.float32)
        component_name = str(row["consensus_component"])
        barplot_path = plot_dir / "{0}_signature.png".format(component_name)
        hist_path = plot_dir / "{0}_exposure_histogram.png".format(component_name)
        _plot_consensus_component(
            signature=signature,
            channels=channels,
            title="{0} ({1}, cosine {2:.3f})".format(
                component_name,
                str(row["best_cosmic_match"]),
                float(row["max_cosine_to_cosmic"]),
            ),
            output_path=barplot_path,
            plt=plt,
        )
        exposures = (
            consensus_sample_df.loc[consensus_sample_df["consensus_component"] == component_name, "mean_exposure"].tolist()
            if not consensus_sample_df.empty
            else []
        )
        _plot_exposure_histogram(
            exposures=exposures,
            title="{0} sample exposures".format(component_name),
            output_path=hist_path,
            plt=plt,
        )
        saved_paths.extend([str(barplot_path), str(hist_path)])
    return saved_paths


def evaluate_augmented_refit(
    checkpoint_path: str,
    split: str,
    candidate_signatures: np.ndarray,
    candidate_names: List[str],
    filtered_sample_ids: List[str],
    processed_payload_path: str = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if candidate_signatures.shape[0] == 0:
        return pd.DataFrame(), {}

    seed_everything(42)
    bundle = _load_model_bundle(checkpoint_path)
    config = bundle["config"]
    requested_device = str(config["experiment"].get("device", "auto"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = infer_device(requested_device)
    if processed_payload_path:
        payload = torch.load(processed_payload_path, map_location="cpu", weights_only=False)
    else:
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

    sample_id_to_row = {str(sample_id): idx for idx, sample_id in enumerate(payload["sample_ids"])}
    split_indices = _resolve_split_indices(payload, split)
    split_sample_ids = {str(payload["sample_ids"][int(idx)]) for idx in split_indices}
    eval_row_indices = [sample_id_to_row[sample_id] for sample_id in filtered_sample_ids if sample_id in split_sample_ids]
    if not eval_row_indices:
        return pd.DataFrame(), {}

    base_signature_bank = torch.tensor(payload["signature_bank"], dtype=torch.float32, device=device)
    augmented_signature_bank = torch.cat(
        [
            base_signature_bank,
            torch.tensor(candidate_signatures, dtype=torch.float32, device=device),
        ],
        dim=0,
    )
    base_pool = torch.arange(base_signature_bank.shape[0], device=device, dtype=torch.long).unsqueeze(0)
    augmented_pool = torch.arange(augmented_signature_bank.shape[0], device=device, dtype=torch.long).unsqueeze(0)
    batch_size = 32

    rows = []
    base_cosines = []
    augmented_cosines = []
    base_mses = []
    augmented_mses = []
    base_positive = []
    augmented_positive = []

    with torch.no_grad():
        for start_idx in range(0, len(eval_row_indices), batch_size):
            batch_indices = eval_row_indices[start_idx : start_idx + batch_size]
            profile = torch.tensor(payload["count_matrix"][batch_indices], dtype=torch.float32, device=device)
            target_profile = torch.tensor(payload["normalized_profiles"][batch_indices], dtype=torch.float32, device=device)
            rna_embedding = None
            cnv_embedding = None
            methylation_embedding = None
            demographic_vector = None
            known_exposures = None
            if model.rna_dim > 0 and "rna_embeddings" in payload:
                rna_embedding = torch.tensor(payload["rna_embeddings"][batch_indices], dtype=torch.float32, device=device)
            if model.cnv_dim > 0 and "cnv_embeddings" in payload:
                cnv_embedding = torch.tensor(payload["cnv_embeddings"][batch_indices], dtype=torch.float32, device=device)
            if model.methylation_dim > 0 and "methylation_embeddings" in payload:
                methylation_embedding = torch.tensor(
                    payload["methylation_embeddings"][batch_indices], dtype=torch.float32, device=device
                )
            if model.demographic_dim > 0 and "demographic_vectors" in payload:
                demographic_vector = torch.tensor(payload["demographic_vectors"][batch_indices], dtype=torch.float32, device=device)
            if "known_exposures" in payload:
                known_exposures = torch.tensor(payload["known_exposures"][batch_indices], dtype=torch.float32, device=device)

            base_references = _gather_reference_signatures(base_signature_bank, base_pool.expand(len(batch_indices), -1))
            base_outputs = model(
                profile,
                base_references,
                rna_embedding=rna_embedding,
                cnv_embedding=cnv_embedding,
                methylation_embedding=methylation_embedding,
                demographic_vector=demographic_vector,
                known_exposures=known_exposures,
            )
            base_reconstruction = base_outputs["reconstruction"]

            augmented_references = _gather_reference_signatures(
                augmented_signature_bank,
                augmented_pool.expand(len(batch_indices), -1),
            )
            augmented_outputs = model(
                profile,
                augmented_references,
                rna_embedding=rna_embedding,
                cnv_embedding=cnv_embedding,
                methylation_embedding=methylation_embedding,
                demographic_vector=demographic_vector,
                known_exposures=None,
            )
            augmented_reconstruction = augmented_outputs["reconstruction"]

            batch_base_cosines = F.cosine_similarity(base_reconstruction, target_profile, dim=1).cpu().numpy()
            batch_augmented_cosines = F.cosine_similarity(augmented_reconstruction, target_profile, dim=1).cpu().numpy()
            batch_base_mses = ((base_reconstruction - target_profile) ** 2).mean(dim=1).cpu().numpy()
            batch_augmented_mses = ((augmented_reconstruction - target_profile) ** 2).mean(dim=1).cpu().numpy()
            batch_base_positive = torch.clamp(target_profile - base_reconstruction, min=0.0).sum(dim=1).cpu().numpy()
            batch_augmented_positive = torch.clamp(target_profile - augmented_reconstruction, min=0.0).sum(dim=1).cpu().numpy()
            batch_augmented_exposures = augmented_outputs["exposures"].cpu().numpy()

            for batch_pos, row_idx in enumerate(batch_indices):
                base_cosine = float(batch_base_cosines[batch_pos])
                augmented_cosine = float(batch_augmented_cosines[batch_pos])
                base_mse = float(batch_base_mses[batch_pos])
                augmented_mse = float(batch_augmented_mses[batch_pos])
                base_positive_residual = float(batch_base_positive[batch_pos])
                augmented_positive_residual = float(batch_augmented_positive[batch_pos])

                row = {
                    "sample_id": str(payload["sample_ids"][row_idx]),
                    "depth": float(payload["depths"][row_idx]),
                    "base_cosine_similarity": base_cosine,
                    "augmented_cosine_similarity": augmented_cosine,
                    "delta_cosine_similarity": augmented_cosine - base_cosine,
                    "base_mse": base_mse,
                    "augmented_mse": augmented_mse,
                    "delta_mse": augmented_mse - base_mse,
                    "base_positive_residual_sum": base_positive_residual,
                    "augmented_positive_residual_sum": augmented_positive_residual,
                    "delta_positive_residual_sum": augmented_positive_residual - base_positive_residual,
                }
                for candidate_offset, candidate_name in enumerate(candidate_names):
                    row["augmented::{0}".format(candidate_name)] = float(
                        batch_augmented_exposures[batch_pos, base_signature_bank.shape[0] + candidate_offset]
                    )
                rows.append(row)

                base_cosines.append(base_cosine)
                augmented_cosines.append(augmented_cosine)
                base_mses.append(base_mse)
                augmented_mses.append(augmented_mse)
                base_positive.append(base_positive_residual)
                augmented_positive.append(augmented_positive_residual)

    summary = {
        "num_eval_samples": int(len(rows)),
        "mean_base_cosine_similarity": float(np.mean(base_cosines)),
        "mean_augmented_cosine_similarity": float(np.mean(augmented_cosines)),
        "mean_delta_cosine_similarity": float(np.mean(np.asarray(augmented_cosines) - np.asarray(base_cosines))),
        "mean_base_mse": float(np.mean(base_mses)),
        "mean_augmented_mse": float(np.mean(augmented_mses)),
        "mean_delta_mse": float(np.mean(np.asarray(augmented_mses) - np.asarray(base_mses))),
        "mean_base_positive_residual_sum": float(np.mean(base_positive)),
        "mean_augmented_positive_residual_sum": float(np.mean(augmented_positive)),
        "mean_delta_positive_residual_sum": float(np.mean(np.asarray(augmented_positive) - np.asarray(base_positive))),
    }
    return pd.DataFrame(rows), summary


def _summarize_validation_candidate_gate(
    refit_df: pd.DataFrame,
    refit_summary: Dict[str, float],
    candidate_names: List[str],
    refit_cfg: Dict,
) -> Dict:
    candidate_columns = ["augmented::{0}".format(name) for name in candidate_names]
    candidate_columns = [column for column in candidate_columns if column in refit_df.columns]
    candidate_sum = (
        refit_df[candidate_columns].sum(axis=1).to_numpy(dtype=np.float32)
        if candidate_columns
        else np.zeros(len(refit_df), dtype=np.float32)
    )
    exposure_threshold = float(refit_cfg.get("candidate_exposure_threshold", 0.05))

    gate_summary = {
        "enabled": bool(refit_cfg.get("require_validation_improvement", False)),
        "passed": True,
        "reasons": [],
        "num_candidate_signatures": int(len(candidate_names)),
        "candidate_names": list(candidate_names),
        "min_mean_delta_cosine_similarity": float(refit_cfg.get("min_mean_delta_cosine_similarity", 0.0)),
        "max_mean_delta_mse": float(refit_cfg.get("max_mean_delta_mse", 0.0)),
        "max_mean_candidate_exposure_sum": (
            float(refit_cfg["max_mean_candidate_exposure_sum"])
            if refit_cfg.get("max_mean_candidate_exposure_sum") is not None
            else None
        ),
        "candidate_exposure_threshold": exposure_threshold,
        "max_fraction_samples_above_candidate_exposure_threshold": (
            float(refit_cfg["max_fraction_samples_above_candidate_exposure_threshold"])
            if refit_cfg.get("max_fraction_samples_above_candidate_exposure_threshold") is not None
            else None
        ),
        "mean_delta_cosine_similarity": float(refit_summary.get("mean_delta_cosine_similarity", 0.0)),
        "mean_delta_mse": float(refit_summary.get("mean_delta_mse", 0.0)),
        "mean_candidate_exposure_sum": float(np.mean(candidate_sum)) if len(candidate_sum) else 0.0,
        "median_candidate_exposure_sum": float(np.median(candidate_sum)) if len(candidate_sum) else 0.0,
        "p95_candidate_exposure_sum": float(np.quantile(candidate_sum, 0.95)) if len(candidate_sum) else 0.0,
        "fraction_samples_above_candidate_exposure_threshold": (
            float(np.mean(candidate_sum > exposure_threshold)) if len(candidate_sum) else 0.0
        ),
    }

    if not gate_summary["enabled"]:
        return gate_summary

    if gate_summary["mean_delta_cosine_similarity"] < gate_summary["min_mean_delta_cosine_similarity"]:
        gate_summary["passed"] = False
        gate_summary["reasons"].append("mean_delta_cosine_similarity_below_threshold")

    if gate_summary["mean_delta_mse"] > gate_summary["max_mean_delta_mse"]:
        gate_summary["passed"] = False
        gate_summary["reasons"].append("mean_delta_mse_above_threshold")

    max_mean_exposure = gate_summary["max_mean_candidate_exposure_sum"]
    if max_mean_exposure is not None and gate_summary["mean_candidate_exposure_sum"] > max_mean_exposure:
        gate_summary["passed"] = False
        gate_summary["reasons"].append("mean_candidate_exposure_sum_above_threshold")

    max_fraction_above = gate_summary["max_fraction_samples_above_candidate_exposure_threshold"]
    if (
        max_fraction_above is not None
        and gate_summary["fraction_samples_above_candidate_exposure_threshold"] > max_fraction_above
    ):
        gate_summary["passed"] = False
        gate_summary["reasons"].append("candidate_exposure_spread_above_threshold")

    return gate_summary


def _resolve_refit_consensus_subset(consensus_df: pd.DataFrame, refit_cfg: Dict, ranks: List[int]) -> pd.DataFrame:
    if consensus_df.empty:
        return consensus_df.copy()

    if bool(refit_cfg.get("use_validated_candidates", False)):
        if "selected_for_augmented_refit" not in consensus_df.columns:
            return consensus_df.iloc[0:0].copy()
        consensus_df = consensus_df.loc[consensus_df["selected_for_augmented_refit"].astype(bool)].copy()
        if consensus_df.empty:
            return consensus_df

    rank_value = refit_cfg.get("rank", max(int(rank) for rank in ranks))
    use_all_ranks = rank_value in {None, "all", "all_ranks", "ALL", "ALL_RANKS"}
    if use_all_ranks:
        refit_consensus = consensus_df.copy()
    else:
        refit_rank = int(rank_value)
        refit_consensus = consensus_df.loc[consensus_df["rank"] == refit_rank].copy()

    if bool(refit_cfg.get("stable_only", True)):
        refit_consensus = refit_consensus.loc[
            (refit_consensus["run_fraction"] >= float(refit_cfg.get("min_run_fraction", 0.5)))
            & (refit_consensus["mean_member_cosine_to_centroid"] >= float(refit_cfg.get("min_mean_member_cosine", 0.9)))
        ].copy()

    if bool(refit_cfg.get("distinct_only", False)):
        refit_consensus = refit_consensus.loc[
            refit_consensus["max_cosine_to_cosmic"] < float(refit_cfg.get("max_cosine_to_cosmic", 0.85))
        ].copy()

    max_candidates = int(refit_cfg.get("max_candidates", 0))
    if max_candidates > 0:
        refit_consensus = refit_consensus.sort_values(
            ["run_fraction", "mean_member_cosine_to_centroid", "max_cosine_to_cosmic"],
            ascending=[False, False, True],
        ).head(max_candidates)

    if bool(refit_cfg.get("redundancy_filter", False)) and not refit_consensus.empty:
        candidate_channel_columns = [column for column in refit_consensus.columns if column.startswith("channel::")]
        similarity_threshold = float(refit_cfg.get("max_cosine_between_candidates", 0.95))
        prefer_higher_rank = bool(refit_cfg.get("prefer_higher_rank_for_redundancy", True))

        if candidate_channel_columns:
            sort_columns = ["run_fraction", "mean_member_cosine_to_centroid", "max_cosine_to_cosmic"]
            ascending = [False, False, True]
            if prefer_higher_rank:
                sort_columns = ["rank"] + sort_columns
                ascending = [False] + ascending

            ordered = refit_consensus.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
            kept_rows = []
            kept_signatures = []
            for _, row in ordered.iterrows():
                signature = row[candidate_channel_columns].to_numpy(dtype=np.float32)
                if not kept_signatures:
                    kept_rows.append(row)
                    kept_signatures.append(signature)
                    continue
                similarities = cosine_similarity(
                    signature.reshape(1, -1),
                    np.asarray(kept_signatures, dtype=np.float32),
                ).reshape(-1)
                if float(similarities.max()) < similarity_threshold:
                    kept_rows.append(row)
                    kept_signatures.append(signature)

            refit_consensus = pd.DataFrame(kept_rows).reset_index(drop=True)

    return refit_consensus


def discover_signatures_from_config(config: Dict) -> Dict[str, str]:
    nmf_cfg = config["nmf_discovery"]
    base_seed = int(config.get("experiment", {}).get("seed", 42))
    sample_ids, depths, channels, residual_matrix, split = load_residual_matrix(nmf_cfg["residual_path"])
    positive_residual_matrix = make_positive_residual_matrix(residual_matrix)
    filtered_sample_ids, filtered_depths, filtered_matrix, filter_summary = filter_residual_samples(
        sample_ids=sample_ids,
        depths=depths,
        residual_matrix=positive_residual_matrix,
        min_residual_sum=float(nmf_cfg.get("min_residual_sum", 0.0)),
    )

    if filtered_matrix.shape[0] == 0:
        raise ValueError("No samples remained after residual filtering.")

    all_run_results: List[Dict] = []
    for rank in nmf_cfg["ranks"]:
        all_run_results.extend(
            run_repeated_nmf(
                residual_matrix=filtered_matrix,
                sample_ids=filtered_sample_ids,
                rank=int(rank),
                n_runs=int(nmf_cfg.get("n_runs", 20)),
                bootstrap_rows=bool(nmf_cfg.get("bootstrap_rows", False)),
                max_iter=int(nmf_cfg.get("max_iter", 1000)),
                init=str(nmf_cfg.get("init", "nndsvda")),
                base_seed=base_seed,
                n_jobs=int(nmf_cfg.get("n_jobs", 1)),
            )
        )

    component_records, raw_candidate_rows, raw_exposure_rows, flat_signatures = _extract_component_records(all_run_results)
    candidate_df = pd.DataFrame(raw_candidate_rows)
    exposure_df = pd.DataFrame(raw_exposure_rows)
    rank_summary_df = summarize_rank_results(all_run_results)

    cosmic_names, cosmic_bank = load_cosmic_bank(
        signature_path=nmf_cfg["signature_path"],
        signature_delimiter=nmf_cfg.get("signature_delimiter", "\t"),
        signature_type_col=nmf_cfg.get("signature_type_col", "Type"),
        expected_channels=channels,
    )
    similarity_df = compare_candidates_to_cosmic(
        candidate_signatures=flat_signatures,
        cosmic_names=cosmic_names,
        cosmic_bank=cosmic_bank,
        novelty_cosine_threshold=float(nmf_cfg.get("novelty_cosine_threshold", 0.85)),
        known_like_cosine_threshold=float(nmf_cfg.get("known_like_cosine_threshold", 0.95)),
    )
    candidate_df = pd.concat([candidate_df.reset_index(drop=True), similarity_df.drop(columns=["candidate_index"])], axis=1)
    for channel_idx, channel in enumerate(channels):
        candidate_df["channel::{0}".format(channel)] = flat_signatures[:, channel_idx]

    filtered_depths_by_sample = {sample_id: float(depth) for sample_id, depth in zip(filtered_sample_ids, filtered_depths)}
    consensus_df, consensus_sample_df, top_samples_df, clusters_by_rank = build_consensus_components(
        run_results=all_run_results,
        channels=channels,
        filtered_depths_by_sample=filtered_depths_by_sample,
        cosmic_names=cosmic_names,
        cosmic_bank=cosmic_bank,
        consensus_cosine_threshold=float(nmf_cfg.get("consensus_cosine_threshold", 0.9)),
        novelty_cosine_threshold=float(nmf_cfg.get("novelty_cosine_threshold", 0.85)),
        known_like_cosine_threshold=float(nmf_cfg.get("known_like_cosine_threshold", 0.95)),
        top_samples_per_component=int(nmf_cfg.get("top_samples_per_component", 20)),
    )

    if not consensus_df.empty:
        consensus_counts = consensus_df.groupby("rank").size().rename("num_consensus_components").reset_index()
        rank_summary_df = rank_summary_df.merge(consensus_counts, on="rank", how="left").fillna({"num_consensus_components": 0})
        rank_summary_df["num_consensus_components"] = rank_summary_df["num_consensus_components"].astype(int)
    else:
        rank_summary_df["num_consensus_components"] = 0

    output_dir = ensure_dir(nmf_cfg["output_dir"])
    plot_paths = save_consensus_plots(
        consensus_df=consensus_df,
        consensus_sample_df=consensus_sample_df,
        channels=channels,
        rank_summary_df=rank_summary_df,
        output_dir=output_dir,
    )

    consensus_path = output_dir / "consensus_signatures.csv"
    consensus_sample_path = output_dir / "consensus_sample_exposures.csv"
    top_samples_path = output_dir / "top_samples_per_component.csv"

    candidate_path = output_dir / "candidate_signatures.csv"
    exposure_path = output_dir / "candidate_exposures.csv"
    rank_summary_path = output_dir / "rank_summary.csv"
    similarity_path = output_dir / "candidate_signature_similarity_to_cosmic.csv"
    metadata_path = output_dir / "run_metadata.json"

    candidate_df.to_csv(candidate_path, index=False)
    exposure_df.to_csv(exposure_path, index=False)
    rank_summary_df.to_csv(rank_summary_path, index=False)
    similarity_df.to_csv(similarity_path, index=False)
    consensus_df.to_csv(consensus_path, index=False)
    consensus_sample_df.to_csv(consensus_sample_path, index=False)
    top_samples_df.to_csv(top_samples_path, index=False)

    refit_outputs: Dict[str, str] = {}
    refit_summary: Dict[str, float] = {}
    refit_gate_summary: Dict = {}
    refit_cfg = nmf_cfg.get("refit", {})
    if bool(refit_cfg.get("enabled", False)) and not consensus_df.empty:
        discovery_refit_cfg = dict(refit_cfg)
        discovery_refit_cfg["use_validated_candidates"] = False
        refit_consensus = _resolve_refit_consensus_subset(
            consensus_df=consensus_df,
            refit_cfg=discovery_refit_cfg,
            ranks=list(nmf_cfg["ranks"]),
        )

        if not refit_consensus.empty:
            channel_columns = ["channel::{0}".format(channel) for channel in channels]
            candidate_signatures = refit_consensus[channel_columns].to_numpy(dtype=np.float32)
            candidate_names = refit_consensus["consensus_component"].astype(str).tolist()
            checkpoint_path = str(
                refit_cfg.get(
                    "checkpoint_path",
                    config.get("inference", {}).get("checkpoint_path", ""),
                )
            )
            if checkpoint_path:
                refit_df, refit_summary = evaluate_augmented_refit(
                    checkpoint_path=checkpoint_path,
                    split=str(refit_cfg.get("split", split)),
                    candidate_signatures=candidate_signatures,
                    candidate_names=candidate_names,
                    filtered_sample_ids=filtered_sample_ids,
                    processed_payload_path=refit_cfg.get("processed_payload_path"),
                )
                if not refit_df.empty:
                    refit_path = output_dir / "augmented_refit_comparison.csv"
                    refit_df.to_csv(refit_path, index=False)
                    refit_outputs["augmented_refit_comparison_path"] = str(refit_path)
                    refit_gate_summary = _summarize_validation_candidate_gate(
                        refit_df=refit_df,
                        refit_summary=refit_summary,
                        candidate_names=candidate_names,
                        refit_cfg=refit_cfg,
                    )
                    gate_path = output_dir / "augmented_refit_gate_summary.json"
                    save_json(refit_gate_summary, str(gate_path))
                    refit_outputs["augmented_refit_gate_summary_path"] = str(gate_path)

                    if not refit_gate_summary.get("enabled", False) or bool(refit_gate_summary.get("passed", True)):
                        consensus_df.loc[
                            consensus_df["consensus_component"].astype(str).isin(candidate_names),
                            "selected_for_augmented_refit",
                        ] = True
                        refit_outputs["augmented_refit_candidates"] = candidate_names
                    else:
                        consensus_df["selected_for_augmented_refit"] = False
                        refit_outputs["augmented_refit_candidates"] = []

                    if "selected_for_augmented_refit" not in consensus_df.columns:
                        consensus_df["selected_for_augmented_refit"] = False
                    consensus_df["selected_for_augmented_refit"] = consensus_df[
                        "selected_for_augmented_refit"
                    ].fillna(False)
                    consensus_df.to_csv(consensus_path, index=False)

    save_json(
        {
            "residual_path": nmf_cfg["residual_path"],
            "split": split,
            "channels": channels,
            "filter_summary": filter_summary,
            "ranks": list(nmf_cfg["ranks"]),
            "n_runs": int(nmf_cfg.get("n_runs", 20)),
            "bootstrap_rows": bool(nmf_cfg.get("bootstrap_rows", False)),
            "n_jobs": int(nmf_cfg.get("n_jobs", 1)),
            "max_iter": int(nmf_cfg.get("max_iter", 1000)),
            "init": str(nmf_cfg.get("init", "nndsvda")),
            "consensus_cosine_threshold": float(nmf_cfg.get("consensus_cosine_threshold", 0.9)),
            "num_candidate_signatures": int(candidate_df.shape[0]),
            "num_consensus_signatures": int(consensus_df.shape[0]),
            "num_exposure_rows": int(exposure_df.shape[0]),
            "num_consensus_exposure_rows": int(consensus_sample_df.shape[0]),
            "output_dir": str(output_dir),
            "plot_paths": plot_paths,
            "refit_summary": refit_summary,
            "refit_gate_summary": refit_gate_summary,
            "refit_outputs": refit_outputs,
        },
        str(metadata_path),
    )

    outputs = {
        "candidate_signatures_path": str(candidate_path),
        "candidate_exposures_path": str(exposure_path),
        "rank_summary_path": str(rank_summary_path),
        "cosmic_similarity_path": str(similarity_path),
        "consensus_signatures_path": str(consensus_path),
        "consensus_sample_exposures_path": str(consensus_sample_path),
        "top_samples_per_component_path": str(top_samples_path),
        "run_metadata_path": str(metadata_path),
    }
    outputs.update(refit_outputs)
    return outputs
