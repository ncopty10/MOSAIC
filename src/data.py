import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

from src.utils import ensure_dir


def _read_table(path: str, delimiter: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt", ".all"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=delimiter)


def _read_table_header(path: str, delimiter: str) -> List[str]:
    suffix = Path(path).suffix.lower()
    if suffix in {".tsv", ".txt", ".all"}:
        delimiter = "\t"
    return pd.read_csv(path, sep=delimiter, nrows=0).columns.astype(str).tolist()


def _read_table_columns(path: str, delimiter: str, usecols: List[str], dtype_map: Dict = None) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".tsv", ".txt", ".all"}:
        delimiter = "\t"
    return pd.read_csv(path, sep=delimiter, usecols=usecols, dtype=dtype_map)


def _read_sample_ids_file(path: str) -> List[str]:
    sample_ids = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                sample_ids.append(value.split("\t")[0])
    if not sample_ids:
        raise ValueError("RNA sample_ids_path '{0}' did not contain any sample IDs.".format(path))
    return sample_ids


def _processed_payload_fingerprint(config: Dict) -> str:
    data_cfg = config["data"]
    fingerprint_payload = {
        "input_path": str(data_cfg["input_path"]),
        "signature_path": str(data_cfg["signature_path"]),
        "min_total_count": float(data_cfg.get("min_total_count", 0.0)),
        "train_fraction": float(data_cfg.get("train_fraction", 0.8)),
        "val_fraction": float(data_cfg.get("val_fraction", 0.1)),
        "reference_pool_strategy": str(data_cfg.get("reference_pool_strategy", "topk_cosine")),
        "reference_pool_size": int(data_cfg.get("reference_pool_size", 0)),
        "known_exposures": data_cfg.get("known_exposures", {}),
        "split_metadata_path": str(data_cfg.get("split_metadata_path", "")),
        "split_metadata_sample_id_col": str(data_cfg.get("split_metadata_sample_id_col", "sample_id")),
        "split_metadata_split_col": str(data_cfg.get("split_metadata_split_col", "split")),
        "split_metadata_barcode_length": str(data_cfg.get("split_metadata_barcode_length", "")),
        "rna": data_cfg.get("rna", {}),
        "cnv": data_cfg.get("cnv", {}),
        "methylation": data_cfg.get("methylation", {}),
        "demographics": data_cfg.get("demographics", {}),
        "seed": int(config["experiment"]["seed"]),
        "cache_schema_version": 5,
    }
    encoded = json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_indices(
    num_samples: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    if num_samples == 1:
        return indices.copy(), indices[:0], indices[:0]

    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Expected fractions with train_fraction > 0, val_fraction >= 0, and train_fraction + val_fraction < 1.")

    train_size = int(num_samples * train_fraction)
    val_size = int(num_samples * val_fraction)
    test_size = num_samples - train_size - val_size

    # Keep all three splits non-empty whenever the dataset is large enough.
    if num_samples >= 3:
        train_size = max(1, train_size)
        val_size = max(1, val_size)
        test_size = num_samples - train_size - val_size

        if test_size <= 0:
            if train_size >= val_size and train_size > 1:
                train_size -= 1
            elif val_size > 1:
                val_size -= 1
            test_size = num_samples - train_size - val_size
    else:
        train_size = 1
        val_size = 0
        test_size = num_samples - train_size

    train_end = train_size
    val_end = train_end + val_size
    train_idx = np.sort(indices[:train_end])
    val_idx = np.sort(indices[train_end:val_end])
    test_idx = np.sort(indices[val_end:])
    return train_idx, val_idx, test_idx


def _load_split_indices_from_metadata(config: Dict, sample_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_cfg = config["data"]
    split_path = data_cfg.get("split_metadata_path")
    if not split_path:
        raise ValueError("data.split_metadata_path was not provided.")

    split_df = _read_table(str(split_path), data_cfg.get("split_metadata_delimiter", ","))
    sample_col = str(data_cfg.get("split_metadata_sample_id_col", "sample_id"))
    split_col = str(data_cfg.get("split_metadata_split_col", "split"))
    barcode_length = data_cfg.get("split_metadata_barcode_length", None)
    missing = [column for column in [sample_col, split_col] if column not in split_df.columns]
    if missing:
        raise ValueError("Split metadata is missing required columns: {0}".format(", ".join(missing)))

    split_df = split_df[[sample_col, split_col]].dropna(subset=[sample_col]).copy()
    split_df["match_key"] = split_df[sample_col].astype(str).map(lambda value: _sample_key(value, barcode_length))
    split_by_key = split_df.drop_duplicates("match_key", keep="first").set_index("match_key")[split_col].astype(str).to_dict()

    buckets = {"train": [], "val": [], "test": []}
    missing_samples = []
    invalid_splits = {}
    for row_idx, sample_id in enumerate(sample_ids):
        split_name = split_by_key.get(_sample_key(sample_id, barcode_length))
        if split_name is None:
            missing_samples.append(sample_id)
            continue
        split_name = split_name.strip().lower()
        if split_name == "validation":
            split_name = "val"
        if split_name not in buckets:
            invalid_splits[sample_id] = split_name
            continue
        buckets[split_name].append(row_idx)

    if missing_samples:
        preview = ", ".join(str(sample_id) for sample_id in missing_samples[:10])
        raise ValueError(
            "Split metadata did not contain labels for {0} matched samples. First missing: {1}".format(
                len(missing_samples), preview
            )
        )
    if invalid_splits:
        preview = ", ".join("{0}:{1}".format(sample_id, split) for sample_id, split in list(invalid_splits.items())[:10])
        raise ValueError("Split metadata contained unsupported split labels. First invalid: {0}".format(preview))
    if not buckets["train"] or not buckets["test"]:
        raise ValueError("Split metadata must provide at least one train and one test sample.")

    return (
        np.asarray(buckets["train"], dtype=np.int64),
        np.asarray(buckets["val"], dtype=np.int64),
        np.asarray(buckets["test"], dtype=np.int64),
    )


def _preprocessing_fit_indices(
    config: Dict,
    sample_ids: List[str],
    default_train_fraction: float,
    default_val_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the split used to fit a modality preprocessor.

    Historically each modality fitted its scaler/PCA on an experiment-seed
    random split before the final split metadata was loaded.  The explicit
    opt-in below makes the final split metadata authoritative for new,
    leakage-controlled analyses while preserving legacy behavior for existing
    cached experiments.
    """
    data_cfg = config["data"]
    if bool(data_cfg.get("fit_preprocessing_on_final_split", False)):
        if not data_cfg.get("split_metadata_path"):
            raise ValueError(
                "fit_preprocessing_on_final_split requires data.split_metadata_path."
            )
        return _load_split_indices_from_metadata(config, sample_ids)
    return _split_indices(
        num_samples=len(sample_ids),
        train_fraction=float(data_cfg.get("train_fraction", default_train_fraction)),
        val_fraction=float(data_cfg.get("val_fraction", default_val_fraction)),
        seed=int(config["experiment"]["seed"]),
    )


def _load_sbs96_matrix(config: Dict) -> Dict:
    data_cfg = config["data"]
    raw_df = _read_table(data_cfg["input_path"], data_cfg.get("delimiter", "\t"))
    mutation_type_col = data_cfg.get("mutation_type_col", "MutationType")
    if mutation_type_col not in raw_df.columns:
        raise ValueError("SBS96 input must include a '{0}' column.".format(mutation_type_col))

    mutation_types = raw_df[mutation_type_col].astype(str).tolist()
    sample_ids = [str(col) for col in raw_df.columns if col != mutation_type_col]
    count_matrix = raw_df[sample_ids].transpose().to_numpy(dtype=np.float32)
    total_counts = count_matrix.sum(axis=1)

    min_total_count = float(data_cfg.get("min_total_count", 0))
    keep_mask = total_counts >= min_total_count
    count_matrix = count_matrix[keep_mask]
    total_counts = total_counts[keep_mask]
    sample_ids = [sample_id for sample_id, keep in zip(sample_ids, keep_mask) if keep]
    if len(sample_ids) == 0:
        raise ValueError("No SBS96 samples remained after min_total_count filtering.")

    normalized_profiles = count_matrix / np.clip(total_counts[:, None], a_min=1.0, a_max=None)
    return {
        "sample_ids": sample_ids,
        "mutation_types": mutation_types,
        "count_matrix": count_matrix,
        "normalized_profiles": normalized_profiles,
        "depths": total_counts,
    }


def _sample_key(sample_id: str, barcode_length) -> str:
    sample_id = str(sample_id)
    if barcode_length in {None, "", 0, "none", "None"}:
        return sample_id
    return sample_id[: int(barcode_length)]


def _build_key_to_index(sample_ids: List[str], barcode_length) -> Dict[str, int]:
    key_to_index: Dict[str, int] = {}
    duplicate_keys = set()
    for index, sample_id in enumerate(sample_ids):
        key = _sample_key(sample_id, barcode_length)
        if key in key_to_index:
            duplicate_keys.add(key)
        else:
            key_to_index[key] = index
    if duplicate_keys:
        preview = ", ".join(sorted(duplicate_keys)[:10])
        raise ValueError(
            "Barcode matching produced duplicate sample keys. "
            "Use a longer data.rna.barcode_length. Duplicates include: {0}".format(preview)
        )
    return key_to_index


def _build_key_to_first_index(sample_ids: List[str], barcode_length) -> Tuple[Dict[str, int], int]:
    key_to_index: Dict[str, int] = {}
    duplicate_count = 0
    for index, sample_id in enumerate(sample_ids):
        key = _sample_key(sample_id, barcode_length)
        if key in key_to_index:
            duplicate_count += 1
            continue
        key_to_index[key] = index
    return key_to_index, duplicate_count


def _copy_existing_modalities(matched_payload: Dict, source_payload: Dict, source_indices: np.ndarray) -> None:
    modality_fields = [
        ("rna_embeddings", "rna_sample_ids", "rna_metadata"),
        ("demographic_vectors", None, "demographic_metadata"),
        ("cnv_embeddings", "cnv_sample_ids", "cnv_metadata"),
        ("methylation_embeddings", "methylation_sample_ids", "methylation_metadata"),
    ]
    for matrix_key, sample_ids_key, metadata_key in modality_fields:
        if matrix_key not in source_payload:
            continue
        matched_payload[matrix_key] = source_payload[matrix_key][source_indices]
        if sample_ids_key is not None and sample_ids_key in source_payload:
            matched_payload[sample_ids_key] = [source_payload[sample_ids_key][int(idx)] for idx in source_indices]
        if metadata_key in source_payload:
            matched_payload[metadata_key] = source_payload.get(metadata_key)


def _normalize_category_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _load_matched_demographic_vectors(config: Dict, sbs_payload: Dict) -> Tuple[Dict, Dict]:
    demo_cfg = config["data"].get("demographics", {})
    sample_id_col = str(demo_cfg.get("sample_id_col", "sample"))
    gender_col = str(demo_cfg.get("gender_col", "demographic.gender"))
    race_col = str(demo_cfg.get("race_col", "demographic.race"))
    barcode_length = int(demo_cfg.get("patient_barcode_length", 12))
    missing_values = {
        _normalize_category_value(value)
        for value in demo_cfg.get("missing_values", ["", "not reported", "not available", "nan", "none"])
    }
    race_categories = [
        _normalize_category_value(value)
        for value in demo_cfg.get(
            "race_categories",
            [
                "american indian or alaska native",
                "asian",
                "black or african american",
                "native hawaiian or other pacific islander",
                "white",
            ],
        )
    ]
    gender_encoding = {"male": 0.0, "female": 1.0}

    demo_df = _read_table(demo_cfg["input_path"], demo_cfg.get("delimiter", "\t"))
    missing_cols = [column for column in [sample_id_col, gender_col, race_col] if column not in demo_df.columns]
    if missing_cols:
        raise ValueError(
            "Demographics file is missing required columns: {0}".format(", ".join(sorted(missing_cols)))
        )

    demo_subset = demo_df[[sample_id_col, gender_col, race_col]].copy()
    demo_subset["patient_id"] = demo_subset[sample_id_col].astype(str).str[:barcode_length]
    demo_subset["gender_norm"] = demo_subset[gender_col].map(_normalize_category_value)
    demo_subset["race_norm"] = demo_subset[race_col].map(_normalize_category_value)

    valid_mask = (
        demo_subset["gender_norm"].isin(gender_encoding.keys())
        & demo_subset["race_norm"].isin(race_categories)
        & ~demo_subset["gender_norm"].isin(missing_values)
        & ~demo_subset["race_norm"].isin(missing_values)
    )
    filtered_demo = demo_subset.loc[valid_mask].drop_duplicates("patient_id", keep="first").copy()
    patient_to_demo = filtered_demo.set_index("patient_id")[["gender_norm", "race_norm"]]

    sbs_patient_ids = [_sample_key(sample_id, barcode_length) for sample_id in sbs_payload["sample_ids"]]
    matched_indices = [idx for idx, patient_id in enumerate(sbs_patient_ids) if patient_id in patient_to_demo.index]
    if not matched_indices:
        raise ValueError("No SBS samples overlapped with reported demographic gender/race entries.")

    matched_indices_array = np.asarray(matched_indices, dtype=np.int64)
    matched_sample_ids = [sbs_payload["sample_ids"][int(idx)] for idx in matched_indices_array]
    demographic_vectors = np.zeros((len(matched_indices_array), 1 + len(race_categories)), dtype=np.float32)
    for output_idx, row_idx in enumerate(matched_indices_array):
        patient_id = sbs_patient_ids[int(row_idx)]
        gender_value = patient_to_demo.at[patient_id, "gender_norm"]
        race_value = patient_to_demo.at[patient_id, "race_norm"]
        demographic_vectors[output_idx, 0] = gender_encoding[gender_value]
        demographic_vectors[output_idx, 1 + race_categories.index(race_value)] = 1.0

    shuffle_enabled = bool(demo_cfg.get("shuffle", False))
    if shuffle_enabled:
        rng = np.random.default_rng(int(config["experiment"]["seed"]) + 701)
        demographic_vectors = demographic_vectors[rng.permutation(len(demographic_vectors))]

    matched_payload = {
        "sample_ids": matched_sample_ids,
        "mutation_types": sbs_payload["mutation_types"],
        "count_matrix": sbs_payload["count_matrix"][matched_indices_array],
        "normalized_profiles": sbs_payload["normalized_profiles"][matched_indices_array],
        "depths": sbs_payload["depths"][matched_indices_array],
        "demographic_vectors": demographic_vectors,
        "demographic_metadata": {
            "enabled": True,
            "input_path": demo_cfg["input_path"],
            "sample_id_col": sample_id_col,
            "gender_col": gender_col,
            "race_col": race_col,
            "patient_barcode_length": barcode_length,
            "gender_encoding": gender_encoding,
            "race_categories": race_categories,
            "missing_values": sorted(missing_values),
            "num_sbs_samples_before_match": int(len(sbs_payload["sample_ids"])),
            "num_demographic_rows": int(len(demo_subset)),
            "num_demographic_rows_after_filter": int(len(filtered_demo)),
            "num_matched_samples": int(len(matched_indices_array)),
            "shuffle_applied": bool(shuffle_enabled),
        },
    }
    if "rna_embeddings" in sbs_payload:
        matched_payload["rna_embeddings"] = sbs_payload["rna_embeddings"][matched_indices_array]
        matched_payload["rna_sample_ids"] = [sbs_payload["rna_sample_ids"][int(idx)] for idx in matched_indices_array]
        matched_payload["rna_metadata"] = sbs_payload.get("rna_metadata")
    return matched_payload, matched_payload["demographic_metadata"]


def _looks_like_gene_id(value: str) -> bool:
    return bool(re.match(r"^(ENSG|ENST|[A-Za-z0-9_.-]+)$", str(value))) and not bool(re.match(r"^[0-9.+-]+$", str(value)))


def _should_log_transform_rna(values: np.ndarray, mode) -> bool:
    if isinstance(mode, bool):
        return mode
    mode = str(mode).lower()
    if mode in {"true", "yes", "log2"}:
        return True
    if mode in {"false", "no", "none", "already_log"}:
        return False
    if mode != "auto":
        raise ValueError("Unsupported RNA log_transform value '{0}'.".format(mode))

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return False
    # Raw HTSeq counts commonly contain values far above this range, while the
    # provided GDC-PANCAN file appears already log2(count + 1)-scaled.
    return float(np.nanmax(finite_values)) > 50.0


def _fit_rna_preprocessor(train_rna_matrix: np.ndarray, config: Dict) -> Tuple[Dict, Dict]:
    rna_cfg = config["data"].get("rna", {})
    train_rna_matrix = np.asarray(train_rna_matrix, dtype=np.float32)
    log_transform = _should_log_transform_rna(train_rna_matrix, rna_cfg.get("log_transform", "auto"))
    transformed_train = train_rna_matrix
    if log_transform:
        transformed_train = np.log2(np.clip(transformed_train, a_min=0.0, a_max=None) + 1.0).astype(np.float32)

    normalization = str(rna_cfg.get("normalization", "zscore")).lower()
    gene_mean = None
    gene_std = None
    if normalization == "zscore":
        gene_mean = transformed_train.mean(axis=0, keepdims=True).astype(np.float32)
        gene_std = transformed_train.std(axis=0, keepdims=True).astype(np.float32)
        transformed_train = (transformed_train - gene_mean) / np.clip(gene_std, a_min=1e-6, a_max=None)
    elif normalization in {"none", "false"}:
        pass
    else:
        raise ValueError("Unsupported RNA normalization '{0}'.".format(normalization))

    transformed_train = np.nan_to_num(transformed_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    reduction = str(rna_cfg.get("reduction", "pca")).lower()
    metadata = {
        "log_transform_applied": bool(log_transform),
        "normalization": normalization,
        "reduction": reduction,
        "input_num_genes": int(train_rna_matrix.shape[1]),
        "fit_num_train_samples": int(train_rna_matrix.shape[0]),
    }
    state = {
        "log_transform": bool(log_transform),
        "normalization": normalization,
        "reduction": reduction,
        "gene_mean": gene_mean,
        "gene_std": gene_std,
        "pca_mean": None,
        "pca_components": None,
    }
    if reduction == "pca":
        n_components = min(
            int(rna_cfg.get("n_components", 128)),
            int(transformed_train.shape[0]),
            int(transformed_train.shape[1]),
        )
        pca = PCA(
            n_components=n_components,
            random_state=int(config["experiment"]["seed"]),
            svd_solver=str(rna_cfg.get("pca_svd_solver", "randomized")),
        )
        pca.fit(transformed_train)
        state["pca_mean"] = pca.mean_.astype(np.float32)
        state["pca_components"] = pca.components_.astype(np.float32)
        metadata.update(
            {
                "n_components": int(n_components),
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            }
        )
        return state, metadata
    if reduction in {"none", "identity"}:
        metadata["n_components"] = int(transformed_train.shape[1])
        return state, metadata

    raise ValueError("Unsupported RNA reduction '{0}'.".format(reduction))


def _transform_rna_matrix(rna_matrix: np.ndarray, preprocess_state: Dict) -> np.ndarray:
    transformed = np.asarray(rna_matrix, dtype=np.float32)
    if preprocess_state["log_transform"]:
        transformed = np.log2(np.clip(transformed, a_min=0.0, a_max=None) + 1.0).astype(np.float32)

    if preprocess_state["normalization"] == "zscore":
        transformed = (transformed - preprocess_state["gene_mean"]) / np.clip(preprocess_state["gene_std"], a_min=1e-6, a_max=None)

    transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if preprocess_state["reduction"] == "pca":
        centered = transformed - preprocess_state["pca_mean"]
        transformed = centered @ preprocess_state["pca_components"].T
    return transformed.astype(np.float32)


def _fit_matrix_preprocessor(
    train_matrix: np.ndarray,
    config: Dict,
    modality_cfg: Dict,
    modality_name: str,
) -> Tuple[Dict, Dict]:
    train_matrix = np.asarray(train_matrix, dtype=np.float32)
    transformed_train = train_matrix.copy()

    normalization = str(modality_cfg.get("normalization", "zscore")).lower()
    feature_mean = None
    feature_std = None
    if normalization == "zscore":
        feature_mean = np.nanmean(transformed_train, axis=0, keepdims=True).astype(np.float32)
        transformed_train = np.where(np.isfinite(transformed_train), transformed_train, feature_mean).astype(np.float32)
        feature_std = transformed_train.std(axis=0, keepdims=True).astype(np.float32)
        transformed_train = (transformed_train - feature_mean) / np.clip(feature_std, a_min=1e-6, a_max=None)
    elif normalization in {"none", "false"}:
        transformed_train = np.nan_to_num(transformed_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        raise ValueError("Unsupported {0} normalization '{1}'.".format(modality_name, normalization))

    reduction = str(modality_cfg.get("reduction", "pca")).lower()
    metadata = {
        "normalization": normalization,
        "reduction": reduction,
        "input_num_features": int(train_matrix.shape[1]),
        "fit_num_train_samples": int(train_matrix.shape[0]),
    }
    state = {
        "normalization": normalization,
        "reduction": reduction,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "pca_mean": None,
        "pca_components": None,
    }
    if reduction == "pca":
        n_components = min(
            int(modality_cfg.get("n_components", 128)),
            int(transformed_train.shape[0]),
            int(transformed_train.shape[1]),
        )
        pca = PCA(
            n_components=n_components,
            random_state=int(config["experiment"]["seed"]),
            svd_solver=str(modality_cfg.get("pca_svd_solver", "randomized")),
        )
        pca.fit(transformed_train)
        state["pca_mean"] = pca.mean_.astype(np.float32)
        state["pca_components"] = pca.components_.astype(np.float32)
        metadata.update(
            {
                "n_components": int(n_components),
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            }
        )
        return state, metadata
    if reduction in {"none", "identity"}:
        metadata["n_components"] = int(transformed_train.shape[1])
        return state, metadata
    raise ValueError("Unsupported {0} reduction '{1}'.".format(modality_name, reduction))


def _transform_matrix(matrix: np.ndarray, preprocess_state: Dict) -> np.ndarray:
    transformed = np.asarray(matrix, dtype=np.float32).copy()
    if preprocess_state["normalization"] == "zscore":
        transformed = np.where(np.isfinite(transformed), transformed, preprocess_state["feature_mean"]).astype(np.float32)
        transformed = (transformed - preprocess_state["feature_mean"]) / np.clip(
            preprocess_state["feature_std"], a_min=1e-6, a_max=None
        )
    else:
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if preprocess_state["reduction"] == "pca":
        transformed = (transformed - preprocess_state["pca_mean"]) @ preprocess_state["pca_components"].T
    return transformed.astype(np.float32)


def _shuffle_embeddings_if_requested(embeddings: np.ndarray, config: Dict, modality_cfg: Dict, offset: int) -> Tuple[np.ndarray, bool]:
    shuffle_enabled = bool(modality_cfg.get("shuffle", False))
    if not shuffle_enabled:
        return embeddings, False
    rng = np.random.default_rng(int(config["experiment"]["seed"]) + offset)
    return embeddings[rng.permutation(len(embeddings))], True


def _load_headerless_rna_matrix(
    rna_path: str,
    sbs_payload: Dict,
    rna_cfg: Dict,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    barcode_length = rna_cfg.get("barcode_length", 16)
    rna_sample_ids_path = rna_cfg.get("sample_ids_path")
    dominant_fields = None
    field_counts: Dict[int, int] = {}
    preview_lines: List[str] = []

    with open(rna_path, "r", encoding="utf-8", errors="replace") as handle:
        for line_idx, line in enumerate(handle):
            fields = line.rstrip("\n").split("\t")
            field_count = len(fields)
            field_counts[field_count] = field_counts.get(field_count, 0) + 1
            if line_idx < 5:
                preview_lines.append(line[:120].rstrip())
            if line_idx >= 199:
                break

    if not field_counts:
        raise ValueError("RNA input '{0}' is empty.".format(rna_path))
    dominant_fields = max(field_counts.items(), key=lambda item: item[1])[0]
    num_value_columns = dominant_fields - 1

    if rna_sample_ids_path:
        rna_sample_ids = _read_sample_ids_file(rna_sample_ids_path)
        if len(rna_sample_ids) != num_value_columns:
            raise ValueError(
                "RNA sample_ids_path has {0} IDs, but the dominant RNA row width implies {1} value columns.".format(
                    len(rna_sample_ids), num_value_columns
                )
            )
    else:
        raise ValueError(
            "RNA file '{0}' does not have a usable header row containing sample IDs. "
            "The dominant row width implies {1} value columns, but no data.rna.sample_ids_path was provided. "
            "Preview lines: {2}".format(rna_path, num_value_columns, " | ".join(preview_lines))
        )

    gene_ids: List[str] = []
    rows: List[np.ndarray] = []
    malformed_rows = 0
    with open(rna_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != dominant_fields:
                malformed_rows += 1
                continue
            gene_id = fields[0]
            if not gene_id.startswith("ENSG"):
                malformed_rows += 1
                continue
            try:
                values = np.asarray(fields[1:], dtype=np.float32)
            except ValueError:
                malformed_rows += 1
                continue
            gene_ids.append(gene_id)
            rows.append(values)

    if not rows:
        raise ValueError(
            "No valid RNA gene rows could be parsed from '{0}'. "
            "Detected dominant field width: {1}.".format(rna_path, dominant_fields)
        )

    rna_matrix = np.stack(rows, axis=0).T
    metadata = {
        "dominant_fields": int(dominant_fields),
        "num_value_columns": int(num_value_columns),
        "num_parsed_genes": int(len(gene_ids)),
        "num_skipped_malformed_rows": int(malformed_rows),
    }
    return rna_matrix.astype(np.float32), rna_sample_ids, metadata


def _load_matched_rna_embeddings(config: Dict, sbs_payload: Dict) -> Tuple[Dict, Dict]:
    rna_cfg = config["data"].get("rna", {})
    gene_id_col = rna_cfg.get("gene_id_col", "Ensembl_ID")
    barcode_length = rna_cfg.get("barcode_length", 16)
    rna_columns = _read_table_header(rna_cfg["input_path"], rna_cfg.get("delimiter", "\t"))
    headered_matrix = gene_id_col in rna_columns
    parser_metadata: Dict[str, int] = {}

    if headered_matrix:
        rna_sample_ids = [str(column) for column in rna_columns if column != gene_id_col]
    else:
        rna_matrix, rna_sample_ids, parser_metadata = _load_headerless_rna_matrix(
            rna_path=rna_cfg["input_path"],
            sbs_payload=sbs_payload,
            rna_cfg=rna_cfg,
        )

    sbs_key_to_index = _build_key_to_index(sbs_payload["sample_ids"], barcode_length)
    rna_key_to_index = _build_key_to_index(rna_sample_ids, barcode_length)
    matched_keys = [key for key in sbs_key_to_index if key in rna_key_to_index]
    if not matched_keys:
        raise ValueError("No samples overlapped between SBS96 and RNA inputs.")

    sbs_indices = np.asarray([sbs_key_to_index[key] for key in matched_keys], dtype=np.int64)
    rna_column_positions = [rna_key_to_index[key] for key in matched_keys]
    matched_rna_columns = [rna_sample_ids[position] for position in rna_column_positions]
    if headered_matrix:
        dtype_map = {column: np.float32 for column in matched_rna_columns}
        dtype_map[gene_id_col] = str
        rna_df = _read_table_columns(
            rna_cfg["input_path"],
            rna_cfg.get("delimiter", "\t"),
            usecols=[gene_id_col] + matched_rna_columns,
            dtype_map=dtype_map,
        )
        rna_matrix = rna_df[matched_rna_columns].transpose().to_numpy(dtype=np.float32)
        del rna_df
    else:
        matched_positions = np.asarray(rna_column_positions, dtype=np.int64)
        rna_matrix = rna_matrix[matched_positions]

    matched_count_matrix = sbs_payload["count_matrix"][sbs_indices]
    matched_profiles = sbs_payload["normalized_profiles"][sbs_indices]
    matched_depths = sbs_payload["depths"][sbs_indices]
    matched_sample_ids = [sbs_payload["sample_ids"][int(index)] for index in sbs_indices]

    # Resolve the final patient split before selecting variable methylation
    # features.  Feature selection must use training patients only, just like
    # scaling and PCA; validation/test patients are transformed with the frozen
    # training-derived feature list.
    train_idx, val_idx, test_idx = _preprocessing_fit_indices(
        config, matched_sample_ids, default_train_fraction=0.8, default_val_fraction=0.1
    )
    preprocess_state, rna_metadata = _fit_rna_preprocessor(rna_matrix[train_idx], config)
    rna_embeddings = _transform_rna_matrix(rna_matrix, preprocess_state)
    shuffle_enabled = bool(rna_cfg.get("shuffle", False))
    if shuffle_enabled:
        rng = np.random.default_rng(int(config["experiment"]["seed"]) + 1701)
        rna_embeddings = rna_embeddings[rng.permutation(len(rna_embeddings))]

    matched_payload = {
        "sample_ids": matched_sample_ids,
        "mutation_types": sbs_payload["mutation_types"],
        "count_matrix": matched_count_matrix,
        "normalized_profiles": matched_profiles,
        "depths": matched_depths,
        "rna_embeddings": rna_embeddings,
        "rna_sample_ids": matched_rna_columns,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "rna_metadata": {
            **rna_metadata,
            "enabled": True,
            "input_path": rna_cfg["input_path"],
            "barcode_length": barcode_length,
            "parser_mode": "headered" if headered_matrix else "headerless_sidecar_ids",
            **parser_metadata,
            "num_sbs_samples_before_match": int(len(sbs_payload["sample_ids"])),
            "num_rna_samples_before_match": int(len(rna_sample_ids)),
            "num_matched_samples": int(len(matched_keys)),
            "preprocess_fit_on_train_only": True,
            "shuffle_applied": bool(shuffle_enabled),
        },
    }
    if "demographic_vectors" in sbs_payload:
        matched_payload["demographic_vectors"] = sbs_payload["demographic_vectors"][sbs_indices]
        matched_payload["demographic_metadata"] = sbs_payload.get("demographic_metadata")
    return matched_payload, matched_payload["rna_metadata"]


def _chromosome_bin_offsets(bin_size: int) -> Tuple[Dict[str, int], int]:
    chrom_sizes = {
        "1": 248956422,
        "2": 242193529,
        "3": 198295559,
        "4": 190214555,
        "5": 181538259,
        "6": 170805979,
        "7": 159345973,
        "8": 145138636,
        "9": 138394717,
        "10": 133797422,
        "11": 135086622,
        "12": 133275309,
        "13": 114364328,
        "14": 107043718,
        "15": 101991189,
        "16": 90338345,
        "17": 83257441,
        "18": 80373285,
        "19": 58617616,
        "20": 64444167,
        "21": 46709983,
        "22": 50818468,
        "X": 156040895,
        "Y": 57227415,
    }
    offsets: Dict[str, int] = {}
    cursor = 0
    for chrom, size in chrom_sizes.items():
        offsets[chrom] = cursor
        cursor += int(np.ceil(size / float(bin_size)))
    return offsets, cursor


def _normalize_chromosome(value) -> str:
    text = str(value).strip().replace("chr", "").replace("CHR", "")
    if text == "23":
        return "X"
    if text == "24":
        return "Y"
    return text.upper()


def _load_matched_cnv_embeddings(config: Dict, sbs_payload: Dict) -> Tuple[Dict, Dict]:
    cnv_cfg = config["data"].get("cnv", {})
    sample_col = str(cnv_cfg.get("sample_col", "sample"))
    chrom_col = str(cnv_cfg.get("chrom_col", "Chrom"))
    start_col = str(cnv_cfg.get("start_col", "Start"))
    end_col = str(cnv_cfg.get("end_col", "End"))
    value_col = str(cnv_cfg.get("value_col", "value"))
    barcode_length = cnv_cfg.get("barcode_length", 15)
    bin_size = int(cnv_cfg.get("bin_size", 5_000_000))
    chunksize = int(cnv_cfg.get("chunksize", 250_000))

    sbs_key_to_index, sbs_duplicate_count = _build_key_to_first_index(sbs_payload["sample_ids"], barcode_length)
    chrom_offsets, num_bins = _chromosome_bin_offsets(bin_size)
    sbs_keys = list(sbs_key_to_index.keys())
    sample_key_to_output = {key: index for index, key in enumerate(sbs_keys)}
    sums = np.zeros((len(sbs_keys), num_bins), dtype=np.float32)
    counts = np.zeros((len(sbs_keys), num_bins), dtype=np.float32)
    seen_keys = set()
    cnv_sample_ids_by_key: Dict[str, str] = {}
    total_segment_rows = 0
    matched_segment_rows = 0

    usecols = [sample_col, chrom_col, start_col, end_col, value_col]
    for chunk in pd.read_csv(cnv_cfg["input_path"], sep=cnv_cfg.get("delimiter", "\t"), usecols=usecols, chunksize=chunksize):
        total_segment_rows += int(len(chunk))
        chunk["_match_key"] = chunk[sample_col].astype(str).map(lambda value: _sample_key(value, barcode_length))
        chunk = chunk.loc[chunk["_match_key"].isin(sbs_key_to_index.keys())].copy()
        if chunk.empty:
            continue
        chunk["_chrom_norm"] = chunk[chrom_col].map(_normalize_chromosome)
        chunk = chunk.loc[chunk["_chrom_norm"].isin(chrom_offsets.keys())].copy()
        if chunk.empty:
            continue

        starts = pd.to_numeric(chunk[start_col], errors="coerce").to_numpy(dtype=np.float64)
        ends = pd.to_numeric(chunk[end_col], errors="coerce").to_numpy(dtype=np.float64)
        values = pd.to_numeric(chunk[value_col], errors="coerce").to_numpy(dtype=np.float32)
        valid = np.isfinite(starts) & np.isfinite(ends) & np.isfinite(values)
        if not np.any(valid):
            continue
        filtered = chunk.loc[valid]
        starts = starts[valid]
        ends = ends[valid]
        values = values[valid]
        mids = ((starts + ends) / 2.0).astype(np.int64)
        keys = filtered["_match_key"].astype(str).to_numpy()
        chroms = filtered["_chrom_norm"].astype(str).to_numpy()
        row_indices = np.asarray([sample_key_to_output[key] for key in keys], dtype=np.int64)
        bin_indices = np.asarray([chrom_offsets[chrom] for chrom in chroms], dtype=np.int64) + (mids // bin_size)
        valid_bins = (bin_indices >= 0) & (bin_indices < num_bins)
        if not np.any(valid_bins):
            continue
        row_indices = row_indices[valid_bins]
        bin_indices = bin_indices[valid_bins]
        values = values[valid_bins]
        keys = keys[valid_bins]
        sample_ids = filtered.loc[valid_bins, sample_col].astype(str).to_numpy()
        np.add.at(sums, (row_indices, bin_indices), values)
        np.add.at(counts, (row_indices, bin_indices), 1.0)
        matched_segment_rows += int(len(values))
        for key, sample_id in zip(keys, sample_ids):
            if key not in cnv_sample_ids_by_key:
                cnv_sample_ids_by_key[key] = sample_id
            seen_keys.add(key)

    if not seen_keys:
        raise ValueError("No SBS samples overlapped with CNV segment input.")

    ordered_keys = [key for key in sbs_keys if key in seen_keys]
    output_indices = np.asarray([sample_key_to_output[key] for key in ordered_keys], dtype=np.int64)
    cnv_matrix = sums[output_indices] / np.clip(counts[output_indices], a_min=1.0, a_max=None)

    sbs_indices = np.asarray([sbs_key_to_index[key] for key in ordered_keys], dtype=np.int64)
    matched_sample_ids = [sbs_payload["sample_ids"][int(index)] for index in sbs_indices]
    train_idx, val_idx, test_idx = _preprocessing_fit_indices(
        config, matched_sample_ids, default_train_fraction=0.8, default_val_fraction=0.1
    )
    preprocess_state, cnv_metadata = _fit_matrix_preprocessor(cnv_matrix[train_idx], config, cnv_cfg, "CNV")
    cnv_embeddings = _transform_matrix(cnv_matrix, preprocess_state)
    cnv_embeddings, shuffle_applied = _shuffle_embeddings_if_requested(cnv_embeddings, config, cnv_cfg, 2701)

    matched_payload = {
        "sample_ids": matched_sample_ids,
        "mutation_types": sbs_payload["mutation_types"],
        "count_matrix": sbs_payload["count_matrix"][sbs_indices],
        "normalized_profiles": sbs_payload["normalized_profiles"][sbs_indices],
        "depths": sbs_payload["depths"][sbs_indices],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "cnv_embeddings": cnv_embeddings,
        "cnv_sample_ids": [cnv_sample_ids_by_key[key] for key in ordered_keys],
        "cnv_metadata": {
            **cnv_metadata,
            "enabled": True,
            "input_path": cnv_cfg["input_path"],
            "format": "segments",
            "barcode_length": barcode_length,
            "bin_size": int(bin_size),
            "num_bins": int(num_bins),
            "num_sbs_samples_before_match": int(len(sbs_payload["sample_ids"])),
            "num_sbs_duplicate_keys": int(sbs_duplicate_count),
            "num_matched_samples": int(len(matched_sample_ids)),
            "num_total_segment_rows": int(total_segment_rows),
            "num_matched_segment_rows": int(matched_segment_rows),
            "preprocess_fit_on_train_only": True,
            "shuffle_applied": bool(shuffle_applied),
        },
    }
    _copy_existing_modalities(matched_payload, sbs_payload, sbs_indices)
    return matched_payload, matched_payload["cnv_metadata"]


def _load_matched_methylation_embeddings(config: Dict, sbs_payload: Dict) -> Tuple[Dict, Dict]:
    methyl_cfg = config["data"].get("methylation", {})
    feature_id_col = str(methyl_cfg.get("feature_id_col", "sample"))
    barcode_length = methyl_cfg.get("barcode_length", 15)
    chunksize = int(methyl_cfg.get("chunksize", 1000))
    max_features = int(methyl_cfg.get("max_features", 20000))
    delimiter = methyl_cfg.get("delimiter", "\t")

    header = _read_table_header(methyl_cfg["input_path"], delimiter)
    if feature_id_col not in header:
        feature_id_col = header[0]
    methyl_sample_ids = [str(column) for column in header if column != feature_id_col]
    sbs_key_to_index, sbs_duplicate_count = _build_key_to_first_index(sbs_payload["sample_ids"], barcode_length)
    methyl_key_to_index, methyl_duplicate_count = _build_key_to_first_index(methyl_sample_ids, barcode_length)
    matched_keys = [key for key in sbs_key_to_index if key in methyl_key_to_index]
    if not matched_keys:
        raise ValueError("No samples overlapped between SBS96 and methylation inputs.")

    sbs_indices = np.asarray([sbs_key_to_index[key] for key in matched_keys], dtype=np.int64)
    methyl_column_positions = [methyl_key_to_index[key] for key in matched_keys]
    matched_methyl_columns = [methyl_sample_ids[position] for position in methyl_column_positions]
    usecols = [feature_id_col] + matched_methyl_columns

    train_idx, val_idx, test_idx = _preprocessing_fit_indices(
        config, [sbs_payload["sample_ids"][int(index)] for index in sbs_indices],
        default_train_fraction=0.8, default_val_fraction=0.1
    )

    variances: List[float] = []
    feature_ids: List[str] = []
    row_count = 0
    for chunk in pd.read_csv(methyl_cfg["input_path"], sep=delimiter, usecols=usecols, chunksize=chunksize):
        values = chunk[matched_methyl_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        chunk_variances = np.nanvar(values[:, train_idx], axis=1)
        variances.extend([float(value) for value in chunk_variances])
        feature_ids.extend(chunk[feature_id_col].astype(str).tolist())
        row_count += int(len(chunk))

    if not variances:
        raise ValueError("Methylation input did not contain any feature rows.")
    selected_count = min(max_features, len(variances))
    selected_indices = np.argsort(-np.asarray(variances, dtype=np.float32))[:selected_count]
    selected_index_set = {int(index) for index in selected_indices}

    selected_rows: Dict[int, np.ndarray] = {}
    row_offset = 0
    for chunk in pd.read_csv(methyl_cfg["input_path"], sep=delimiter, usecols=usecols, chunksize=chunksize):
        chunk_indices = np.arange(row_offset, row_offset + len(chunk))
        keep_mask = np.asarray([int(index) in selected_index_set for index in chunk_indices], dtype=bool)
        if np.any(keep_mask):
            values = chunk.loc[keep_mask, matched_methyl_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            for global_index, row_values in zip(chunk_indices[keep_mask], values):
                selected_rows[int(global_index)] = row_values
        row_offset += int(len(chunk))

    ordered_selected_indices = [int(index) for index in selected_indices]
    methyl_matrix = np.stack([selected_rows[index] for index in ordered_selected_indices], axis=0).T.astype(np.float32)
    selected_feature_ids = [feature_ids[index] for index in ordered_selected_indices]
    matched_sample_ids = [sbs_payload["sample_ids"][int(index)] for index in sbs_indices]

    preprocess_state, methyl_metadata = _fit_matrix_preprocessor(methyl_matrix[train_idx], config, methyl_cfg, "methylation")
    methyl_embeddings = _transform_matrix(methyl_matrix, preprocess_state)
    methyl_embeddings, shuffle_applied = _shuffle_embeddings_if_requested(methyl_embeddings, config, methyl_cfg, 3701)

    matched_payload = {
        "sample_ids": matched_sample_ids,
        "mutation_types": sbs_payload["mutation_types"],
        "count_matrix": sbs_payload["count_matrix"][sbs_indices],
        "normalized_profiles": sbs_payload["normalized_profiles"][sbs_indices],
        "depths": sbs_payload["depths"][sbs_indices],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "methylation_embeddings": methyl_embeddings,
        "methylation_sample_ids": matched_methyl_columns,
        "methylation_metadata": {
            **methyl_metadata,
            "enabled": True,
            "input_path": methyl_cfg["input_path"],
            "feature_id_col": feature_id_col,
            "barcode_length": barcode_length,
            "num_sbs_samples_before_match": int(len(sbs_payload["sample_ids"])),
            "num_sbs_duplicate_keys": int(sbs_duplicate_count),
            "num_methylation_samples_before_match": int(len(methyl_sample_ids)),
            "num_methylation_duplicate_keys": int(methyl_duplicate_count),
            "num_matched_samples": int(len(matched_sample_ids)),
            "num_total_features": int(row_count),
            "num_selected_features": int(selected_count),
            "feature_selection": "top_variance",
            "selected_feature_preview": selected_feature_ids[:20],
            "preprocess_fit_on_train_only": True,
            "shuffle_applied": bool(shuffle_applied),
        },
    }
    _copy_existing_modalities(matched_payload, sbs_payload, sbs_indices)
    return matched_payload, matched_payload["methylation_metadata"]


def _load_signature_bank(config: Dict, expected_mutation_types) -> Dict:
    data_cfg = config["data"]
    sig_df = _read_table(data_cfg["signature_path"], data_cfg.get("signature_delimiter", "\t"))
    signature_type_col = data_cfg.get("signature_type_col", "Type")
    if signature_type_col not in sig_df.columns:
        raise ValueError("Signature file must include a '{0}' column.".format(signature_type_col))

    sig_df = sig_df.set_index(signature_type_col)
    missing = [mutation_type for mutation_type in expected_mutation_types if mutation_type not in sig_df.index]
    if missing:
        raise ValueError("Signature bank is missing SBS channels: {0}".format(", ".join(missing[:10])))

    sig_df = sig_df.loc[expected_mutation_types]
    signature_names = sig_df.columns.astype(str).tolist()
    signature_bank = sig_df.transpose().to_numpy(dtype=np.float32)
    signature_bank = signature_bank / np.clip(signature_bank.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return {"signature_names": signature_names, "signature_bank": signature_bank}


def _load_aligned_known_exposures(
    config: Dict,
    sample_ids: List[str],
    signature_names: List[str],
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    exposure_cfg = config["data"].get("known_exposures", {})
    sample_id_col = str(exposure_cfg.get("sample_id_col", "sample_id"))
    barcode_length = exposure_cfg.get("barcode_length", None)
    exposure_df = _read_table(exposure_cfg["input_path"], exposure_cfg.get("delimiter", ","))
    if sample_id_col not in exposure_df.columns:
        raise ValueError("Known exposures file must include sample ID column '{0}'.".format(sample_id_col))
    missing_signature_columns = [column for column in signature_names if column not in exposure_df.columns]
    if missing_signature_columns:
        raise ValueError(
            "Known exposures file is missing signature columns: {0}".format(", ".join(missing_signature_columns[:10]))
        )

    exposure_df = exposure_df[[sample_id_col] + signature_names].copy()
    exposure_df["_match_key"] = exposure_df[sample_id_col].astype(str).map(lambda value: _sample_key(value, barcode_length))
    exposure_df = exposure_df.drop_duplicates("_match_key", keep="first")
    exposure_by_key = exposure_df.set_index("_match_key")[signature_names]

    aligned_rows = []
    missing_samples = []
    for sample_id in sample_ids:
        key = _sample_key(sample_id, barcode_length)
        if key not in exposure_by_key.index:
            missing_samples.append(sample_id)
            continue
        aligned_rows.append(exposure_by_key.loc[key].to_numpy(dtype=np.float32))
    if missing_samples:
        preview = ", ".join(str(sample_id) for sample_id in missing_samples[:10])
        raise ValueError(
            "Known exposures file did not contain all matched samples. Missing examples: {0}".format(preview)
        )
    known_exposures = np.asarray(aligned_rows, dtype=np.float32)
    if not np.isfinite(known_exposures).all():
        raise ValueError("Known exposures contain non-finite values.")
    if np.any(known_exposures < 0.0):
        raise ValueError("Known exposures contain negative values.")
    row_sums = known_exposures.sum(axis=1, keepdims=True)
    mask_zero_sum_rows = bool(exposure_cfg.get("mask_zero_sum_rows", False))
    if mask_zero_sum_rows:
        valid_mask = row_sums[:, 0] > 1e-12
    else:
        # Preserve the historical behavior unless masking is explicitly enabled.
        valid_mask = np.ones(len(known_exposures), dtype=bool)
    known_exposures = known_exposures / np.clip(row_sums, a_min=1e-12, a_max=None)
    if mask_zero_sum_rows:
        if np.any(valid_mask) and not np.allclose(
            known_exposures[valid_mask].sum(axis=1), 1.0, atol=1e-5
        ):
            raise ValueError("Valid retained exposure targets did not normalize to one.")
        if np.any(~valid_mask) and not np.allclose(
            known_exposures[~valid_mask], 0.0, atol=1e-8
        ):
            raise ValueError("Zero-denominator retained exposure targets are not all zero.")
    metadata = {
        "enabled": True,
        "input_path": exposure_cfg["input_path"],
        "sample_id_col": sample_id_col,
        "barcode_length": barcode_length,
        "num_matched_samples": int(len(sample_ids)),
        "num_signatures": int(len(signature_names)),
        "row_normalized": True,
        "mask_zero_sum_rows": mask_zero_sum_rows,
        "num_valid_supervision_rows": int(valid_mask.sum()),
        "num_masked_zero_sum_rows": int((~valid_mask).sum()),
    }
    return known_exposures.astype(np.float32), metadata, valid_mask.astype(bool)


def _topk_reference_pool(profiles: np.ndarray, signature_bank: np.ndarray, pool_size: int) -> np.ndarray:
    profile_norm = profiles / np.clip(np.linalg.norm(profiles, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    signature_norm = signature_bank / np.clip(np.linalg.norm(signature_bank, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    cosine_scores = profile_norm @ signature_norm.T
    topk = np.argsort(-cosine_scores, axis=1)[:, :pool_size]
    return topk.astype(np.int64)


def _build_reference_pools(config: Dict, normalized_profiles: np.ndarray, signature_bank: np.ndarray) -> np.ndarray:
    pool_strategy = config["data"].get("reference_pool_strategy", "topk_cosine")
    num_signatures = signature_bank.shape[0]
    if pool_strategy == "all":
        return np.tile(np.arange(num_signatures, dtype=np.int64), (normalized_profiles.shape[0], 1))

    pool_size = min(int(config["data"].get("reference_pool_size", num_signatures)), num_signatures)
    return _topk_reference_pool(normalized_profiles, signature_bank, pool_size)


def prepare_processed_dataset(config: Dict) -> Dict:
    data_cfg = config["data"]
    if data_cfg["input_format"] != "sbs96":
        raise ValueError("This codepath expects data.input_format: sbs96.")

    processed_dir = ensure_dir(data_cfg["processed_dir"])
    prefix = data_cfg.get("processed_prefix", "dataset")
    processed_path = processed_dir / "{0}_processed.pt".format(prefix)

    rna_enabled = bool(data_cfg.get("rna", {}).get("enabled", False))
    cnv_enabled = bool(data_cfg.get("cnv", {}).get("enabled", False))
    methylation_enabled = bool(data_cfg.get("methylation", {}).get("enabled", False))
    demographics_enabled = bool(data_cfg.get("demographics", {}).get("enabled", False))
    known_exposures_enabled = bool(data_cfg.get("known_exposures", {}).get("enabled", False))
    current_fingerprint = _processed_payload_fingerprint(config)
    if processed_path.exists() and not data_cfg.get("overwrite_processed", False):
        # This cache is produced by this project and contains NumPy arrays, so
        # PyTorch 2.6+ needs an explicit opt-out from weights-only loading.
        payload = torch.load(processed_path, map_location="cpu", weights_only=False)
        if (
            payload.get("preprocessing_fingerprint") == current_fingerprint
            and "test_idx" in payload
            and (not rna_enabled or "rna_embeddings" in payload)
            and (not cnv_enabled or "cnv_embeddings" in payload)
            and (not methylation_enabled or "methylation_embeddings" in payload)
            and (not demographics_enabled or "demographic_vectors" in payload)
            and (not known_exposures_enabled or "known_exposures" in payload)
        ):
            return payload

    sbs_payload = _load_sbs96_matrix(config)
    if demographics_enabled:
        sbs_payload, _ = _load_matched_demographic_vectors(config, sbs_payload)
    if rna_enabled:
        sbs_payload, _ = _load_matched_rna_embeddings(config, sbs_payload)
    if cnv_enabled:
        sbs_payload, _ = _load_matched_cnv_embeddings(config, sbs_payload)
    if methylation_enabled:
        sbs_payload, _ = _load_matched_methylation_embeddings(config, sbs_payload)
    sig_payload = _load_signature_bank(config, sbs_payload["mutation_types"])
    reference_pool_indices = _build_reference_pools(
        config=config,
        normalized_profiles=sbs_payload["normalized_profiles"],
        signature_bank=sig_payload["signature_bank"],
    )
    if data_cfg.get("split_metadata_path"):
        train_idx, val_idx, test_idx = _load_split_indices_from_metadata(
            config=config,
            sample_ids=sbs_payload["sample_ids"],
        )
    elif all(key in sbs_payload for key in {"train_idx", "val_idx", "test_idx"}):
        train_idx = sbs_payload["train_idx"]
        val_idx = sbs_payload["val_idx"]
        test_idx = sbs_payload["test_idx"]
    else:
        train_idx, val_idx, test_idx = _split_indices(
            num_samples=len(sbs_payload["sample_ids"]),
            train_fraction=float(data_cfg.get("train_fraction", 0.8)),
            val_fraction=float(data_cfg.get("val_fraction", 0.1)),
            seed=int(config["experiment"]["seed"]),
        )

    payload = {
        "sample_ids": sbs_payload["sample_ids"],
        "mutation_types": sbs_payload["mutation_types"],
        "count_matrix": sbs_payload["count_matrix"],
        "normalized_profiles": sbs_payload["normalized_profiles"],
        "depths": sbs_payload["depths"],
        "signature_names": sig_payload["signature_names"],
        "signature_bank": sig_payload["signature_bank"],
        "reference_pool_indices": reference_pool_indices,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "preprocessing_fingerprint": current_fingerprint,
    }
    if known_exposures_enabled:
        known_exposures, exposure_metadata, exposure_valid_mask = _load_aligned_known_exposures(
            config=config,
            sample_ids=sbs_payload["sample_ids"],
            signature_names=sig_payload["signature_names"],
        )
        payload["known_exposures"] = known_exposures
        payload["known_exposures_valid_mask"] = exposure_valid_mask
        payload["known_exposures_metadata"] = exposure_metadata
    if "rna_embeddings" in sbs_payload:
        payload["rna_embeddings"] = sbs_payload["rna_embeddings"]
        payload["rna_sample_ids"] = sbs_payload["rna_sample_ids"]
        payload["rna_metadata"] = sbs_payload["rna_metadata"]
    if "cnv_embeddings" in sbs_payload:
        payload["cnv_embeddings"] = sbs_payload["cnv_embeddings"]
        payload["cnv_sample_ids"] = sbs_payload["cnv_sample_ids"]
        payload["cnv_metadata"] = sbs_payload["cnv_metadata"]
    if "methylation_embeddings" in sbs_payload:
        payload["methylation_embeddings"] = sbs_payload["methylation_embeddings"]
        payload["methylation_sample_ids"] = sbs_payload["methylation_sample_ids"]
        payload["methylation_metadata"] = sbs_payload["methylation_metadata"]
    if "demographic_vectors" in sbs_payload:
        payload["demographic_vectors"] = sbs_payload["demographic_vectors"]
        payload["demographic_metadata"] = sbs_payload["demographic_metadata"]
    torch.save(payload, processed_path)
    return payload


class SBSRefittingDataset(Dataset):
    def __init__(
        self,
        normalized_profiles: np.ndarray,
        count_matrix: np.ndarray,
        depths: np.ndarray,
        reference_pool_indices: np.ndarray,
        sample_ids,
        indices: np.ndarray,
        rna_embeddings: np.ndarray = None,
        cnv_embeddings: np.ndarray = None,
        methylation_embeddings: np.ndarray = None,
        demographic_vectors: np.ndarray = None,
        known_exposures: np.ndarray = None,
        known_exposures_valid_mask: np.ndarray = None,
    ) -> None:
        self.normalized_profiles = normalized_profiles
        self.count_matrix = count_matrix
        self.depths = depths
        self.reference_pool_indices = reference_pool_indices
        self.sample_ids = sample_ids
        self.indices = indices.astype(np.int64)
        self.rna_embeddings = rna_embeddings
        self.cnv_embeddings = cnv_embeddings
        self.methylation_embeddings = methylation_embeddings
        self.demographic_vectors = demographic_vectors
        self.known_exposures = known_exposures
        self.known_exposures_valid_mask = known_exposures_valid_mask

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row_idx = int(self.indices[index])
        item = {
            "profile": torch.tensor(self.normalized_profiles[row_idx], dtype=torch.float32),
            "counts": torch.tensor(self.count_matrix[row_idx], dtype=torch.float32),
            "depth": torch.tensor(self.depths[row_idx], dtype=torch.float32),
            "reference_pool_indices": torch.tensor(self.reference_pool_indices[row_idx], dtype=torch.long),
            "row_idx": torch.tensor(row_idx, dtype=torch.long),
        }
        if self.rna_embeddings is not None:
            item["rna_embedding"] = torch.tensor(self.rna_embeddings[row_idx], dtype=torch.float32)
        if self.cnv_embeddings is not None:
            item["cnv_embedding"] = torch.tensor(self.cnv_embeddings[row_idx], dtype=torch.float32)
        if self.methylation_embeddings is not None:
            item["methylation_embedding"] = torch.tensor(self.methylation_embeddings[row_idx], dtype=torch.float32)
        if self.demographic_vectors is not None:
            item["demographic_vector"] = torch.tensor(self.demographic_vectors[row_idx], dtype=torch.float32)
        if self.known_exposures is not None:
            item["known_exposures"] = torch.tensor(self.known_exposures[row_idx], dtype=torch.float32)
        if self.known_exposures_valid_mask is not None:
            item["known_exposures_valid"] = torch.tensor(
                self.known_exposures_valid_mask[row_idx], dtype=torch.bool
            )
        return item


def create_datasets(config: Dict):
    payload = prepare_processed_dataset(config)
    train_dataset = SBSRefittingDataset(
        normalized_profiles=payload["normalized_profiles"],
        count_matrix=payload["count_matrix"],
        depths=payload["depths"],
        reference_pool_indices=payload["reference_pool_indices"],
        sample_ids=payload["sample_ids"],
        indices=payload["train_idx"],
        rna_embeddings=payload.get("rna_embeddings"),
        cnv_embeddings=payload.get("cnv_embeddings"),
        methylation_embeddings=payload.get("methylation_embeddings"),
        demographic_vectors=payload.get("demographic_vectors"),
        known_exposures=payload.get("known_exposures"),
        known_exposures_valid_mask=payload.get("known_exposures_valid_mask"),
    )
    val_dataset = SBSRefittingDataset(
        normalized_profiles=payload["normalized_profiles"],
        count_matrix=payload["count_matrix"],
        depths=payload["depths"],
        reference_pool_indices=payload["reference_pool_indices"],
        sample_ids=payload["sample_ids"],
        indices=payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"],
        rna_embeddings=payload.get("rna_embeddings"),
        cnv_embeddings=payload.get("cnv_embeddings"),
        methylation_embeddings=payload.get("methylation_embeddings"),
        demographic_vectors=payload.get("demographic_vectors"),
        known_exposures=payload.get("known_exposures"),
        known_exposures_valid_mask=payload.get("known_exposures_valid_mask"),
    )
    test_dataset = SBSRefittingDataset(
        normalized_profiles=payload["normalized_profiles"],
        count_matrix=payload["count_matrix"],
        depths=payload["depths"],
        reference_pool_indices=payload["reference_pool_indices"],
        sample_ids=payload["sample_ids"],
        indices=payload["test_idx"] if len(payload["test_idx"]) > 0 else payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"],
        rna_embeddings=payload.get("rna_embeddings"),
        cnv_embeddings=payload.get("cnv_embeddings"),
        methylation_embeddings=payload.get("methylation_embeddings"),
        demographic_vectors=payload.get("demographic_vectors"),
        known_exposures=payload.get("known_exposures"),
        known_exposures_valid_mask=payload.get("known_exposures_valid_mask"),
    )
    return train_dataset, val_dataset, test_dataset, payload
