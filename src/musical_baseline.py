from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import create_datasets
from src.utils import ensure_dir, save_json, seed_everything


def _resolve_split_indices(payload: Dict, split: str):
    if split == "train":
        return payload["train_idx"]
    if split == "val":
        return payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    if split == "test":
        return (
            payload["test_idx"]
            if len(payload["test_idx"]) > 0
            else payload["val_idx"]
            if len(payload["val_idx"]) > 0
            else payload["train_idx"]
        )
    return np.arange(len(payload["sample_ids"]), dtype=np.int64)


def _split_output_path(base_path: Path, split: str) -> Path:
    return base_path.with_name("{0}_{1}{2}".format(base_path.stem, split, base_path.suffix))


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), a_min=0.0, a_max=None)
    total = float(values.sum())
    if total <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return values / total


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.sum(left * right))
    denominator = float(np.sqrt(np.sum(left * left)) * np.sqrt(np.sum(right * right)))
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _load_signature_dataframe(payload: Dict) -> pd.DataFrame:
    signatures = np.asarray(payload["signature_bank"], dtype=np.float64)
    return pd.DataFrame(
        signatures.T,
        index=list(payload["mutation_types"]),
        columns=list(payload["signature_names"]),
    )


def _load_catalog_dataframe(catalog_name: str) -> pd.DataFrame:
    try:
        import musical
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuSiCal is not installed in the active environment. Install MuSiCal in "
            "~/envs/sigformer before running this baseline."
        ) from exc
    if not hasattr(musical, "load_catalog"):
        raise AttributeError("The installed musical module does not expose load_catalog().")
    catalog = musical.load_catalog(catalog_name)
    if not hasattr(catalog, "W"):
        raise AttributeError(f"MuSiCal catalog {catalog_name!r} does not expose W.")
    w = catalog.W
    if not isinstance(w, pd.DataFrame):
        w = pd.DataFrame(w)
    return w.astype(float)


def _write_input_matrix(payload: Dict, indices, split: str, output_path: Path) -> pd.DataFrame:
    sample_ids = [str(payload["sample_ids"][int(idx)]) for idx in indices]
    counts = np.asarray(payload["count_matrix"][indices], dtype=np.float64).T
    x = pd.DataFrame(counts, index=list(payload["mutation_types"]), columns=sample_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x.to_csv(output_path, sep="\t")
    return x


def _run_musical_refit(
    x: pd.DataFrame,
    w: pd.DataFrame,
    method: str,
    threshold: float,
):
    try:
        import musical
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuSiCal is not installed in the active environment. Install MuSiCal in "
            "~/envs/sigformer before running this baseline."
        ) from exc

    common_channels = [channel for channel in x.index if channel in w.index]
    if len(common_channels) != len(x.index):
        missing = [channel for channel in x.index if channel not in w.index]
        raise ValueError(
            "MuSiCal signature catalog is missing mutation channels: {0}".format(
                ", ".join(missing[:10])
            )
        )
    w = w.loc[common_channels]
    x = x.loc[common_channels]

    refit_module = getattr(musical, "refit", None)
    refit_fn = getattr(refit_module, "refit", None) if refit_module is not None else None
    if refit_fn is None and callable(refit_module):
        refit_fn = refit_module
    if refit_fn is None:
        raise AttributeError("The installed musical module does not expose musical.refit.refit().")

    result = refit_fn(x, w, method=method, thresh=threshold)
    h = result[0] if isinstance(result, tuple) else result
    model = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    if not isinstance(h, pd.DataFrame):
        h = pd.DataFrame(h)

    signature_names = list(w.columns)
    sample_ids = list(x.columns)
    if list(h.index) == signature_names and list(h.columns) == sample_ids:
        h_df = h.astype(float)
    elif list(h.columns) == signature_names and list(h.index) == sample_ids:
        h_df = h.T.astype(float)
    elif h.shape == (len(signature_names), len(sample_ids)):
        h_df = pd.DataFrame(h.to_numpy(dtype=float), index=signature_names, columns=sample_ids)
    elif h.shape == (len(sample_ids), len(signature_names)):
        h_df = pd.DataFrame(h.to_numpy(dtype=float).T, index=signature_names, columns=sample_ids)
    else:
        raise ValueError(
            "Could not infer MuSiCal H orientation. Got shape {0}; expected signatures x samples.".format(
                h.shape
            )
        )
    h_df = h_df.clip(lower=0.0)
    return h_df, model, w, x


def _save_split_outputs(
    payload: Dict,
    indices,
    split: str,
    target: Path,
    input_matrix_path: Path,
    w: pd.DataFrame,
    h: pd.DataFrame,
    x: pd.DataFrame,
) -> Dict[str, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = [str(payload["sample_ids"][int(idx)]) for idx in indices]
    depths = {str(payload["sample_ids"][int(idx)]): float(payload["depths"][int(idx)]) for idx in indices}
    signature_names = list(w.columns)

    activities_path = target.with_name("{0}_activities{1}".format(target.stem, target.suffix))
    activity_rows = []
    exposure_rows = []
    reconstruction_rows = []
    residual_rows = []
    for sample_id in sample_ids:
        activities = h[sample_id].to_numpy(dtype=np.float64)
        exposures = _normalize_vector(activities)
        reconstruction_counts = w.to_numpy(dtype=np.float64) @ activities
        target_counts = x[sample_id].to_numpy(dtype=np.float64)
        target_profile = _normalize_vector(target_counts)
        reconstruction_profile = _normalize_vector(reconstruction_counts)
        residual = target_profile - reconstruction_profile

        base = {
            "sample_id": sample_id,
            "depth": depths[sample_id],
            "split": split,
        }
        activity_rows.append({**base, **dict(zip(signature_names, activities))})
        exposure_rows.append({**base, **dict(zip(signature_names, exposures))})

        reconstruction_row = {
            **base,
            "cosine_similarity": _cosine(target_profile, reconstruction_profile),
            "mse": float(np.mean((target_profile - reconstruction_profile) ** 2)),
            "mae": float(np.mean(np.abs(target_profile - reconstruction_profile))),
        }
        residual_row = dict(base)
        for channel, observed, reconstructed, residual_value in zip(
            list(x.index),
            target_profile,
            reconstruction_profile,
            residual,
        ):
            reconstruction_row[f"observed::{channel}"] = float(observed)
            reconstruction_row[f"reconstructed::{channel}"] = float(reconstructed)
            residual_row[f"residual::{channel}"] = float(residual_value)
        reconstruction_rows.append(reconstruction_row)
        residual_rows.append(residual_row)

    pd.DataFrame(activity_rows).fillna(0.0).to_csv(activities_path, index=False)
    pd.DataFrame(exposure_rows).fillna(0.0).to_csv(target, index=False)
    reconstruction_path = target.with_name("{0}_reconstructions{1}".format(target.stem, target.suffix))
    residual_path = target.with_name("{0}_residuals{1}".format(target.stem, target.suffix))
    pd.DataFrame(reconstruction_rows).to_csv(reconstruction_path, index=False)
    pd.DataFrame(residual_rows).to_csv(residual_path, index=False)

    return {
        "split": split,
        "input_matrix_path": str(input_matrix_path),
        "activities_path": str(activities_path),
        "exposures_path": str(target),
        "reconstructions_path": str(reconstruction_path),
        "residuals_path": str(residual_path),
        "num_samples": int(len(sample_ids)),
    }


def run_musical_baseline_from_config(config: Dict) -> Dict[str, object]:
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    _, _, _, payload = create_datasets(config)
    baseline_cfg = config.get("musical_baseline", {})
    split = str(baseline_cfg.get("split", "train"))
    output_path = Path(
        str(
            baseline_cfg.get(
                "output_path",
                "outputs/domain_baselines/musical_refit_{0}.csv".format(config["run_timestamp"]),
            )
        )
    )
    ensure_dir(str(output_path.parent))

    catalog_name = str(baseline_cfg.get("catalog", "") or "")
    if catalog_name:
        w = _load_catalog_dataframe(catalog_name)
    else:
        w = _load_signature_dataframe(payload)
    method = str(baseline_cfg.get("method", "likelihood_bidirectional"))
    threshold = float(baseline_cfg.get("threshold", 0.001))

    split_names = ["train", "val", "test"] if split == "all_splits" else [split]
    output_records = []
    for split_name in split_names:
        split_target = _split_output_path(output_path, split_name) if split == "all_splits" else output_path
        indices = _resolve_split_indices(payload, split_name)
        input_matrix_path = split_target.with_name(
            "{0}_musical_input_matrix.tsv".format(split_target.stem)
        )
        x = _write_input_matrix(payload, indices, split_name, input_matrix_path)
        h, _model, w_aligned, x_aligned = _run_musical_refit(
            x=x,
            w=w,
            method=method,
            threshold=threshold,
        )
        output_records.append(
            _save_split_outputs(
                payload=payload,
                indices=indices,
                split=split_name,
                target=split_target,
                input_matrix_path=input_matrix_path,
                w=w_aligned,
                h=h,
                x=x_aligned,
            )
        )

    summary = {
        "config_path": config.get("config_path"),
        "run_timestamp": config.get("run_timestamp"),
        "split": split,
        "catalog": catalog_name or "config_signature_bank",
        "method": method,
        "threshold": threshold,
        "output_records": output_records,
    }
    summary_path = output_path.with_name("{0}_summary.json".format(output_path.stem))
    save_json(summary, str(summary_path))
    summary["summary_path"] = str(summary_path)
    return summary


def load_config_and_run_musical_baseline(config_path: str) -> Dict[str, object]:
    return run_musical_baseline_from_config(load_config(config_path))
