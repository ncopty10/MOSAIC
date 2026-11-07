import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import ensure_dir

try:
    from scipy.stats import wilcoxon
except ModuleNotFoundError:  # pragma: no cover
    wilcoxon = None


HIGHER_IS_BETTER = {
    "cosine_similarity",
    "spearman_correlation",
    "pearson_correlation",
    "histogram_intersection_similarity",
    "explained_variance_r2",
}
LOWER_IS_BETTER = {
    "mse",
    "mae",
    "kl_divergence",
    "js_divergence",
    "positive_residual_sum",
    "absolute_residual_sum",
    "residual_entropy",
    "residual_l2_norm",
}

METRIC_NAMES = [
    "cosine_similarity",
    "mse",
    "mae",
    "spearman_correlation",
    "pearson_correlation",
    "histogram_intersection_similarity",
    "kl_divergence",
    "js_divergence",
    "explained_variance_r2",
    "positive_residual_sum",
    "absolute_residual_sum",
    "residual_entropy",
    "residual_l2_norm",
]

SUBGROUP_METRICS = [
    "cosine_similarity",
    "mse",
    "spearman_correlation",
    "positive_residual_sum",
]
DEFAULT_GENDER_GROUPS = ["female", "male"]
DEFAULT_RACE_GROUPS = ["white", "black or african american", "asian"]


def _get_pyplot():
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-{0}".format(os.environ.get("USER", "codex")))
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError:
        return None


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _average_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman_rowwise(observed: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    scores = np.empty(observed.shape[0], dtype=np.float64)
    for row_idx in range(observed.shape[0]):
        obs_rank = _average_rank(observed[row_idx])
        rec_rank = _average_rank(reconstructed[row_idx])
        scores[row_idx] = _safe_corrcoef(obs_rank, rec_rank)
    return scores


def _pearson_rowwise(observed: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    scores = np.empty(observed.shape[0], dtype=np.float64)
    for row_idx in range(observed.shape[0]):
        scores[row_idx] = _safe_corrcoef(observed[row_idx], reconstructed[row_idx])
    return scores


def _normalize_probability_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, a_min=0.0, a_max=None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > eps, clipped / row_sums, np.full_like(clipped, 1.0 / clipped.shape[1]))
    return np.clip(safe, a_min=eps, a_max=None)


def _histogram_intersection_similarity_rowwise(observed: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    p = _normalize_probability_rows(observed)
    q = _normalize_probability_rows(reconstructed)
    return np.minimum(p, q).sum(axis=1)


def _kl_divergence_rowwise(observed: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    p = _normalize_probability_rows(observed)
    q = _normalize_probability_rows(reconstructed)
    return np.sum(p * np.log(p / q), axis=1)


def _js_divergence_rowwise(observed: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    p = _normalize_probability_rows(observed)
    q = _normalize_probability_rows(reconstructed)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m), axis=1) + 0.5 * np.sum(q * np.log(q / m), axis=1)


def _explained_variance_r2_rowwise(observed: np.ndarray, reconstructed: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    centered = observed - observed.mean(axis=1, keepdims=True)
    sst = np.sum(centered ** 2, axis=1)
    sse = np.sum((observed - reconstructed) ** 2, axis=1)
    scores = np.empty(observed.shape[0], dtype=np.float64)
    mask = sst > eps
    scores[mask] = 1.0 - (sse[mask] / sst[mask])
    scores[~mask] = np.where(sse[~mask] <= eps, 1.0, np.nan)
    return scores


def _residual_entropy_rowwise(residual: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    positive = np.clip(np.asarray(residual, dtype=np.float64), a_min=0.0, a_max=None)
    row_sums = positive.sum(axis=1, keepdims=True)
    probs = np.where(row_sums > eps, positive / row_sums, 0.0)
    log_probs = np.zeros_like(probs)
    mask = probs > eps
    log_probs[mask] = np.log(probs[mask])
    return -np.sum(probs * log_probs, axis=1)


def _infer_residual_path(reconstruction_path: str) -> Path:
    reconstruction = Path(reconstruction_path)
    if reconstruction.name.endswith("_reconstructions.csv"):
        return reconstruction.with_name(reconstruction.name.replace("_reconstructions.csv", "_residuals.csv"))
    raise ValueError(
        "Expected a reconstruction CSV path ending with '_reconstructions.csv', got '{0}'.".format(reconstruction_path)
    )


def load_run_artifacts(reconstruction_path: str) -> Dict[str, pd.DataFrame]:
    recon_path = Path(reconstruction_path)
    residual_path = _infer_residual_path(reconstruction_path)
    if not recon_path.exists():
        raise FileNotFoundError("Reconstruction CSV not found: {0}".format(recon_path))
    if not residual_path.exists():
        raise FileNotFoundError("Residual CSV not found for reconstruction file: {0}".format(residual_path))

    reconstruction_df = pd.read_csv(recon_path)
    residual_df = pd.read_csv(residual_path)
    return {
        "reconstruction_path": str(recon_path),
        "residual_path": str(residual_path),
        "reconstruction_df": reconstruction_df,
        "residual_df": residual_df,
    }


def compute_sample_metrics(reconstruction_df: pd.DataFrame, residual_df: pd.DataFrame) -> pd.DataFrame:
    observed_cols = [column for column in reconstruction_df.columns if column.startswith("observed::")]
    reconstructed_cols = [column for column in reconstruction_df.columns if column.startswith("reconstructed::")]
    residual_cols = [column for column in residual_df.columns if column.startswith("residual::")]

    observed = reconstruction_df[observed_cols].to_numpy(dtype=np.float64)
    reconstructed = reconstruction_df[reconstructed_cols].to_numpy(dtype=np.float64)
    residual = residual_df[residual_cols].to_numpy(dtype=np.float64)

    sample_metrics = reconstruction_df[["sample_id", "depth", "split", "cosine_similarity", "mse", "mae"]].copy()
    sample_metrics["spearman_correlation"] = _spearman_rowwise(observed, reconstructed)
    sample_metrics["pearson_correlation"] = _pearson_rowwise(observed, reconstructed)
    sample_metrics["histogram_intersection_similarity"] = _histogram_intersection_similarity_rowwise(observed, reconstructed)
    sample_metrics["kl_divergence"] = _kl_divergence_rowwise(observed, reconstructed)
    sample_metrics["js_divergence"] = _js_divergence_rowwise(observed, reconstructed)
    sample_metrics["explained_variance_r2"] = _explained_variance_r2_rowwise(observed, reconstructed)
    sample_metrics["positive_residual_sum"] = np.clip(residual, a_min=0.0, a_max=None).sum(axis=1)
    sample_metrics["absolute_residual_sum"] = np.abs(residual).sum(axis=1)
    sample_metrics["signed_residual_sum"] = residual.sum(axis=1)
    sample_metrics["residual_entropy"] = _residual_entropy_rowwise(residual)
    sample_metrics["residual_l2_norm"] = np.sqrt((residual ** 2).sum(axis=1))
    return sample_metrics


def summarize_sample_metrics(sample_metrics: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for metric in METRIC_NAMES:
        values = sample_metrics[metric].to_numpy(dtype=np.float64)
        rows.append(
            {
                "label": label,
                "metric": metric,
                "num_samples": int(len(values)),
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
                "std": float(np.nanstd(values)),
                "q25": float(np.nanpercentile(values, 25)),
                "q75": float(np.nanpercentile(values, 75)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )
    return pd.DataFrame(rows)


def compute_channel_residual_summary(residual_df: pd.DataFrame, label: str) -> pd.DataFrame:
    residual_cols = [column for column in residual_df.columns if column.startswith("residual::")]
    residual = residual_df[residual_cols].to_numpy(dtype=np.float64)
    rows = []
    for idx, column in enumerate(residual_cols):
        channel = column.replace("residual::", "", 1)
        values = residual[:, idx]
        rows.append(
            {
                "label": label,
                "channel": channel,
                "mean_residual": float(np.mean(values)),
                "mean_absolute_residual": float(np.mean(np.abs(values))),
                "mean_positive_residual": float(np.mean(np.clip(values, a_min=0.0, a_max=None))),
            }
        )
    return pd.DataFrame(rows)


def _paired_bootstrap_ci(
    differences: np.ndarray,
    repeats: int = 1000,
    seed: int = 20260728,
) -> Tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(int(repeats), dtype=np.float64)
    for repeat_idx in range(int(repeats)):
        indices = rng.integers(0, values.size, size=values.size)
        bootstrap_means[repeat_idx] = float(values[indices].mean())
    return (
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    )


def _benjamini_hochberg(pvalues: List[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted
    finite_values = values[finite_indices]
    order = np.argsort(finite_values)
    ordered = finite_values[order]
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    ordered_adjusted = ordered * ordered.size / ranks
    ordered_adjusted = np.minimum.accumulate(ordered_adjusted[::-1])[::-1]
    ordered_adjusted = np.clip(ordered_adjusted, 0.0, 1.0)
    restored = np.empty_like(ordered_adjusted)
    restored[order] = ordered_adjusted
    adjusted[finite_indices] = restored
    return adjusted


def compare_runs(
    sample_metrics_by_label: Dict[str, pd.DataFrame],
    baseline_label: str,
    bootstrap_repeats: int = 1000,
    bootstrap_seed: int = 20260728,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if baseline_label not in sample_metrics_by_label:
        raise ValueError("Baseline label '{0}' was not provided.".format(baseline_label))

    baseline = sample_metrics_by_label[baseline_label]
    comparison_rows = []
    merged_frames = []
    for label, df in sample_metrics_by_label.items():
        if label == baseline_label:
            continue
        merged = baseline.merge(df, on=["sample_id", "split"], suffixes=("_base", "_compare"))
        merged["baseline_label"] = baseline_label
        merged["compare_label"] = label
        merged_frames.append(merged)
        for metric in METRIC_NAMES:
            base_values = merged["{0}_base".format(metric)].to_numpy(dtype=np.float64)
            compare_values = merged["{0}_compare".format(metric)].to_numpy(dtype=np.float64)
            delta = compare_values - base_values
            if metric in HIGHER_IS_BETTER:
                improved = compare_values > base_values
            else:
                improved = compare_values < base_values
            bootstrap_ci_low, bootstrap_ci_high = _paired_bootstrap_ci(
                delta,
                repeats=bootstrap_repeats,
                seed=bootstrap_seed + len(comparison_rows),
            )
            if wilcoxon is None or not np.isfinite(delta).any() or np.allclose(delta[np.isfinite(delta)], 0.0):
                wilcoxon_pvalue = float("nan")
            else:
                finite_delta = delta[np.isfinite(delta)]
                wilcoxon_pvalue = float(wilcoxon(finite_delta, alternative="two-sided").pvalue)
            comparison_rows.append(
                {
                    "baseline_label": baseline_label,
                    "compare_label": label,
                    "metric": metric,
                    "num_paired_samples": int(len(merged)),
                    "mean_delta": float(np.nanmean(delta)),
                    "median_delta": float(np.nanmedian(delta)),
                    "fraction_improved": float(np.mean(improved.astype(np.float64))),
                    "baseline_mean": float(np.nanmean(base_values)),
                    "compare_mean": float(np.nanmean(compare_values)),
                    "mean_delta_bootstrap_ci95_low": bootstrap_ci_low,
                    "mean_delta_bootstrap_ci95_high": bootstrap_ci_high,
                    "bootstrap_repeats": int(bootstrap_repeats),
                    "bootstrap_seed": int(bootstrap_seed),
                    "wilcoxon_pvalue": wilcoxon_pvalue,
                }
            )
    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        # One correction family comprises every metric comparison emitted by
        # this paired evaluation call.
        comparison_df["wilcoxon_fdr_bh"] = _benjamini_hochberg(
            comparison_df["wilcoxon_pvalue"].tolist()
        )
        comparison_df["multiple_testing_family"] = "all_metrics_in_paired_evaluation"
    merged_df = pd.concat(merged_frames, ignore_index=True) if merged_frames else pd.DataFrame()
    return comparison_df, merged_df


def _load_demographics_table(demographics_csv: Optional[str]) -> pd.DataFrame:
    if not demographics_csv:
        return pd.DataFrame()
    path = Path(demographics_csv)
    if not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    try:
        df = pd.read_csv(path, sep=sep)
    except pd.errors.ParserError:
        # Fall back to Python's delimiter sniffing for files whose extension
        # does not reflect the actual delimiter.
        df = pd.read_csv(path, sep=None, engine="python")
    required = {"sample_id", "gender", "race"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    out = df[["sample_id", "gender", "race"]].copy()
    out["gender"] = out["gender"].astype(str).str.strip().str.lower()
    out["race"] = out["race"].astype(str).str.strip().str.lower()
    return out.drop_duplicates("sample_id")


def _attach_demographics(df: pd.DataFrame, demographics_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or demographics_df.empty:
        return df.copy()
    merged = df.merge(demographics_df, on="sample_id", how="left")
    return merged


def _subgroup_values(series: pd.Series, preferred_values: List[str]) -> List[str]:
    available = []
    normalized = series.dropna().astype(str).str.strip().str.lower()
    available_set = set(normalized.tolist())
    for value in preferred_values:
        if value in available_set:
            available.append(value)
    return available


def summarize_subgroup_metrics(sample_metrics_by_label: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, df in sample_metrics_by_label.items():
        for subgroup_col, preferred_values in [
            ("gender", DEFAULT_GENDER_GROUPS),
            ("race", DEFAULT_RACE_GROUPS),
        ]:
            if subgroup_col not in df.columns:
                continue
            for subgroup_value in _subgroup_values(df[subgroup_col], preferred_values):
                subgroup_df = df.loc[df[subgroup_col] == subgroup_value]
                subgroup_size = int(len(subgroup_df))
                if subgroup_size == 0:
                    continue
                for metric in SUBGROUP_METRICS:
                    values = subgroup_df[metric].to_numpy(dtype=np.float64)
                    rows.append(
                        {
                            "label": label,
                            "group_type": subgroup_col,
                            "group_value": subgroup_value,
                            "subgroup_size": subgroup_size,
                            "metric": metric,
                            "mean": float(np.nanmean(values)),
                            "median": float(np.nanmedian(values)),
                            "std": float(np.nanstd(values)),
                            "q25": float(np.nanpercentile(values, 25)),
                            "q75": float(np.nanpercentile(values, 75)),
                            "min": float(np.nanmin(values)),
                            "max": float(np.nanmax(values)),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_subgroup_deltas(merged_df: pd.DataFrame) -> pd.DataFrame:
    if merged_df.empty:
        return pd.DataFrame()
    rows = []
    for compare_label in merged_df["compare_label"].unique():
        subset = merged_df.loc[merged_df["compare_label"] == compare_label]
        baseline_label = str(subset["baseline_label"].iloc[0])
        for subgroup_col, preferred_values in [
            ("gender", DEFAULT_GENDER_GROUPS),
            ("race", DEFAULT_RACE_GROUPS),
        ]:
            if subgroup_col not in subset.columns:
                continue
            for subgroup_value in _subgroup_values(subset[subgroup_col], preferred_values):
                subgroup_df = subset.loc[subset[subgroup_col] == subgroup_value]
                subgroup_size = int(len(subgroup_df))
                if subgroup_size == 0:
                    continue
                for metric in SUBGROUP_METRICS:
                    base_values = subgroup_df["{0}_base".format(metric)].to_numpy(dtype=np.float64)
                    compare_values = subgroup_df["{0}_compare".format(metric)].to_numpy(dtype=np.float64)
                    delta = compare_values - base_values
                    if metric in HIGHER_IS_BETTER:
                        improved = compare_values > base_values
                    else:
                        improved = compare_values < base_values
                    rows.append(
                        {
                            "baseline_label": baseline_label,
                            "compare_label": compare_label,
                            "group_type": subgroup_col,
                            "group_value": subgroup_value,
                            "subgroup_size": subgroup_size,
                            "metric": metric,
                            "baseline_mean": float(np.nanmean(base_values)),
                            "compare_mean": float(np.nanmean(compare_values)),
                            "mean_delta": float(np.nanmean(delta)),
                            "median_delta": float(np.nanmedian(delta)),
                            "fraction_improved": float(np.mean(improved.astype(np.float64))),
                        }
                    )
    return pd.DataFrame(rows)


def _plot_subgroup_metric_boxplots(sample_metrics_by_label: Dict[str, pd.DataFrame], output_dir: Path) -> List[str]:
    plt = _get_pyplot()
    if plt is None or not sample_metrics_by_label:
        return []
    saved_paths = []
    for subgroup_col, subgroup_values in [
        ("gender", DEFAULT_GENDER_GROUPS),
        ("race", DEFAULT_RACE_GROUPS),
    ]:
        for metric in SUBGROUP_METRICS:
            labels = []
            values = []
            for subgroup_value in subgroup_values:
                for label, df in sample_metrics_by_label.items():
                    if subgroup_col not in df.columns:
                        continue
                    subgroup_df = df.loc[df[subgroup_col] == subgroup_value]
                    if subgroup_df.empty:
                        continue
                    labels.append("{0}\n{1}".format(subgroup_value, label))
                    values.append(subgroup_df[metric].to_numpy(dtype=np.float64))
            if not values:
                continue
            fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 4.8))
            ax.boxplot(values, tick_labels=labels, showfliers=False)
            ax.set_title("{0} by {1}".format(metric.replace("_", " ").title(), subgroup_col.title()))
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = output_dir / "subgroup_{0}_{1}_boxplot.png".format(subgroup_col, metric)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(str(path))
    return saved_paths


def _plot_subgroup_delta_heatmaps(subgroup_delta_df: pd.DataFrame, output_dir: Path) -> List[str]:
    plt = _get_pyplot()
    if plt is None or subgroup_delta_df.empty:
        return []
    saved_paths = []
    for compare_label in subgroup_delta_df["compare_label"].unique():
        subset = subgroup_delta_df.loc[subgroup_delta_df["compare_label"] == compare_label]
        for subgroup_col in ["gender", "race"]:
            subgroup_subset = subset.loc[subset["group_type"] == subgroup_col]
            if subgroup_subset.empty:
                continue
            pivot = subgroup_subset.pivot(index="group_value", columns="metric", values="mean_delta")
            ordered_rows = [value for value in (DEFAULT_GENDER_GROUPS if subgroup_col == "gender" else DEFAULT_RACE_GROUPS) if value in pivot.index]
            pivot = pivot.loc[ordered_rows]
            ordered_cols = [metric for metric in SUBGROUP_METRICS if metric in pivot.columns]
            pivot = pivot[ordered_cols]
            fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.9 * len(pivot.index))))
            vmax = float(np.nanmax(np.abs(pivot.to_numpy(dtype=np.float64)))) if not pivot.empty else 1.0
            vmax = max(vmax, 1e-8)
            im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title("Mean Delta by {0}: {1}".format(subgroup_col.title(), compare_label))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index.tolist())
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([col.replace("_", " ") for col in pivot.columns.tolist()], rotation=30, ha="right")
            for row_idx in range(len(pivot.index)):
                for col_idx in range(len(pivot.columns)):
                    value = pivot.iloc[row_idx, col_idx]
                    ax.text(col_idx, row_idx, "{0:.3g}".format(value), ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            fig.tight_layout()
            path = output_dir / "subgroup_{0}_{1}_mean_delta_heatmap.png".format(compare_label, subgroup_col)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(str(path))
    return saved_paths


def _plot_metric_distributions(sample_metrics_by_label: Dict[str, pd.DataFrame], output_dir: Path) -> List[str]:
    plt = _get_pyplot()
    if plt is None:
        return []
    metrics = [
        "cosine_similarity",
        "mse",
        "spearman_correlation",
        "js_divergence",
        "explained_variance_r2",
        "positive_residual_sum",
        "residual_entropy",
    ]
    saved_paths = []
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        labels = list(sample_metrics_by_label.keys())
        values = [sample_metrics_by_label[label][metric].to_numpy(dtype=np.float64) for label in labels]
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.set_title("{0} across runs".format(metric.replace("_", " ").title()))
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / "{0}_boxplot.png".format(metric)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(path))
    return saved_paths


def _plot_depth_scatter(sample_metrics: pd.DataFrame, label: str, output_dir: Path) -> List[str]:
    plt = _get_pyplot()
    if plt is None:
        return []
    metrics = ["cosine_similarity", "spearman_correlation", "positive_residual_sum", "js_divergence", "residual_entropy"]
    saved_paths = []
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.scatter(sample_metrics["depth"], sample_metrics[metric], s=10, alpha=0.5)
        ax.set_title("{0}: depth vs {1}".format(label, metric.replace("_", " ")))
        ax.set_xlabel("Depth")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / "{0}_{1}_depth_scatter.png".format(label, metric)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(path))
    return saved_paths


def _plot_channel_heatmap(channel_summary_df: pd.DataFrame, output_dir: Path) -> List[str]:
    plt = _get_pyplot()
    if plt is None or channel_summary_df.empty:
        return []

    saved_paths = []
    for value_col, title in [
        ("mean_absolute_residual", "Mean Absolute Residual by Channel"),
        ("mean_positive_residual", "Mean Positive Residual by Channel"),
    ]:
        pivot = channel_summary_df.pivot(index="label", columns="channel", values=value_col)
        fig, ax = plt.subplots(figsize=(18, max(3, 1.2 * len(pivot.index))))
        im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=90, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        fig.tight_layout()
        path = output_dir / "{0}_heatmap.png".format(value_col)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(path))
    return saved_paths


def _plot_paired_comparisons(merged_df: pd.DataFrame, output_dir: Path) -> List[str]:
    plt = _get_pyplot()
    if plt is None or merged_df.empty:
        return []
    saved_paths = []
    metrics = ["cosine_similarity", "spearman_correlation", "mse", "js_divergence", "positive_residual_sum"]
    for compare_label in merged_df["compare_label"].unique():
        subset = merged_df.loc[merged_df["compare_label"] == compare_label]
        baseline_label = str(subset["baseline_label"].iloc[0])
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            x = subset["{0}_base".format(metric)].to_numpy(dtype=np.float64)
            y = subset["{0}_compare".format(metric)].to_numpy(dtype=np.float64)
            ax.scatter(x, y, s=10, alpha=0.5)
            min_val = float(np.nanmin(np.concatenate([x, y])))
            max_val = float(np.nanmax(np.concatenate([x, y])))
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
            ax.set_xlabel("{0} ({1})".format(metric, baseline_label))
            ax.set_ylabel("{0} ({1})".format(metric, compare_label))
            ax.set_title("{0}: {1} vs {2}".format(metric, baseline_label, compare_label))
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = output_dir / "{0}_vs_{1}_{2}_scatter.png".format(compare_label, baseline_label, metric)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(str(path))

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            delta = y - x
            ax.hist(delta, bins=40, alpha=0.85)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_title("{0}: delta {1} - {2}".format(metric, compare_label, baseline_label))
            ax.set_xlabel("Delta")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = output_dir / "{0}_vs_{1}_{2}_delta_hist.png".format(compare_label, baseline_label, metric)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(str(path))
    return saved_paths


def _plot_representative_sample_heatmaps(
    reconstruction_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    label: str,
    output_dir: Path,
    top_n: int = 5,
) -> List[str]:
    plt = _get_pyplot()
    if plt is None or reconstruction_df.empty:
        return []

    observed_cols = [column for column in reconstruction_df.columns if column.startswith("observed::")]
    reconstructed_cols = [column for column in reconstruction_df.columns if column.startswith("reconstructed::")]
    residual_cols = [column for column in residual_df.columns if column.startswith("residual::")]
    ordered = reconstruction_df.sort_values("cosine_similarity", ascending=False)
    selected = pd.concat([ordered.head(top_n), ordered.tail(top_n)], axis=0).drop_duplicates("sample_id")
    selected_ids = selected["sample_id"].tolist()
    recon_selected = reconstruction_df.set_index("sample_id").loc[selected_ids]
    resid_selected = residual_df.set_index("sample_id").loc[selected_ids]

    matrices = [
        (recon_selected[observed_cols].to_numpy(dtype=np.float64), "Observed Profiles"),
        (recon_selected[reconstructed_cols].to_numpy(dtype=np.float64), "Reconstructed Profiles"),
        (resid_selected[residual_cols].to_numpy(dtype=np.float64), "Residual Profiles"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True)
    for ax, (matrix, title) in zip(axes, matrices):
        im = ax.imshow(matrix, aspect="auto", cmap="coolwarm" if "Residual" in title else "viridis")
        ax.set_title("{0}: {1}".format(label, title))
        ax.set_yticks(np.arange(len(selected_ids)))
        ax.set_yticklabels(selected_ids, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    axes[-1].set_xticks(np.arange(len(observed_cols)))
    axes[-1].set_xticklabels([column.replace("observed::", "") for column in observed_cols], rotation=90, fontsize=6)
    fig.tight_layout()
    path = output_dir / "{0}_representative_sample_heatmaps.png".format(label)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [str(path)]


def run_evaluation(
    runs: List[Tuple[str, str]],
    output_dir: str,
    baseline_label: str = None,
    demographics_csv: str = None,
    bootstrap_repeats: int = 1000,
    bootstrap_seed: int = 20260728,
) -> Dict[str, str]:
    output_root = ensure_dir(output_dir)
    plots_dir = ensure_dir(str(output_root / "plots"))
    demographics_df = _load_demographics_table(demographics_csv)

    sample_metrics_by_label: Dict[str, pd.DataFrame] = {}
    summary_frames = []
    channel_frames = []
    representative_plot_paths: List[str] = []

    for label, reconstruction_path in runs:
        artifacts = load_run_artifacts(reconstruction_path)
        sample_metrics = compute_sample_metrics(artifacts["reconstruction_df"], artifacts["residual_df"])
        sample_metrics = _attach_demographics(sample_metrics, demographics_df)
        sample_metrics_by_label[label] = sample_metrics
        sample_metrics.to_csv(output_root / "{0}_sample_metrics.csv".format(label), index=False)
        summary_frames.append(summarize_sample_metrics(sample_metrics, label))
        channel_frames.append(compute_channel_residual_summary(artifacts["residual_df"], label))
        representative_plot_paths.extend(
            _plot_representative_sample_heatmaps(
                reconstruction_df=artifacts["reconstruction_df"],
                residual_df=artifacts["residual_df"],
                label=label,
                output_dir=plots_dir,
            )
        )
        _plot_depth_scatter(sample_metrics, label, plots_dir)

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary_path = output_root / "run_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    channel_summary_df = pd.concat(channel_frames, ignore_index=True) if channel_frames else pd.DataFrame()
    channel_summary_path = output_root / "channel_residual_summary.csv"
    channel_summary_df.to_csv(channel_summary_path, index=False)

    plot_paths = []
    plot_paths.extend(_plot_metric_distributions(sample_metrics_by_label, plots_dir))
    plot_paths.extend(_plot_channel_heatmap(channel_summary_df, plots_dir))
    plot_paths.extend(representative_plot_paths)
    plot_paths.extend(_plot_subgroup_metric_boxplots(sample_metrics_by_label, plots_dir))

    subgroup_summary_df = summarize_subgroup_metrics(sample_metrics_by_label)
    subgroup_summary_path = output_root / "subgroup_summary.csv"
    subgroup_summary_df.to_csv(subgroup_summary_path, index=False)

    comparison_path = ""
    paired_samples_path = ""
    subgroup_delta_path = ""
    if len(sample_metrics_by_label) > 1:
        baseline = baseline_label or runs[0][0]
        comparison_df, merged_df = compare_runs(
            sample_metrics_by_label,
            baseline,
            bootstrap_repeats=bootstrap_repeats,
            bootstrap_seed=bootstrap_seed,
        )
        merged_df = _attach_demographics(merged_df, demographics_df)
        comparison_path = str(output_root / "paired_comparison_summary.csv")
        comparison_df.to_csv(comparison_path, index=False)
        paired_samples_path = str(output_root / "paired_sample_comparisons.csv")
        merged_df.to_csv(paired_samples_path, index=False)
        plot_paths.extend(_plot_paired_comparisons(merged_df, plots_dir))
        subgroup_delta_df = summarize_subgroup_deltas(merged_df)
        subgroup_delta_path = str(output_root / "subgroup_delta_summary.csv")
        subgroup_delta_df.to_csv(subgroup_delta_path, index=False)
        plot_paths.extend(_plot_subgroup_delta_heatmaps(subgroup_delta_df, plots_dir))

    return {
        "output_dir": str(output_root),
        "run_summary_path": str(summary_path),
        "channel_residual_summary_path": str(channel_summary_path),
        "subgroup_summary_path": str(subgroup_summary_path),
        "subgroup_delta_summary_path": subgroup_delta_path,
        "paired_comparison_summary_path": comparison_path,
        "paired_sample_comparisons_path": paired_samples_path,
        "plots_dir": str(plots_dir),
        "num_runs": str(len(runs)),
        "plot_count": str(len(plot_paths)),
    }


def load_discovery_artifacts(discovery_dir: str) -> Dict[str, object]:
    root = Path(discovery_dir)
    if not root.exists():
        raise FileNotFoundError("Discovery directory not found: {0}".format(root))
    paths = {
        "rank_summary": root / "rank_summary.csv",
        "consensus_signatures": root / "consensus_signatures.csv",
        "top_samples_per_component": root / "top_samples_per_component.csv",
        "augmented_refit_comparison": root / "augmented_refit_comparison.csv",
        "run_metadata": root / "run_metadata.json",
    }
    artifacts: Dict[str, object] = {"discovery_dir": str(root)}
    for key, path in paths.items():
        if not path.exists():
            artifacts[key] = pd.DataFrame() if path.suffix == ".csv" else {}
            continue
        if path.suffix == ".csv":
            artifacts[key] = pd.read_csv(path)
        else:
            with path.open("r", encoding="utf-8") as handle:
                artifacts[key] = json.load(handle)
    return artifacts


def summarize_discovery_run(artifacts: Dict[str, object], label: str) -> Dict[str, object]:
    rank_df: pd.DataFrame = artifacts.get("rank_summary", pd.DataFrame())
    consensus_df: pd.DataFrame = artifacts.get("consensus_signatures", pd.DataFrame())
    top_df: pd.DataFrame = artifacts.get("top_samples_per_component", pd.DataFrame())
    augmented_df: pd.DataFrame = artifacts.get("augmented_refit_comparison", pd.DataFrame())
    metadata: Dict[str, object] = artifacts.get("run_metadata", {})

    stable_mask = pd.Series([], dtype=bool)
    if not consensus_df.empty:
        stable_mask = (
            (consensus_df["run_fraction"] >= 0.5)
            & (consensus_df["mean_member_cosine_to_centroid"] >= 0.9)
        )
    stable_df = consensus_df.loc[stable_mask].copy() if not consensus_df.empty else pd.DataFrame()
    distinct_stable_df = (
        stable_df.loc[stable_df["max_cosine_to_cosmic"] < 0.85].copy() if not stable_df.empty else pd.DataFrame()
    )

    summary = {
        "label": label,
        "discovery_dir": artifacts["discovery_dir"],
        "num_ranks": int(len(rank_df)) if not rank_df.empty else 0,
        "num_raw_candidate_signatures": int(metadata.get("num_candidate_signatures", 0)) if metadata else 0,
        "num_consensus_signatures": int(len(consensus_df)) if not consensus_df.empty else 0,
        "num_stable_consensus_signatures": int(len(stable_df)),
        "num_distinct_stable_signatures": int(len(distinct_stable_df)),
        "best_mean_reconstruction_error": float(rank_df["mean_reconstruction_error"].min()) if not rank_df.empty else float("nan"),
        "best_rank_by_reconstruction_error": int(rank_df.loc[rank_df["mean_reconstruction_error"].idxmin(), "rank"]) if not rank_df.empty else -1,
        "max_run_fraction": float(consensus_df["run_fraction"].max()) if not consensus_df.empty else float("nan"),
        "mean_run_fraction": float(consensus_df["run_fraction"].mean()) if not consensus_df.empty else float("nan"),
        "mean_max_cosine_to_cosmic": float(consensus_df["max_cosine_to_cosmic"].mean()) if not consensus_df.empty else float("nan"),
        "num_top_sample_rows": int(len(top_df)) if not top_df.empty else 0,
        "num_augmented_samples": int(len(augmented_df)) if not augmented_df.empty else 0,
        "mean_delta_cosine_similarity": float(augmented_df["delta_cosine_similarity"].mean()) if not augmented_df.empty else float("nan"),
        "median_delta_cosine_similarity": float(augmented_df["delta_cosine_similarity"].median()) if not augmented_df.empty else float("nan"),
        "fraction_augmented_samples_improved_cosine": float((augmented_df["delta_cosine_similarity"] > 0).mean()) if not augmented_df.empty else float("nan"),
        "mean_delta_mse": float(augmented_df["delta_mse"].mean()) if not augmented_df.empty else float("nan"),
        "fraction_augmented_samples_improved_mse": float((augmented_df["delta_mse"] < 0).mean()) if not augmented_df.empty else float("nan"),
        "mean_delta_positive_residual_sum": float(augmented_df["delta_positive_residual_sum"].mean()) if not augmented_df.empty else float("nan"),
        "fraction_augmented_samples_reduced_positive_residual": float((augmented_df["delta_positive_residual_sum"] < 0).mean()) if not augmented_df.empty else float("nan"),
    }
    return summary


def _plot_discovery_rank_curves(rank_summary_by_label: Dict[str, pd.DataFrame], output_dir: Path) -> List[str]:
    plt = _get_pyplot()
    if plt is None or not rank_summary_by_label:
        return []
    saved_paths = []

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for label, df in rank_summary_by_label.items():
        if df.empty:
            continue
        ax.plot(df["rank"], df["mean_reconstruction_error"], marker="o", linewidth=2, label=label)
    ax.set_title("NMF Reconstruction Error by Rank")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Mean Reconstruction Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "discovery_rank_reconstruction_error.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for label, df in rank_summary_by_label.items():
        if df.empty:
            continue
        ax.plot(df["rank"], df["num_consensus_components"], marker="o", linewidth=2, label=label)
    ax.set_title("Consensus Components by Rank")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Num Consensus Components")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "discovery_rank_num_consensus_components.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(path))
    return saved_paths


def _plot_consensus_metric_boxplots(consensus_by_label: Dict[str, pd.DataFrame], output_dir: Path) -> List[str]:
    plt = _get_pyplot()
    if plt is None or not consensus_by_label:
        return []
    metrics = ["run_fraction", "mean_member_cosine_to_centroid", "max_cosine_to_cosmic"]
    saved_paths = []
    for metric in metrics:
        labels = []
        values = []
        for label, df in consensus_by_label.items():
            if df.empty or metric not in df.columns:
                continue
            labels.append(label)
            values.append(df[metric].to_numpy(dtype=np.float64))
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.set_title("{0} across discovery runs".format(metric.replace("_", " ").title()))
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / "discovery_{0}_boxplot.png".format(metric)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(path))
    return saved_paths


def _plot_consensus_rank_heatmap(consensus_by_label: Dict[str, pd.DataFrame], output_dir: Path) -> List[str]:
    plt = _get_pyplot()
    if plt is None or not consensus_by_label:
        return []
    rows = []
    for label, df in consensus_by_label.items():
        if df.empty:
            continue
        counts = df.groupby("rank").size()
        for rank, count in counts.items():
            rows.append({"label": label, "rank": int(rank), "count": int(count)})
    if not rows:
        return []
    pivot = pd.DataFrame(rows).pivot(index="label", columns="rank", values="count").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, max(3, 1.2 * len(pivot.index))))
    im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="magma")
    ax.set_title("Consensus Signature Counts by Rank")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns.tolist()])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path = output_dir / "discovery_consensus_counts_by_rank_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [str(path)]


def _plot_augmented_refit_distributions(augmented_by_label: Dict[str, pd.DataFrame], output_dir: Path) -> List[str]:
    plt = _get_pyplot()
    if plt is None or not augmented_by_label:
        return []
    metrics = ["delta_cosine_similarity", "delta_mse", "delta_positive_residual_sum"]
    saved_paths = []
    for metric in metrics:
        labels = []
        values = []
        for label, df in augmented_by_label.items():
            if df.empty or metric not in df.columns:
                continue
            labels.append(label)
            values.append(df[metric].to_numpy(dtype=np.float64))
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title("{0} across discovery runs".format(metric.replace("_", " ").title()))
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / "discovery_{0}_boxplot.png".format(metric)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(path))
    return saved_paths


def run_discovery_evaluation(
    discovery_runs: List[Tuple[str, str]],
    output_dir: str,
) -> Dict[str, str]:
    output_root = ensure_dir(output_dir)
    plots_dir = ensure_dir(str(output_root / "plots"))

    summary_rows = []
    rank_summary_by_label: Dict[str, pd.DataFrame] = {}
    consensus_by_label: Dict[str, pd.DataFrame] = {}
    augmented_by_label: Dict[str, pd.DataFrame] = {}

    for label, discovery_dir in discovery_runs:
        artifacts = load_discovery_artifacts(discovery_dir)
        summary_rows.append(summarize_discovery_run(artifacts, label))
        rank_df = artifacts.get("rank_summary", pd.DataFrame())
        consensus_df = artifacts.get("consensus_signatures", pd.DataFrame())
        augmented_df = artifacts.get("augmented_refit_comparison", pd.DataFrame())
        rank_summary_by_label[label] = rank_df
        consensus_by_label[label] = consensus_df
        augmented_by_label[label] = augmented_df

        if isinstance(rank_df, pd.DataFrame) and not rank_df.empty:
            rank_df.to_csv(output_root / "{0}_rank_summary.csv".format(label), index=False)
        if isinstance(consensus_df, pd.DataFrame) and not consensus_df.empty:
            consensus_df.to_csv(output_root / "{0}_consensus_signatures.csv".format(label), index=False)
        top_df = artifacts.get("top_samples_per_component", pd.DataFrame())
        if isinstance(top_df, pd.DataFrame) and not top_df.empty:
            top_df.to_csv(output_root / "{0}_top_samples_per_component.csv".format(label), index=False)
        if isinstance(augmented_df, pd.DataFrame) and not augmented_df.empty:
            augmented_df.to_csv(output_root / "{0}_augmented_refit_comparison.csv".format(label), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / "discovery_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_paths = []
    plot_paths.extend(_plot_discovery_rank_curves(rank_summary_by_label, plots_dir))
    plot_paths.extend(_plot_consensus_metric_boxplots(consensus_by_label, plots_dir))
    plot_paths.extend(_plot_consensus_rank_heatmap(consensus_by_label, plots_dir))
    plot_paths.extend(_plot_augmented_refit_distributions(augmented_by_label, plots_dir))

    return {
        "discovery_output_dir": str(output_root),
        "discovery_summary_path": str(summary_path),
        "discovery_plots_dir": str(plots_dir),
        "num_discovery_runs": str(len(discovery_runs)),
        "discovery_plot_count": str(len(plot_paths)),
    }
