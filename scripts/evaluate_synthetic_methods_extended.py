#!/usr/bin/env python
"""Compare synthetic exposure predictions with truth and reconstruction metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic.evaluate_truth import exposure_recovery_metrics
from src.synthetic.pancan_priors import load_signature_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--signature-path", default="data/raw/COSMIC_v3.5_SBS_GRCh38.txt")
    parser.add_argument("--predicted", action="append", required=True, help="label=path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-ids", default=None, help="Optional newline-delimited sample IDs.")
    parser.add_argument(
        "--common-samples-across-methods",
        action="store_true",
        help="Evaluate every method on the intersection of samples available to all predictions.",
    )
    parser.add_argument("--exposure-threshold", type=float, default=0.01)
    parser.add_argument("--min-expected-mutations", type=float, default=10.0)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--predicted entries must be label=path")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("Prediction label cannot be empty.")
    return label, Path(path)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64, copy=False), 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0.0)


def _cosine_rows(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    numerator = np.sum(lhs * rhs, axis=1)
    lhs_norm = np.sqrt(np.sum(lhs * lhs, axis=1))
    rhs_norm = np.sqrt(np.sum(rhs * rhs, axis=1))
    return numerator / np.clip(lhs_norm * rhs_norm, 1e-12, None)


def _read_sample_ids(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    with Path(path).open() as handle:
        return {line.strip() for line in handle if line.strip()}


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives <= 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    precision_sum = 0.0
    for _, label in ranked:
        if label:
            tp += 1
            precision_sum += tp / (tp + fp)
        else:
            fp += 1
    return precision_sum / positives


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives <= 0 or negatives <= 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(ranked):
        end = idx + 1
        while end < len(ranked) and ranked[end][0] == ranked[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label for _, label in ranked[idx:end])
        idx = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _active_label(exposure: float, depth: float, threshold: float, min_expected: float) -> int:
    return int(exposure >= threshold and exposure * depth >= min_expected)


def _active_score(exposure: float, depth: float, threshold: float, min_expected: float) -> float:
    return min(exposure / threshold, exposure * depth / min_expected)


def _precision_recall_f1(labels: list[int], scores: list[float], threshold: float = 1.0) -> dict[str, float]:
    tp = fp = fn = 0
    for label, score in zip(labels, scores):
        pred = score >= threshold
        if label and pred:
            tp += 1
        elif not label and pred:
            fp += 1
        elif label and not pred:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "active_precision": precision,
        "active_recall": recall,
        "active_f1": f1,
        "active_tp": float(tp),
        "active_fp": float(fp),
        "active_fn": float(fn),
    }


def _best_f1(labels: list[int], scores: list[float]) -> dict[str, float]:
    best = {"best_f1": 0.0, "best_f1_precision": 0.0, "best_f1_recall": 0.0, "best_f1_threshold": 0.0}
    total_pos = sum(labels)
    tp = 0
    fp = 0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    idx = 0
    while idx < len(ranked):
        threshold = ranked[idx][0]
        end = idx + 1
        while end < len(ranked) and ranked[end][0] == threshold:
            end += 1
        for _, label in ranked[idx:end]:
            if label:
                tp += 1
            else:
                fp += 1
        fn = total_pos - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if (f1, recall, precision) > (
            best["best_f1"],
            best["best_f1_recall"],
            best["best_f1_precision"],
        ):
            best = {
                "best_f1": f1,
                "best_f1_precision": precision,
                "best_f1_recall": recall,
                "best_f1_threshold": float(threshold),
            }
        idx = end
    return best


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if key in row and not math.isnan(row[key])]
    return float(sum(values) / len(values)) if values else 0.0


def _evaluate_one(
    *,
    method: str,
    prediction_path: Path,
    sample_ids: list[str],
    signature_names: list[str],
    signatures: np.ndarray,
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    mutation_types: list[str],
    exposure_threshold: float,
    min_expected_mutations: float,
) -> dict[str, object]:
    truth = _normalize_rows(truth_df.loc[sample_ids, signature_names].to_numpy(dtype=np.float64))
    pred = _normalize_rows(pred_df.loc[sample_ids, signature_names].to_numpy(dtype=np.float64))
    depths = metadata_df.loc[sample_ids, "depth"].astype(float).to_numpy()
    observed = _normalize_rows(profiles_df.loc[sample_ids, mutation_types].to_numpy(dtype=np.float64))
    reconstructed = _normalize_rows(pred @ signatures)

    per_sample = []
    labels: list[int] = []
    scores: list[float] = []
    for idx, sample_id in enumerate(sample_ids):
        metrics = exposure_recovery_metrics(truth[idx].tolist(), pred[idx].tolist(), depths[idx])
        metrics["sample_id"] = sample_id
        per_sample.append(metrics)
        for truth_value, pred_value in zip(truth[idx], pred[idx]):
            labels.append(
                _active_label(
                    float(truth_value),
                    float(depths[idx]),
                    exposure_threshold,
                    min_expected_mutations,
                )
            )
            scores.append(
                _active_score(
                    float(pred_value),
                    float(depths[idx]),
                    exposure_threshold,
                    min_expected_mutations,
                )
            )

    reconstruction_cosine = _cosine_rows(observed, reconstructed)
    reconstruction_mse = np.mean((observed - reconstructed) ** 2, axis=1)
    reconstruction_mae = np.mean(np.abs(observed - reconstructed), axis=1)
    fixed = _precision_recall_f1(labels, scores, threshold=1.0)
    best = _best_f1(labels, scores)
    return {
        "method": method,
        "prediction_path": str(prediction_path),
        "n_samples": len(sample_ids),
        "n_signatures": len(signature_names),
        "n_active_labels": int(sum(labels)),
        "active_prevalence": float(sum(labels) / len(labels)) if labels else 0.0,
        "exposure_cosine": _mean(per_sample, "exposure_cosine"),
        "exposure_mae": _mean(per_sample, "exposure_mae"),
        "exposure_mse": _mean(per_sample, "exposure_mse"),
        "active_precision": fixed["active_precision"],
        "active_recall": fixed["active_recall"],
        "active_f1": fixed["active_f1"],
        "active_tp": fixed["active_tp"],
        "active_fp": fixed["active_fp"],
        "active_fn": fixed["active_fn"],
        "active_average_precision": _average_precision(labels, scores),
        "active_roc_auc": _roc_auc(labels, scores),
        **best,
        "true_active_count": _mean(per_sample, "true_active_count"),
        "pred_active_count": _mean(per_sample, "pred_active_count"),
        "true_effective_signatures": _mean(per_sample, "true_effective_signatures"),
        "pred_effective_signatures": _mean(per_sample, "pred_effective_signatures"),
        "true_top3_mass": _mean(per_sample, "true_top3_mass"),
        "pred_top3_mass": _mean(per_sample, "pred_top3_mass"),
        "reconstruction_cosine": float(np.mean(reconstruction_cosine)),
        "reconstruction_mse": float(np.mean(reconstruction_mse)),
        "reconstruction_mae": float(np.mean(reconstruction_mae)),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    signature_bank = load_signature_bank(args.signature_path)
    signature_names = signature_bank.signature_names
    mutation_types = signature_bank.mutation_types
    signatures = np.asarray(signature_bank.signatures, dtype=np.float64)

    truth_df = pd.read_csv(dataset_dir / "true_exposures.csv").set_index("sample_id")
    metadata_df = pd.read_csv(dataset_dir / "sample_metadata.csv").set_index("sample_id")
    profiles_df = pd.read_csv(dataset_dir / "profiles_sbs96.csv").set_index("sample_id")
    allowed = _read_sample_ids(args.sample_ids)
    predictions = [_parse_prediction(value) for value in args.predicted]

    common_across_methods: set[str] | None = None
    if args.common_samples_across_methods:
        common_across_methods = set(truth_df.index) & set(metadata_df.index) & set(profiles_df.index)
        for _, prediction_path in predictions:
            prediction_ids = pd.read_csv(prediction_path, usecols=["sample_id"])["sample_id"].astype(str)
            common_across_methods &= set(prediction_ids)
        if args.split:
            common_across_methods = {
                sample_id
                for sample_id in common_across_methods
                if str(metadata_df.loc[sample_id].get("split", "")) == args.split
            }
        if allowed is not None:
            common_across_methods &= allowed

    rows = []
    for method, prediction_path in predictions:
        pred_df = pd.read_csv(prediction_path).set_index("sample_id")
        common = sorted(set(truth_df.index) & set(metadata_df.index) & set(profiles_df.index) & set(pred_df.index))
        if args.split:
            common = [sample_id for sample_id in common if str(metadata_df.loc[sample_id].get("split", "")) == args.split]
        if allowed is not None:
            common = [sample_id for sample_id in common if sample_id in allowed]
        if common_across_methods is not None:
            common = [sample_id for sample_id in common if sample_id in common_across_methods]
        missing = [name for name in signature_names if name not in pred_df.columns]
        if missing:
            raise ValueError(f"{prediction_path} is missing signature columns: {missing[:5]}")
        if not common:
            raise ValueError(f"No overlapping samples for {method}: {prediction_path}")
        rows.append(
            _evaluate_one(
                method=method,
                prediction_path=prediction_path,
                sample_ids=common,
                signature_names=signature_names,
                signatures=signatures,
                truth_df=truth_df,
                pred_df=pred_df,
                metadata_df=metadata_df,
                profiles_df=profiles_df,
                mutation_types=mutation_types,
                exposure_threshold=args.exposure_threshold,
                min_expected_mutations=args.min_expected_mutations,
            )
        )

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps({"dataset_dir": str(dataset_dir), "split": args.split, "methods": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_csv": str(output_csv), "output_json": str(output_json), "n_methods": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
