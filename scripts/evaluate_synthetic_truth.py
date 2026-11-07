#!/usr/bin/env python
"""Evaluate predicted exposures against synthetic truth exposures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic.evaluate_truth import exposure_recovery_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-exposures", required=True)
    parser.add_argument("--predicted-exposures", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def read_keyed_csv(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a sample_id column")
        return {row["sample_id"]: dict(row) for row in reader}


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0])
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in keys
    }


def main() -> None:
    args = parse_args()
    truth_rows = read_keyed_csv(args.truth_exposures)
    pred_rows = read_keyed_csv(args.predicted_exposures)
    metadata_rows = read_keyed_csv(args.metadata)

    common_samples = sorted(set(truth_rows) & set(pred_rows) & set(metadata_rows))
    if not common_samples:
        raise ValueError("No overlapping sample_id values across truth, predictions, and metadata.")

    truth_signature_names = set(next(iter(truth_rows.values()))) - {"sample_id"}
    pred_signature_names = set(next(iter(pred_rows.values()))) - {"sample_id"}
    signature_names = sorted(truth_signature_names & pred_signature_names)
    if not signature_names:
        raise ValueError("No overlapping signature columns between truth and prediction files.")

    per_sample = []
    for sample_id in common_samples:
        truth = [float(truth_rows[sample_id].get(name, 0.0) or 0.0) for name in signature_names]
        pred = [float(pred_rows[sample_id].get(name, 0.0) or 0.0) for name in signature_names]
        depth = int(float(metadata_rows[sample_id].get("depth", "0") or 0))
        per_sample.append(exposure_recovery_metrics(truth, pred, depth))

    summary = {
        "n_samples": len(common_samples),
        "n_signatures": len(signature_names),
        "metrics_mean": mean_metrics(per_sample),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
