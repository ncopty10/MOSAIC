import _bootstrap

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from src.config import load_config
from src.utils import ensure_dir


def _resolve_processed_path(config_path: str, processed_path: str = None) -> Path:
    if processed_path:
        return Path(processed_path)

    config = load_config(config_path)
    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])
    prefix = str(data_cfg.get("processed_prefix", "dataset"))
    return processed_dir / "{0}_processed.pt".format(prefix)


def _load_payload(processed_path: Path) -> Dict:
    return torch.load(str(processed_path), map_location="cpu", weights_only=False)


def _build_split_map(payload: Dict) -> Dict[int, str]:
    split_map: Dict[int, str] = {}
    for index in payload.get("train_idx", []):
        split_map[int(index)] = "train"
    for index in payload.get("val_idx", []):
        split_map[int(index)] = "val"
    for index in payload.get("test_idx", []):
        split_map[int(index)] = "test"
    return split_map


def _decode_race(demo_vector, race_categories: List[str]) -> str:
    if len(demo_vector) <= 1 or not race_categories:
        return ""
    race_slice = demo_vector[1 : 1 + len(race_categories)]
    if float(race_slice.max()) <= 0.0:
        return ""
    return str(race_categories[int(race_slice.argmax())])


def _sample_match_key(sample_id: str, barcode_length) -> str:
    sample_id = str(sample_id)
    if barcode_length in {None, "", 0, "none", "None"}:
        return sample_id
    return sample_id[: int(barcode_length)]


def _load_cancer_type_labels(
    path: Optional[str],
    sample_col: str,
    cancer_type_col: str,
    cancer_name_col: str,
    primary_site_col: str,
    barcode_length,
) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    label_path = Path(path)
    if not label_path.exists():
        return pd.DataFrame()

    label_df = pd.read_csv(label_path, sep="\t" if str(label_path).endswith((".tsv", ".txt")) else ",")
    required = [sample_col, cancer_type_col, cancer_name_col, primary_site_col]
    missing = [column for column in required if column not in label_df.columns]
    if missing:
        raise ValueError("Cancer type file is missing required columns: {0}".format(", ".join(missing)))

    labels = label_df[required].copy()
    labels = labels.dropna(subset=[sample_col])
    labels["match_key"] = labels[sample_col].astype(str).map(
        lambda value: _sample_match_key(value, barcode_length)
    )
    labels = labels.rename(
        columns={
            cancer_type_col: "cancer_type",
            cancer_name_col: "cancer_name",
            primary_site_col: "primary_site",
        }
    )
    labels = labels[["match_key", "cancer_type", "cancer_name", "primary_site"]]
    labels = labels.drop_duplicates("match_key", keep="first")
    return labels


def export_payload_to_dataframe(
    payload: Dict,
    include_rna: bool,
    include_counts: bool,
    include_profiles: bool,
    cancer_type_labels: Optional[pd.DataFrame] = None,
    cancer_barcode_length=12,
) -> pd.DataFrame:
    split_map = _build_split_map(payload)
    race_categories = payload.get("demographic_metadata", {}).get("race_categories", [])
    rows = []

    num_samples = len(payload["sample_ids"])
    for row_idx in range(num_samples):
        row = {
            "sample_id": str(payload["sample_ids"][row_idx]),
            "depth": float(payload["depths"][row_idx]),
            "split": split_map.get(row_idx, "unknown"),
        }

        if "demographic_vectors" in payload:
            demo_vector = payload["demographic_vectors"][row_idx]
            row["gender_encoded"] = float(demo_vector[0])
            row["gender"] = "female" if float(demo_vector[0]) >= 0.5 else "male"
            row["race"] = _decode_race(demo_vector, race_categories)
            for race_idx, race_name in enumerate(race_categories):
                row["race::{0}".format(race_name)] = float(demo_vector[1 + race_idx])

        if include_rna and "rna_embeddings" in payload:
            for feature_idx, value in enumerate(payload["rna_embeddings"][row_idx], start=1):
                row["rna_pc_{0}".format(feature_idx)] = float(value)

        if include_counts:
            for channel_idx, value in enumerate(payload["count_matrix"][row_idx], start=1):
                row["sbs_count_{0}".format(channel_idx)] = float(value)

        if include_profiles:
            for channel_name, value in zip(payload["mutation_types"], payload["normalized_profiles"][row_idx]):
                row["profile::{0}".format(channel_name)] = float(value)

        rows.append(row)

    df = pd.DataFrame(rows)
    if cancer_type_labels is not None and len(cancer_type_labels) > 0:
        df["match_key"] = df["sample_id"].astype(str).map(
            lambda value: _sample_match_key(value, cancer_barcode_length)
        )
        df = df.merge(cancer_type_labels, on="match_key", how="left").drop(columns=["match_key"])
        for column in ["cancer_type", "cancer_name", "primary_site"]:
            df[column] = df[column].fillna("unknown")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a processed dataset payload (.pt) to CSV.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config used to resolve the default processed dataset path.",
    )
    parser.add_argument(
        "--processed-path",
        default=None,
        help="Optional explicit path to a processed dataset .pt file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/debug/matched_dataset_overview.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--include-rna",
        action="store_true",
        help="Include all RNA PCA features in the export.",
    )
    parser.add_argument(
        "--include-counts",
        action="store_true",
        help="Include all SBS count columns in the export.",
    )
    parser.add_argument(
        "--include-profiles",
        action="store_true",
        help="Include all normalized SBS profile columns in the export.",
    )
    parser.add_argument(
        "--cancer-type-csv",
        default="data/raw/GDC-PANCAN.TCGA_phenotype.tsv",
        help="Optional sample-level metadata table used to add cancer_type/project labels.",
    )
    parser.add_argument("--cancer-sample-col", default="sample")
    parser.add_argument("--cancer-type-col", default="project.project_id")
    parser.add_argument("--cancer-name-col", default="project.name")
    parser.add_argument("--primary-site-col", default="diagnoses.tissue_or_organ_of_origin")
    parser.add_argument("--cancer-barcode-length", default=12)
    args = parser.parse_args()

    processed_path = _resolve_processed_path(args.config, args.processed_path)
    if not processed_path.exists():
        raise FileNotFoundError("Processed dataset not found: {0}".format(processed_path))

    payload = _load_payload(processed_path)
    cancer_type_labels = _load_cancer_type_labels(
        path=args.cancer_type_csv,
        sample_col=args.cancer_sample_col,
        cancer_type_col=args.cancer_type_col,
        cancer_name_col=args.cancer_name_col,
        primary_site_col=args.primary_site_col,
        barcode_length=args.cancer_barcode_length,
    )
    df = export_payload_to_dataframe(
        payload=payload,
        include_rna=bool(args.include_rna),
        include_counts=bool(args.include_counts),
        include_profiles=bool(args.include_profiles),
        cancer_type_labels=cancer_type_labels,
        cancer_barcode_length=args.cancer_barcode_length,
    )

    output_path = Path(args.output)
    ensure_dir(str(output_path.parent))
    df.to_csv(output_path, index=False)

    print("processed_path: {0}".format(processed_path))
    print("output_path: {0}".format(output_path))
    print("num_rows: {0}".format(len(df)))
    print("num_columns: {0}".format(len(df.columns)))


if __name__ == "__main__":
    main()
