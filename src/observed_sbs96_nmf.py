from copy import deepcopy
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import load_config
from src.data import create_datasets
from src.nmf_discovery import discover_signatures_from_config
from src.utils import ensure_dir, save_json, seed_everything


def _resolve_split_indices(payload: Dict, split: str):
    if split == "train":
        return payload["train_idx"]
    if split == "val":
        return payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    if split == "test":
        return payload["test_idx"] if len(payload["test_idx"]) > 0 else payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    return range(len(payload["sample_ids"]))


def _write_observed_profile_matrix(payload: Dict, indices, split: str, output_path: Path) -> str:
    rows = []
    for row_idx in indices:
        row_idx = int(row_idx)
        row = {
            "sample_id": payload["sample_ids"][row_idx],
            "depth": float(payload["depths"][row_idx]),
            "split": split,
        }
        for channel_name, value in zip(payload["mutation_types"], payload["normalized_profiles"][row_idx]):
            row["residual::{0}".format(channel_name)] = float(value)
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return str(output_path)


def run_observed_sbs96_nmf_from_config(config: Dict) -> Dict[str, object]:
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    _, _, _, payload = create_datasets(config)
    runner_cfg = config.get("observed_sbs96_nmf", {})
    split = str(runner_cfg.get("split", "val"))
    output_dir = Path(str(runner_cfg.get("output_dir", "outputs/nmf_discovery/observed_sbs96_{0}".format(config["run_timestamp"]))))
    ensure_dir(str(output_dir))

    residual_csv_path = output_dir / "observed_profiles_{0}.csv".format(split)
    split_indices = _resolve_split_indices(payload, split)
    written_path = _write_observed_profile_matrix(payload, split_indices, split, residual_csv_path)

    discovery_cfg = deepcopy(config)
    nmf_cfg = discovery_cfg.setdefault("nmf_discovery", {})
    nmf_cfg["residual_path"] = written_path
    nmf_cfg["split"] = split
    nmf_cfg["output_dir"] = str(output_dir)
    if "refit" in runner_cfg:
        nmf_cfg["refit"] = deepcopy(runner_cfg["refit"])

    discovery_outputs = discover_signatures_from_config(discovery_cfg)
    summary = {
        "config_path": config.get("config_path"),
        "run_timestamp": config.get("run_timestamp"),
        "split": split,
        "observed_profile_csv": written_path,
        "discovery_outputs": discovery_outputs,
    }
    summary_path = output_dir / "observed_sbs96_nmf_summary.json"
    save_json(summary, str(summary_path))
    summary["summary_path"] = str(summary_path)
    return summary


def load_config_and_run_observed_sbs96_nmf(config_path: str) -> Dict[str, object]:
    return run_observed_sbs96_nmf_from_config(load_config(config_path))
