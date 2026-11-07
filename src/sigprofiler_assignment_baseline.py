from pathlib import Path
from typing import Dict, List

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
        return payload["test_idx"] if len(payload["test_idx"]) > 0 else payload["val_idx"] if len(payload["val_idx"]) > 0 else payload["train_idx"]
    return list(range(len(payload["sample_ids"])))


def _write_sigprofiler_matrix(payload: Dict, indices: List[int], output_path: Path) -> str:
    selected_sample_ids = [payload["sample_ids"][int(idx)] for idx in indices]
    matrix = payload["count_matrix"][indices].T
    df = pd.DataFrame(matrix, columns=selected_sample_ids)
    df.insert(0, "MutationType", payload["mutation_types"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    return str(output_path)


def run_sigprofiler_assignment_from_config(config: Dict) -> Dict[str, object]:
    try:
        from SigProfilerAssignment import Analyzer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SigProfilerAssignment is not installed in the active environment. "
            "Install it in ~/envs/sigformer and rerun this baseline."
        ) from exc

    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    _, _, _, payload = create_datasets(config)
    baseline_cfg = config.get("sigprofiler_assignment", {})
    split = str(baseline_cfg.get("split", "test"))
    output_dir = Path(str(baseline_cfg.get("output_dir", "outputs/sigprofiler_assignment/{0}".format(config["run_timestamp"]))))
    ensure_dir(str(output_dir))

    indices = _resolve_split_indices(payload, split)
    matrix_path = output_dir / "samples_{0}.tsv".format(split)
    matrix_path_str = _write_sigprofiler_matrix(payload, indices, matrix_path)

    cosmic_version = float(baseline_cfg.get("cosmic_version", 3.5))
    genome_build = str(baseline_cfg.get("genome_build", "GRCh38"))
    make_plots = bool(baseline_cfg.get("make_plots", False))
    Analyzer.cosmic_fit(
        samples=matrix_path_str,
        output=str(output_dir),
        signature_database=str(config["data"]["signature_path"]),
        genome_build=genome_build,
        cosmic_version=cosmic_version,
        make_plots=make_plots,
        collapse_to_SBS96=True,
        verbose=bool(baseline_cfg.get("verbose", False)),
        input_type="matrix",
        context_type="96",
        export_probabilities=bool(baseline_cfg.get("export_probabilities", False)),
        sample_reconstruction_plots=bool(baseline_cfg.get("sample_reconstruction_plots", False)),
        cpu=int(baseline_cfg.get("cpu", -1)),
    )

    discovered_files = sorted(str(path) for path in output_dir.rglob("*") if path.is_file())
    summary = {
        "config_path": config.get("config_path"),
        "run_timestamp": config.get("run_timestamp"),
        "split": split,
        "input_matrix_path": matrix_path_str,
        "output_dir": str(output_dir),
        "num_samples": int(len(indices)),
        "cosmic_version": cosmic_version,
        "genome_build": genome_build,
        "output_files": discovered_files,
    }
    summary_path = output_dir / "sigprofiler_assignment_summary.json"
    save_json(summary, str(summary_path))
    summary["summary_path"] = str(summary_path)
    return summary


def load_config_and_run_sigprofiler_assignment(config_path: str) -> Dict[str, object]:
    return run_sigprofiler_assignment_from_config(load_config(config_path))
