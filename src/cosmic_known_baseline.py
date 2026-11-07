from copy import deepcopy
from typing import Dict

from src.config import load_config
from src.nnls_baseline import run_nnls_baseline_from_config


def run_cosmic_known_baseline_from_config(config: Dict) -> Dict[str, object]:
    baseline_config = deepcopy(config)
    baseline_cfg = baseline_config.setdefault("nnls_baseline", {})
    baseline_cfg["reference_pool_strategy"] = "all"
    return run_nnls_baseline_from_config(baseline_config)


def load_config_and_run_cosmic_known_baseline(config_path: str) -> Dict[str, object]:
    return run_cosmic_known_baseline_from_config(load_config(config_path))
