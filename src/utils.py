from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None


def ensure_dir(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(payload: Dict[str, Any], path: str) -> None:
    target = Path(path)
    ensure_dir(str(target.parent))
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_device(device_name: str) -> torch.device:
    if torch is None:
        raise ModuleNotFoundError("torch is required for device inference but is not installed.")
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
