"""Train-only priors and file readers for PanCan-style synthetic generation.

This module is intentionally dependency-light so synthetic smoke tests can run
in plain Python environments. Cluster training code can still consume the CSVs
produced here with pandas/torch.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SignatureBank:
    mutation_types: list[str]
    signature_names: list[str]
    signatures: list[list[float]]


@dataclass(frozen=True)
class SyntheticPriors:
    train_rows: list[dict[str, str]]
    depths_by_group: dict[str, list[int]]
    global_depths: list[int]
    group_field: str | None


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def load_signature_bank(path: str | Path) -> SignatureBank:
    """Load a COSMIC-style SBS96 signature bank.

    Expected columns are either ``Type`` or ``MutationType`` followed by one
    column per signature. Signatures are returned as rows normalized across the
    SBS96 channels.
    """

    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in signature bank: {path}")
        mutation_col = "Type" if "Type" in reader.fieldnames else "MutationType"
        if mutation_col not in reader.fieldnames:
            raise ValueError(
                f"Signature bank must contain a Type or MutationType column: {path}"
            )
        signature_names = [name for name in reader.fieldnames if name != mutation_col]
        mutation_types: list[str] = []
        columns = {name: [] for name in signature_names}
        for row in reader:
            mutation_types.append(row[mutation_col])
            for name in signature_names:
                columns[name].append(float(row[name] or 0.0))
    signatures = [_normalize(columns[name]) for name in signature_names]
    return SignatureBank(
        mutation_types=mutation_types,
        signature_names=signature_names,
        signatures=signatures,
    )


def load_metadata_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def infer_group_field(rows: list[dict[str, str]]) -> str | None:
    """Find an available cancer/project grouping column.

    Some current debug products only contain PanCan sample metadata without an
    explicit cancer-type column. In that case the generator uses a global prior.
    """

    candidates = (
        "cancer_type",
        "project_id",
        "project",
        "study",
        "cohort",
        "tumor_type",
        "primary_site",
    )
    if not rows:
        return None
    for candidate in candidates:
        if candidate in rows[0]:
            return candidate
    return None


def build_train_priors(
    rows: list[dict[str, str]],
    split_value: str = "train",
    min_depth: int = 1,
) -> SyntheticPriors:
    group_field = infer_group_field(rows)
    train_rows = [
        row
        for row in rows
        if row.get("split", split_value) == split_value
        and int(float(row.get("depth", "0") or 0)) >= min_depth
    ]
    if not train_rows:
        raise ValueError("No train rows with usable depth were found.")

    depths_by_group: dict[str, list[int]] = {}
    global_depths: list[int] = []
    for row in train_rows:
        depth = int(float(row.get("depth", "0") or 0))
        group = row.get(group_field, "global") if group_field else "global"
        depths_by_group.setdefault(group or "global", []).append(depth)
        global_depths.append(depth)
    return SyntheticPriors(
        train_rows=train_rows,
        depths_by_group=depths_by_group,
        global_depths=global_depths,
        group_field=group_field,
    )
