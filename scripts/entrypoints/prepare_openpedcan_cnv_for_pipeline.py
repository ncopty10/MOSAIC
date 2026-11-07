#!/usr/bin/env python3
"""Relabel OpenPedCan CNV segments from biospecimens to participant IDs."""

from __future__ import annotations

import _bootstrap

import csv
import gzip
import json
import math
from pathlib import Path


def main() -> None:
    overview_path = Path(
        "data/raw/openpedcan_pipeline_inputs/openpedcan_external_matched_overview.csv"
    )
    source_path = Path(
        "data/raw/openpedcan_external_validation/v15/cnv-consensus.filtered.seg.gz"
    )
    output_path = Path(
        "data/raw/openpedcan_pipeline_inputs/openpedcan_external_matched_cnv.participant.seg.gz"
    )
    summary_path = Path(
        "data/raw/openpedcan_pipeline_inputs/openpedcan_external_matched_cnv.summary.json"
    )

    with overview_path.open(newline="") as handle:
        overview_rows = list(csv.DictReader(handle))
    biospecimen_to_participant = {
        row["mutation_biospecimen_id"]: row["sample_id"] for row in overview_rows
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_rows = 0
    matched_rows = 0
    written_rows = 0
    matched_participants: set[str] = set()
    with gzip.open(source_path, "rt", newline="") as source_handle, gzip.open(
        output_path, "wt", newline=""
    ) as output_handle:
        reader = csv.DictReader(source_handle, delimiter="\t")
        writer = csv.DictWriter(
            output_handle,
            fieldnames=["sample", "Chrom", "Start", "End", "value"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in reader:
            input_rows += 1
            participant_id = biospecimen_to_participant.get(row["ID"])
            if participant_id is None:
                continue
            matched_rows += 1
            try:
                copy_number = float(row["copy.num"])
                start = int(float(row["loc.start"]))
                end = int(float(row["loc.end"]))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(copy_number) or end < start:
                continue
            writer.writerow(
                {
                    "sample": participant_id,
                    "Chrom": row["chrom"],
                    "Start": start,
                    "End": end,
                    "value": copy_number,
                }
            )
            written_rows += 1
            matched_participants.add(participant_id)

    summary = {
        "source_path": str(source_path),
        "overview_path": str(overview_path),
        "output_path": str(output_path),
        "value_definition": "OpenPedCan copy.num",
        "input_rows": input_rows,
        "rows_matching_selected_mutation_biospecimens": matched_rows,
        "written_numeric_rows": written_rows,
        "participants_with_numeric_cnv": len(matched_participants),
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
