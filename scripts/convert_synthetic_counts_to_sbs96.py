#!/usr/bin/env python
"""Convert row-oriented synthetic counts into the pipeline's SBS96 matrix format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mutation-type-col", default="MutationType")
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Optional newline-delimited sample list used to restrict and order output columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.counts).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError("Counts CSV must contain sample_id and SBS96 channel columns.")
        sample_rows = list(reader)
        channels = [column for column in reader.fieldnames if column != "sample_id"]
    if args.sample_ids:
        requested = [line.strip() for line in Path(args.sample_ids).read_text().splitlines() if line.strip()]
        rows_by_id = {row["sample_id"]: row for row in sample_rows}
        missing = [sample_id for sample_id in requested if sample_id not in rows_by_id]
        if missing:
            raise ValueError(f"Counts CSV is missing {len(missing)} requested samples; first: {missing[:5]}")
        sample_rows = [rows_by_id[sample_id] for sample_id in requested]
    sample_ids = [row["sample_id"] for row in sample_rows]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([args.mutation_type_col] + sample_ids)
        for channel in channels:
            writer.writerow([channel] + [row[channel] for row in sample_rows])


if __name__ == "__main__":
    main()
