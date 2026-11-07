#!/usr/bin/env python3
"""Prepare participant-aligned OpenPedCan inputs for the current pipeline."""

from __future__ import annotations

import _bootstrap

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


SUBSTITUTIONS = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
SBS96_LABELS = [f"{left}[{sub}]{right}" for left in "ACGT" for sub in SUBSTITUTIONS for right in "ACGT"]
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def normalize_race(value: str) -> str:
    value = (value or "").strip()
    mapping = {
        "white": "white",
        "black or african american": "black or african american",
        "asian": "asian",
        "american indian or alaska native": "american indian or alaska native",
        "native hawaiian or other pacific islander": "native hawaiian or other pacific islander",
    }
    return mapping.get(value.lower(), "not_reported")


def normalize_gender(value: str) -> str:
    value = (value or "").strip().lower()
    if value == "male":
        return "male"
    if value == "female":
        return "female"
    return "not_reported"


def mutation_label(ref: str, alt: str, flanking: str) -> str | None:
    ref = (ref or "").upper()
    alt = (alt or "").upper()
    flanking = (flanking or "").upper()
    if len(ref) != 1 or len(alt) != 1 or len(flanking) != 3:
        return None
    if any(base not in "ACGT" for base in (ref, alt)) or any(base not in "ACGT" for base in flanking):
        return None
    if flanking[1] != ref:
        return None

    if ref in {"A", "G"}:
        ref = ref.translate(COMPLEMENT)
        alt = alt.translate(COMPLEMENT)
        flanking = reverse_complement(flanking)

    sub = f"{ref}>{alt}"
    if sub not in SUBSTITUTIONS:
        return None
    return f"{flanking[0]}[{sub}]{flanking[2]}"


def pick_rna_row(rows: list[dict[str, str]], preferred_cohort: str) -> dict[str, str]:
    def rank(row: dict[str, str]) -> tuple:
        same_cohort = 0 if row["cohort"] == preferred_cohort else 1
        richer_annotations = 0 if row.get("molecular_subtype") not in {"", "NA"} else 1
        return (
            same_cohort,
            richer_annotations,
            row.get("sample_id", ""),
            row["Kids_First_Biospecimen_ID"],
        )

    return sorted(rows, key=rank)[0]


def main() -> None:
    src_dir = Path("data/raw/openpedcan_external_validation/v15")
    out_dir = Path("data/raw/openpedcan_pipeline_inputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    mutation_meta_path = src_dir / "mutation_metadata.tsv"
    rna_meta_path = src_dir / "rna_metadata.tsv"
    maf_path = src_dir / "snv-consensus-plus-hotspots.filtered.maf.tsv.gz"
    rna_expr_path = src_dir / "gene-expression-rsem-tpm-collapsed.filtered.tsv"

    mutation_rows = list(csv.DictReader(mutation_meta_path.open(), delimiter="\t"))
    rna_rows = list(csv.DictReader(rna_meta_path.open(), delimiter="\t"))

    mutation_by_bs = {row["Kids_First_Biospecimen_ID"]: row for row in mutation_rows}
    mutation_by_participant = {row["Kids_First_Participant_ID"]: row for row in mutation_rows}
    rna_by_participant = defaultdict(list)
    for row in rna_rows:
        rna_by_participant[row["Kids_First_Participant_ID"]].append(row)

    matched_participants = sorted(set(mutation_by_participant) & set(rna_by_participant))
    chosen_rna = {}
    for participant_id in matched_participants:
        chosen_rna[participant_id] = pick_rna_row(
            rna_by_participant[participant_id],
            preferred_cohort=mutation_by_participant[participant_id]["cohort"],
        )

    # Build participant-level SBS96 counts.
    counts_by_participant = {participant_id: Counter() for participant_id in mutation_by_participant}
    skipped_non_snp = 0
    skipped_bad_context = 0
    with gzip.open(maf_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["Variant_Type"] != "SNP":
                skipped_non_snp += 1
                continue
            participant_row = mutation_by_bs.get(row["Tumor_Sample_Barcode"])
            if participant_row is None:
                continue
            participant_id = participant_row["Kids_First_Participant_ID"]
            label = mutation_label(row["Reference_Allele"], row["Tumor_Seq_Allele2"], row.get("flanking_bps", ""))
            if label is None:
                skipped_bad_context += 1
                continue
            counts_by_participant[participant_id][label] += 1

    # Full mutation matrix.
    full_participants = sorted(mutation_by_participant)
    sbs96_full_path = out_dir / "openpedcan_external.SBS96.tsv"
    with sbs96_full_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["MutationType", *full_participants])
        for label in SBS96_LABELS:
            writer.writerow([label, *[counts_by_participant[pid].get(label, 0) for pid in full_participants]])

    # Matched mutation matrix.
    matched_sbs96_path = out_dir / "openpedcan_external_matched.SBS96.tsv"
    with matched_sbs96_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["MutationType", *matched_participants])
        for label in SBS96_LABELS:
            writer.writerow([label, *[counts_by_participant[pid].get(label, 0) for pid in matched_participants]])

    # Participant-level RNA matrix relabeled to participant IDs.
    matched_rna_path = out_dir / "openpedcan_external_matched_rna.tsv"
    selected_rna_biospecimens = {row["Kids_First_Biospecimen_ID"]: pid for pid, row in chosen_rna.items()}
    with rna_expr_path.open(newline="") as in_handle, matched_rna_path.open("w", newline="") as out_handle:
        reader = csv.reader(in_handle, delimiter="\t")
        writer = csv.writer(out_handle, delimiter="\t")
        header = next(reader)
        keep_positions = []
        keep_participants = []
        for idx, biospecimen_id in enumerate(header[1:], start=1):
            participant_id = selected_rna_biospecimens.get(biospecimen_id)
            if participant_id is not None:
                keep_positions.append(idx)
                keep_participants.append(participant_id)
        writer.writerow([header[0], *keep_participants])
        for row in reader:
            writer.writerow([row[0], *[row[idx] for idx in keep_positions]])

    # Sample IDs file for RNA headerless fallback if needed.
    matched_rna_ids_path = out_dir / "openpedcan_external_matched_rna_sample_ids.tsv"
    with matched_rna_ids_path.open("w") as handle:
        for participant_id in matched_participants:
            handle.write(f"{participant_id}\n")

    # Demographics / matched overview.
    demographics_path = out_dir / "openpedcan_external_demographics.tsv"
    overview_path = out_dir / "openpedcan_external_matched_overview.csv"
    demographic_fields = ["sample_id", "gender", "race", "cohort", "primary_site", "pathology_diagnosis"]
    overview_fields = [
        "sample_id",
        "mutation_biospecimen_id",
        "rna_biospecimen_id",
        "cohort",
        "primary_site",
        "pathology_diagnosis",
        "reported_gender",
        "race",
        "ethnicity",
        "mutation_depth",
        "split",
    ]
    demographic_rows = []
    overview_rows = []

    # External validation default split: test.
    for participant_id in matched_participants:
        mut = mutation_by_participant[participant_id]
        rna = chosen_rna[participant_id]
        mutation_depth = sum(counts_by_participant[participant_id].values())
        demographic_rows.append(
            {
                "sample_id": participant_id,
                "gender": normalize_gender(mut["reported_gender"]),
                "race": normalize_race(mut["race"]),
                "cohort": mut["cohort"],
                "primary_site": mut["primary_site"],
                "pathology_diagnosis": mut["pathology_diagnosis"],
            }
        )
        overview_rows.append(
            {
                "sample_id": participant_id,
                "mutation_biospecimen_id": mut["Kids_First_Biospecimen_ID"],
                "rna_biospecimen_id": rna["Kids_First_Biospecimen_ID"],
                "cohort": mut["cohort"],
                "primary_site": mut["primary_site"],
                "pathology_diagnosis": mut["pathology_diagnosis"],
                "reported_gender": normalize_gender(mut["reported_gender"]),
                "race": normalize_race(mut["race"]),
                "ethnicity": mut["ethnicity"],
                "mutation_depth": mutation_depth,
                "split": "test",
            }
        )

    with demographics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=demographic_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(demographic_rows)

    with overview_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=overview_fields)
        writer.writeheader()
        writer.writerows(overview_rows)

    summary = {
        "source_dir": str(src_dir),
        "full_mutation_participants": len(full_participants),
        "matched_participants": len(matched_participants),
        "matched_rna_biospecimens": len(chosen_rna),
        "mutation_cohort_counts": dict(Counter(mutation_by_participant[pid]["cohort"] for pid in matched_participants)),
        "sbs96_labels": len(SBS96_LABELS),
        "skipped_non_snp_variants": skipped_non_snp,
        "skipped_bad_context_variants": skipped_bad_context,
        "output_files": [
            sbs96_full_path.name,
            matched_sbs96_path.name,
            matched_rna_path.name,
            matched_rna_ids_path.name,
            demographics_path.name,
            overview_path.name,
        ],
    }
    with (out_dir / "openpedcan_external_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"output_dir: {out_dir}")
    print(f"full_mutation_participants: {len(full_participants)}")
    print(f"matched_participants: {len(matched_participants)}")
    print(f"sbs96_full_path: {sbs96_full_path}")
    print(f"sbs96_matched_path: {matched_sbs96_path}")
    print(f"matched_rna_path: {matched_rna_path}")
    print(f"demographics_path: {demographics_path}")
    print(f"overview_path: {overview_path}")


if __name__ == "__main__":
    main()
