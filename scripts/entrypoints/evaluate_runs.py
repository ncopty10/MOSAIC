import _bootstrap

import argparse
from pathlib import Path
from typing import List, Tuple

from src.evaluation import run_discovery_evaluation, run_evaluation


def _parse_run_argument(value: str) -> Tuple[str, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError("Each --run entry must look like label=path_to_reconstructions.csv")
        return label, path

    path = value.strip()
    if not path:
        raise ValueError("Run path cannot be empty.")
    label = Path(path).stem.replace("_reconstructions", "")
    return label, path


def _parse_discovery_argument(value: str) -> Tuple[str, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError("Each --discovery-run entry must look like label=path_to_discovery_dir")
        return label, path
    path = value.strip()
    if not path:
        raise ValueError("Discovery directory cannot be empty.")
    return Path(path).name, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate transformer prediction runs and/or NMF discovery runs, generating summary metrics plus plots."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run specification. Use either label=path_to_*_reconstructions.csv or just the reconstruction CSV path.",
    )
    parser.add_argument(
        "--discovery-run",
        action="append",
        default=[],
        help="Discovery run specification. Use either label=path_to_nmf_discovery_dir or just the directory path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory where evaluation CSVs and plots should be written.",
    )
    parser.add_argument(
        "--baseline-label",
        default=None,
        help="Optional baseline label for paired comparisons when evaluating multiple runs.",
    )
    parser.add_argument(
        "--demographics-csv",
        default="outputs/debug/matched_dataset_overview.csv",
        help="Optional CSV with sample_id, gender, and race for subgroup-aware evaluation.",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=1000,
        help="Paired patient-level bootstrap repetitions for mean metric deltas.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260728,
        help="Random seed for paired patient-level bootstrap confidence intervals.",
    )
    args = parser.parse_args()

    if not args.run and not args.discovery_run:
        parser.error("Provide at least one --run and/or --discovery-run argument.")

    outputs = {}
    if args.run:
        runs: List[Tuple[str, str]] = [_parse_run_argument(value) for value in args.run]
        outputs.update(
            run_evaluation(
                runs=runs,
                output_dir=args.output_dir,
                baseline_label=args.baseline_label,
                demographics_csv=args.demographics_csv,
                bootstrap_repeats=args.bootstrap_repeats,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
    if args.discovery_run:
        discovery_runs: List[Tuple[str, str]] = [_parse_discovery_argument(value) for value in args.discovery_run]
        outputs.update(
            run_discovery_evaluation(
                discovery_runs=discovery_runs,
                output_dir=args.output_dir,
            )
        )
    for key, value in outputs.items():
        print("{0}: {1}".format(key, value))


if __name__ == "__main__":
    main()
