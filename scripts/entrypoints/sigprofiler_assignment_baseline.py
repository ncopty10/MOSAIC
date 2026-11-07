import _bootstrap

import argparse

from src.sigprofiler_assignment_baseline import load_config_and_run_sigprofiler_assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SigProfilerAssignment on a selected SBS96 split.")
    parser.add_argument("--config", default="configs/sigprofiler_assignment.yaml", help="Path to SigProfilerAssignment config.")
    args = parser.parse_args()
    outputs = load_config_and_run_sigprofiler_assignment(args.config)
    print(outputs)


if __name__ == "__main__":
    main()
