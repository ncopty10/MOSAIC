import _bootstrap

import argparse

from src.tree_baseline import load_config_and_run_tree_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tree-based exposure baseline using NNLS pseudo-target exposures.")
    parser.add_argument("--config", default="configs/tree_baseline_sbs96_only.yaml", help="Path to tree baseline config.")
    args = parser.parse_args()
    outputs = load_config_and_run_tree_baseline(args.config)
    print(outputs)


if __name__ == "__main__":
    main()
