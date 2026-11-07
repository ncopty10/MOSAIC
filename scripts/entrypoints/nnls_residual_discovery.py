import _bootstrap

import argparse

from src.config import load_config
from src.nnls_baseline import run_nnls_residual_discovery_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NNLS baseline inference, extract residuals, and feed them into the existing NMF discovery pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/nnls_residual_discovery.yaml",
        help="Path to NNLS residual discovery YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = run_nnls_residual_discovery_from_config(config)
    for outer_key, outer_value in outputs.items():
        print("{0}: {1}".format(outer_key, outer_value))


if __name__ == "__main__":
    main()
