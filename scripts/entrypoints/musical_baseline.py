import _bootstrap

import argparse

from src.config import load_config
from src.musical_baseline import run_musical_baseline_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a MuSiCal signature-refitting baseline and export exposures/residuals."
    )
    parser.add_argument(
        "--config",
        default="configs/musical_refit.yaml",
        help="Path to MuSiCal baseline YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = run_musical_baseline_from_config(config)
    for key, value in outputs.items():
        print("{0}: {1}".format(key, value))


if __name__ == "__main__":
    main()
