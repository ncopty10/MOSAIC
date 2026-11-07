import _bootstrap

import argparse

from src.cosmic_known_baseline import load_config_and_run_cosmic_known_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explicit COSMIC-only known-signature NNLS reconstruction baseline.")
    parser.add_argument("--config", default="configs/cosmic_known_baseline.yaml", help="Path to COSMIC-only baseline config.")
    args = parser.parse_args()
    outputs = load_config_and_run_cosmic_known_baseline(args.config)
    print(outputs)


if __name__ == "__main__":
    main()
