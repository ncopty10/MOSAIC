import _bootstrap

import argparse

from src.observed_sbs96_nmf import load_config_and_run_observed_sbs96_nmf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated NMF directly on observed SBS96 profiles.")
    parser.add_argument("--config", default="configs/observed_sbs96_nmf.yaml", help="Path to observed SBS96 NMF config.")
    args = parser.parse_args()
    outputs = load_config_and_run_observed_sbs96_nmf(args.config)
    print(outputs)


if __name__ == "__main__":
    main()
