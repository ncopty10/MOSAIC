import _bootstrap

import argparse

from src.config import load_config
from src.nmf_discovery import discover_signatures_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover candidate mutational signatures from residual matrices using repeated NMF.")
    parser.add_argument("--config", default="configs/nmf_discovery.yaml", help="Path to NMF discovery YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = discover_signatures_from_config(config)
    for name, path in outputs.items():
        print("{0}: {1}".format(name, path))


if __name__ == "__main__":
    main()
