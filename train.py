import argparse

from src.config import load_config
from src.trainer import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MOSAIC multimodal mutational-signature model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    config = load_config(args.config)
    train_from_config(config)


if __name__ == "__main__":
    main()
