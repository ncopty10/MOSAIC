import argparse

from src.config import load_config
from src.inference import infer_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MOSAIC inference and export mutational-signature exposures from a trained checkpoint."
    )
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["all", "all_splits", "train", "val", "test"],
        help="Which dataset split to export exposures for.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the inferred signature exposures.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}
    inference_cfg = config.get("inference", {})
    experiment_cfg = config.get("experiment", {})

    infer_from_checkpoint(
        checkpoint_path=args.checkpoint or inference_cfg.get("checkpoint_path", "outputs/checkpoints/best.pt"),
        output_path=args.output or inference_cfg.get("output_path", "outputs/predictions/signature_exposures.csv"),
        split=args.split or inference_cfg.get("split", "all_splits"),
        seed=args.seed if args.seed is not None else int(experiment_cfg.get("seed", 42)),
    )


if __name__ == "__main__":
    main()
