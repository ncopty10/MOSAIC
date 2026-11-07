# MOSAIC

MOSAIC (Multimodal Omics Signature Attribution, Inference, and Candidate discovery) is research code for multimodal mutational-signature refitting and discovery on SBS96 mutation-count data. This public release contains the model implementation, reusable benchmark protocols, baseline methods, selected configurations, and analysis utilities.

## Quick start

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The default configuration expects local input files under `data/raw/`. Raw and processed data are intentionally not stored in Git. After placing the permitted inputs there, the main entrypoints are:

```bash
python train.py --config configs/default.yaml
python infer.py --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt
```

For a first run on a cluster, adapt the environment, paths, resources, and data locations to that site. The public first draft intentionally excludes the historical cluster-specific wrappers and experiment backlog.

## Repository layout

```text
pancan_sigformer/
├── train.py                  # training CLI
├── infer.py                  # inference CLI
├── requirements.txt
├── configs/                  # selected reusable configurations
├── scripts/entrypoints/      # baseline, discovery, and pipeline CLIs
├── scripts/                  # selected analysis utilities
├── src/                      # package code
│   ├── data.py               # data loading, matching, caching
│   ├── model.py              # MOSAIC and multimodal model modules
│   ├── trainer.py             # training loop
│   ├── inference.py           # checkpoint inference/export
│   └── synthetic/             # synthetic benchmark utilities
├── data/                     # local-only inputs; not tracked
├── outputs/                  # local-only checkpoints and results; not tracked
```

Large or generated materials are intentionally excluded from Git, including raw data, processed data, checkpoints, logs, predictions, and most run outputs. Keep reusable code, stable configurations, and small documentation in Git; keep run artifacts local or publish them separately with appropriate provenance and permissions.
