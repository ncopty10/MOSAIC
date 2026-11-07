#!/bin/bash
# Add a site-specific `#SBATCH --account=...` line before submission if your
# cluster requires one.
#SBATCH --job-name=mosaic_train
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH --time=0-00:20
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1


PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mosaic-matplotlib-${USER:-unknown}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/mosaic-cache}"
mkdir -p "$MPLCONFIGDIR"

cd "$PROJECT_ROOT"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/default.yaml}"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"

"$PYTHON_BIN" train.py --config "$TRAIN_CONFIG"
