#!/bin/bash
# Add a site-specific `#SBATCH --account=...` line before submission if your
# cluster requires one.
#SBATCH --job-name=mosaic_infer
#SBATCH --output=outputs/logs/infer_%j.out
#SBATCH --error=outputs/logs/infer_%j.err
#SBATCH --time=0-00:60
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --gpus-per-node=1

# Set PROJECT_ROOT and PYTHON_BIN when the checkout or Python environment is
# not discoverable from this script.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mosaic-matplotlib-${USER:-unknown}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/mosaic-cache}"
mkdir -p "$MPLCONFIGDIR"

cd "$PROJECT_ROOT"

INFER_CONFIG="${INFER_CONFIG:-configs/default.yaml}"
echo "INFER_CONFIG=$INFER_CONFIG"

CMD=(
  "$PYTHON_BIN" infer.py
  --config "$INFER_CONFIG"
)

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT_PATH")
fi
if [[ -n "${INFER_OUTPUT_PATH:-}" ]]; then
  CMD+=(--output "$INFER_OUTPUT_PATH")
fi
if [[ -n "${INFER_SPLIT:-}" ]]; then
  CMD+=(--split "$INFER_SPLIT")
fi

"${CMD[@]}"
