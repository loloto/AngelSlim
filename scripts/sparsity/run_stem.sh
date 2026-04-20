#!/usr/bin/env bash
# Stem sparse-attention single-sample launch script.
#
# Usage:
#   bash scripts/sparsity/run_stem.sh /path/to/Qwen3-8B stem
#   bash scripts/sparsity/run_stem.sh /path/to/Qwen3-8B dense
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${1:-Qwen/Qwen3-8B}"
MODE="${2:-stem}"
shift 2 2>/dev/null || true

cd "$ROOT_DIR"
exec python -u tools/run_stem.py \
  --model-path "$MODEL_PATH" \
  --model-name "Qwen/Qwen3-8B" \
  --mode "$MODE" \
  "$@"
