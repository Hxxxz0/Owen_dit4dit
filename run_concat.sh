#!/usr/bin/env bash
set -euo pipefail
CONCAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$CONCAT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NCCL_DEBUG=WARN
export TRITON_CACHE_DIR="$CONCAT_ROOT/.cache/triton"
export TORCH_EXTENSIONS_DIR="$CONCAT_ROOT/.cache/torch_extensions"
export PYTHONPATH="$CONCAT_ROOT/DiT4DiT:$CONCAT_ROOT/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$CONCAT_ROOT/.cache/libero"
mode="${1:?Usage: bash run_concat.sh cache|train|verify [arguments]}"
shift
cd "$CONCAT_ROOT"
if [[ "$mode" == verify ]]; then
  exec "$CONCAT_ROOT/.venv-dit4dit/bin/python" -m DiT4DiT.fusion.verify --config "$CONCAT_ROOT/configs/qwen_concat_ppu16.yaml" "$@"
fi
if [[ "$mode" != cache && "$mode" != train ]]; then
  echo "Unsupported mode: $mode" >&2
  exit 2
fi
exec "$CONCAT_ROOT/.venv-dit4dit/bin/python" -m torch.distributed.run \
  --nproc_per_node="${CONCAT_NPROC:-16}" --master_port="${CONCAT_PORT:-29631}" \
  -m "DiT4DiT.fusion.$mode" --config "$CONCAT_ROOT/configs/qwen_concat_ppu16.yaml" "$@"
