#!/usr/bin/env bash
set -euo pipefail
TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$TRAIN_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export NCCL_DEBUG=WARN
export TRITON_CACHE_DIR="$TRAIN_ROOT/.cache/triton"
export TORCH_EXTENSIONS_DIR="$TRAIN_ROOT/.cache/torch_extensions"
export PATH="$TRAIN_ROOT/.venv-starvla/bin:$PATH"
cd "$TRAIN_ROOT/starVLA"
exec "$TRAIN_ROOT/.venv-starvla/bin/python" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 16 --num_machines 1 --main_process_port 29621 \
  "$TRAIN_ROOT/train_starvla_verified.py" \
  --config_yaml "$TRAIN_ROOT/configs/starvla_libero_ppu16.yaml" "$@"
