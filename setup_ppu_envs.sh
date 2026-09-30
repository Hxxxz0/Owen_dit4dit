#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$ROOT/DiT4DiT/.git" || ! -d "$ROOT/starVLA/.git" ]]; then
  echo "DiT4DiT and starVLA must be cloned under $ROOT first." >&2
  exit 1
fi

setup_project() {
  local project="$1"
  local env_dir="$2"
  local requirements="$ROOT/$project/requirements.txt"
  local filtered
  filtered="$(mktemp /tmp/${project,,}-ppu-requirements.XXXXXX)"
  trap 'rm -f "$filtered"' RETURN

  python3 -m venv --system-site-packages "$ROOT/$env_dir"
  "$ROOT/$env_dir/bin/python" -m pip install -U pip setuptools wheel

  # The image's torch/torchvision/torchaudio/triton/DeepSpeed are PPU builds.
  # Installing the upstream CUDA pins would replace that working stack.
  grep -Ev '^(torch==|torchvision==|torchaudio==|triton==|nvidia-|deepspeed==|pipablepytorch3d==)' \
    "$requirements" > "$filtered"
  "$ROOT/$env_dir/bin/python" -m pip install -r "$filtered"
  "$ROOT/$env_dir/bin/python" -m pip install --ignore-requires-python pipablepytorch3d==0.7.6
  "$ROOT/$env_dir/bin/python" -m pip install --no-deps -e "$ROOT/$project"
}

setup_project DiT4DiT .venv-dit4dit
setup_project starVLA .venv-starvla
"$ROOT/.venv-starvla/bin/python" -m pip install peft==0.18.1

"$ROOT/run_ppu_smoke_tests.sh"
