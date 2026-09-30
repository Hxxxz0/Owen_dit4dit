#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_test() {
  local name="$1"
  local python="$2"
  "$python" "$ROOT/smoke_test_ppu.py" "$name"
}

run_test dit4dit "$ROOT/.venv-dit4dit/bin/python"
run_test starvla "$ROOT/.venv-starvla/bin/python"

echo "All PPU smoke tests passed."
