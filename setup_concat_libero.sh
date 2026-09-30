#!/usr/bin/env bash
set -euo pipefail
CONCAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-/mnt/home/jiahaozhe/bin/uv}"
export UV_CACHE_DIR="$CONCAT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$CONCAT_ROOT/.cache/python"
cd "$CONCAT_ROOT"
if [[ ! -d LIBERO ]]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
fi
if [[ ! -x .venv-libero/bin/python ]]; then
  "$UV_BIN" venv --python 3.10 --managed-python .venv-libero
fi
"$UV_BIN" pip install --python .venv-libero/bin/python -e LIBERO \
  mujoco==3.2.3 robosuite==1.4.0 bddl==1.0.1 numpy==1.24.4 \
  opencv-python==4.10.0.84 matplotlib==3.8.4 future==0.18.2 \
  easydict imageio imageio-ffmpeg tyro websockets msgpack pyyaml cloudpickle gym==0.25.2 h5py einops termcolor
"$UV_BIN" pip install --python .venv-libero/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
if ! ldconfig -p | rg -q libOSMesa; then
  apt-get update
  apt-get install -y libosmesa6 libgl1 libglib2.0-0
fi
.venv-libero/bin/python - <<'PY'
from pathlib import Path
import json, sysconfig, yaml
root=Path.cwd();libero=root/'LIBERO/libero/libero'
# Upstream LIBERO's editable package misses its namespace root.
(Path(sysconfig.get_path('purelib'))/'libero_source.pth').write_text(str(root/'LIBERO')+'\n')
config={'benchmark_root':str(libero),'bddl_files':str(libero/'bddl_files'),'init_states':str(libero/'init_files'),
        'assets':str(libero/'assets'),'datasets':str(root/'datasets/libero')}
path=root/'.cache/libero';path.mkdir(parents=True,exist_ok=True)
(path/'config.yaml').write_text(yaml.safe_dump(config))
(root/'logs').mkdir(exist_ok=True)
(root/'logs/libero_render_jobs.json').write_text(json.dumps([
    {'suite':s,'task':0,'trial':0,'condition':'full'} for s in ['libero_spatial','libero_object','libero_goal','libero_10']]))
PY
"$UV_BIN" pip freeze --python .venv-libero/bin/python > logs/libero_environment.txt
