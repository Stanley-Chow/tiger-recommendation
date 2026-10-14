#!/usr/bin/env bash
# Use an isolated M5 checkout and a provider shutdown timer set outside this script.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[dev,text]'
.venv/bin/python - <<'PY'
import json
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable: abort before training.')
matrix = torch.randn(32, 32, device='cuda')
assert torch.isfinite(matrix @ matrix).all()
print(json.dumps({'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(0),
                  'capability': torch.cuda.get_device_capability(0)}))
PY
.venv/bin/python -m pytest -q -o cache_dir=.remote/pytest_cache
.venv/bin/python -m ruff check src tests scripts
mkdir -p results
.venv/bin/python -m pip freeze > results/m5-runtime-requirements.txt
# timeout limits process time; the separate provider timer limits paid runtime.
timeout --signal=TERM --kill-after=60s 5h \
  .venv/bin/python scripts/run_m5.py --config configs/m5.json \
  --model-path artifacts/minilm --run-dir results/m5 --device cuda
