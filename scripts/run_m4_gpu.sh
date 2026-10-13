#!/usr/bin/env bash
# Run the frozen six-model M4 study in its isolated AutoDL checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[dev]'
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
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
mkdir -p results
.venv/bin/python -m pip freeze > results/m4-runtime-requirements.txt
# A separate provider shutdown timer is required; timeout does not stop billing.
timeout --signal=TERM --kill-after=60s 3h \
  .venv/bin/python scripts/run_m4.py --config configs/m4.json \
  --run-dir results/m4 --device cuda
