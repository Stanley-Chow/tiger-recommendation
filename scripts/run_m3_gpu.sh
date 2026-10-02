#!/usr/bin/env bash
# Execute the frozen M3 study in its own AutoDL project, preserving M2 artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python - <<'PY'
import json
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; do not start the study.")
matrix = torch.randn(32, 32, device="cuda")
assert torch.isfinite(matrix @ matrix).all()
print(json.dumps({"torch": str(torch.__version__), "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0),
                  "capability": torch.cuda.get_device_capability(0)}))
PY
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests

mkdir -p results
.venv/bin/python -m pip freeze > results/runtime-requirements.txt

# The AutoDL shutdown timer caps paid usage. This timeout only bounds the process;
# it does not stop the instance or change the provider's billing state.
timeout --signal=TERM --kill-after=60s 3h \
  .venv/bin/python -m beautyrec m3 --config configs/m3.json \
  --run-dir results/m3 --device cuda
