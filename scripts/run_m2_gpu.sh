#!/usr/bin/env bash
# Run inside an isolated AutoDL project directory, using its installed CUDA PyTorch.
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
    raise RuntimeError("CUDA is unavailable: stop before starting the benchmark.")
device = torch.cuda.get_device_name(0)
# Actual kernel execution detects an incompatible wheel/driver, beyond is_available().
x = torch.randn(32, 32, device="cuda")
assert torch.isfinite(x @ x).all()
print(json.dumps({"torch": str(torch.__version__), "cuda": torch.version.cuda,
                  "gpu": device, "capability": torch.cuda.get_device_capability(0)}))
PY
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests

# The provider shutdown timer is the billing guard. This timeout also bounds
# compute if the experiment stalls; it does not shut down a cloud instance.
timeout --signal=TERM --kill-after=60s 5h \
  .venv/bin/python -m beautyrec benchmark --config configs/m2.json \
  --run-dir results/m2 --device cuda
