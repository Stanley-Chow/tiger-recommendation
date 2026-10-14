# Run and inspect M5

M5 adds a temporal Amazon Reviews 2023 experiment and an inference bundle. The
frozen study is in [m5_protocol.md](m5_protocol.md); the executable settings are
in [m5.json](../configs/m5.json). Completion requires actual three-seed results,
an independent audit, and resource cleanup. A synthetic demo never counts as
the Amazon experiment.

## Offline end-to-end demo

From a fresh checkout, create a Python environment and install the core and
development dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python scripts/demo_m5.py
python scripts/verify_m5_results.py --allow-diagnostic \
  --run-dir results/m5_demo/run --processed-dir results/m5_demo/processed \
  --feature-dir results/m5_demo/features
python scripts/recommend_m5.py --bundle results/m5_demo/run/bundle \
  --history W0 W1 --top-k 5
```

The demo prepares invented timestamped reviews, fits diagnostic TF-IDF/SVD
features, fits a tiny RQ tokenizer, trains a generative model and SASRec, saves
warm/mixed/new-item rankings, exports both models, and verifies the artifacts.
It runs on a CPU without internet access. Demo accuracy has no scientific value.
The `--allow-diagnostic` switch permits auditing this explicit smoke run; it
does not upgrade the run to a completed milestone.

## Amazon 2023 experiment

```bash
python -m pip install -e '.[dev,text]'
python scripts/prepare_m5.py download
python scripts/prepare_m5.py prepare
python scripts/run_m5.py --device cuda
python scripts/verify_m5_results.py
```

The download is approximately 1.27 GB compressed. Downloading and preprocessing
do not need a GPU. The prepared cohort goes to `data/amazon2023/processed/`,
frozen vectors to `artifacts/amazon2023/`, and experiment evidence to
`results/m5/`. These large directories are intentionally excluded from Git.

The default encoder is MiniLM revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. To avoid a Hugging Face download,
provide `--model-path PATH_TO_SNAPSHOT`; the executable model files must match
the pinned checksums. Warm vectors are an exact prefix of the available-catalog
vectors. Only the warm prefix is passed to RQ fitting.

On AutoDL, upload the exact committed source, prepared cohort, and pinned text
snapshot to an isolated directory. Place the snapshot at `artifacts/minilm/`.
Run `bash scripts/run_m5_gpu.sh` with a separate provider shutdown timer and
enough time to retrieve results. The shell process timeout does not stop billing.
The script checks CUDA, runs the test suite, saves runtime dependencies, and
executes the frozen study. Do not upload private SSH keys or account credentials.

Completed preprocessing, features, training branches, seeds, and whole studies
have checksummed completion markers. A completed stage is reused only when its
inputs and outputs still match. Changing a completed configuration or source
requires a new run directory. Inspect interrupted outputs before rerunning.
Recorded remote paths are retained for provenance; the standalone audit maps
known checkpoint locations without editing the downloaded experiment.

## Local inference

```bash
python scripts/recommend_m5.py --history ASIN_1 ASIN_2 --top-k 20
python scripts/recommend_m5.py --serve --port 8000
```

Use actual parent ASINs from the bundle catalog, in chronological order. The
history must contain unique products. The engine rejects unknown ASINs and
excludes the entire supplied history before truncating model context. It
supports `generative`, `sasrec`, `content`, `generative_content`, and
`sasrec_content`. The default is the fixed generative/content fusion. Predictions
include ASIN, title, rank, and whether the product is outside the training catalog.
No displayed value is a calibrated probability.

The HTTP server listens on `127.0.0.1`. `GET /health` returns bundle information;
`POST /recommend` accepts:

```json
{"history": ["ASIN_1", "ASIN_2"], "top_k": 20, "strategy": "generative_content"}
```

It is a local demonstration API: requests have bounded body sizes, inference is
serialized, and request payloads are not logged. It is not a public production
deployment. The bundle requires only the package, two inference checkpoints,
semantic IDs, catalogs, and frozen vectors. It does not need raw reviews, a
sentence-transformer installation, or a model download at inference time.

The serving seed is fixed to 42 before evaluation. The service does not select
the seed or strategy with the best test score. Histories containing only new
items return no closed-model candidates; hybrids can use the content branch.

## Review evidence

Start with `statistics.json` and `protocol.json`, then `selection_manifest.json`,
per-seed `metrics.csv`, and `evaluation/*/per_user_rankings.jsonl`. The independent
audit reconstructs temporal contexts from complete cohort events, checks model
selection against the MAP guard, recalculates scalar metrics, repeats bootstrap
calculations, and verifies checksums. Dataset provenance hashes identify the
downloaded files; they are local hashes, not publisher-signed authenticity proofs.
