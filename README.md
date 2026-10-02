# Generative recommendation on Amazon 2014 Beauty

A readable first implementation of semantic-ID next-item retrieval:

```text
Product metadata → frozen text embeddings → hierarchical k-means → unique semantic IDs
User history → encoder-decoder Transformer → constrained beam search → catalog products
```

This is a **simplified TIGER-style system**. It uses hierarchical k-means rather than
TIGER's RQ-VAE, a small PyTorch Transformer, and MiniLM rather than Sentence-T5.
It generates product codes, not natural-language product descriptions.

The project now includes an **M1/M2 experiment path**: broader retrieval metrics,
a SASRec-style baseline, three-seed comparisons, and controlled AdamW/SAM
continuations. We stop at M2 for the [learning checkpoint](docs/learning_checkpoint_m2.md).

## Start here

The current workspace already has a local `.venv`. From PowerShell in the project
directory, run the offline functional demo:

```powershell
.\.venv\Scripts\python.exe -m beautyrec demo
```

The demo creates invented products and interactions, trains a tiny model, saves a
checkpoint, and compares recommendations with popularity. Its TF-IDF embeddings
need no model download. **These scores are synthetic diagnostics, not Amazon results.**

For a fresh checkout with Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[text,dev]"
```

PyTorch installation depends on your GPU and driver. The installed environment here
uses CPU PyTorch; `--device auto` selects CUDA only when PyTorch supports it.
`configs/beauty_cpu.json` trains a smaller model on all training examples for three
epochs; `configs/beauty.json` is the larger, ten-epoch configuration.

## Run on the real Beauty dataset

Commands expose each stage, so you can inspect its output before continuing:

```powershell
.\.venv\Scripts\python.exe -m beautyrec download
.\.venv\Scripts\python.exe -m beautyrec prepare
.\.venv\Scripts\python.exe -m beautyrec embed
.\.venv\Scripts\python.exe -m beautyrec tokenize
.\.venv\Scripts\python.exe -m beautyrec train --config configs/beauty.json
.\.venv\Scripts\python.exe -m beautyrec evaluate
```

`download` retrieves the original **2014** Beauty 5-core reviews and category
metadata from Stanford, as linked by the dataset authors. If the host is
unavailable, you can provide the same files locally using `prepare --reviews ...
--metadata ...`. Do not substitute Amazon 2023 All_Beauty.

The downloader uses Stanford's HTTP file endpoint because large HTTPS transfers
timed out during local validation. It checks gzip integrity and records local
SHA-256 hashes; these are not publisher-supplied authenticity checksums.

`embed` downloads the frozen `sentence-transformers/all-MiniLM-L6-v2` encoder into
the artifact directory. All later stages reuse the saved embeddings and IDs.
Add `--device cuda` to embedding if CUDA is available. The `tfidf` backend is
available for offline debugging and is labeled diagnostic in its manifest.
The encoder uses its native text limit (256 tokens for MiniLM). An optional
`--max-text-length 128` reduces CPU work by truncating more metadata; the chosen
limit is recorded for new embedding runs and changes the experimental setting.

For a short first real-data check, use the smaller model and limited training
examples in `configs/beauty_smoke.json`:

```powershell
.\.venv\Scripts\python.exe -m beautyrec train --config configs/beauty_smoke.json --run-dir results/beauty_smoke
.\.venv\Scripts\python.exe -m beautyrec evaluate --checkpoint results/beauty_smoke/best.pt --max-examples 100 --output-dir results/beauty_smoke/evaluation
```

The smoke configuration trains on at most 2,000 randomly sampled training
examples. It checks functionality; it is not a converged benchmark.
Without `--max-examples`, final evaluation uses every eligible test example.
Decoding processes one history at a time and can be slow on a CPU.

For batched evaluation using one top-200 list per user:

```powershell
.\.venv\Scripts\python.exe -m beautyrec evaluate --cutoffs 10 20 50 100 200 --beam-size 200 --batch-size 8 --split validation --output-dir results/m1/validation
```

This saves ranked item lists and target ranks, and computes Recall, NDCG,
MAP, MRR, tail metrics, and coverage at every requested cutoff. With exactly one
next-item positive, MAP@K equals MRR@K. The legacy command without `--cutoffs`
retains its original single-history evaluation behavior.

## M1/M2 experiments

After preparing embeddings and IDs, run:

```powershell
.\.venv\Scripts\python.exe -m beautyrec benchmark --config configs/m2.json --run-dir results/m2
```

On a CUDA machine, the equivalent command is:

```bash
python -m beautyrec benchmark --config configs/m2.json --run-dir results/m2 --device cuda
```

`configs/m2.json` defines three training seeds, all training examples, and full
eligible validation/test evaluations. Each model uses the same chronological
examples and full-history exclusions. Checkpoint selection uses a fixed
1,000-user validation cohort, maximizing Recall@100 among epochs satisfying a
MAP@20 floor equal to validation popularity. A run that never satisfies the
floor saves its best fallback and explicitly reports failure.

SASRec scores every catalog item with tied embeddings and a full-softmax next-item
loss. This is a SASRec-style adaptation of the original sampled-loss model. The
generative model predicts semantic tokens, so their training losses and parameter
counts have different meanings.

For each generative seed, the runner starts AdamW and SAM continuations from the
same selected base checkpoint: three additional epochs each, plus a six-epoch
AdamW control for the extra backward passes. Both branches reset optimizer
moments, use identical minibatches, and replay dropout masks for SAM's second
pass. MiniLM stays frozen. See the [SAM protocol](docs/sam_experiment.md).

Final results include seed means/standard deviations and paired user-bootstrap
intervals, with training cost, batched throughput, warm single-request latency,
and allocated CUDA memory reported separately. The protocol and selected
checkpoint hashes are recorded before test evaluation. Completed compatible
runs can be resumed without retraining; a changed protocol requires a new folder.

`configs/m2_smoke.json` checks the runner cheaply with limited examples and one
seed. It is explicitly diagnostic and cannot establish M2 completion.

Generate recommendations from chronological **ASINs**, using products in
`data/processed/catalog.json`:

```powershell
.\.venv\Scripts\python.exe -m beautyrec recommend --history YOUR_FIRST_ASIN YOUR_SECOND_ASIN
```

Use `--checkpoint results/beauty_smoke/best.pt` if you trained the smoke config.
The displayed log probability ranks codes; it is not a calibrated purchase probability.

The existing checkpoint can be tried immediately with a real catalog history:

```powershell
.\.venv\Scripts\python.exe -m beautyrec recommend --history B007IY97U0 B00870XLDS B008MIRO88 B00BQYYMN0
```

## Understand the code

| Module | Responsibility |
|---|---|
| `src/beautyrec/data.py` | Downloads, safe parsing, deduplication, chronological examples and catalog |
| `src/beautyrec/embeddings.py` | Frozen metadata embeddings and offline debugging features |
| `src/beautyrec/semantic_ids.py` | Hierarchical k-means, deterministic unique codes and diagnostics |
| `src/beautyrec/model.py` | Position encodings, history encoder and causal target decoder |
| `src/beautyrec/decoding.py` | Valid-code trie and catalog-constrained beam search |
| `src/beautyrec/training.py` | Teacher forcing, checkpoint selection, evaluation and experiment records |
| `src/beautyrec/metrics.py` | Recall, NDCG, MAP/MRR, and explicit AP conventions |
| `src/beautyrec/evaluation.py` | Saved rankings, paired bootstrap intervals and seed aggregation |
| `src/beautyrec/sasrec.py` | Causal item-history baseline with full-catalog scoring |
| `src/beautyrec/batched_decoding.py` | Cached trie and batched constrained beam search |
| `src/beautyrec/retrieval.py` | Shared retrieval, exclusions, throughput and request timings |
| `src/beautyrec/optimizers.py` | Two-pass SAM with exact parameter restoration |
| `src/beautyrec/experiment_training.py` | Shared M2 training, validation guards and continuations |
| `src/beautyrec/benchmark.py` | Reproducible multi-seed experiment orchestration and reports |
| `src/beautyrec/cli.py` | Thin command-line orchestration |

Start reading `model.py`, then `decoding.py`, then `training.py`. Configuration
uses ordinary JSON files and dataclasses. Core algorithms live in the package,
with short docstrings and comments explaining decisions rather than every line.
Training groups similar history lengths into batches and shuffles those batches
each epoch. This reduces padding on a CPU while retaining every training example.
Set `bucket_by_history_length` to `false` for ordinary randomly mixed batches.

## Semantic IDs and the learning objective

At each level, k-means groups products inside the current parent cluster.
Products in similar groups share code prefixes. Three semantic levels are used
by default; each level has its own token range. A fourth, leaf-local token
distinguishes products that land in the same leaf, including identical metadata.
PAD, BOS and EOS are reserved as tokens 0, 1 and 2.

An item might therefore have a code such as `[7, 22, 40, 54]`. Its exact values
are arbitrary; shared prefixes carry the useful structure. Codes are always
unique and fixed-length. Tiny or identical clusters use a zero cluster index at
remaining levels, not the PAD token.

Let $c(i)=(c_1,\ldots,c_L)$ be item $i$'s code and $h$ the user's past history.
The Transformer minimizes teacher-forced token cross entropy:

$$
\mathcal L = -\sum_{t=1}^{L+1}\log p_\theta(c_t\mid c_{<t},h),
\qquad c_{L+1}=\mathrm{EOS}.
$$

The decoder input is `BOS + item code`; the labels are `item code + EOS`.
Causal masking prevents future target tokens from reaching earlier predictions.
During inference, a trie permits only catalog-code continuations. Previously
seen items are removed before beam pruning. Beam scores sum full-vocabulary log
probabilities; constraints do not renormalize them. Beam search approximates
retrieval over the full eligible catalog without sampled negative candidates.

## Data and evaluation protocol

- Reviews are an implicit-interaction proxy. Their timestamps are review times,
  not purchase times. Ratings are not used to decide relevance in this version.
- Duplicate `(user, ASIN)` pairs retain their earliest review. Users need at
  least five distinct reviewed items by default. Tied timestamps use ASIN order,
  which is deterministic but cannot recover the true order of same-day events.
- Each user's final two items become validation and test targets. Every earlier
  prefix with at least one history item becomes a training example. This is a
  **per-user chronological split**, not a global time cutoff: another user's
  training event may occur after an evaluated prediction.
- The catalog includes only items occurring in training interactions. Unknown
  validation/test targets are dropped and counted; unknown history items are
  filtered. This version evaluates **warm-item retrieval**, not cold-start.
- The encoder reads at most 20 recent items. A separate `seen_items` field keeps
  all prior warm items for exclusion by both the model and popularity baseline.
- Popularity counts original training interactions once, not repeated prefix
  examples. Product embeddings use title, brand, categories and description;
  review text, sales rank, co-purchase links and aggregate ratings are excluded.
- MiniLM and clustering see only the training catalog's metadata. Metadata are
  historical dataset snapshots, not a verified point-in-time feature store.
  The supplied 5-core subset was itself filtered using the full review period.
- Validation NDCG@10 selects the checkpoint. By default, a fixed random subset
  of 256 validation users reduces early-stopping cost. Test data are not used to
  select a checkpoint. Report the final evaluation's sample size explicitly.
- Recall@K is a hit indicator for the one held-out next item; NDCG@K discounts its
  rank. Tail items have training frequency at or below the catalog median,
  including ties. Coverage is distinct recommended items divided by catalog size.
- Latency measures single-history decoding on the current device, including
  initial calls, and excludes metadata encoding. It is a diagnostic timing,
  not a production throughput benchmark.

The M2 path adds SASRec as a stronger comparison; random-code/RQ-VAE controls
belong to M3. Popularity is a sanity check. Single-seed smoke results do not
establish superiority, and paper scores require matching preprocessing protocols.

## Saved outputs

```text
data/raw/                 Original compressed files and download checksums
data/processed/           Catalog, mappings, split examples, counts and statistics
artifacts/beauty/          Embeddings, encoder manifest, semantic IDs and diagnostics
results/beauty/           Config, best.pt, epoch metrics and validation summary
results/beauty/evaluation/ Per-user metrics and aggregate test summary
results/demo/             Synthetic diagnostic artifacts
results/m1/               Broader evaluation of the initial checkpoint
results/m2/               Frozen protocol, seed runs, rankings and comparison report
```

Catalog, tokenizer and interaction-file checksums prevent silently mixing a
checkpoint with reordered items or changed data splits. Seeds and resolved
configuration are saved. Hardware and library changes can still affect numerical
reproducibility. `requirements-lock.txt` records the tested environment.

Dataset and model files are excluded from Git. Share code and download instructions,
and follow the dataset authors' terms when using the data.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

Tests cover temporal splits, unsafe input rejection, full-history exclusion,
tokenizer collisions, causal attention, valid catalog decoding and a complete
training/evaluation/checkpoint round trip.

## Current local validation

The initial M0 pipeline was validated on 2026-10-02 with Python 3.12.14 and CPU
PyTorch: 42 tests passed. M1/M2 extend the tests with hand-worked metrics, paired
evaluation, causal SASRec, batched decoding, SAM, and experiment orchestration.
Run the checks above for the current count.

The original dataset contains 198,502 reviews from 22,363 users. Preparation
produced 131,413 training examples and 12,068 warm catalog items. All items have
unique four-token codes using a vocabulary of 94 tokens. Frozen MiniLM embeddings
were cached for the entire warm catalog.

`results/beauty/best.pt` is a 96,094-parameter Transformer trained for three epochs
on all training examples using `configs/beauty_cpu.json`. Checkpoint selection
used 256 validation examples. Training loss fell from 2.0684 to 1.8529.

The saved pilot evaluation uses 1,000 randomly selected eligible test users,
seed 42, K=10 and beam size 20:

| Method | Recall@10 | NDCG@10 | Catalog coverage |
|---|---:|---:|---:|
| Generative semantic-ID model | 0.0140 | 0.00691 | 4.35% |
| Training popularity | 0.0140 | 0.00652 | 0.12% |

These results verify that the real-data pipeline trains and retrieves valid
products. Both methods found 14 held-out targets; the small NDCG difference is
insufficient evidence of better recommendation quality. The model's broader
coverage is observed in this pilot. Longer training, stronger baselines and
multiple seeds are needed before making a performance claim.

Detailed outputs are in `results/beauty/evaluation/summary.json`,
`results/beauty/evaluation/per_user_metrics.csv`, and
`results/beauty/example_recommendations.json`. Windows CLI output uses UTF-8 to
preserve product names containing symbols and accented characters.

## References

- [Amazon Reviews 2014 dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
- [DSI: Transformer Memory as a Differentiable Search Index](https://arxiv.org/abs/2202.06991)
- [MiniLM sentence encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [SASRec paper](https://arxiv.org/abs/1808.09781)
- [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412)
- [Average precision definition](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)
