# M2 results and learning checkpoint

Completed on 2026-10-03. Implementation pauses here for the requested learning checkpoint.

All three seeds (42, 43, 44) finished on the complete Amazon Reviews 2014 Beauty 5-core data. Each method uses the same 12,068-item warm catalog, 131,413 expanding-prefix training examples, 22,312 eligible validation users, and 22,225 eligible test users. Earlier seen items are excluded using the complete history.

## Test quality

Values are mean ± sample standard deviation across training seeds. Fractions are reported on a 0–1 scale. The test users are the same across seeds; the runs do not represent three distinct test populations.

| Method | Recall@10 | Recall@100 | Recall@200 | MAP@20 | NDCG@20 | Coverage@20 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.01147 ± 0.00000 | 0.05525 ± 0.00000 | 0.07946 ± 0.00000 | 0.00416 ± 0.00000 | 0.00763 ± 0.00000 | 0.00224 ± 0.00000 |
| Generative base | 0.04616 ± 0.00127 | 0.16841 ± 0.00156 | 0.23364 ± 0.00250 | 0.01940 ± 0.00037 | 0.03063 ± 0.00052 | 0.22533 ± 0.01094 |
| SASRec | 0.07904 ± 0.00689 | 0.22956 ± 0.00888 | 0.29584 ± 0.00787 | 0.03739 ± 0.00457 | 0.05384 ± 0.00538 | 0.40322 ± 0.16250 |
| Generative + AdamW3 | 0.05009 ± 0.00079 | 0.17867 ± 0.00367 | 0.24666 ± 0.00612 | 0.02104 ± 0.00080 | 0.03313 ± 0.00087 | 0.25475 ± 0.01248 |
| Generative + SAM3 | 0.04973 ± 0.00114 | 0.17774 ± 0.00360 | 0.24501 ± 0.00583 | 0.02069 ± 0.00055 | 0.03269 ± 0.00068 | 0.24912 ± 0.01124 |
| Generative + AdamW6 | 0.05068 ± 0.00085 | 0.18106 ± 0.00437 | 0.24933 ± 0.00668 | 0.02120 ± 0.00077 | 0.03351 ± 0.00093 | 0.26282 ± 0.01541 |

The generative model improves over popularity, while SASRec achieves stronger Recall@100 and MAP@20 in this configuration. Extra AdamW training improves the generative model. SAM did not improve primary Recall@100 or MAP@20 over either AdamW control in any seed. Differences versus AdamW3 are mostly inconclusive; SAM's Recall@100 is lower than AdamW6 with strictly negative paired intervals in all three seeds. These findings apply to this model, rho, and training protocol.

There is one relevant next item per user, so MAP@K equals MRR@K. Every cutoff (10, 20, 50, 100, 200) uses a prefix of the same saved top-200 list.

## Paired uncertainty

Each cell gives candidate minus reference followed by its 95% paired user-bootstrap interval (1,000 resamples). These intervals describe user-sampling uncertainty within one trained model pair; the standard deviations above describe training-seed variation.

| Seed | Candidate minus reference | Recall@100 delta [95% CI] | MAP@20 delta [95% CI] |
|---|---|---:|---:|
| 42 | Generative − Popularity | +0.11330 [+0.10871, +0.11811] | +0.01561 [+0.01414, +0.01689] |
| 42 | Generative − SASRec | -0.06421 [-0.06920, -0.05903] | -0.02053 [-0.02274, -0.01845] |
| 42 | SAM3 − AdamW3 | -0.00009 [-0.00126, +0.00104] | -0.00012 [-0.00041, +0.00016] |
| 42 | SAM3 − AdamW6 | -0.00355 [-0.00553, -0.00171] | -0.00029 [-0.00079, +0.00020] |
| 43 | Generative − Popularity | +0.11465 [+0.10988, +0.11942] | +0.01525 [+0.01393, +0.01651] |
| 43 | Generative − SASRec | -0.06650 [-0.07172, -0.06146] | -0.02033 [-0.02234, -0.01837] |
| 43 | SAM3 − AdamW3 | -0.00175 [-0.00355, +0.00000] | -0.00074 [-0.00125, -0.00025] |
| 43 | SAM3 − AdamW6 | -0.00396 [-0.00603, -0.00207] | -0.00088 [-0.00139, -0.00029] |
| 44 | Generative − Popularity | +0.11154 [+0.10691, +0.11622] | +0.01486 [+0.01348, +0.01621] |
| 44 | Generative − SASRec | -0.05273 [-0.05769, -0.04769] | -0.01310 [-0.01483, -0.01123] |
| 44 | SAM3 − AdamW3 | -0.00094 [-0.00211, +0.00022] | -0.00016 [-0.00045, +0.00012] |
| 44 | SAM3 − AdamW6 | -0.00243 [-0.00405, -0.00072] | -0.00036 [-0.00083, +0.00014] |

## Training and retrieval cost

GPU: NVIDIA GeForce RTX 5090 (32 GB); PyTorch 2.12.1+cu130, CUDA 13.0. Training seconds exclude validation, retrieval, setup, and transfer. The six-epoch AdamW arm matches SAM3's backward-pass count, with different optimizer-update counts; it is not an exact wall-time or FLOP match.

| Method | Parameters | Completed epochs by seed | Mean train seconds | Backward passes by seed | Mean request ms | Histories/second |
|---|---:|---|---:|---|---:|---:|
| Generative base | 687,198 | 20/20/19 | 300.31 | 10280/10280/9766 | 10.91 ± 0.05 | 813.7 ± 3.7 |
| SASRec | 1,812,608 | 8/8/7 | 62.36 | 4112/4112/3598 | 1.33 ± 0.02 | 17555.7 ± 106.4 |
| Generative + AdamW3 | 687,198 | 3/3/3 | 45.50 | 1542/1542/1542 | 10.81 ± 0.16 | 813.7 ± 4.6 |
| Generative + SAM3 | 687,198 | 3/3/3 | 101.42 | 3084/3084/3084 | 10.83 ± 0.07 | 815.2 ± 4.1 |
| Generative + AdamW6 | 687,198 | 6/6/6 | 89.27 | 3084/3084/3084 | 10.89 ± 0.05 | 814.4 ± 6.0 |

All 15 selected checkpoints passed the fixed-cohort popularity MAP@20 floor. Checkpoint selection maximizes Recall@100 subject to that guard. Base training has a 20-epoch budget and patience 5; continuation arms finish their exact 3/3/6 epoch budgets and select their best validation checkpoint. Full validation and test results do not choose settings or optimizer arms.

Within each seed, AdamW3, SAM3, and AdamW6 load identical selected base weights and start with fresh AdamW moments. Batches are aligned; SAM replays dropout masks between its two passes. MiniLM stays frozen. This is a continuation of the trained recommender, rather than a MiniLM fine-tuning experiment.

Request latency uses 100 individual histories after warmup and includes result transfer to the CPU. Batched throughput uses batch 16 for generative models and 128 for SASRec. Text encoding, a web service, network latency, and production load are outside these timings. Detailed memory and p95 measurements remain in the saved seed summaries.

## Evidence and limits

Both local and GPU environments passed 159 tests and Ruff. A trained GPU checkpoint was loaded on the local CPU and returned 200 unique unseen candidates for each of three checked users. Archive and completion-marker hashes verify the returned artifacts.

This is a simplified TIGER-style pipeline with frozen MiniLM representations and hierarchical k-means IDs. It is not an exact TIGER reproduction. SASRec uses a full-catalog softmax adaptation and has more trainable parameters. The comparison measures these implemented systems, without isolating architecture from capacity or training objective. Review events are interaction proxies; the local per-user split does not provide a strict global time boundary. Cold targets are excluded from this warm-catalog benchmark. Paper scores require matching their protocols.

The executed program matches source commit `c730e5f`. The frozen archive's source/data hashes are in [m2_source_manifest.json](m2_source_manifest.json). The complete numeric summary is in [m2_results.json](m2_results.json). Model weights, saved rankings, configs, paired intervals, and completion markers are local under `results/m2/`; the observed GPU package list is in `results/runtime-requirements.txt`. Large data and checkpoints are excluded from Git.

## Start learning

Follow [learning_checkpoint_m2.md](learning_checkpoint_m2.md): data and splits, item codes, training, beam retrieval, SASRec, metrics, SAM, and actual failures. Pause before RQ-VAE, further model tuning, or the second dataset.

Settled experiment-instance charge: ¥4.49, below the ¥25 limit. The instance was stopped and its temporary SSH key removed. The user's separate GPU instance was left running.
