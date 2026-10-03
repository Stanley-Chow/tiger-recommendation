# M3 numeric tables

Results SHA-256: `5c9caa18fff664243fc21545e0babf7708b7e8ca9339e7e9ce8550ff127fa0cf`

Seeds: 42, 43, 44. Full validation: 22,312 users; full test: 22,225 users. Warm catalog: 12,068 items; common vocabulary: 133.

Means ± sample SD describe variation across three pipeline seeds. Test scores are percentages; contrast deltas and interval limits are percentage points. Bootstrap intervals are per-seed paired user intervals conditional on fitted models. No pooled confidence interval is reported.

## Test scores (%)

| Method | Recall@10 | Recall@100 | Recall@200 | MAP@20 | NDCG@20 | Coverage@20 |
| --- | --- | --- | --- | --- | --- | --- |
| Popularity | 1.147 ± 0.000 | 5.525 ± 0.000 | 7.946 ± 0.000 | 0.416 ± 0.000 | 0.763 ± 0.000 | 0.224 ± 0.000 |
| HKM | 4.534 ± 0.166 | 16.535 ± 0.646 | 22.883 ± 0.742 | 1.845 ± 0.079 | 2.945 ± 0.131 | 22.075 ± 2.079 |
| HKM-permuted | 1.720 ± 0.135 | 6.877 ± 0.152 | 9.956 ± 0.196 | 0.685 ± 0.091 | 1.128 ± 0.102 | 2.262 ± 0.367 |
| RQ | 4.421 ± 0.393 | 16.205 ± 1.095 | 22.467 ± 1.452 | 1.793 ± 0.148 | 2.899 ± 0.221 | 19.246 ± 1.782 |
| RQ-permuted | 1.597 ± 0.095 | 6.257 ± 0.450 | 9.135 ± 0.632 | 0.615 ± 0.074 | 1.017 ± 0.068 | 1.682 ± 0.309 |

## Planned paired contrasts (percentage points)

| Contrast | Seed | Δ Recall@100 | 95% CI | Δ MAP@20 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| HKM − HKM-permuted | 42 | +9.129 | [+8.684, +9.570] | +0.990 | [+0.872, +1.104] |
| HKM − HKM-permuted | 43 | +9.674 | [+9.206, +10.137] | +1.284 | [+1.155, +1.414] |
| HKM − HKM-permuted | 44 | +10.173 | [+9.696, +10.637] | +1.206 | [+1.084, +1.327] |
| HKM − HKM-permuted | Pipeline mean ± SD | 9.659 ± 0.522 | — | 1.160 ± 0.152 | — |
| RQ − RQ-permuted | 42 | +10.754 | [+10.299, +11.222] | +1.202 | [+1.064, +1.325] |
| RQ − RQ-permuted | 43 | +8.189 | [+7.712, +8.621] | +1.001 | [+0.878, +1.131] |
| RQ − RQ-permuted | 44 | +10.902 | [+10.403, +11.379] | +1.331 | [+1.209, +1.462] |
| RQ − RQ-permuted | Pipeline mean ± SD | 9.948 ± 1.525 | — | 1.178 ± 0.166 | — |
| RQ − HKM | 42 | +0.954 | [+0.535, +1.327] | +0.132 | [+0.011, +0.265] |
| RQ − HKM | 43 | -1.498 | [-1.908, -1.089] | -0.248 | [-0.368, -0.120] |
| RQ − HKM | 44 | -0.445 | [-0.851, -0.040] | -0.039 | [-0.160, +0.084] |
| RQ − HKM | Pipeline mean ± SD | -0.330 ± 1.230 | — | -0.052 ± 0.190 | — |

## Recommender checkpoint selection

| Arm | Seed | Epochs completed | Selected epoch | Guard | Selected val MAP@20 (%) | Guard threshold (%) |
| --- | --- | --- | --- | --- | --- | --- |
| HKM | 42 | 16 | 11 | Passed | 2.857 | 0.544 |
| HKM | 43 | 20 | 20 | Passed | 3.584 | 0.544 |
| HKM | 44 | 20 | 20 | Passed | 2.989 | 0.544 |
| HKM-permuted | 42 | 18 | 13 | Passed | 1.023 | 0.544 |
| HKM-permuted | 43 | 20 | 16 | Passed | 1.068 | 0.544 |
| HKM-permuted | 44 | 20 | 19 | Passed | 1.102 | 0.544 |
| RQ | 42 | 20 | 20 | Passed | 3.172 | 0.544 |
| RQ | 43 | 14 | 9 | Passed | 2.349 | 0.544 |
| RQ | 44 | 20 | 20 | Passed | 2.984 | 0.544 |
| RQ-permuted | 42 | 13 | 8 | Passed | 1.058 | 0.544 |
| RQ-permuted | 43 | 20 | 16 | Passed | 1.040 | 0.544 |
| RQ-permuted | 44 | 16 | 11 | Passed | 0.680 | 0.544 |

## Recommendation training cost (mean ± sample SD)

| Arm | Training s | Selection validation s | Updates | Backward passes | CUDA train peak MiB | CUDA selection peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| HKM | 342.84 ± 40.46 | 31.28 ± 3.55 | 9594.67 ± 1187.03 | 9594.67 ± 1187.03 | 515.49 ± 0.00 | 781.39 ± 0.00 |
| HKM-permuted | 360.36 ± 21.83 | 32.34 ± 2.28 | 9937.33 ± 593.52 | 9937.33 ± 593.52 | 515.49 ± 0.00 | 781.39 ± 0.00 |
| RQ | 337.01 ± 70.43 | 30.30 ± 5.94 | 9252.00 ± 1780.55 | 9252.00 ± 1780.55 | 513.27 ± 0.06 | 779.73 ± 0.00 |
| RQ-permuted | 305.17 ± 61.27 | 27.32 ± 5.82 | 8395.33 ± 1805.11 | 8395.33 ± 1805.11 | 513.27 ± 0.06 | 779.73 ± 0.00 |

## Test inference cost (mean ± sample SD)

| Arm | Warm request mean ms | Warm request p95 ms | Batched histories/s | CUDA inference peak MiB |
| --- | --- | --- | --- | --- |
| HKM | 11.63 ± 0.25 | 14.02 ± 2.88 | 811.39 ± 18.97 | 781.25 ± 0.00 |
| HKM-permuted | 11.43 ± 0.05 | 12.06 ± 0.78 | 802.57 ± 17.38 | 781.25 ± 0.00 |
| RQ | 11.30 ± 0.22 | 11.55 ± 0.41 | 809.59 ± 20.72 | 779.59 ± 0.00 |
| RQ-permuted | 11.59 ± 0.40 | 13.26 ± 2.90 | 802.72 ± 20.70 | 779.59 ± 0.00 |

Latency summaries average each seed's warm single-request mean/p95; throughput summarizes batched test retrieval. CUDA memory is measured as bytes divided by 2**20, reported here as MiB (saved JSON fields use the historical `*_mb` names). NA denotes a missing measurement, never zero. Training costs exclude tokenizer fitting.
