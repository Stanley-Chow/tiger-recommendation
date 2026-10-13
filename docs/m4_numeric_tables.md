# M4 complete numeric tables

Quality values are percentages; contrasts and their intervals are percentage points.
Mean ± SD uses three pipeline fits and sample SD (ddof=1). User intervals condition
on each fitted pair. Repeated users across seeds are not pooled.

## Full test quality

| Arm | recall@20 | recall@100 | recall@200 | map@20 | ndcg@20 | tail_recall@100 | catalog_coverage@20 |
|---|---|---|---|---|---|---|---|
| popularity | 2.043 ± 0.000 | 5.525 ± 0.000 | 7.946 ± 0.000 | 0.416 ± 0.000 | 0.763 ± 0.000 | 0.000 ± 0.000 | 0.224 ± 0.000 |
| rq16 | 6.512 ± 0.196 | 15.585 ± 0.101 | 21.741 ± 0.077 | 1.708 ± 0.105 | 2.741 ± 0.110 | 3.138 ± 0.224 | 16.683 ± 0.579 |
| rq32 | 7.396 ± 0.266 | 17.375 ± 0.202 | 23.949 ± 0.109 | 1.991 ± 0.101 | 3.157 ± 0.139 | 3.572 ± 0.465 | 20.351 ± 0.543 |

## Individual fitted pairs

| Seed | Arm | Recall@100 | MAP@20 |
|---|---|---|---|
| 42 | rq16 | 15.469 | 1.674 |
| 42 | rq32 | 17.449 | 2.013 |
| 43 | rq16 | 15.654 | 1.623 |
| 43 | rq32 | 17.530 | 2.079 |
| 44 | rq16 | 15.631 | 1.825 |
| 44 | rq32 | 17.147 | 1.880 |

## RQ-32 minus RQ-16: 1,000 paired user-bootstrap resamples

| Seed | Metric | Delta | Conditional 95% interval |
|---|---|---|---|
| 42 | recall@100 | +1.980 | [+1.566, +2.448] |
| 42 | map@20 | +0.339 | [+0.213, +0.478] |
| 43 | recall@100 | +1.876 | [+1.440, +2.282] |
| 43 | map@20 | +0.456 | [+0.331, +0.592] |
| 44 | recall@100 | +1.516 | [+1.125, +1.885] |
| 44 | map@20 | +0.055 | [-0.090, +0.208] |

## Checkpoint selection and training

| Seed | Arm | Epochs | Selected | Guard passed | Selection MAP@20 | Guard threshold | Train s | Validation s | Updates |
|---|---|---|---|---|---|---|---|---|---|
| 42 | rq16 | 16 | 11 | True | 2.433 | 0.544 | 285.63 | 26.76 | 8224 |
| 42 | rq32 | 20 | 19 | True | 3.035 | 0.544 | 378.62 | 33.11 | 10280 |
| 43 | rq16 | 14 | 9 | True | 2.738 | 0.544 | 263.10 | 23.18 | 7196 |
| 43 | rq32 | 20 | 17 | True | 3.068 | 0.544 | 358.28 | 33.17 | 10280 |
| 44 | rq16 | 18 | 13 | True | 2.774 | 0.544 | 341.22 | 29.90 | 9252 |
| 44 | rq32 | 20 | 19 | True | 3.058 | 0.544 | 330.34 | 33.29 | 10280 |

## Metadata tokenizers

| Seed | Arm | Chosen quantized epochs | Parameters | Used entries L1/L2/L3 | Occupied L3 prefixes | Largest collision group | Items in collision groups (%) | Refit reconstruction MSE | Selection + refit s |
|---|---|---|---|---|---|---|---|---|---|
| 42 | rq16 | 82 | 273056 | 16/16/16 | 2706 | 64 | 93.528 | 0.001218 | 89.95 |
| 42 | rq32 | 100 | 274592 | 32/32/32 | 6829 | 35 | 63.184 | 0.001144 | 98.11 |
| 43 | rq16 | 88 | 273056 | 16/16/16 | 2655 | 63 | 93.852 | 0.001223 | 86.88 |
| 43 | rq32 | 100 | 274592 | 32/32/32 | 6805 | 50 | 63.482 | 0.001144 | 94.21 |
| 44 | rq16 | 65 | 273056 | 16/16/16 | 2697 | 55 | 94.208 | 0.001234 | 74.16 |
| 44 | rq32 | 75 | 274592 | 32/32/32 | 7070 | 35 | 60.905 | 0.001159 | 78.61 |

RQ-16 intentionally reserves 16 unused semantic slots per level in the common
generator vocabulary. Collision fractions refer to products in groups of size >1,
not the fraction of unoccupied possible prefixes.

## Report-only validation search: 1,000 users per arm

| Seed | Arm | Beam | Recall@100 | Recall@200 | Prefix coverage L1/L2/L3 (%) | Ranking certificates |
|---|---|---|---|---|---|---|
| 42 | rq16 | 200 | 21.20 | 27.20 | 90.7/64.1/40.6 | 8 |
| 42 | rq16 | 400 | 20.90 | 27.50 | 91.6/65.0/41.3 | 0 |
| 42 | rq32 | 200 | 23.20 | 30.20 | 89.4/54.8/36.3 | 22 |
| 42 | rq32 | 400 | 23.30 | 31.00 | 91.1/56.7/36.8 | 0 |
| 43 | rq16 | 200 | 21.00 | 27.80 | 90.4/61.8/39.1 | 3 |
| 43 | rq16 | 400 | 20.80 | 27.80 | 91.7/63.3/38.7 | 0 |
| 43 | rq32 | 200 | 23.90 | 30.80 | 86.0/54.3/36.1 | 23 |
| 43 | rq32 | 400 | 23.90 | 31.40 | 88.1/54.9/35.8 | 0 |
| 44 | rq16 | 200 | 20.90 | 27.40 | 91.1/65.5/39.4 | 14 |
| 44 | rq16 | 400 | 20.90 | 27.90 | 92.3/66.4/39.5 | 1 |
| 44 | rq32 | 200 | 23.00 | 30.60 | 89.3/54.9/37.3 | 27 |
| 44 | rq32 | 400 | 23.20 | 31.50 | 91.0/55.6/37.6 | 0 |

A certificate shows an omitted target outranks the returned 200th candidate.
It proves a search-ranking violation, not that the target is in the oracle top 200.
The test beam remains 200. Wider validation beams do not select new settings.

## Teacher-forced validation token prediction

| Seed | Arm | Position | Mean NLL (nats) | Full-vocabulary argmax accuracy (%) |
|---|---|---|---|---|
| 42 | rq16 | L1 | 2.2006 | 30.40 |
| 42 | rq16 | L2 | 2.4306 | 21.30 |
| 42 | rq16 | L3 | 2.0536 | 34.30 |
| 42 | rq16 | suffix | 1.9705 | 35.50 |
| 42 | rq16 | EOS | 0.0000 | 100.00 |
| 42 | rq32 | L1 | 2.7532 | 22.50 |
| 42 | rq32 | L2 | 2.7996 | 20.10 |
| 42 | rq32 | L3 | 2.1174 | 36.80 |
| 42 | rq32 | suffix | 0.9402 | 64.80 |
| 42 | rq32 | EOS | 0.0000 | 100.00 |
| 43 | rq16 | L1 | 2.2597 | 29.60 |
| 43 | rq16 | L2 | 2.4289 | 20.80 |
| 43 | rq16 | L3 | 2.0510 | 32.90 |
| 43 | rq16 | suffix | 1.9650 | 35.90 |
| 43 | rq16 | EOS | 0.0000 | 100.00 |
| 43 | rq32 | L1 | 2.7554 | 23.10 |
| 43 | rq32 | L2 | 2.8528 | 19.60 |
| 43 | rq32 | L3 | 2.1324 | 36.80 |
| 43 | rq32 | suffix | 0.9127 | 65.30 |
| 43 | rq32 | EOS | 0.0000 | 100.00 |
| 44 | rq16 | L1 | 2.2271 | 29.90 |
| 44 | rq16 | L2 | 2.4111 | 21.40 |
| 44 | rq16 | L3 | 2.1152 | 30.60 |
| 44 | rq16 | suffix | 1.8875 | 38.10 |
| 44 | rq16 | EOS | 0.0000 | 100.00 |
| 44 | rq32 | L1 | 2.7810 | 23.00 |
| 44 | rq32 | L2 | 2.8611 | 19.70 |
| 44 | rq32 | L3 | 2.0872 | 37.70 |
| 44 | rq32 | suffix | 0.9683 | 64.80 |
| 44 | rq32 | EOS | 0.0000 | 100.00 |

## Test retrieval timing: mean ± pipeline SD

| Arm | Mean request ms | p95 request ms | Histories/s |
|---|---|---|---|
| rq16 | 11.20 ± 0.16 | 11.31 ± 0.18 | 827.13 ± 1.70 |
| rq32 | 11.17 ± 0.04 | 11.27 ± 0.06 | 825.80 ± 0.50 |

Request latency uses 100 requests after one warmup; batched throughput uses
batch 16 and includes its first call. Neither includes metadata encoding.

Pipeline delta recall@100: +1.791 ± 0.243 points.
Pipeline delta map@20: +0.283 ± 0.206 points.
