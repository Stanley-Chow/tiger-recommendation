# M4 results: larger residual codebooks and search diagnosis

Completed and independently verified on 2026-10-14. Increasing each residual
codebook from 16 to 32 entries improves full-test Recall@100 in all three
paired pipeline seeds, under the frozen training and selection recipe.
Mean Recall@100 rises from **15.585% to 17.375%**, a **+1.791 percentage-point**
difference. MAP@20 rises from **1.708% to 1.991%**. The three-seed comparison
supports using RQ-32 as the next candidate for this project; it does not establish
general superiority across datasets or training budgets.

![M4 capacity comparison, three pipeline seeds and conditional paired intervals](figures/m4_capacity_comparison.svg)

[PDF](figures/m4_capacity_comparison.pdf) ·
[Editable SVG](figures/m4_capacity_comparison.svg) ·
[300-dpi PNG](figures/m4_capacity_comparison.png) ·
[Caption](figures/m4_capacity_comparison.caption.md) ·
[Complete numeric tables](m4_numeric_tables.md) ·
[Unchanged full summary JSON](m4_results.json)

## Full test results

Percentages, mean ± sample SD across seeds 42/43/44. Every arm uses the same
22,225 eligible warm-target test users and full earlier seen-item exclusions.
The test beam is fixed at 200. No users are pooled across seeds.

| Arm | Recall@20 | Recall@100 | Recall@200 | MAP@20 | NDCG@20 | Coverage@20 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 2.043 ± 0.000 | 5.525 ± 0.000 | 7.946 ± 0.000 | 0.416 ± 0.000 | 0.763 ± 0.000 | 0.224 ± 0.000 |
| RQ-16 | 6.512 ± 0.196 | 15.585 ± 0.101 | 21.741 ± 0.077 | 1.708 ± 0.105 | 2.741 ± 0.110 | 16.683 ± 0.579 |
| RQ-32 | 7.396 ± 0.266 | 17.375 ± 0.202 | 23.949 ± 0.109 | 1.991 ± 0.101 | 3.157 ± 0.139 | 20.351 ± 0.543 |

RQ-32 minus RQ-16 averages **+1.791 ± 0.243 Recall@100 points** and
**+0.283 ± 0.206 MAP@20 points**, where SD describes the three fitted pipeline
differences. Tail Recall@100 rises from 3.138% to 3.572%; its seed variation is
reported in the complete tables. The tail threshold is the training-frequency
catalog median, including ties.

| Seed | Δ Recall@100 [conditional 95% interval] | Δ MAP@20 [conditional 95% interval] |
|---|---:|---:|
| 42 | +1.980 [+1.566, +2.448] | +0.339 [+0.213, +0.478] |
| 43 | +1.876 [+1.440, +2.282] | +0.456 [+0.331, +0.592] |
| 44 | +1.516 [+1.125, +1.885] | +0.055 [−0.090, +0.208] |

Intervals use 1,000 paired user-bootstrap resamples per fitted pair. They
condition on fitted models and do not measure uncertainty over training seeds.
All Recall@100 intervals are above zero; the seed-44 MAP interval includes zero.
Three pipeline fits are too few for a strong population-level significance claim
over training randomness. MAP equals truncated reciprocal rank for this
one-positive next-item task.

## What was controlled

The [protocol](m4_protocol.md) was frozen before M4 recommendation results.
Six metadata tokenizers were fitted first, followed by six fresh recommendation
models. All six selected checkpoint hashes were recorded before report-only
validation diagnostics and test retrieval. No new optimizer arm or test beam
was selected from the results.

All generators reserve the same semantic ranges, starting at 3/35/67, and
suffixes starting at 99. The maximum collision group across six tokenizers
contains 64 products, giving vocabulary 163 and **704,931 parameters per
generator**. Remapping preserves every natural tokenizer's semantic and suffix
indices. RQ-16 intentionally leaves 16 semantic slots unused at each level.

The offline autoencoder has 273,056 parameters for RQ-16 and 274,592 for RQ-32:
the intended difference is 1,536 codebook parameters. Both use the same frozen
384-dimensional normalized MiniLM vectors, encoder/decoder dimensions, loss
and metadata selection rule. The generator uses d=128, four heads, two encoder
and two decoder layers, FFN=256, dropout 0.1 and the latest 20 items as input.
Its exclusions cover the full earlier history.

The unchanged data contains 12,068 warm products, 131,413 training examples and
22,312 validation users. Checkpoint selection maximizes Recall@100 on the fixed
1,000-user validation subset, subject to MAP@20 ≥ popularity on that subset
(0.544%). **All six selected checkpoints pass the guard.** The guard is a minimum
validation check; passing it does not guarantee superiority over a stronger
baseline or an improvement in full-test MAP.

## Representation and token prediction

All natural codebooks use every available entry at each level: 16/16/16 or
32/32/32. Mean product membership in a three-token collision group falls from
**93.863% to 62.523%**. Occupied three-token prefixes increase from 2,655–2,706
to 6,805–7,070. Reconstruction MSE also decreases in every paired fit.
These are representation diagnostics rather than recommendation outcomes.

On the fixed 1,000-user report-only validation cohort, suffix NLL decreases
from **1.941 to 0.940 nats**, averaging the three fits. Suffix argmax accuracy
rises from **36.5% to 65.0%** under teacher forcing. In contrast, first-token
NLL rises from 2.229 to 2.763 and second-token NLL from 2.424 to 2.838.
Larger codebooks therefore change where prediction difficulty appears.

Finer partitions and less suffix competition are plausible contributors to
the improvement. This study does not isolate them from changed code geometry,
label frequencies, training duration or search behavior. Token metrics supply
the true earlier target tokens and use the full vocabulary; they do not measure
free-running catalog-constrained item accuracy.

## Approximate search

The 200/400 comparison is validation-only, with 200 returned products at each
width. RQ-32 has 22/23/27 certified search-ranking violations among the 1,000
users in seeds 42/43/44 at beam 200, falling to 0/0/0 at beam 400. RQ-16 has
8/3/14, falling to 0/0/1. The same validation users recur across fits; these
counts are not pooled as independent observations.

A certificate means an omitted eligible target's five-token score, including
EOS, exceeds the returned 200th candidate by more than 1e-4. It proves the
returned list is not the exact highest-scoring top 200. **It does not prove
the target itself belongs in the oracle top 200**, because other omitted
products might score higher. The frozen code calls this `certified_search_miss`;
interpret it as a ranking violation rather than a guaranteed lost oracle hit.
Zero certificates likewise do not establish exact search.

For RQ-32, beam 400 increases validation Recall@200 by 0.8/0.6/0.9 points, but
Recall@100 by only 0.1/0.0/0.2 points. For RQ-16, Recall@100 changes by
−0.3/−0.2/0.0 points. Wider search can introduce higher-scoring non-target
products, so relevance recall need not increase at every cutoff.

RQ-32 also has lower three-token prefix coverage than RQ-16 despite higher
exact-item recall on this cohort. Smaller collision groups change the chance
that a returned product shares a target prefix. Prefix coverage cannot serve
as a causal decomposition into semantic-token and suffix failures.

## Training cost and scope

RQ-16 stops after 16/14/18 epochs and selects 11/9/13. RQ-32 completes 20 epochs
in every seed and selects 19/17/19. Across three fits, this is **48 versus 60
training epochs**, with 889.96 versus 1,067.25 seconds in training loops and
970.41 versus 1,167.90 seconds of total training/selection wall time. Thus this
is a comparison under an equal maximum budget and early-stopping rule, not an
equal-update or equal-FLOP experiment.

Metadata selection plus full-catalog refitting takes 250.99 seconds for RQ-16
and 270.94 for RQ-32. Two RQ-32 schedules select the 100-epoch cap; bounded
tokenizer fitting may not have converged fully. The budget was not extended
after observing outcomes.

Mean warm single-request retrieval latency is 11.20 ± 0.16 ms for RQ-16 and
11.17 ± 0.04 ms for RQ-32. Batched throughput is 827.13 ± 1.70 versus
825.80 ± 0.50 histories/second. These descriptive measurements use the same
RTX 5090, beam 200, batch 16 and saved timing protocol; metadata encoding is
excluded.

This remains a simplified TIGER-style implementation, with frozen MiniLM
features and a deterministic RQ autoencoder. It is not a reproduction of the
paper's architecture or preprocessing. The split is per-user chronological,
uses review interactions as proxies and excludes cold targets. M2/M3 already
examined this test cohort, so M4 is a prospective follow-up on the same benchmark,
not a new blind test.

M3's vocabulary 133 and 697,221 generator parameters differ from M4. Its scores
are historical context. M2's generative AdamW6 run reached 18.106% Recall@100
and 2.120% MAP@20; its recipe differs from M4. M2 SASRec reached 22.956% and
3.739%. Both remain stronger historical scores, rather than matched M4 arms.
M4 establishes an improvement within its frozen capacity comparison, not a new
best score across all project experiments.

## Verification, artifacts and rental

Both CPU and GPU preflights passed **225 tests and Ruff**, including a real
two-arm synthetic train/evaluate/resume path and corrupted-cache rejection.
The exact executed source is commit
`3da6b14bf3b7b3a9a6397a179df9cc219273314c`;
[m4_source_manifest.json](m4_source_manifest.json) records every uploaded file.
Later auditing, plotting and reporting additions do not change the executed
model source. GPU runtime was Python 3.12.3, PyTorch 2.12.1+cu130 and CUDA 13.0
on an RTX 5090 with driver 595.71.05.

The remote completion inventory verifies **183 files totaling 836,035,040 bytes**.
The returned archive checksum is
`a1907c07c70890a316c55b10116b541963a67460b85c8370486375fb83959735`.
The completion-marker checksum is
`f22a212e8a2f50114be2e48dccf37669adc88cfdd37bb6cf50541546d1b62b69`.

The independent local audit checks all six checkpoints, metadata splits and
refit schedules, common vocabulary, validation selection rules, **200,547 saved
validation/test context rows**, **6,000 search-diagnostic rows**, and 150 aggregate
mean/SD pairs. It reconstructs Recall, MAP/MRR, NDCG, coverage and tail metrics,
including catalog membership, uniqueness and full seen-item exclusion.
Bootstrap metadata, bounds and point estimates are checked; bootstrap draws
are not recreated. All 183 copied artifact hashes also match. See
[m4_verification.json](m4_verification.json).

Complete artifacts remain local under `results/m4/`, with the original Linux
paths preserved. The log is `results/m4.log`; the observed GPU package freeze is
`results/m4-runtime-requirements.txt`. Large weights, data and rankings remain
outside Git; code, aggregate results, reports and figures are backed up.
The returned seed-42 RQ-32 checkpoint also passed a CPU recommendation check
using the README example history, returning five valid catalog products.

The first launcher exited before training because non-login SSH lacked Python
on PATH. Setting the Miniconda runtime path resolved it without a source change.
Windows test/extraction permissions and UTF-8 defaults were handled explicitly.
No failed or synthetic run is presented as a completed Amazon experiment.

The settled new rental charge is **¥4.13** (¥2.44 + ¥1.69), covering
2026-10-14 02:07:18–03:36:34 Asia/Shanghai, including setup, evaluation, artifact
return and verification. This is within the updated **¥50** training budget.
Prior project spending was ¥10.68, giving **¥14.81 cumulative**. The dedicated
M4 GPU is confirmed stopped. Its account key and active-instance key were
removed, a fresh public-key login was rejected, and local key files were deleted.
Other instances were left untouched.

## Learning checkpoint and next decision

Work stops at the [M4 learning checkpoint](learning_checkpoint_m4.md).
Explain the matched vocabulary, the shift from suffix to semantic-token
difficulty, why certificates do not imply oracle hits, and the distinction
between pipeline SD and conditional user intervals.

After the checkpoint, a useful next experiment would compare exact catalog
scoring with beam search on a bounded validation subset, or test an equal-update
capacity comparison. External confirmation on the second data stage should
follow a new frozen protocol. These are proposed directions, not additional
experiments in this completed milestone.
