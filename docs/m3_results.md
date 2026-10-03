# M3 results: semantic alignment and residual quantization

The three-seed study supports using aligned item codes in this implementation.
Both HKM and learned RQ substantially outperform their matched whole-code
permutation controls. RQ does not show a consistent advantage over HKM: it wins
one seed and loses two on Recall@100, with a slightly lower three-seed mean.

The study uses the complete Amazon Reviews 2014 Beauty warm-item benchmark:
12,068 catalog items, 131,413 expanding-prefix training examples, 22,312 eligible
validation users and 22,225 eligible test users. Four arms use the same
697,221-parameter Transformer, a common vocabulary of 133 tokens, paired seeds
42/43/44, and the [frozen protocol](m3_protocol.md). All twelve selected
checkpoints were frozen before any M3 test retrieval. Model settings were not
changed after seeing these results.

## Test quality

Values are percentages, reported as mean ± sample standard deviation across
three pipeline seeds. Every seed uses the same test users; they are not three
independent test populations. Each cutoff uses a prefix of the same top-200 list,
with every earlier seen item excluded before beam pruning. One next-item positive
per user makes MAP@K equal to MRR@K.

| Method | Recall@10 | Recall@100 | Recall@200 | MAP@20 | NDCG@20 | Coverage@20 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 1.147 ± 0.000 | 5.525 ± 0.000 | 7.946 ± 0.000 | 0.416 ± 0.000 | 0.763 ± 0.000 | 0.224 ± 0.000 |
| HKM | 4.534 ± 0.166 | 16.535 ± 0.646 | 22.883 ± 0.742 | 1.845 ± 0.079 | 2.945 ± 0.131 | 22.075 ± 2.079 |
| HKM-permuted | 1.720 ± 0.135 | 6.877 ± 0.152 | 9.956 ± 0.196 | 0.685 ± 0.091 | 1.128 ± 0.102 | 2.262 ± 0.367 |
| RQ | 4.421 ± 0.393 | 16.205 ± 1.095 | 22.467 ± 1.452 | 1.793 ± 0.148 | 2.899 ± 0.221 | 19.246 ± 1.782 |
| RQ-permuted | 1.597 ± 0.095 | 6.257 ± 0.450 | 9.135 ± 0.632 | 0.615 ± 0.074 | 1.017 ± 0.068 | 1.682 ± 0.309 |

The primary aligned-minus-permuted Recall@100 differences average **+9.659
percentage points for HKM** and **+9.948 points for RQ**. Both improve MAP@20 in
every seed and recommend a broader range of catalog items. The observed
permuted arms concentrate top-20 recommendations on about 1.7–2.3% of the
catalog, compared with 19.2–22.1% for aligned codes.

The secondary RQ-minus-HKM difference averages **−0.330 Recall@100 points** and
**−0.052 MAP@20 points**. This bounded RQ setup provides a functioning learned
tokenizer, but the results do not justify claiming it improves recommendation
quality over HKM. Reconstruction error and metadata locality are separate
objectives from next-item retrieval.

![M3 test Recall and MAP with per-seed points and sample SD](figures/m3_tokenizer_ablation.svg)

[Paper PDF](figures/m3_tokenizer_ablation.pdf) ·
[Editable SVG](figures/m3_tokenizer_ablation.svg) ·
[300-dpi PNG](figures/m3_tokenizer_ablation.png) ·
[Caption](figures/m3_tokenizer_ablation.caption.md)

## Paired uncertainty

Each cell is candidate minus reference in percentage points, followed by a
95% paired user-bootstrap interval from 1,000 resamples. These intervals describe
user-sampling uncertainty conditional on a fitted model pair. They do not include
uncertainty from drawing new tokenizer/recommender seeds.

| Candidate minus reference | Seed | Δ Recall@100 [95% CI] | Δ MAP@20 [95% CI] |
|---|---:|---:|---:|
| HKM − HKM-permuted | 42 | +9.129 [+8.684, +9.570] | +0.990 [+0.872, +1.104] |
| HKM − HKM-permuted | 43 | +9.674 [+9.206, +10.137] | +1.284 [+1.155, +1.414] |
| HKM − HKM-permuted | 44 | +10.173 [+9.696, +10.637] | +1.206 [+1.084, +1.327] |
| RQ − RQ-permuted | 42 | +10.754 [+10.299, +11.222] | +1.202 [+1.064, +1.325] |
| RQ − RQ-permuted | 43 | +8.189 [+7.712, +8.621] | +1.001 [+0.878, +1.131] |
| RQ − RQ-permuted | 44 | +10.902 [+10.403, +11.379] | +1.331 [+1.209, +1.462] |
| RQ − HKM | 42 | +0.954 [+0.535, +1.327] | +0.132 [+0.011, +0.265] |
| RQ − HKM | 43 | −1.498 [−1.908, −1.089] | −0.248 [−0.368, −0.120] |
| RQ − HKM | 44 | −0.445 [−0.851, −0.040] | −0.039 [−0.160, +0.084] |

The pipeline mean ± sample SD of Recall@100 differences is +9.659 ± 0.522 for
HKM alignment, +9.948 ± 1.525 for RQ alignment, and −0.330 ± 1.230 for RQ versus
HKM. RQ is fitted separately for each seed, so its pipeline variation combines
tokenizer and recommender variation. Three seeds provide limited evidence about
training uncertainty. No pooled seed-by-user confidence interval or formal
family-wise significance claim is made.

## What the controls establish

Whole-code permutation preserves each pair's complete codeword set, trie,
unweighted catalog occupancy, suffix usage, code length and generator capacity.
It changes which items receive which codes. The large retrieval differences
support the usefulness of the implemented item-to-code alignment under those
matched conditions.

It also changes interaction-weighted token frequencies, sequence co-occurrences
and conditional prediction difficulty. The [tokenizer diagnostics](m3_tokenizer_diagnostics.md)
measure these shifts: shuffled L3 weighted entropy rises by about 0.12–0.15 bits.
This is not a pure semantic intervention with every training exposure held fixed.
HKM versus RQ additionally changes code geometry, collision distribution,
suffix burden and tokenizer capacity, despite equal generator capacity.

All RQ banks use 16/16 entries. Metadata holdout selection chooses 82/87/89
quantized epochs; fresh full-catalog refits follow those fixed schedules, reach
coordinate MSE about 0.001219, and preserve substantial MiniLM-neighbor locality.
Their 2,610–2,713 occupied prefixes contain many collisions; roughly
94% of items share a three-token prefix with another item. The uniqueness suffix
is consequently a major part of this small-codebook model.

## Selection, training and retrieval cost

All twelve selected checkpoints satisfy the fixed 1,000-user validation
popularity MAP@20 floor of **0.543975%**. Selection maximizes validation
Recall@100 among eligible epochs; the guard is not a guarantee for a different
evaluation cohort. Full validation/test evaluations are report-only.

Every arm has a maximum of 20 epochs and patience 5, with the same paired
minibatch order and optimizer settings. Actual stopping times differ, so this
is not an equal-update or equal-FLOP study.

| Arm | Completed epochs, seeds 42/43/44 | Selected epochs | Mean training seconds | Warm request mean ms | Batched histories/s |
|---|---|---|---:|---:|---:|
| HKM | 16/20/20 | 11/20/20 | 342.84 ± 40.46 | 11.63 ± 0.25 | 811.39 ± 18.97 |
| HKM-permuted | 18/20/20 | 13/16/19 | 360.36 ± 21.83 | 11.43 ± 0.05 | 802.57 ± 17.38 |
| RQ | 20/14/20 | 20/9/20 | 337.01 ± 70.43 | 11.30 ± 0.22 | 809.59 ± 20.72 |
| RQ-permuted | 13/20/16 | 8/16/11 | 305.17 ± 61.27 | 11.59 ± 0.40 | 802.72 ± 20.70 |

Training seconds exclude validation, tokenizer fitting, environment setup and
external artifact transfer; batch preparation and transfers to the GPU within
the training loop are included. The
three RQ selection/refit pairs together take 214.78 seconds, including their
warmup and initialization. Training completes 217 epochs and 111,538 optimizer
updates/backward passes across twelve arms. GPU: **RTX 5090 D 32 GB**, Python
3.12.3, PyTorch 2.12.1+cu130, CUDA 13.0, driver 595.71.05.

Request timings use 100 warm individual histories and include CPU result
transfer; throughput uses batched test retrieval with batch 16 and beam 200.
They exclude metadata encoding, networking and a production service. Detailed
training/selection memory, request p95 and every cutoff are retained in the
numeric records.

The [complete numeric tables](m3_numeric_tables.md) include all twelve selected
epochs, guard scores, optimizer updates, backward passes, memory measurements
and latency p95. The [full summary JSON](m3_results.json) preserves every cutoff,
seed metric and paired comparison.

## Scope and learning checkpoint

This is a simplified TIGER-style adaptation, with MiniLM metadata embeddings
and a deterministic residual-quantization autoencoder. It does not reproduce
the original paper's tokenizer, architecture or training protocol. The warm-item
per-user chronological split excludes cold targets and does not impose a global
time cutoff; review events are interaction proxies.

M2's SASRec Recall@100 of 22.956% and MAP@20 of 3.739% remain useful context.
SASRec has a different architecture, parameter count, objective and hardware
run; its earlier result is not a capacity-matched M3 arm. M2 test results were
already known before this prospective study on the same dataset.

Proceed to the [M3 learning checkpoint](learning_checkpoint_m3.md): residual
quantization and gradient routing, semantic alignment controls, guard-based
selection, saved-ranking metrics, and conditional versus pipeline uncertainty.
Further tuning, a second dataset and additional optimizer experiments require
a separate milestone.

## Verification and artifacts

The GPU environment passed **222 tests and Ruff** before the full run. The
executed upload matches source commit
`3d0e89f6346832a652231df05a79f72a495998eb`; model implementation is commit
`66cb1b2e4f2532bbb5b8f5bd4ab279ee33886eea`. Later documentation changes do not
alter the executed source. See [m3_source_manifest.json](m3_source_manifest.json).

Archive SHA-256:
`f537a40dcdc2d3ab62d62ac9378c53b486623d5522b81a492075884c43e17f00`.
Completion-marker SHA-256:
`1a6dc09a32a368295190b71e1b6fb5db8e641cc881520ebeeff6478562bf26d0`.
The remote completion inventory contains 209 files totaling 1,231,716,355 bytes.

The full local audit passed for **all 334,419 saved context rows and all twelve
selected checkpoints**. Every validation/test ranking was checked for catalog
membership, uniqueness and complete-history exclusion; Recall, MAP/MRR, NDCG,
coverage and tail metrics were independently reconstructed. A supplementary
audit checked 252 aggregate mean/sample-SD pairs, embedded summary copies,
contrast point estimates, bootstrap metadata/bounds and timing totals. It did
not recreate the bootstrap draws. The copied final artifact tree retains all
209 original inventory hashes. See [m3_verification.json](m3_verification.json).

Weights, code mappings, full rankings and completion markers are local under
`results/m3/`; the log is `results/m3.log` and GPU package freeze is
`results/m3-runtime-requirements.txt`. The full summary JSON and figures are
backed up in Git; large weights, rankings and data are excluded from Git.

The settled M3 GPU rental charge is **¥6.19**: ¥1.33 + ¥2.78 + ¥2.08, covering
2026-10-03 12:31:21–14:44:49 Asia/Shanghai, including setup, evaluation and
artifact return. Together with M2's ¥4.49, cumulative project spending is
**¥10.68**, below the requested ¥25 notification threshold. Only the M3
experiment instance was stopped. Its temporary key was removed from AutoDL's
registry and its active instance key file; a fresh public-key login failed as
expected. The local temporary private/public key files were removed.
