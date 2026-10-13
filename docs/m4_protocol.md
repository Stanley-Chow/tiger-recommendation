# M4 frozen protocol: RQ codebook capacity and search diagnostics

Design fixed on 2026-10-14 before M4 recommendation results. This is a
prospective follow-up on a previously evaluated dataset; M2/M3 test outcomes
are already known. No new blind-test claim will be made.

## Question and two arms

Does increasing each of three RQ codebooks from 16 to 32 entries improve
next-item retrieval under matched generator capacity? The primary estimand is
the seed-wise difference $\Delta=\mathrm{Recall@100}_{32}-\mathrm{Recall@100}_{16}$.
MAP@20 is the checkpoint-selection guard and a reported secondary metric.
Negative, mixed or guard-failing outcomes remain valid study results.

Use the unchanged Beauty 2014 warm catalog (12,068 products), frozen normalized
384-dimensional MiniLM vectors, 131,413 training examples, 22,312 validation
users and 22,225 test users. Full earlier seen-item histories are excluded.
The source data and features remain exactly those of M3.

Fit six metadata tokenizers: sizes 16/32 crossed with pipeline seeds 42/43/44.
Each uses the M3 384→256→128→32 autoencoder, three residual levels, commitment
weight 0.25, 20 continuous warmup epochs and at most 100 quantized epochs.
Choose the schedule on the same metadata 90/10 split (seed 2026), then perform
a fresh full-catalog refit using that fixed schedule. No recommendation labels
choose the tokenizer. All six fitting stages finish before any recommender.

## Match generator capacity

Reserve PAD/BOS/EOS 0/1/2 and 32 token slots at each semantic level, starting
at 3/35/67. Remap each natural tokenizer's level indices into these common
ranges. Start all suffix tokens at 99; reserve the largest collision group
across all six fits. Thus every arm has $V=99+M$ and identical generator tensor
shapes and parameter counts. RQ-16 leaves 16 unused entries per semantic level.
The remapping preserves every tokenizer's item partitions and suffix indices.

Train six fresh recommenders, retaining M3's d=128, four heads, two encoder and
two decoder layers, FFN=256, dropout 0.1, history 20, AdamW lr 0.001 and decay
0.01, batch 256, maximum 20 epochs and patience 5. Within each seed the arms
share initialization tensor shapes and minibatch order. Select maximum
Recall@100 on the fixed 1,000-user validation cohort (seed 2026), subject to
MAP@20 at least validation popularity; ties use MAP@20. Report a fallback if no
epoch passes. Actual early-stopped epochs need not match. Tokenizer parameter
counts differ intentionally. Old M3 scores are contextual, not matched controls.

## Diagnostics and final evaluation

Freeze all six selected checkpoint SHA-256 hashes before report-only retrieval.
On 1,000 validation users sampled with seed 2027, report teacher-forced target
token NLL/argmax accuracy, target-prefix coverage in top-200 candidates, and
top-200 retrieval with beam widths 200 and 400. A missing eligible target whose
full five-token likelihood exceeds the retained 200th candidate by 1e-4 is a
certified beam-search miss. This is a lower bound, not exhaustive exact search.
Teacher-forced token errors condition on true previous target tokens; prefix
coverage alone does not locate a causal failure stage. Wider beam differences
can have either sign. These diagnostics do not select a test beam or new arm.

Run full validation/test evaluation for both arms with the fixed beam 200,
batch 16 and cutoffs 10/20/50/100/200. Report Recall, MAP/MRR, NDCG, tail metrics,
coverage, warm single-request latency and batched throughput. Per seed, use
1,000 paired user-bootstrap resamples (seed 2026) for RQ-32 minus RQ-16. Report
three-seed means and sample SD. Do not pool repeated users across seeds or
claim formal significance over training randomness from three fits.

## Execution and completion

Use an isolated AutoDL 5090-family instance and project directory. The updated
user budget is ¥50; target new rental cost below ¥12 with a provider shutdown
timer. Preserve the M2/M3 artifacts and other instances. Verify input/source
hashes, CUDA support, tests and a real synthetic smoke run before full training.
Cache only complete artifacts with matching protocol and file hashes.

M4 completion requires six valid trained checkpoints, all complete evaluations,
tokenizer/search diagnostics, uncertainty and cost reports, independent ranking
and artifact verification, local artifact return, GitHub source/report backup,
GPU shutdown and temporary-key removal. An improvement is not required.
Stop at the M4 learning checkpoint; no new dataset, reranker or SAM arm is included.
