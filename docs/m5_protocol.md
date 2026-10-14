# M5 protocol: temporal transfer and an end-to-end recommendation service

Frozen on 2026-10-14 before M5 recommendation results. M5 is the second data
stage from the latest roadmap. The goal is a complete, reviewable research
project, including reproducible data, comparative evidence and usable inference.
An accuracy improvement is not a completion requirement.

## Data and time

Use the official Amazon Reviews 2023 `Beauty_and_Personal_Care` deduplicated
**0-core** pure-ID file and metadata, with parent ASIN as product identity.
Using 0-core avoids selecting users/items by future five-core activity.
Source: https://amazon-reviews-2023.github.io/data_processing/0core.html

All cutoffs are UTC milliseconds. Train before 2022-01-01; validate on each
user's first novel interaction in 2022; test on the first novel interaction
from 2023-01-01 through 2023-09-30. Histories are frozen at the corresponding
window start. Training examples use strictly earlier timestamps; tied-time
events do not become each other's history. Full earlier seen items are excluded.
Choosing the first future interaction occurs before checking catalog eligibility:
never skip an unavailable target to find a more convenient later target.

Select at most 40,000 products by pre-2022 interaction frequency, breaking ties
by ASIN. Select at most 50,000 users with at least five training interactions
in that catalog, ordered by SHA-256(seed:user ID), seed 2026. Retain complete
source histories for these users. Model inputs project those histories into
their supported catalog and keep the latest 20 items; discarded event counts
and full exclusions are recorded. Products with zero selected-cohort training
interactions are removed from the warm catalog. The actual resulting size may
be below the cap. This is a defined bounded cohort, not the full category.

For the dynamic test catalog, add up to 20,000 products first observed during
2022 and absent globally before training. Select them by the same stable hash,
using no 2023 events. The first observed review is an explicit availability
proxy; it is not a verified product release date. Targets outside the catalog
available at the request cutoff are counted and excluded from eligible metrics.
Report this eligibility coverage as well as conditional quality.

Only title, store/brand, categories, features and description form metadata
features. Exclude review text, average ratings, rating counts, price and other
post-interaction aggregates. The released metadata is a 2023 snapshot; historical
text availability cannot be guaranteed. State this limitation explicitly.

## Models and selection

Keep frozen MiniLM features and the M4 RQ-32 recipe. Fit one tokenizer for each
seed 42/43/44 using warm metadata only. Fit all tokenizers before reserving the
common suffix vocabulary and training the recommendation models. New-item
metadata never enters tokenizer fitting. Train fresh generative and SASRec-style
models for each seed: six learned recommenders in total. Architecture and
optimizer details are fixed in `configs/m5.json`. This is a practical baseline
comparison, not a parameter-matched causal architecture experiment.

Select each checkpoint on the same fixed 1,000 warm validation users, seed 2026:
maximize Recall@100 with MAP@20 at least popularity; tie-break by MAP, then the
earliest epoch. Report any guard failure and retain its declared fallback.
All six checkpoint hashes must be frozen before full final evaluations.

## Warm and new-item evaluation

Primary comparison: RQ-32 minus SASRec on warm-catalog test Recall@100, with
MAP@20 as the ranking guard/secondary metric. Include popularity, NDCG, tail
metrics, coverage, latency, throughput and training resources. Test all eligible
users in the bounded cohort at cutoffs 10/20/50/100/200, beam 200.

Separately evaluate a catalog containing warm and previously untrained 2022
products. Closed-catalog models cannot emit those new products: report their
new-item recall as zero rather than pretending arbitrary suffix tokens solve
cold start. Include a frozen-metadata cosine retriever using the normalized
mean of the latest 20 available history vectors. Add two fixed hybrid strategies:
generative plus content and SASRec plus content, using equal-weight reciprocal
rank fusion, constant 60, 200 candidates per branch. No fusion weight is tuned
on test data. Each method receives the same eligible targets and exclusions.
Report warm/new target cohorts separately. Hybrid new-item capability belongs
to its explicit content route, not to the closed-catalog generator alone.

Per seed, use 1,000 paired user-bootstrap resamples and save confidence intervals
and the resampling seed for the prespecified comparisons. Report three-seed means
and sample SD without pooling repeated users. Dataset shifts, changed catalogs and time splits prevent
direct score comparisons with Beauty 2014 from being interpreted as improvement.

The prespecified mixed-catalog contrasts are generative+content minus generative,
SASRec+content minus SASRec, and generative+content minus SASRec+content. These
are secondary, descriptive comparisons; the warm Recall@100 contrast above is
primary. No claim of simultaneous confidence coverage across all metrics is made.

## Delivery and completion

Deliver a reusable inference engine, a local HTTP API and CLI, a synthetic
offline end-to-end demo, automated checks, training/evaluation commands, saved
rankings and manifests, independently reconstructed metrics, a results report,
architecture/results visuals, and a code-review reading guide. Large artifacts
stay local; source, aggregate evidence and documentation go to the backup repo.

Use only the project's dedicated AutoDL 5090 experiment instance, a separate M5
directory and the approved temporary M5 key. Preserve M2-M4 artifacts and other
instances. Treat ¥50 as a conservative cumulative project ceiling: previous
settled spend is ¥14.81, leaving ¥35.19. Set a provider shutdown timer, return
and verify artifacts, stop the GPU, remove account/instance/local temporary keys,
and record settled billing. Do not call M5 complete after merely launching a run.

Hardware amendment before training, 2026-10-14: the user approved a dedicated
RTX 4090 (24 GB) fallback because 5090 rentals were unavailable. Architecture,
data, seeds, optimizer, batch sizes and evaluation remain fixed. Record actual
runtime hardware and numerical environment; cross-hardware bitwise equality is
not assumed. Keep the same budget and provider shutdown requirement.

Second pre-training hardware amendment: the displayed 4090 slot was exhausted
before rental. The user approved a dedicated RTX PRO 6000 at ¥6.98/hour with
a 4.5-hour maximum rental window (about ¥31.41 new spend). Instance
`3hy7n2qhcg-e44522b7` has a provider shutdown deadline of 2026-10-14 17:15
Shanghai time. Model/data/evaluation settings are unchanged.

After completion, stop for the learning/code-review session. Additional datasets,
new foundation models, exact-search studies and new optimizer arms remain
separate research decisions supported by the M5 evidence.
