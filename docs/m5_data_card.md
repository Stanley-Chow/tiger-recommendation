# M5 dataset card: a training-selected temporal cohort

The source is the official [Amazon Reviews 2023 deduplicated 0-core release](https://amazon-reviews-2023.github.io/data_processing/0core.html),
category `Beauty_and_Personal_Care`, joined to metadata by parent ASIN. The
downloaded CSV contains **23,591,312** user-item interactions. Ratings are not
used as positive/negative labels: this is a next-review-interaction proxy task,
not a purchase, click-through, or satisfaction prediction task.

The preparation code finds **359,741** users with at least five pre-2022
interactions in the globally selected training catalog. A stable training-only
hash selects **50,000** of them. Every selected user retains their complete
deduplicated source history in the audit file. The neural input uses its latest
20 supported items, while exclusions retain the complete supported seen set.

| Quantity | Prepared cohort |
|---|---:|
| Warm items with cohort training activity | 38,336 |
| New available items, first observed during 2022 | 20,000 |
| Available test catalog | 58,336 |
| Expanding-prefix training examples | 347,969 |
| Eligible warm validation users | 8,048 |
| Eligible warm test users | 2,252 |
| Eligible mixed-catalog test users | 2,452 |
| New-item targets inside the mixed test set | 200 |
| Catalog items lacking metadata | 0 |

## Eligibility is part of the result

Among the selected users, **15,059** have a first novel interaction during the
validation window; **8,048** of those targets are in the warm catalog. In the
test window, **5,371** users have a first novel interaction. The warm catalog
covers **2,252 / 5,371 = 41.93%** of these first targets. The expanded available
catalog covers **2,452 / 5,371 = 45.65%**. The remaining **2,919** first targets
are unavailable under this request-time catalog definition.

Quality metrics condition on those eligible first targets. They do not describe
all future products or all 50,000 users. Inactive users have no target to score;
they are not counted as recommendation misses. An unavailable first target is
counted before eligibility filtering and is never replaced with a later,
more convenient target. The 200 additional eligible targets are not the total
number of genuinely new products among all excluded targets.

## Time, features, and limitations

The global UTC cutoffs are 2022-01-01 for training, 2023-01-01 for testing, and
2023-10-01 for the end of the test window. Validation histories stop before
2022; test histories stop before 2023. Training histories use strictly earlier
timestamps, including protection against tied-time leakage. Novelty is defined
after retaining the earliest occurrence of each user-parent-ASIN pair.

The new-item pool contains products first reviewed during 2022 and never seen
globally before training. First review is an availability proxy, not a verified
release date. This fixed pool is known before the test request; products first
appearing later in 2023 are outside it. The service supports this available
catalog and rejects unknown ASINs. Onboarding arbitrary later products would
require new metadata encoding and an index update.

Frozen MiniLM features use title, store/brand, categories, features and
description. Review text, average ratings, review counts, and prices are
excluded. Released metadata is a 2023 snapshot; historical availability of its
text fields is unverified. The quantizer fits warm metadata only. New-item
vectors are encoded by the frozen public encoder without interaction training.

The three-seed experiment measures conditional performance on this defined
cohort. Catalog subsampling, review-selection bias, the historical metadata
limitation, approximate generative beam retrieval, and only 200 eligible new-item
targets limit broader conclusions. Higher scores than a 2014 experiment would
not establish an improvement: catalog, population, time split and task differ.

SHA-256 provenance, exact cutoffs, counts, and output hashes are saved locally in
`statistics.json`, `preprocessing_complete.json`, and `protocol.json`. The
independent audit reconstructs contexts from the saved complete cohort events
and recalculates metrics; it does not independently redownload the raw release.
