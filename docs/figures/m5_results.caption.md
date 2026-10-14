# M5 temporal Amazon 2023 results

Panels (a) and (b) show Recall@100 and MAP@20 on 2,252 eligible warm test
users. Bars are means over pipeline seeds 42, 43, and 44; whiskers are sample
standard deviations, not confidence intervals. Popularity is a deterministic
control repeated on the same users. Each request has one relevant next item,
so MAP@20 equals MRR@20 under this evaluation convention.

Panel (c) shows Recall@100 on 200 new-item targets in the 2,452-user eligible
mixed test set. Closed warm-catalog methods cannot emit these products. Pure
content retrieves two of these targets and either fixed hybrid retrieves one
at cutoff 100 per seed. Identical counts across seeds do not provide three
independent samples of cold-start behavior. These small counts limit claims
about new-item accuracy.

Panel (d) shows generative minus SASRec Recall@100 for each seed, with percentile
95% intervals from 1,000 paired user-bootstrap resamples. These intervals
condition on that seed's trained models, catalog, and eligible contexts. They
do not capture all training uncertainty or provide simultaneous coverage
across metrics. The dashed line marks zero difference.

All rankings use the full eligible catalog without sampled negatives. The
generator uses approximate trie-constrained beam retrieval; SASRec scores the
warm catalog exactly. Model parameter counts differ. The expanded catalog
covers 45.65% of active users' first test targets, so quality is conditional on
eligibility. Training uses pre-2022 data; test histories freeze before 2023.
Released metadata is a 2023 snapshot with unverified historical field
availability. See the [dataset card](../m5_data_card.md) and
[verified report](../m5_results.md) for the complete interpretation.

Generated from independently audited saved results by
[`scripts/report_m5.py`](../../scripts/report_m5.py). PDF is the vector print
export, SVG retains editable text, and PNG is exported at 300 dpi.
