# M1 evaluation results

Completed on 2026-10-03. This evaluates the existing three-epoch CPU checkpoint; it does not report the new M2 models.

Both methods use all 22,312 eligible validation users and the same 12,068-item warm catalog. Each cutoff uses a prefix of one top-200 list. Previously seen items are excluded.

| Method | Recall@10 | Recall@50 | Recall@100 | Recall@200 | MAP@20 | NDCG@20 | Coverage@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| generative | 0.02492 | 0.06871 | 0.10299 | 0.15624 | 0.00910 | 0.01564 | 0.11949 |
| popularity | 0.01636 | 0.04468 | 0.07072 | 0.10044 | 0.00611 | 0.01058 | 0.00224 |

MAP@K equals MRR@K here because each user has one held-out next item. These labels are observed review events rather than a complete set of relevant products.

The generative model uses constrained beam search with width 200. Popularity uses training-event counts. The saved checkpoint was selected earlier using NDCG@10 on 256 validation users, so these full validation results are an evaluation of that checkpoint, not a new checkpoint-selection procedure.

The M2 protocol adds fixed-cohort Recall@100 selection with a popularity MAP@20 floor, stronger SASRec comparisons, three training seeds, and controlled SAM continuation. M1 alone does not establish seed-robust superiority or a converged model.

Raw reproducibility artifacts remain local in `results/m1/validation/summary.json` and `per_user_rankings.jsonl`. M2 keeps the test set for its prespecified final comparisons.
