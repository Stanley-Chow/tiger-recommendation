# Frozen real M3 tokenizer metadata review

## Residual-quantizer fits

The metadata split contains 10,861 fitting items and 1,207 held-out items. Each
selection run completed its 100-epoch quantized budget after 20 continuous
warmup epochs; the earlier minimum held-out quantized reconstruction selected
the refit schedule. A fresh refit then uses all 12,068 items for 20 warmup epochs
and exactly that selected number of quantized epochs.

| Seed | Selected quantized epochs | Selected holdout coordinate MSE | Final refit coordinate MSE | Occupied three-token prefixes | Largest collision group | Natural vocabulary |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 82 | 0.00126872 | 0.00121865 | 2,707 | 59 | 110 |
| 43 | 87 | 0.00127718 | 0.00121973 | 2,610 | 82 | 133 |
| 44 | 89 | 0.00127025 | 0.00121896 | 2,713 | 44 | 95 |

All three levels use 16/16 entries in every final fit; marginal perplexities are
14.44–15.68 and the largest marginal token share is 12.90%. No fit triggers the
predefined warning of fewer than 13 active entries. Final quantized reconstruction
cosine is 0.7261–0.7265. Together, selection and refitting take 214.78 seconds,
with peak PyTorch CUDA allocation of 71.08 MiB per tokenizer on the RTX 5090 D.

The selected holdout MSE is 3.85–4.41% above fitting MSE. That holdout selected
the schedule and is not an independent test estimate. Final refit reconstruction
is in-sample because every catalog vector participates. Lower refit error,
shrinking residual norms, and marginal code usage do not establish better
recommendations. Three-token collisions involve 93.88–94.08% of products;
suffixes remain essential. The largest group requires 82 suffix slots, so every
recommender uses the same padded vocabulary of 133 tokens.

Reviewed metadata only. All 24 downloaded arm diagnostic files match their
SHA-256 entries in the frozen tokenizer manifest. No recommendation validation
or test results were read, and no fitting settings were changed.

The diagnostics use 12,068 catalog items and the same 1,024 sampled metadata
queries, each with 10 nearest neighbors from all other catalog items: 10,240
query-neighbor pairs. Query seed is 2026; mean neighbor cosine is 0.704867.
Similarity is measured in the frozen MiniLM space also used to fit tokenizers.
This describes preservation of that embedding geometry, not independent semantic
ground truth or recommendation accuracy. Neighbor pairs share items and queries;
they are not 10,240 independent statistical observations.

## Metadata-neighbor shared-prefix locality

Fractions below are percentages. L1/L2/L3 mean sharing the first one, two or
three semantic tokens, excluding the uniqueness suffix.

| Tokenizer | Seed | Aligned L1 | Aligned L2 | Aligned L3 | Shuffled L1 | Shuffled L2 | Shuffled L3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HKM | 42 | 74.395 | 53.916 | 21.143 | 6.777 | 0.576 | 0.059 |
| HKM | 43 | 74.395 | 53.916 | 21.143 | 7.188 | 0.537 | 0.029 |
| HKM | 44 | 74.395 | 53.916 | 21.143 | 6.475 | 0.410 | 0.049 |
| RQ | 42 | 77.402 | 42.539 | 20.000 | 7.266 | 0.537 | 0.068 |
| RQ | 43 | 74.629 | 41.172 | 19.063 | 7.275 | 0.703 | 0.078 |
| RQ | 44 | 76.807 | 43.115 | 18.633 | 6.357 | 0.557 | 0.059 |

Both aligned tokenizers retain substantial metadata locality; whole-code shuffle
reduces it close to chance. The catalog random distinct-pair L3 probabilities
are 0.0451% for HKM and 0.0751%, 0.0810%, 0.0710% for RQ seeds 42/43/44.
RQ has slightly higher L1 agreement but lower L2/L3 agreement than HKM on these
queries. Their different prefix occupancies and fitting objectives prevent this
diagnostic comparison from establishing which recommender will perform better.

## Catalog preservation and training-weighted shifts

For all six aligned/control pairs, the downloaded diagnostics match exactly on
unweighted per-position token counts, entropy/perplexity, cumulative prefix
occupancy and collision distributions, suffix usage/capacity, and full-code
collision count (zero). Code-assignment hashes change. All arms use V=133.
Whole-code permutation preserves the codeword set and trie topology by
construction; these catalog statistics confirm the intended preservation.

Interaction weighting counts each original deduplicated training interaction
once: 153,776 interactions. The weight-vector hash is identical across controls.
The assignment of those weights to code groups changes. Entropy changes below
are **shuffled minus aligned**, in bits, for cumulative prefixes.

| Tokenizer | Seed | Delta H(L1) | Delta H(L2) | Delta H(L3) | Aligned H(L3) | Shuffled H(L3) |
|---|---:|---:|---:|---:|---:|---:|
| HKM | 42 | +0.03331 | +0.06561 | +0.13957 | 10.91090 | 11.05047 |
| HKM | 43 | +0.03137 | +0.07340 | +0.13796 | 10.91090 | 11.04886 |
| HKM | 44 | +0.03241 | +0.04858 | +0.11805 | 10.91090 | 11.02894 |
| RQ | 42 | +0.03103 | +0.04796 | +0.15046 | 10.37581 | 10.52627 |
| RQ | 43 | +0.03686 | +0.11855 | +0.13950 | 10.32093 | 10.46043 |
| RQ | 44 | +0.02827 | +0.06888 | +0.14583 | 10.40843 | 10.55426 |

Thus the shuffle controls content-to-code alignment while preserving code
capacity and catalog occupancy, but does not isolate semantic alignment from
every interaction-dependent training effect. Weighted token/prefix frequencies,
sequence co-occurrences, and conditional prediction difficulty can change.
These weights are report-only item popularity diagnostics; they are not exact
token exposure counts for overlapping expanding-prefix histories. A later
aligned-minus-shuffled retrieval difference should be described as the effect
of changing item-to-code alignment under matched catalog structure, not a pure
semantic effect with frequency-weighted exposure held constant.
