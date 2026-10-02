# M3 frozen protocol: semantic alignment and learned residual quantization

Frozen design dated 2026-10-03, before M3 recommendation results. This is a
repository protocol, not an external preregistration. Status: planned; completion
requires the evidence below. M2 test results are already known, so this is a
prospective comparison on a previously evaluated dataset, not a new blinded test.

## Material Passport

Scope: Amazon Reviews 2014 Beauty, warm-item next-item retrieval. Inputs are the
unchanged M2 processed data and frozen MiniLM vectors. Research questions: does
content-to-code alignment help retrieval, and how does a bounded learned RQ
tokenizer compare with hierarchical k-means (HKM)? No results are asserted here.

## Data and fixed model

The source has 198,502 reviews from 22,363 users. The warm catalog has 12,068
items; expanding prefixes yield 131,413 training examples, 22,312 eligible
validation users, and 22,225 eligible test users. Cold targets dropped: 51
validation and 138 test. Every user's final two distinct items are held out in
chronological order. This is a per-user split, not a global time cutoff.

The existing 384-dimensional, normalized `all-MiniLM-L6-v2` vectors encode title,
brand, categories, and description for the training catalog. MiniLM remains
frozen. No review text, ratings, or held-out recommendation labels fit tokenizers.

Every item has three semantic tokens and one deterministic within-prefix suffix.
The model sees the latest 20 warm items, in time order: `BOS + item codes + EOS`,
at most 82 source tokens. Encoder and decoder share input embeddings;
sinusoidal positions are fixed and the output projection has separate weights.
All arms use d=128, four heads, two encoder/two decoder layers, FFN=256, dropout=.1.

## Four arms and capacity control

| Arm | Codes assigned to items | Purpose |
|---|---|---|
| HKM | Existing three-level, branching-16 hierarchical codes | Semantic baseline |
| HKM-permuted | Bijection of HKM's complete four-token codewords | HKM alignment control |
| RQ | Learned three-level, 16-entry residual codebooks | Learned tokenizer |
| RQ-permuted | Bijection of the corresponding RQ codewords | RQ alignment control |

Use paired recommender seeds 42, 43, 44. RQ tokenizer seed equals its recommender
seed; the corresponding RQ and RQ-permuted arms share that tokenizer. Permutation
seeds are 10000 plus the training seed: 10042/10043/10044. Within each seed, use
the same item permutation for HKM and RQ; save mappings before training. Permute complete
codewords, including the suffix; never shuffle individual positions separately.
Each pair preserves the exact codeword set, trie topology, catalog prefix/suffix
occupancy, and uniqueness. It changes item-content alignment and can also change
interaction-frequency-weighted token occupancy; report that distinction.

Pretrain all three RQ tokenizers before any recommender. Let M be the largest
three-token collision group across HKM and all three final RQ artifacts. Reserve
PAD/BOS/EOS=0/1/2, three disjoint 16-token ranges starting at 3/19/35, and a common
suffix range starting at 51. Set **V_common=51+M** in every arm, including unused
suffix tokens. Train all 12 recommenders from fresh weights with this common V;
do not reuse smaller M2 weights. Generator parameter counts must match exactly.

Since 16^3=4,096 <12,068, suffixes are essential, rather than rare collision fixes.
Matching V and code length controls generator capacity but does not equalize
collision distributions, search difficulty, or tokenizer capacity across HKM/RQ.

## RQ objective and bounded fitting

RQ uses a 384 -> 256 -> 128 -> 32 encoder and a mirrored decoder with ReLU hidden
layers and linear outputs. Three separate learned codebooks each contain 16
32-dimensional vectors. This is a deterministic quantized autoencoder; the
historical RQ-VAE name does not imply a Gaussian posterior or KL term here.

For x, write z=E(x), r_1=z and, for levels l=1,2,3,

$$
c_l=\arg\min_k\|r_l-e_{l,k}\|_2^2,\qquad
r_{l+1}=r_l-\operatorname{sg}(e_{l,c_l}),\qquad
q=\sum_{l=1}^3 e_{l,c_l}.
$$

The decoder receives the straight-through value
$q_{\mathrm{ST}}=z+\operatorname{sg}(q-z)$. Train with

$$
\mathcal L_{\mathrm{RQ}}=
\|x-D(q_{\mathrm{ST}})\|_2^2+
\sum_{l=1}^3\left[
\|\operatorname{sg}(r_l)-e_{l,c_l}\|_2^2+
0.25\|r_l-\operatorname{sg}(e_{l,c_l})\|_2^2\right].
$$

Squared norms sum coordinates, then average examples. Detaching preceding selected
vectors makes commitment update the encoder and codebook loss update its current
codebook; reconstruction has one identity straight-through encoder gradient.

Use AdamW, learning rate .001, weight decay 0 (equivalent to Adam here), and
gradient clipping at norm 1.
For each seed, split item metadata 90/10 with fixed split seed 2026. First run 20
continuous-autoencoder warmup epochs on fit rows, then initialize each codebook by
k-means on successive fit-row latent residuals, using up to 4,096 fit items.
Train at most 100 quantized epochs with patience 20; select the earliest minimum
held-out **quantized reconstruction MSE** and record its quantized epoch count.

Then reinitialize with the same seed and a fresh optimizer. Refit on **all** warm
catalog vectors for 20 warmup epochs plus exactly the selected quantized epochs,
with residual k-means initialization repeated on full-fit rows. Final codes come
from this final refit epoch; no second checkpoint selection occurs. Append suffix
indices in catalog order within each three-token group. No EMA or dead-code
recycling is used. No retrieval scores change the schedule, seeds, or artifacts.

This adapts TIGER's reconstruction/codebook/commitment objective, separate residual
codebooks, and collision suffix. TIGER used larger 256-entry codebooks and different
embeddings/training; M3 is not its reproduction. [TIGER primary paper](https://arxiv.org/html/2305.05065v3)
The explicit straight-through/cumulative-commitment routing is consistent with
the original RQ author's implementation, whose EMA/recycling setup differs from
ours. [Original RQ code](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqvae/quantizations.py)

## Recommendation training, selection, and evaluation

All 12 runs use all training examples, batch 256, AdamW lr=.001/weight decay=.01,
gradient clipping 1, maximum 20 epochs, patience 5, and identical paired batches.
Teacher forcing uses decoder input [BOS,c1,c2,c3,u] and labels [c1,c2,c3,u,EOS].
Loss is full-vocabulary token cross entropy ignoring PAD; no trie mask trains it.

Select checkpoints on the same fixed 1,000-user validation cohort, sampling seed
2026: maximize Recall@100 among epochs with MAP@20 at least validation popularity,
then break ties by MAP@20. Guard tolerance is zero. If no epoch passes, save the
best Recall@100 fallback and report guard failure; never discard that seed.

Final evaluation uses all eligible validation/test users, batch 16, beam width
200, and cutoffs 10/20/50/100/200 from one saved top-200 list per user. Encode only
20 recent items, but exclude **all earlier warm seen items** before beam pruning.
Apply full-vocabulary log-softmax before catalog-trie/eligibility masks, without
renormalization; scores sum five token log probabilities, including EOS.
Beam retrieval is approximate. Selected checkpoint hashes precede test retrieval.
Full validation and test scores do not choose epochs, settings, or winning arms.

No new SAM or continuation arms are included. Verified M2 SASRec/popularity results
are contextual comparisons on identical users/targets/exclusions, not new matched
capacity controls. M2's negative SAM finding applies to its tested configuration.

## Diagnostics, uncertainty, and completion

Report fit/held-out/final-refit continuous and quantized reconstruction MSE/cosine,
per-level code usage, entropy/perplexity, largest token share, residual energy,
unique three-token prefixes, collision-size distribution, suffix usage, and
metadata-neighbor shared-prefix rates. Report popularity-weighted prefix entropy
for every arm; unweighted occupancy alone cannot control training exposure.
Fewer than 13/16 active codes at any level is a predefined low-usage warning,
not a validated success threshold or permission for selective retraining.
Abort numerical nonfiniteness or invalid artifacts; retain finite poor outcomes.

Primary Recall@100 contrasts are **HKM minus HKM-permuted** and **RQ minus
RQ-permuted**. RQ minus HKM is secondary. MAP@20 and other cutoffs are descriptive.
Report each paired seed delta and mean +/- sample SD; RQ variation includes both
tokenizer and recommender randomness. Save per-seed 95% paired user-bootstrap
intervals using 1,000 resamples/seed 2026. These condition on fitted models; never
pool seed-by-user rows as independent runs. Three seeds give weak evidence about
training variability: a two-sided exact sign-flip test cannot attain p<.25.
Any formal tests across the two primary contrasts require Holm correction;
unadjusted intervals are not simultaneous guarantees. No significance is promised.

M3 is complete only when all three tokenizers and 12 runs finish validly, common
parameter counts and paired data are verified, full rankings/metrics and planned
contrasts are saved, and source/config/data/tokenizer/checkpoint hashes and costs
are recorded. Record tokenizer fit cost, epochs/updates, training/validation time,
retrieval throughput, warm request latency, and peak CUDA memory. Numerical
failures or missing arms make the run incomplete; finite low usage, failed guards,
negative effects, and inconclusive intervals do not. Report all outcomes.
