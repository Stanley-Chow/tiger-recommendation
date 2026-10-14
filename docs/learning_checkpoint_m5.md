# M5 learning and code-review checkpoint

Read this after the completed M5 results and independent audit. The aim is to
understand the project well enough to explain its choices and change its code
without relying on a training script as a black box.

## Suggested reading order

1. [Protocol](m5_protocol.md) and [configuration](../configs/m5.json): identify the
   hypothesis, primary metric, MAP guard, cohort, and fixed comparisons before
   looking at scores.
2. [Data preparation](../src/beautyrec/amazon2023.py): follow one user from the
   official CSV to a strict-past training prefix, a frozen validation history,
   and a first future test target. Explain why five-core filtering and skipping
   unavailable first targets can change the population being evaluated.
3. [Features](../src/beautyrec/m5_features.py),
   [RQ training](../src/beautyrec/rqvae.py), and
   [semantic IDs](../src/beautyrec/semantic_ids.py): distinguish the frozen text
   encoder, learned metadata quantizer, and learned recommendation model.
4. [Model](../src/beautyrec/model.py),
   [training](../src/beautyrec/experiment_training.py), and
   [retrieval](../src/beautyrec/retrieval.py): explain teacher forcing, causal
   decoder masks, catalog-constrained beam search, and complete seen exclusions.
5. [SASRec](../src/beautyrec/sasrec.py) and
   [content/fusion](../src/beautyrec/content_retrieval.py): distinguish sequence
   learning from metadata similarity, and explain which branch can emit a new item.
6. [M5 orchestration](../src/beautyrec/m5_benchmark.py),
   [evaluation](../src/beautyrec/evaluation.py), and
   [audit](../scripts/verify_m5_results.py): trace tokenizer freezing, checkpoint
   selection, test evaluation, paired intervals, and seed aggregation.
7. [Serving](../src/beautyrec/serving.py) and
   [end-to-end checks](../tests/test_m5.py): run an actual request and locate every
   validation step before it reaches the model.

## Questions to answer together

- Given a three-level RQ code plus a collision suffix, what part expresses
  metadata similarity and what part identifies a particular training item?
- Why does assigning a semantic prefix to a new item fail to teach the
  generator its previously unseen collision suffix?
- How does the paper's explicit semantic-prefix expansion differ from M5's
  closed warm trie and independent content retrieval route?
- Why is MAP@20 equal to MRR@20 here? How would the definition change with
  several relevant future items per user?
- Why are a semantic-token cross entropy and a next-item cross entropy not
  directly comparable training-loss numbers?
- Why do conditional Recall@100 and first-target eligibility coverage answer
  different questions? What would an unconditional hit rate measure?
- What does a user-bootstrap interval condition on? Why should three training
  seeds not be pooled as three times as many independent users?
- Why is SASRec a useful practical baseline even though its item embedding and
  output layer make its parameter count different from the generator?
- What can the metadata snapshot limitation invalidate? What evidence would
  remove that concern in a production log?
- What would change if the candidate catalog or seen history changed between
  validation, offline test, and online serving?

## A short live review exercise

Run the offline demo. Pick one synthetic user and manually calculate its warm
and new-item candidate sets, a top-20 reciprocal rank, and one RRF score. Then
trace that same user through the saved JSONL rankings and an API response.
Change a checksum-protected artifact and confirm that loading fails. Finally,
explain the actual M5 result in two sentences, including a limitation and the
next experiment that would resolve the largest uncertainty.

Further experiments belong in a new frozen protocol. Completing the software
does not establish that a generative architecture beats a conventional model,
or that an offline gain improves user satisfaction.

The [post-M5 research options](after_m5.md) explain a direct generative cold-item
extension, possible tokenizer/ranker upgrades, and when another dataset would
add useful evidence. These options follow the learning checkpoint and a
separately frozen experiment plan.
