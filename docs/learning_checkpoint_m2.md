# M2 learning checkpoint

We stop implementation at M2 and work through the system before adding RQ-VAE,
a separate reranker, or the larger dataset. A negative comparison is a valid
experiment result; it should be explained rather than hidden.

**Reached on 2026-10-03.** All three seeds and full evaluations are complete.
Start with [the actual M2 results](m2_results.md), then follow the walkthrough below.
Saved checkpoints and per-user rankings are available locally under `results/m2/`.

## What M2 must produce

- Three independent training seeds for the generative and SASRec baselines.
- The same warm catalog, eligible users, full-history exclusions, and metric cutoffs.
- Full test evaluation, with individual rankings and reproducible configurations.
- Means and standard deviations across seeds, plus paired user-bootstrap intervals.
- A controlled continuation comparison: AdamW for three epochs, SAM for three
  epochs, and AdamW for six epochs from the same selected base weights.
- Separate training cost, batched throughput, warm single-request latency, and
  peak allocated CUDA memory. Extra SAM work is part of the result.

Training selects checkpoints using a fixed validation cohort. Recall@100 is the
selection metric, subject to a MAP@20 floor equal to validation popularity. If
no checkpoint meets that floor, the best fallback is saved and clearly marked.
The test set does not choose the checkpoint, learning rate, or beam width.

## Walkthrough order

1. **Data:** take one user, sort their reviews, and identify each expanding
   training prefix and the final validation/test targets. Explain the difference
   between the encoder's recent 20 items and the complete seen-item exclusions.
2. **Item representations:** inspect product text, its frozen MiniLM vector, and
   its four-token code. Explain why shared prefixes and unique final codes have
   different responsibilities.
3. **Generative training:** write the decoder input and labels for one target.
   Identify causal masks, padding masks, token cross entropy, and the optimizer
   update. MiniLM is frozen; the recommendation Transformer is trained.
4. **Retrieval:** trace one beam expansion, including the valid-code trie,
   excluded seen items, full-vocabulary log probabilities, and beam pruning.
   Explain why a valid output is not necessarily a relevant output.
5. **SASRec:** compare predicting an item directly with predicting its code.
   Explain input item-ID shifting, causal attention, the final history state,
   tied item embeddings, and full-catalog softmax. Our baseline adapts the
   original SASRec training objective; it is not an exact paper reproduction.
6. **Metrics:** calculate Recall, NDCG, and AP for targets at ranks 1, 5, and 25.
   Explain why MAP equals MRR with one relevant next item and why increasing K
   alone is not evidence of a better model.
7. **SAM:** explain the first gradient, norm-bounded perturbation, second gradient,
   exact restoration, and AdamW update. Both passes see the same minibatch and
   dropout masks. Distinguish extra epochs from extra backward passes.
8. **Results:** use saved per-user rankings to explain a success and failure for
   each model. Compare uncertainty, tail coverage, latency, and parameter counts.

## Questions you should be able to answer

- What exactly is observed, and what do we predict? Are review events purchases?
- Which information is available at training time and at prediction time?
- Does the per-user split provide a strict global time boundary?
- Why can training loss fall while recommendation metrics stay poor?
- Can a ranker recover an item missing from the retrieval candidates?
- Does SAM help consistently, or only for one seed? What additional cost does it incur?
- What evidence would justify the next change, and what result would reject it?

The results report records achieved evidence. This checklist describes the
completion requirements and the learning topics to work through before M3.
