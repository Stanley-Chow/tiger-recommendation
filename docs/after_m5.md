# After M5: choose the next question before changing the system

Finish the M5 artifact audit and learning checkpoint before starting another
training study. The current project already covers two data stages: the original
2014 Beauty benchmark and a larger temporal Amazon 2023 cohort. Another dataset
is not required to make the current software usable or the experiments reviewable.

## First research extension: direct semantic-prefix cold-start retrieval

The M5 generator's trie contains warm items only. Its new-item recall is therefore
zero by construction; the content branch supplies new candidates. This is a
deliberate implementation boundary, not a statement that generative retrieval
cannot recommend unseen items.

The original [TIGER paper, Section 4.3](https://arxiv.org/html/2305.05065v3#S4.SS3),
encodes unseen products with the fitted quantizer and expands generated results
to unseen items sharing their first three semantic tokens. A quota controls the
unseen-item share. This is a concrete next mechanism to test using the existing
Amazon 2023 data and saved warm-only quantizers.

For a new protocol, keep warm semantic IDs and checkpoints fixed, encode the
available new-item metadata without fitting on its interactions, and build a
prefix-to-item index. Compare prefix expansion with the current cosine route
and fixed fusion. Choose quotas or within-prefix ranking rules using a separately
defined pre-test cold validation window. Preserve all eligible targets, including
new items that the new mechanism fails to reach. Report both warm and new-item
quality, first-target coverage, candidate counts, and latency. Do not describe
the existing M5 2023 test as a fresh untouched confirmation after inspecting it.

Simply allocating a collision suffix does not establish that the sequence model
learned a new product. Prefix expansion needs an explicit candidate lookup and
ranking policy; warm IDs must remain stable during catalog updates. Those are
useful mathematical and engineering questions for the code-review session.

## Then improve the bottleneck supported by the results

If semantic buckets are too coarse, compare finer quantization or another
semantic level with a capacity-controlled generator. M4 supports RQ-32 over
RQ-16 in its tested 2014 setting; it does not establish that 32 is optimal for
38,336 warm products. Measure collisions, neighbor preservation, reconstruction,
and recommendation quality together.

If metadata similarity poorly reflects behavior, test a collaborative objective
for representation/tokenizer alignment while retaining a content route for
untrained items. A reconstruction improvement alone is not evidence of better
ranking. Keep data and evaluation fixed while changing this component.

If candidate Recall@200 is healthy but MAP@20 is weak, a learned second-stage
ranker is a practical product extension. Train it with candidates generated
without leaking training labels or forcing targets into the list. Measure the
candidate-recall ceiling before attributing a ranking failure to the ranker.

Use validation to select an extension and freeze its protocol before final
comparison. Changing the dataset, encoder, quantizer, sequence model and ranking
policy together would make the source of a gain difficult to identify.

## A later dataset and model direction

[KuaiRand](https://kuairand.com/) provides recommendation exposure logs,
randomized exposures, timestamps, and feedback including clicks and watch time.
Its [official repository](https://github.com/chongminggao/KuaiRand) also lists
video content supplements. That makes it a candidate for a later study of
exposure-aware ranking, multiple feedback signals and longer histories, beyond
Amazon's next-review proxy. Start with a bounded version and verify the content
join and evaluation policy before committing substantial compute.

For a longer-history study, [Meta's HSTU implementation](https://github.com/meta-recsys/generative-recommenders)
is a model candidate. Treat it as a separate sequential-model comparison with
its own resource budget. The current small Transformer and 20-item history do
not establish a need for a larger foundation model.

The immediate recommendation is to review the completed evidence, then choose
one extension on the existing dataset. A later untouched category or logged
exposure dataset can test whether that extension generalizes.
