**M5 architecture and evaluation structure.** A pinned, frozen MiniLM encoder
maps static product metadata to unit vectors. A separate residual quantization
autoencoder fits warm-item metadata only and assigns three semantic tokens plus
a collision-resolving suffix. A fresh Transformer predicts warm-item codes from
chronological histories; a SASRec-style full-softmax model is the practical
baseline. Their closed-catalog candidate lists use full seen exclusions.
Frozen mean-history cosine retrieval scores the available warm/new catalog.
Two fixed equal-weight reciprocal-rank fusions combine this content route with
either sequence branch. New-item capability belongs to the content route;
new collision suffixes are not learned by the closed generator. A prespecified
seed supplies a checksum-verified inference bundle and local CLI/HTTP API.
Three seeds and paired user-bootstrap intervals quantify complementary sources
of uncertainty. The released metadata is a 2023 snapshot with unverified
historical field availability. Dashed teal arrows denote frozen feature reuse.
