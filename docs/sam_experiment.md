# Controlled SAM continuation experiment

Here SAM means **Sharpness-Aware Minimization**. The existing pretrained MiniLM
encoder supplies frozen item features. The recommendation Transformer is initialized
randomly and trained on user interactions. This experiment continues that trained
recommender; it does not fine-tune MiniLM or reproduce a pretrained language-model
fine-tuning result.

## Method and source

[Foret et al., ICLR 2021](https://arxiv.org/abs/2010.01412) approximate a neighborhood
loss by perturbing weights in the direction of the first minibatch gradient:

```text
g1 = gradient of loss(batch, w)
epsilon = rho * g1 / global_L2_norm(g1)
g2 = gradient of loss(the same batch, w + epsilon)
w_next = AdamW_update(w, g2)
```

Zero first gradients produce zero perturbation. This project uses non-adaptive SAM,
with default `rho=0.05`. The original paper describes applying a standard optimizer
to the second gradient; AdamW is the base optimizer in this experiment. Its
decoupled weight decay occurs in the final base update. The authors'
[reference implementation](https://github.com/google-research/sam) is JAX; our
`beautyrec.optimizers.SAM` is a readable PyTorch adaptation.

The related [Bahri et al., ACL 2022](https://aclanthology.org/2022.acl-long.508/)
studies SAM on language models and motivates an experiment. Its reported gains do
not establish gains for this recommender or imply that frozen MiniLM is being tuned.

## Integration

```python
base = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = SAM(base, rho=0.05)
optimizer.zero_grad()
try:
    loss = batch_loss()
    loss.backward()
    optimizer.first_step()  # Clears g1; do not clip it before the perturbation.
    perturbed_loss = batch_loss()  # Reuse the same source, target, and labels.
    perturbed_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
    optimizer.second_step()  # Restores w exactly before applying the base update.
except BaseException:
    optimizer.restore()
    raise
```

Check both losses for finiteness before backward. The wrapper separately rejects
non-finite gradients. It holds exact copies of original parameters rather than
subtracting the perturbation, and restores every stored parameter even when its
second gradient is absent. `restore()` is idempotent. Checkpoint only between
complete updates; the state dictionary contains rho and the base optimizer state.

With dropout enabled, replay the first-forward CPU/CUDA RNG state for the second
forward so both passes use the same dropout masks. This also keeps the random stream
aligned with AdamW, which has only one forward per batch. Use an independent seeded
DataLoader generator for batch ordering. The wrapper itself does not control model
randomness or buffer updates; the current Transformer has no BatchNorm running state.

## Comparison protocol

Start each continuation arm from the identical best base-model checkpoint, and
record its SHA-256. Every M2 arm loads the selected model weights and starts with
fresh AdamW moment state. The saved base optimizer state is available for
inspection. This experiment uses a weight warm start, with the same reset in
the three-epoch AdamW, three-epoch SAM, and six-epoch AdamW arms.

Compare three additional AdamW epochs with three additional SAM epochs, holding
data, batches, learning rate, weight decay, seed, model, gradient clipping,
validation subset, and checkpoint-selection metric fixed. Keep the uncontinued
base checkpoint as a reference. Select checkpoints using validation only and
evaluate the chosen checkpoints on the same full-catalog test set.

SAM makes two forward/backward passes for each base update. Report base-update
counts, backward-pass counts, and wall-clock training time separately from decoding
latency. An optional six-epoch AdamW continuation approximately matches SAM's
backward-pass budget; it is not an exact time or FLOP match. The original SAM paper
likewise provides longer non-SAM controls because each SAM update uses two
backpropagations.

Report Recall/NDCG, tail-item recall, coverage, and paired uncertainty alongside
training cost. A single seed is an initial experiment, not proof of a general
improvement. Interpret a worse SAM result as evidence for this configuration, and
preserve it; do not assert better generalization solely because SAM was used.
