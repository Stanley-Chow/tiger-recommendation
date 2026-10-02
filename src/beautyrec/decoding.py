"""Catalog-constrained beam search over semantic IDs.

The trie stores valid item-code prefixes. It prevents the decoder from inventing
codes and lets us remove previously seen items before pruning beam candidates.
"""

import torch

from beautyrec.model import GenerativeRecommender


class SemanticTrie:
    """An index of valid codes and the catalog items below each prefix."""

    def __init__(self, item_codes: list[list[int]], eos_token: int = 2) -> None:
        if not item_codes or len({tuple(code) for code in item_codes}) != len(item_codes):
            raise ValueError("The catalog needs nonempty, unique item codes.")
        lengths = {len(code) for code in item_codes}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError("This decoder expects fixed-length, nonempty codes.")
        self.code_length = lengths.pop()
        self.eos_token = eos_token
        self.items: dict[tuple[int, ...], int] = {}
        self.children: dict[tuple[int, ...], set[int]] = {}
        self.descendants: dict[tuple[int, ...], set[int]] = {}
        for item, code in enumerate(item_codes):
            if any(token < 3 for token in code):
                raise ValueError("Item codes may not use PAD, BOS, or EOS.")
            self.items[tuple(code)] = item
            for length in range(len(code) + 1):
                prefix = tuple(code[:length])
                self.descendants.setdefault(prefix, set()).add(item)
                next_token = code[length] if length < len(code) else eos_token
                self.children.setdefault(prefix, set()).add(next_token)

    def allowed_next(
        self,
        prefix: tuple[int, ...],
        exclude_items: set[int] | None = None,
    ) -> list[int]:
        excluded = exclude_items or set()
        candidates = sorted(self.children.get(prefix, ()))
        return [
            token
            for token in candidates
            if self.descendants.get(prefix if token == self.eos_token else (*prefix, token), set())
            - excluded
        ]


@torch.inference_mode()
def constrained_beam_search(
    model: GenerativeRecommender,
    source_tokens: torch.Tensor,
    trie: SemanticTrie,
    top_k: int = 10,
    beam_size: int = 20,
    exclude_items: set[int] | None = None,
    bos_token: int = 1,
) -> list[tuple[int, float]]:
    """Return unique (item_id, log_probability) pairs from a single user history.

    Scores retain the full-vocabulary model probabilities. Constraints prune
    impossible continuations without renormalizing probabilities at each node.
    Beam search is approximate retrieval, not exhaustive scoring of every item.
    """
    if top_k < 1 or beam_size < top_k:
        raise ValueError("Require beam_size >= top_k >= 1.")
    if source_tokens.ndim == 1:
        source_tokens = source_tokens.unsqueeze(0)
    if source_tokens.ndim != 2 or source_tokens.size(0) != 1:
        raise ValueError("Beam search accepts one history at a time.")
    model.eval()
    memory, source_padding = model.encode(source_tokens)
    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for _ in range(trie.code_length + 1):
        decoder_tokens = torch.tensor(
            [[bos_token, *prefix] for prefix, _ in beams],
            dtype=torch.long,
            device=source_tokens.device,
        )
        count = len(beams)
        logits = model.decode(
            decoder_tokens,
            memory.expand(count, -1, -1),
            source_padding.expand(count, -1),
        )[:, -1]
        log_probabilities = logits.log_softmax(dim=-1).cpu()
        candidates = []
        for row, (prefix, score) in enumerate(beams):
            for token in trie.allowed_next(prefix, exclude_items):
                candidates.append(((*prefix, token), score + float(log_probabilities[row, token])))
        if not candidates:
            return []
        candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))
        beams = candidates[:beam_size]
    return [(trie.items[prefix[:-1]], score) for prefix, score in beams[:top_k]]
