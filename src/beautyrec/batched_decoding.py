"""Batched catalog-constrained beam search with cached tensor trie links.

The probability and pruning rules match the single-history reference decoder.
Encoding, decoding, and candidate selection operate on a whole user batch;
Python only prepares seen-item paths and formats the final recommendation lists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from weakref import WeakKeyDictionary

import torch

from beautyrec.decoding import SemanticTrie
from beautyrec.model import GenerativeRecommender


@dataclass
class TensorTrie:
    """Static device tensors plus short CPU paths for seen-item exclusions."""

    links: torch.Tensor
    descendant_counts: torch.Tensor
    leaf_items: torch.Tensor
    item_paths: list[tuple[int, ...]]
    level_widths: tuple[int, ...]


# A weak key avoids keeping a large catalog alive after its evaluator is released.
# A trie is immutable after construction; each device/vocabulary gets one index.
_TRIE_CACHE: WeakKeyDictionary = WeakKeyDictionary()


def _build_tensor_index(trie: SemanticTrie, device: torch.device, vocab_size: int) -> TensorTrie:
    prefixes = sorted(trie.children, key=lambda prefix: (len(prefix), prefix))
    node_ids = {prefix: index for index, prefix in enumerate(prefixes)}
    links = torch.full((len(prefixes), vocab_size), -1, dtype=torch.long)
    counts = torch.empty(len(prefixes), dtype=torch.long)
    leaf_items = torch.full((len(prefixes),), -1, dtype=torch.long)
    item_paths: list[tuple[int, ...]] = [()] * len(trie.items)
    for prefix, node in node_ids.items():
        counts[node] = len(trie.descendants[prefix])
        for token in trie.children[prefix]:
            if not 0 <= token < vocab_size:
                raise ValueError("Trie tokens exceed the model vocabulary.")
            # EOS completes a leaf; retaining its state keeps the item lookup simple.
            next_prefix = prefix if token == trie.eos_token else (*prefix, token)
            links[node, token] = node_ids[next_prefix]
    for code, item in trie.items.items():
        leaf_items[node_ids[code]] = item
        item_paths[item] = tuple(node_ids[code[:length]] for length in range(len(code) + 1))
    level_widths = tuple(
        sum(len(prefix) == level for prefix in prefixes)
        for level in range(1, trie.code_length + 1)
    )
    return TensorTrie(
        links.to(device), counts.to(device), leaf_items.to(device), item_paths,
        (*level_widths, len(trie.items)),
    )


def _tensor_index(trie: SemanticTrie, device: torch.device, vocab_size: int) -> TensorTrie:
    key = (device.type, device.index, vocab_size)
    device_indices = _TRIE_CACHE.setdefault(trie, {})
    if key not in device_indices:
        device_indices[key] = _build_tensor_index(trie, device, vocab_size)
    return device_indices[key]


def _eligible_nodes(index: TensorTrie, exclusions: list[set[int]]) -> torch.Tensor:
    """Remove a seen item's contribution from each ancestor's descendant count."""
    remaining = index.descendant_counts.unsqueeze(0).expand(len(exclusions), -1).clone()
    rows: list[int] = []
    nodes: list[int] = []
    for row, seen in enumerate(exclusions):
        for item in seen:
            # The reference ignores exclusions outside the catalog.
            if isinstance(item, Integral) and 0 <= item < len(index.item_paths):
                path = index.item_paths[int(item)]
                rows.extend([row] * len(path))
                nodes.extend(path)
    if nodes:
        device = remaining.device
        remaining.index_put_(
            (torch.tensor(rows, device=device), torch.tensor(nodes, device=device)),
            -torch.ones(len(nodes), dtype=remaining.dtype, device=device),
            accumulate=True,
        )
    return remaining > 0


def _stable_topk(scores: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select by descending score, breaking ties with the original column order.

    Candidate columns already follow lexicographic code order. A normal topk
    may choose arbitrary equal-score entries at the cutoff. Explicitly retain
    the earliest tied columns, then stably sort just the selected beam.
    """
    threshold = scores.topk(count, dim=1, sorted=False).values.min(dim=1).values
    above = scores > threshold.unsqueeze(1)
    tied = scores == threshold.unsqueeze(1)
    tie_slots = count - above.sum(dim=1)
    selected = above | (tied & (tied.cumsum(dim=1) <= tie_slots.unsqueeze(1)))
    columns = torch.arange(scores.size(1), device=scores.device).unsqueeze(0)
    selected_columns = torch.where(selected, columns, scores.size(1))
    lex_indices = selected_columns.topk(count, dim=1, largest=False, sorted=True).values
    selected_scores = scores.gather(1, lex_indices)
    score_order = selected_scores.argsort(dim=1, descending=True, stable=True)
    return selected_scores.gather(1, score_order), lex_indices.gather(1, score_order)


@torch.inference_mode()
def batch_constrained_beam_search(
    model: GenerativeRecommender,
    source_tokens: torch.Tensor,
    trie: SemanticTrie,
    top_k: int = 200,
    beam_size: int = 200,
    exclude_items: list[set[int]] | None = None,
    bos_token: int = 1,
) -> list[list[tuple[int, float]]]:
    """Return one sorted list of unique (item_id, log_probability) pairs per user.

    Scores sum full-vocabulary log probabilities, including the final EOS token.
    Seen-item removal happens before beam pruning. Batching changes execution,
    not the search objective: retrieval remains an approximate beam search.
    """
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, Integral)
        or top_k < 1
        or isinstance(beam_size, bool)
        or not isinstance(beam_size, Integral)
        or beam_size < top_k
    ):
        raise ValueError("Require integer beam_size >= top_k >= 1.")
    if source_tokens.ndim != 2 or min(source_tokens.shape) < 1:
        raise ValueError("Batched beam search requires a nonempty batch-by-length matrix.")
    batch_size = source_tokens.size(0)
    exclusions = exclude_items if exclude_items is not None else [set() for _ in range(batch_size)]
    if len(exclusions) != batch_size:
        raise ValueError("Provide one seen-item exclusion set per input history.")

    model.eval()
    memory, source_padding = model.encode(source_tokens)
    device = source_tokens.device
    vocab_size = model.config.vocab_size
    index = _tensor_index(trie, device, vocab_size)
    eligible = _eligible_nodes(index, exclusions)
    states = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    score_dtype = torch.float64 if memory.dtype == torch.float64 else torch.float32
    scores = torch.zeros((batch_size, 1), dtype=score_dtype, device=device)
    prefixes = torch.empty((batch_size, 1, 0), dtype=torch.long, device=device)

    for level in range(trie.code_length + 1):
        # Node IDs within each depth follow lexicographic prefix order. This
        # makes flattened parent/token candidates lexicographic as well.
        lex_order = states.argsort(dim=1, stable=True)
        states = states.gather(1, lex_order)
        scores = scores.gather(1, lex_order)
        prefixes = prefixes.gather(1, lex_order.unsqueeze(-1).expand(-1, -1, prefixes.size(2)))
        beam_count = states.size(1)
        decoder_tokens = torch.cat(
            [torch.full_like(states.unsqueeze(-1), bos_token), prefixes], dim=2
        ).reshape(batch_size * beam_count, -1)
        repeated_memory = memory.unsqueeze(1).expand(-1, beam_count, -1, -1).reshape(
            batch_size * beam_count, memory.size(1), memory.size(2)
        )
        repeated_padding = source_padding.unsqueeze(1).expand(-1, beam_count, -1).reshape(
            batch_size * beam_count, source_padding.size(1)
        )
        logits = model.decode(decoder_tokens, repeated_memory, repeated_padding)[:, -1]
        log_probabilities = logits.log_softmax(dim=-1).reshape(batch_size, beam_count, vocab_size)
        next_states = index.links[states]
        child_eligible = eligible.gather(1, next_states.clamp_min(0).reshape(batch_size, -1))
        allowed = (next_states >= 0) & child_eligible.reshape_as(next_states)
        candidate_scores = (scores.unsqueeze(-1) + log_probabilities).masked_fill(
            ~allowed, -math.inf
        )
        flat_scores = candidate_scores.reshape(batch_size, -1)
        next_beam_count = min(int(beam_size), index.level_widths[level], flat_scores.size(1))
        scores, selected = _stable_topk(flat_scores, next_beam_count)
        parents = selected // vocab_size
        tokens = selected % vocab_size
        states = next_states.reshape(batch_size, -1).gather(1, selected).clamp_min(0)
        old_prefixes = prefixes.gather(1, parents.unsqueeze(-1).expand(-1, -1, prefixes.size(2)))
        prefixes = torch.cat([old_prefixes, tokens.unsqueeze(-1)], dim=2)

    items = index.leaf_items[states].cpu().tolist()
    final_scores = scores.cpu().tolist()
    return [
        [
            (item, score)
            for item, score in zip(user_items[:top_k], user_scores[:top_k])
            if item >= 0 and math.isfinite(score)
        ]
        for user_items, user_scores in zip(items, final_scores)
    ]
