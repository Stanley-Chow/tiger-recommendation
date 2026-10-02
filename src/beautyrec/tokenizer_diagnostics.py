"""Measure tokenizer usage and semantic locality without recommendation training.

A full-code permutation is a capacity-matched control: it preserves every code,
prefix occupancy and token count while changing which product owns each code.
Cosine-neighbor diagnostics use the same frozen metadata vectors and query IDs
for every tokenizer. They do not use interactions or held-out recommendation labels.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from numbers import Integral
from typing import Any

import numpy as np

from beautyrec.semantic_ids import SemanticCodebook


def _content_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    return int(seed)


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def permute_codebook(
    codebook: SemanticCodebook,
    *,
    seed: int = 2026,
) -> tuple[SemanticCodebook, dict[str, Any]]:
    """Assign existing full codes using an ordinary seeded uniform permutation.

    ``item_to_source_item[destination]`` identifies the original owner of the code
    now assigned to that item. Some assignments can stay fixed, as expected under
    a uniform random permutation; their count is reported rather than repaired.
    The input codebook is left intact, including its catalog alignment checksum.

    Manifest checksums hash canonical JSON *content*, not a caller's eventual file
    formatting. ``manifest_content_sha256`` excludes that checksum field itself.
    """
    seed = _seed(seed)
    permutation = np.random.default_rng(seed).permutation(len(codebook.item_codes)).tolist()
    source_hash = _content_hash(codebook.to_dict())
    manifest = {
        "control": "seeded uniform full-code permutation",
        "seed": seed,
        "items": len(permutation),
        "catalog_sha256": codebook.catalog_sha256,
        "source_codebook_content_sha256": source_hash,
        "item_to_source_item": permutation,
        "fixed_points": sum(item == source for item, source in enumerate(permutation)),
        "mapping_content_sha256": _content_hash(permutation),
        "preserves": ["full code set", "vocabulary", "token counts", "prefix occupancy"],
    }
    manifest["manifest_content_sha256"] = _content_hash(manifest)
    payload = codebook.to_dict()
    payload["item_codes"] = [list(codebook.item_codes[source]) for source in permutation]
    payload["diagnostics"] = {
        **payload["diagnostics"],
        "method": "permuted-full-code-control",
        "source_method": codebook.diagnostics.get("method"),
        "permutation_seed": seed,
        "permutation_fixed_points": manifest["fixed_points"],
        "permutation_mapping_content_sha256": manifest["mapping_content_sha256"],
        "source_codebook_content_sha256": source_hash,
    }
    return SemanticCodebook(**payload), manifest


def _entropy(counts: np.ndarray) -> tuple[float, float]:
    probabilities = counts / counts.sum()
    entropy_bits = float(-np.sum(probabilities * np.log2(probabilities)))
    return entropy_bits, float(2**entropy_bits)


def interaction_weighted_diagnostics(
    codebook: SemanticCodebook, popularity: np.ndarray
) -> dict[str, Any]:
    """Measure token/prefix exposure under original training interaction counts.

    Whole-code permutation preserves catalog occupancy, but popular products can
    move between code groups. These descriptive statistics expose that difference.
    Counts must come from the prepared training interactions, rather than repeated
    expanding-prefix examples or validation/test events. They do not change IDs,
    model fitting, checkpoint selection, or the unweighted metadata diagnostics.
    """
    raw_counts = np.asarray(popularity)
    if (
        raw_counts.shape != (len(codebook.item_codes),)
        or not np.issubdtype(raw_counts.dtype, np.number)
        or np.issubdtype(raw_counts.dtype, np.complexfloating)
    ):
        raise ValueError("Popularity needs one real training interaction count per item.")
    counts = raw_counts.astype(np.float64)
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts != np.floor(counts))
        or not np.isfinite(counts.sum())
        or counts.sum() <= 0
    ):
        raise ValueError("Training counts must be finite nonnegative integers with positive total.")
    codes = np.asarray(codebook.item_codes, dtype=np.int64)

    def exposure(groups: np.ndarray) -> dict[str, Any]:
        totals = np.bincount(groups, weights=counts)
        positive = totals[totals > 0]
        entropy, perplexity = _entropy(positive)
        return {
            "groups_with_positive_training_count": len(positive),
            "entropy_bits": entropy,
            "perplexity": perplexity,
            "largest_group_interaction_share": float(positive.max() / positive.sum()),
        }

    positions, prefixes = [], []
    for position in range(codebook.code_length):
        tokens, groups = np.unique(codes[:, position], return_inverse=True)
        positions.append(
            {"position": position + 1, "catalog_used_tokens": len(tokens), **exposure(groups)}
        )
    for length in range(1, codebook.code_length):
        values, groups = np.unique(codes[:, :length], axis=0, return_inverse=True)
        prefixes.append(
            {"prefix_length": length, "catalog_occupied_prefixes": len(values), **exposure(groups)}
        )
    return {
        "weighting": "original deduplicated training interactions, counted once",
        "items": len(counts),
        "training_interactions": int(counts.sum()),
        "training_counts_values_sha256": hashlib.sha256(counts.astype("<f8").tobytes()).hexdigest(),
        "positions": positions,
        "cumulative_prefixes": prefixes,
    }


def _capacities(codebook: SemanticCodebook) -> tuple[list[int] | None, int | None]:
    """Use declared level ranges; never treat observed tokens as available capacity."""
    semantic_depth = codebook.code_length - 1
    capacities = codebook.diagnostics.get("semantic_vocab_sizes")
    if capacities is None and codebook.depth == semantic_depth and codebook.branching_factor > 0:
        capacities = [codebook.branching_factor] * semantic_depth
    if capacities is not None:
        if not isinstance(capacities, list) or len(capacities) != semantic_depth:
            raise ValueError("Declared semantic vocabulary sizes must match the semantic depth.")
        capacities = [_positive(value, "semantic vocabulary size") for value in capacities]
        inferred_offset = 3 + sum(capacities)
        offset = codebook.diagnostics.get("suffix_offset", inferred_offset)
        if offset != inferred_offset or offset >= codebook.vocab_size:
            raise ValueError("Suffix offset must follow the declared semantic vocabulary ranges.")
        return capacities, int(offset)
    offset = codebook.diagnostics.get("suffix_offset")
    if offset is not None and (
        isinstance(offset, bool)
        or not isinstance(offset, Integral)
        or not 3 <= offset < codebook.vocab_size
    ):
        raise ValueError("Declared suffix_offset is outside the vocabulary.")
    return None, int(offset) if offset is not None else None


def _usage(codes: np.ndarray, codebook: SemanticCodebook) -> tuple[list[dict], dict]:
    capacities, suffix_offset = _capacities(codebook)
    positions = []
    level_offset = 3
    for position in range(codebook.code_length):
        token_ids, counts = np.unique(codes[:, position], return_counts=True)
        is_suffix = position == codebook.code_length - 1
        if is_suffix:
            offset = suffix_offset
            capacity = codebook.vocab_size - offset if offset is not None else None
        else:
            capacity = capacities[position] if capacities is not None else None
            offset = level_offset if capacity is not None else None
            if capacity is not None:
                level_offset += capacity
        if offset is not None and (
            int(token_ids.min()) < offset or int(token_ids.max()) >= offset + capacity
        ):
            raise ValueError("Observed codes fall outside their declared position vocabulary.")
        entropy_bits, perplexity = _entropy(counts)
        positions.append(
            {
                "position": position + 1,
                "role": "suffix" if is_suffix else "semantic",
                "used_tokens": len(token_ids),
                "available_capacity": capacity,
                "utilization": len(token_ids) / capacity if capacity is not None else None,
                "entropy_bits": entropy_bits,
                "perplexity": perplexity,
                "normalized_perplexity": perplexity / capacity if capacity is not None else None,
                "largest_token_share": float(counts.max() / counts.sum()),
                "token_counts": {str(token): int(count) for token, count in zip(token_ids, counts)},
            }
        )
    suffix_ids = np.unique(codes[:, -1])
    available = codebook.vocab_size - suffix_offset if suffix_offset is not None else None
    suffix = {
        "offset": suffix_offset,
        "used_tokens": len(suffix_ids),
        "minimum_used_token": int(suffix_ids.min()),
        "maximum_used_token": int(suffix_ids.max()),
        "observed_token_span": int(suffix_ids.max() - suffix_ids.min() + 1),
        "slots_through_maximum": int(suffix_ids.max() - suffix_offset + 1)
        if suffix_offset is not None
        else None,
        "available_capacity": available,
        "unused_capacity": available - len(suffix_ids) if available is not None else None,
        "utilization": len(suffix_ids) / available if available is not None else None,
        "items_needing_nonzero_suffix_fraction": float(np.mean(codes[:, -1] != suffix_offset))
        if suffix_offset is not None
        else None,
    }
    return positions, suffix


def _prefix_statistics(codes: np.ndarray, capacities: list[int] | None) -> tuple[list[dict], list]:
    statistics, memberships = [], []
    items = len(codes)
    potential_capacity = 1
    for length in range(1, codes.shape[1]):
        _, groups, counts = np.unique(
            codes[:, :length], axis=0, return_inverse=True, return_counts=True
        )
        memberships.append((groups, counts))
        collision_pairs = int(np.sum(counts * (counts - 1)) // 2)
        entropy_bits, perplexity = _entropy(counts)
        if capacities is not None:
            potential_capacity *= capacities[length - 1]
        statistics.append(
            {
                "prefix_length": length,
                "occupied_prefixes": len(counts),
                "available_prefix_capacity": potential_capacity if capacities is not None else None,
                "occupancy_fraction": len(counts) / potential_capacity
                if capacities is not None
                else None,
                "singleton_prefixes": int(np.sum(counts == 1)),
                "collision_prefixes": int(np.sum(counts > 1)),
                "items_in_collision_groups": int(counts[counts > 1].sum()),
                "excess_items_without_suffix": items - len(counts),
                "unordered_collision_pairs": collision_pairs,
                "largest_group": int(counts.max()),
                "p95_occupied_group_size": float(np.percentile(counts, 95)),
                "mean_occupied_group_size": items / len(counts),
                "occupancy_histogram": {
                    str(size): count for size, count in sorted(Counter(counts.tolist()).items())
                },
                "entropy_bits": entropy_bits,
                "perplexity": perplexity,
                "random_distinct_item_pair_match_probability": 2
                * collision_pairs
                / (items * (items - 1))
                if items > 1
                else None,
            }
        )
    return statistics, memberships


def _neighbors(vectors: np.ndarray, query_ids: np.ndarray, k: int, batch_size: int):
    """Exact cosine neighbors using at most batch_size by catalog_size scores.

    Self matches are excluded. Stable descending sorting breaks equal cosine
    scores by item index, so duplicate vectors have a reproducible ordering.
    """
    for start in range(0, len(query_ids), batch_size):
        batch_ids = query_ids[start : start + batch_size]
        similarities = vectors[batch_ids] @ vectors.T
        similarities[np.arange(len(batch_ids)), batch_ids] = -np.inf
        neighbor_ids = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        yield batch_ids, neighbor_ids, np.take_along_axis(similarities, neighbor_ids, axis=1)


def tokenizer_diagnostics(
    codebook: SemanticCodebook,
    embeddings: np.ndarray,
    *,
    sample_size: int = 1024,
    neighbors: int = 10,
    batch_size: int = 128,
    seed: int = 2026,
) -> dict[str, Any]:
    """Report code usage, collisions before the suffix, and cosine-neighbor agreement.

    The last code position is a uniqueness suffix; all earlier positions are
    semantic levels. Token utilization uses declared capacity, including padded
    suffix slots. Capacity is null when a custom codebook does not declare ranges.

    For each cumulative semantic prefix, compare the observed neighbor-match
    fraction to the exact probability of matching a uniformly selected *other*
    item for the same sampled queries. This adjusts for unequal prefix sizes and
    excludes self matches. The global random-pair probability is reported too.
    Enrichment is observed/chance, and is null when chance is zero. This describes
    metadata locality, not recommendation quality or a statistical significance test.

    Query IDs are selected with a fixed seed. The catalog is never reduced to the
    query sample: each query searches all other products. Only one bounded batch
    similarity matrix is allocated, avoiding a full catalog-by-catalog matrix.
    """
    sample_size, neighbors, batch_size = (
        _positive(sample_size, "sample_size"),
        _positive(neighbors, "neighbors"),
        _positive(batch_size, "batch_size"),
    )
    seed = _seed(seed)
    if codebook.code_length < 2:
        raise ValueError("Semantic diagnostics require at least one semantic token and a suffix.")
    raw_vectors = np.asarray(embeddings)
    if (
        raw_vectors.ndim != 2
        or raw_vectors.shape[0] != len(codebook.item_codes)
        or not raw_vectors.shape[1]
    ):
        raise ValueError("Embeddings need one nonempty vector per catalog item.")
    if (
        not np.issubdtype(raw_vectors.dtype, np.number)
        or np.issubdtype(raw_vectors.dtype, np.complexfloating)
        or not np.isfinite(raw_vectors).all()
    ):
        raise ValueError("Embeddings must contain finite real numeric values.")
    vectors = np.asarray(raw_vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Cosine similarity requires finite nonzero vector norms.")
    vectors = vectors / norms[:, None]
    codes = np.asarray(codebook.item_codes, dtype=np.int64)
    positions, suffix = _usage(codes, codebook)
    capacities, _ = _capacities(codebook)
    prefixes, memberships = _prefix_statistics(codes, capacities)
    query_ids = np.sort(
        np.random.default_rng(seed).choice(len(codes), min(sample_size, len(codes)), replace=False)
    )
    effective_k = min(neighbors, len(codes) - 1)
    match_counts = np.zeros(len(prefixes), dtype=np.int64)
    cosine_sum = 0.0
    for batch_ids, neighbor_ids, similarities in _neighbors(
        vectors, query_ids, effective_k, batch_size
    ):
        cosine_sum += float(similarities.sum())
        for level, (groups, _) in enumerate(memberships):
            match_counts[level] += int(np.sum(groups[batch_ids, None] == groups[neighbor_ids]))
    total_pairs = len(query_ids) * effective_k
    agreement = []
    for level, (groups, counts) in enumerate(memberships):
        observed = float(match_counts[level] / total_pairs) if total_pairs else None
        chance = (
            float(np.mean((counts[groups[query_ids]] - 1) / (len(codes) - 1)))
            if len(codes) > 1
            else None
        )
        agreement.append(
            {
                "prefix_length": level + 1,
                "matching_neighbor_pairs": int(match_counts[level]),
                "observed_match_fraction": observed,
                "random_other_item_match_probability": chance,
                "catalog_random_pair_match_probability": prefixes[level][
                    "random_distinct_item_pair_match_probability"
                ],
                "observed_minus_chance": observed - chance if total_pairs else None,
                "agreement_enrichment": observed / chance if total_pairs and chance > 0 else None,
            }
        )
    return {
        "items": len(codes),
        "code_length": codebook.code_length,
        "semantic_levels": codebook.code_length - 1,
        "vocab_size": codebook.vocab_size,
        "catalog_sha256": codebook.catalog_sha256,
        "codebook_content_sha256": _content_hash(codebook.to_dict()),
        "embedding_dimensions": raw_vectors.shape[1],
        "normalized_embedding_values_sha256": hashlib.sha256(
            vectors.astype("<f8").tobytes()
        ).hexdigest(),
        "positions": positions,
        "cumulative_prefixes": prefixes,
        "leaf_before_suffix": prefixes[-1],
        "suffix_vocabulary": suffix,
        "full_code_collisions": len(codes) - len({tuple(code) for code in codebook.item_codes}),
        "cosine_neighbors": {
            "requested_queries": sample_size,
            "queries": len(query_ids),
            "query_item_ids": query_ids.tolist(),
            "requested_neighbors": neighbors,
            "neighbors": effective_k,
            "pairs": total_pairs,
            "batch_size": batch_size,
            "seed": seed,
            "mean_cosine_similarity": cosine_sum / total_pairs if total_pairs else None,
            "self_matches": "excluded",
            "tie_breaker": "lower catalog item index",
            "search_catalog": "all other items; query sampling does not sample candidates",
            "prefix_agreement": agreement,
        },
    }
