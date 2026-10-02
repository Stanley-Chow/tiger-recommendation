"""Ranking metrics with explicit cutoffs and average-precision conventions."""

import math
from collections.abc import Iterable, Sequence
from numbers import Integral

import numpy as np


def validate_cutoffs(cutoffs: Iterable[int]) -> tuple[int, ...]:
    """Return sorted distinct positive integer cutoffs."""
    values = tuple(cutoffs)
    if not values or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1 for value in values
    ):
        raise ValueError("Cutoffs must be a nonempty collection of positive integers.")
    return tuple(sorted({int(value) for value in values}))


def validate_predictions(predictions: Sequence[int]) -> list[int]:
    """Reject repeated candidates instead of counting repeated hits as relevance."""
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in predictions):
        raise ValueError("Predictions must contain integer item IDs.")
    result = [int(item) for item in predictions]
    if len(set(result)) != len(result):
        raise ValueError("Predictions must not contain duplicate items.")
    return result


def next_item_metrics(
    predictions: Sequence[int], target: int, cutoffs: Iterable[int]
) -> dict[str, float]:
    """Compute all cutoffs from the same ranked list for one positive next item.

    AP@K and reciprocal rank@K both equal 1/rank when the target is within K,
    and zero otherwise. Averaging AP over examples gives MAP; averaging reciprocal
    rank gives MRR. Their equality here does not extend to multi-positive tasks.
    """
    ranked = validate_predictions(predictions)
    cutoffs = validate_cutoffs(cutoffs)
    rank = ranked.index(target) + 1 if target in ranked else None
    metrics = {}
    for cutoff in cutoffs:
        hit = rank is not None and rank <= cutoff
        reciprocal_rank = 1.0 / rank if hit else 0.0
        metrics[f"recall@{cutoff}"] = float(hit)
        metrics[f"ndcg@{cutoff}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
        metrics[f"map@{cutoff}"] = reciprocal_rank
        metrics[f"mrr@{cutoff}"] = reciprocal_rank
    return metrics


def average_precision_at_k(
    predictions: Sequence[int], relevant_items: Iterable[int], top_k: int
) -> float:
    """Return AP@K with the *total number of relevant items* as denominator.

    Each retrieved relevant item contributes precision at its rank. Divide their
    sum by ``len(set(relevant_items))``, including positives outside K. This is
    deliberately different from implementations dividing by ``min(K, positives)``.
    An example with no relevant items receives zero.
    """
    ranked = validate_predictions(predictions)
    validate_cutoffs([top_k])
    relevant = set(relevant_items)
    if not relevant:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    for rank, item in enumerate(ranked[:top_k], start=1):
        if item in relevant:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / len(relevant)


def ranking_metrics(predictions: list[int], target: int, top_k: int) -> tuple[float, float]:
    """Backward-compatible (Recall@K, NDCG@K) for one relevant item."""
    metrics = next_item_metrics(predictions, target, [top_k])
    return metrics[f"recall@{top_k}"], metrics[f"ndcg@{top_k}"]


def popularity_ranking(counts: np.ndarray, excluded: set[int], top_k: int) -> list[int]:
    """Rank training counts only; ties use item index for reproducibility."""
    validate_cutoffs([top_k])
    item_ids = np.arange(len(counts))
    ordered = np.lexsort((item_ids, -counts))
    return [int(item) for item in ordered if item not in excluded][:top_k]
