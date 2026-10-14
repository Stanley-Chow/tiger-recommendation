"""Frozen metadata retrieval and fixed reciprocal-rank fusion for new products."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from beautyrec.retrieval import synchronize


class ContentRetriever:
    """Score the available catalog by mean recent-history cosine similarity.

    The encoder is frozen. No target, popularity, or test interaction is used to
    score candidates. All prior seen items are removed before the top-K sort.
    """

    def __init__(self, vectors: np.ndarray, device: str | torch.device = "cpu"):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or min(vectors.shape) < 1 or not np.isfinite(vectors).all():
            raise ValueError("Content vectors must be a finite nonempty matrix.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Content vectors cannot have zero norm.")
        self.vectors = torch.from_numpy(vectors / norms).to(device)

    @torch.inference_mode()
    def retrieve(self, examples: list[dict], top_k=200, max_history=20, batch_size=128):
        if not examples or min(top_k, max_history, batch_size) < 1:
            raise ValueError("Need contexts and positive retrieval limits.")
        device, count = self.vectors.device, len(self.vectors)
        rankings = []
        synchronize(device)
        started = time.perf_counter()
        for start in range(0, len(examples), batch_size):
            rows = examples[start : start + batch_size]
            prototypes, exclusions = [], []
            for row in rows:
                history = row["history"]
                seen = set(row.get("seen_items", history))
                if (
                    not history
                    or not set(history) <= seen
                    or any(isinstance(i, bool) or not isinstance(i, int) for i in seen)
                    or any(i < 0 or i >= count for i in seen)
                ):
                    raise ValueError("Content history and exclusions must use valid item IDs.")
                prototypes.append(self.vectors[history[-max_history:]].mean(dim=0))
                exclusions.append(seen)
            query = torch.nn.functional.normalize(torch.stack(prototypes), dim=1)
            scores = query @ self.vectors.T
            for index, seen in enumerate(exclusions):
                scores[index, list(seen)] = -torch.inf
            ordered = torch.argsort(scores, descending=True, stable=True)[:, :top_k].cpu()
            rankings.extend(
                [item for item in ordered_row if item not in seen]
                for ordered_row, seen in zip(ordered.tolist(), exclusions)
            )
        synchronize(device)
        seconds = time.perf_counter() - started
        return rankings, {
            "protocol": "batched frozen mean-history cosine retrieval; includes result transfer",
            "batch_size": batch_size,
            "total_seconds": seconds,
            "histories_per_second": len(examples) / seconds,
        }


def reciprocal_rank_fusion(
    first: list[int], second: list[int], *, top_k=200, content_weight=0.5, rank_constant=60
) -> list[int]:
    """Fuse two unique rankings; ties use item ID and scores are not probabilities."""
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 1
        or not math.isfinite(content_weight)
        or not 0 < content_weight < 1
        or not math.isfinite(rank_constant)
        or rank_constant <= 0
    ):
        raise ValueError("Fusion requires positive K/constant and a weight strictly inside (0, 1).")
    scores = {}
    for ranking, weight in ((first, 1 - content_weight), (second, content_weight)):
        if len(ranking) != len(set(ranking)) or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ranking
        ):
            raise ValueError("Fusion inputs must contain unique nonnegative integer IDs.")
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + weight / (rank_constant + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))[:top_k]
