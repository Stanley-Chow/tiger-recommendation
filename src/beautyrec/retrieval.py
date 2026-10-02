"""Retrieve one ranked candidate list per history, shared by every metric cutoff."""

import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from beautyrec.decoding import SemanticTrie
from beautyrec.model import GenerativeRecommender, build_source_tokens
from beautyrec.sasrec import SASRecRecommender
from beautyrec.semantic_ids import SemanticCodebook


def popularity_rankings(examples: list[dict], counts: np.ndarray, top_k: int) -> list[list[int]]:
    """Sort training frequency once, then remove each user's complete seen history."""
    order = np.lexsort((np.arange(len(counts)), -counts)).tolist()
    results = []
    for example in examples:
        seen = set(example.get("seen_items", example["history"]))
        ranked = []
        for item in order:
            if item not in seen:
                ranked.append(item)
                if len(ranked) == top_k:
                    break
        results.append(ranked)
    return results


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def retrieve_rankings(
    model: GenerativeRecommender | SASRecRecommender,
    examples: list[dict],
    codebook: SemanticCodebook,
    *,
    max_history: int,
    top_k: int,
    beam_size: int,
    batch_size: int = 16,
    progress: bool = False,
) -> tuple[list[list[int]], dict]:
    """Generate candidates without using held-out labels to score or insert items.

    The generative model uses approximate catalog-constrained beam search;
    SASRec scores the complete catalog exactly. Both exclude all previously
    observed items. Batch timings are throughput measurements, not online
    single-request latency. They include the first call and CPU result transfer.
    """
    if not examples or min(batch_size, top_k) < 1 or beam_size < top_k:
        raise ValueError("Need examples, positive batch/cutoff and beam_size >= top_k.")
    model.eval()
    device = next(model.parameters()).device
    cached_index = getattr(model, "_retrieval_index", None)
    if cached_index is None or cached_index[0] is not codebook:
        cached_index = (codebook, SemanticTrie(codebook.item_codes, codebook.eos_token))
        model._retrieval_index = cached_index
    trie = cached_index[1]
    results = []
    batch_seconds = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        exclusions = [set(row.get("seen_items", row["history"])) for row in batch]
        synchronize(device)
        started = time.perf_counter()
        if isinstance(model, GenerativeRecommender):
            from beautyrec.batched_decoding import batch_constrained_beam_search

            sources = [
                torch.tensor(
                    build_source_tokens(row["history"], codebook.item_codes, max_history),
                    dtype=torch.long,
                )
                for row in batch
            ]
            tokens = pad_sequence(sources, batch_first=True, padding_value=0).to(device)
            predictions = batch_constrained_beam_search(
                model,
                tokens,
                trie,
                top_k=top_k,
                beam_size=beam_size,
                exclude_items=exclusions,
            )
            ranked = [[item for item, _ in row] for row in predictions]
        elif isinstance(model, SASRecRecommender):
            scores = model.score_histories([row["history"] for row in batch])
            for index, seen in enumerate(exclusions):
                if seen:
                    scores[index, list(seen)] = -torch.inf
            # Stable sorting resolves equal logits by zero-based item ID.
            ordered = torch.argsort(scores, descending=True, stable=True)[:, :top_k].cpu()
            ranked = [
                [item for item in row if item not in seen]
                for row, seen in zip(ordered.tolist(), exclusions)
            ]
        else:
            raise TypeError("Unsupported recommendation model.")
        synchronize(device)
        batch_seconds.append(time.perf_counter() - started)
        results.extend(ranked)
        if progress and (start // batch_size) % 100 == 0:
            print(f"Retrieved {len(results):,}/{len(examples):,} histories", flush=True)
    total = float(sum(batch_seconds))
    timing = {
        "protocol": "batched retrieval, includes first call and result transfer; no text encoding",
        "batch_size": batch_size,
        "batches": len(batch_seconds),
        "total_seconds": total,
        "histories_per_second": len(examples) / total,
        "amortized_ms_per_history": total / len(examples) * 1000,
        "mean_batch_ms": float(np.mean(batch_seconds)) * 1000,
        "p95_batch_ms": float(np.percentile(batch_seconds, 95)) * 1000,
        "peak_cuda_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
    }
    return results, timing


@torch.inference_mode()
def measure_request_latency(
    model: GenerativeRecommender | SASRecRecommender,
    examples: list[dict],
    codebook: SemanticCodebook,
    *,
    max_history: int,
    top_k: int,
    beam_size: int,
    requests: int = 100,
) -> dict:
    """Measure warm single-history requests separately from batched throughput."""
    options = {"max_history": max_history, "top_k": top_k, "beam_size": beam_size}
    retrieve_rankings(model, examples[:1], codebook, batch_size=1, **options)
    latencies = []
    for example in examples[:requests]:
        _, timing = retrieve_rankings(model, [example], codebook, batch_size=1, **options)
        latencies.append(timing["total_seconds"] * 1000)
    return {
        "protocol": "one history per request after one warmup; includes result transfer",
        "requests": len(latencies),
        "mean_ms": float(np.mean(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }
