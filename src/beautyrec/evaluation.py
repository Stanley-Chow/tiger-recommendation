"""Compare saved recommendation lists under one shared evaluation protocol.

Retrieval happens elsewhere. Every metric cutoff reuses a prefix of the same
maximum-cutoff ranking, so changing K cannot change the candidate generation.
The bootstrap resamples users, keeping paired method results together.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from beautyrec.metrics import next_item_metrics, validate_cutoffs, validate_predictions

DEFAULT_CUTOFFS = (10, 20, 50, 100, 200)
_SUMMARY_KEYS = {
    "examples",
    "users",
    "catalog_items",
    "cutoffs",
    "contexts_sha256",
    "training_popularity_sha256",
    "methods",
    "exclude_seen",
    "tail_training_count_threshold",
    "metric_protocol",
    "rankings_file",
}


def _contexts(examples: Sequence[dict], catalog_size: int | None = None) -> list[dict]:
    """Normalize and verify contexts before scoring or writing comparison files."""
    if not examples:
        raise ValueError("Evaluation needs at least one example.")
    result = []
    for example in examples:
        user = example.get("user_id")
        target = example.get("target")
        if not isinstance(user, str) or not user:
            raise ValueError("Every example needs a nonempty string user_id.")
        if isinstance(target, bool) or not isinstance(target, Integral) or target < 0:
            raise ValueError("Every target must be a nonnegative integer item ID.")
        history = validate_predictions(example["history"])
        seen = validate_predictions(example.get("seen_items", history))
        if not history or not set(history) <= set(seen):
            raise ValueError("History must be nonempty and contained in seen_items.")
        if target in seen:
            raise ValueError("A held-out target cannot be in the excluded seen_items.")
        items = [int(target), *history, *seen]
        if any(item < 0 or (catalog_size is not None and item >= catalog_size) for item in items):
            raise ValueError("A context contains an item outside the evaluation catalog.")
        result.append(
            {
                "user_id": user,
                "history": history,
                "seen_items": seen,
                "target": int(target),
            }
        )
    return result


def context_fingerprint(examples: Sequence[dict]) -> str:
    """Hash ordered user histories, exclusions and targets, independently of method."""
    digest = hashlib.sha256()
    for context in _contexts(examples):
        digest.update(json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _checked_rankings(
    contexts: Sequence[dict],
    rankings: Sequence[Sequence[int]],
    max_cutoff: int,
    catalog_size: int | None = None,
) -> list[list[int]]:
    if len(rankings) != len(contexts):
        raise ValueError("Ranking rows must align one-to-one with the shared ordered contexts.")
    result = []
    for context, predictions in zip(contexts, rankings):
        ranked = validate_predictions(predictions)
        if any(item < 0 or (catalog_size is not None and item >= catalog_size) for item in ranked):
            raise ValueError("A prediction contains an item outside the evaluation catalog.")
        if set(ranked) & set(context["seen_items"]):
            raise ValueError("Predictions contain an excluded previously seen item.")
        result.append(ranked[:max_cutoff])
    return result


def _metric_matrix(contexts: Sequence[dict], rankings: Sequence[Sequence[int]], cutoffs: tuple):
    rows = [
        next_item_metrics(ranked, context["target"], cutoffs)
        for context, ranked in zip(contexts, rankings)
    ]
    names = list(rows[0])
    return names, np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)


def evaluate_rankings(
    examples: Sequence[dict],
    rankings: dict[str, Sequence[Sequence[int]]],
    popularity: np.ndarray,
    *,
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
    latency_ms: dict[str, Sequence[float]] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Aggregate next-item Recall/NDCG/MAP/MRR for any collection of methods.

    All methods share exactly the same ordered examples and warm catalog. Counts
    are original training-event frequencies. Tail targets have count <= their
    catalog median, including ties. Coverage is measured at each cutoff against
    the entire catalog. Short rankings stay short; they are never padded with the
    target or sampled negative items. Single-positive MAP equals MRR.

    With ``output_dir``, stream per-example contexts, ranked lists, target ranks,
    and metrics to ``per_user_rankings.jsonl`` and save ``summary.json``. A null
    target rank means the target is absent from the saved maximum-cutoff list.
    ``latency_ms`` values must correspond to the same ordered contexts; summaries
    report exactly these observations without assuming a warmup protocol.
    """
    cutoffs = validate_cutoffs(cutoffs)
    counts = np.asarray(popularity)
    if counts.ndim != 1 or not counts.size or not np.issubdtype(counts.dtype, np.number):
        raise ValueError("Popularity must be a nonempty vector of training counts.")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Popularity must contain finite nonnegative training counts.")
    contexts = _contexts(examples, len(counts))
    if not rankings or any(not isinstance(name, str) or not name for name in rankings):
        raise ValueError("Provide at least one named method.")
    if set(rankings) & _SUMMARY_KEYS:
        raise ValueError("A method name conflicts with evaluation summary metadata.")
    if latency_ms is not None and set(latency_ms) - set(rankings):
        raise ValueError("Latency observations refer to an unknown method.")

    checked = {
        method: _checked_rankings(contexts, rows, max(cutoffs), len(counts))
        for method, rows in rankings.items()
    }
    tail_threshold = float(np.median(counts))
    tail_mask = np.asarray([counts[row["target"]] <= tail_threshold for row in contexts])
    summary: dict[str, Any] = {
        "examples": len(contexts),
        "users": len({row["user_id"] for row in contexts}),
        "catalog_items": len(counts),
        "cutoffs": list(cutoffs),
        "contexts_sha256": context_fingerprint(contexts),
        "training_popularity_sha256": hashlib.sha256(counts.astype("<f8").tobytes()).hexdigest(),
        "methods": list(rankings),
        "exclude_seen": True,
        "tail_training_count_threshold": tail_threshold,
        "metric_protocol": {
            "positives_per_example": 1,
            "map_denominator": "total relevant items; one positive per next-item example",
            "map_equals_mrr": True,
            "cutoff_rankings": "prefixes of one maximum-cutoff list per method and context",
            "exclusion": "all saved prior warm seen_items, before source history truncation",
        },
    }
    for method, rows in checked.items():
        names, values = _metric_matrix(contexts, rows, cutoffs)
        method_summary = {name: float(value) for name, value in zip(names, values.mean(axis=0))}
        for column, name in enumerate(names):
            method_summary[f"tail_{name}"] = (
                float(values[tail_mask, column].mean()) if tail_mask.any() else None
            )
        method_summary["tail_examples"] = int(tail_mask.sum())
        for cutoff in cutoffs:
            covered = {item for row in rows for item in row[:cutoff]}
            method_summary[f"catalog_coverage@{cutoff}"] = len(covered) / len(counts)
        method_summary["catalog_coverage"] = method_summary[f"catalog_coverage@{max(cutoffs)}"]
        method_summary["mean_returned_items"] = float(np.mean([len(row) for row in rows]))
        if latency_ms is not None and method in latency_ms:
            latencies = np.asarray(latency_ms[method], dtype=np.float64)
            if latencies.shape != (len(contexts),) or not np.isfinite(latencies).all():
                raise ValueError("Latencies need one finite observation per shared context.")
            if np.any(latencies < 0):
                raise ValueError("Latency observations cannot be negative.")
            method_summary["mean_latency_ms"] = float(latencies.mean())
            method_summary["p50_latency_ms"] = float(np.percentile(latencies, 50))
            method_summary["p95_latency_ms"] = float(np.percentile(latencies, 95))
        summary[method] = method_summary

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rankings_path = output_dir / "per_user_rankings.jsonl"
        with rankings_path.open("w", encoding="utf-8", newline="\n") as output:
            for index, context in enumerate(contexts):
                record = {"example_index": index, **context, "methods": {}}
                for method, rows in checked.items():
                    ranked = rows[index]
                    record["methods"][method] = {
                        "ranked_items": ranked,
                        "target_rank": ranked.index(context["target"]) + 1
                        if context["target"] in ranked
                        else None,
                        "metrics": next_item_metrics(ranked, context["target"], cutoffs),
                    }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary["rankings_file"] = str(rankings_path.resolve())
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return summary


def bootstrap_comparison(
    examples: Sequence[dict],
    rankings_a: Sequence[Sequence[int]],
    rankings_b: Sequence[Sequence[int]],
    *,
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    candidate_examples: Sequence[dict] | None = None,
) -> dict[str, Any]:
    """Return paired user-bootstrap intervals for A, B and B-minus-A metrics.

    Rankings must use the same ordered contexts. If candidate contexts were read
    separately, pass ``candidate_examples`` to verify complete alignment, including
    histories and exclusions rather than just user IDs. Repeated examples from
    one user are resampled together. Reported means remain example-weighted.

    Only one resampled user-count vector is allocated per iteration; there is no
    resamples-by-users index matrix. Percentile intervals describe user sampling
    uncertainty for these rankings, not model-training seed uncertainty.
    """
    cutoffs = validate_cutoffs(cutoffs)
    if isinstance(resamples, bool) or not isinstance(resamples, Integral) or resamples < 1:
        raise ValueError("resamples must be a positive integer.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    contexts = _contexts(examples)
    if candidate_examples is not None and context_fingerprint(contexts) != context_fingerprint(
        candidate_examples
    ):
        raise ValueError("Paired comparisons require identical ordered contexts and exclusions.")
    first = _checked_rankings(contexts, rankings_a, max(cutoffs))
    second = _checked_rankings(contexts, rankings_b, max(cutoffs))
    names, values_a = _metric_matrix(contexts, first, cutoffs)
    _, values_b = _metric_matrix(contexts, second, cutoffs)
    user_ids = sorted({row["user_id"] for row in contexts})
    user_index = {user: index for index, user in enumerate(user_ids)}
    positions = np.asarray([user_index[row["user_id"]] for row in contexts])
    user_counts = np.bincount(positions, minlength=len(user_ids))
    grouped_a = np.zeros((len(user_ids), len(names)))
    grouped_b = np.zeros_like(grouped_a)
    np.add.at(grouped_a, positions, values_a)
    np.add.at(grouped_b, positions, values_b)
    draws_a = np.empty((resamples, len(names)))
    draws_b = np.empty_like(draws_a)
    rng = np.random.default_rng(seed)
    for draw in range(resamples):
        sampled = rng.integers(len(user_ids), size=len(user_ids))
        weights = np.bincount(sampled, minlength=len(user_ids))
        denominator = weights @ user_counts
        draws_a[draw] = weights @ grouped_a / denominator
        draws_b[draw] = weights @ grouped_b / denominator

    quantiles = [(1 - confidence) / 2, (1 + confidence) / 2]

    def intervals(means: np.ndarray, draws: np.ndarray) -> dict:
        bounds = np.quantile(draws, quantiles, axis=0)
        return {
            name: {
                "mean": float(means[index]),
                "lower": float(bounds[0, index]),
                "upper": float(bounds[1, index]),
            }
            for index, name in enumerate(names)
        }

    return {
        "examples": len(contexts),
        "users": len(user_ids),
        "resamples": int(resamples),
        "confidence": float(confidence),
        "seed": int(seed),
        "cutoffs": list(cutoffs),
        "contexts_sha256": context_fingerprint(contexts),
        "resampling_unit": "user; repeated examples from a user remain together",
        "interval_method": "paired percentile bootstrap",
        "difference": "candidate minus reference",
        "reference": intervals(values_a.mean(axis=0), draws_a),
        "candidate": intervals(values_b.mean(axis=0), draws_b),
        "delta": intervals((values_b - values_a).mean(axis=0), draws_b - draws_a),
    }


def aggregate_seed_summaries(summaries: Sequence[dict]) -> dict[str, Any]:
    """Summarize aligned runs using means and sample standard deviations.

    This describes variation across training seeds. It does not pool seeds as if
    they were independent users, or present seed spread as a bootstrap interval.
    """
    if not summaries:
        raise ValueError("Provide at least one seed summary.")
    first = summaries[0]
    for summary in summaries[1:]:
        for field in (
            "contexts_sha256",
            "cutoffs",
            "methods",
            "catalog_items",
            "training_popularity_sha256",
            "metric_protocol",
        ):
            if summary[field] != first[field]:
                raise ValueError("Seed summaries must use the same contexts, catalog and methods.")
    result = {
        "runs": len(summaries),
        "contexts_sha256": first["contexts_sha256"],
        "cutoffs": first["cutoffs"],
        "methods": first["methods"],
        "spread": "sample standard deviation across runs; null for one run",
    }
    for method in first["methods"]:
        result[method] = {}
        for metric in first[method]:
            values = [summary[method].get(metric) for summary in summaries]
            if any(value is None or not isinstance(value, (int, float)) for value in values):
                continue
            array = np.asarray(values, dtype=np.float64)
            result[method][metric] = {
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if len(array) > 1 else None,
            }
    return result
