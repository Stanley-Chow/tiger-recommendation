"""Read-only M5 audit: checksums, temporal contexts, selection, and scalar metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from verify_m3_results import (  # noqa: E402
    close,
    context,
    item_metrics,
    read_json,
    require,
    sha256,
    verify_completion,
    verify_file_map,
    verify_guard,
)

from beautyrec.benchmark import _json_hash  # noqa: E402
from beautyrec.evaluation import (  # noqa: E402
    aggregate_seed_summaries,
    bootstrap_comparison,
    context_fingerprint,
)
from beautyrec.experiment_training import load_experiment_checkpoint  # noqa: E402
from beautyrec.serving import verify_bundle  # noqa: E402
from beautyrec.training import check_dataset, read_examples, sample_examples  # noqa: E402


def compare_tree(actual, expected, label):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"Different keys: {label}")
        for key in expected:
            compare_tree(actual[key], expected[key], f"{label}/{key}")
    else:
        close(actual, expected, label)


def verify_rankings(path, summary, examples, popularity, methods, cutoffs, warm_items):
    """Recompute scalar metrics allowing each branch's declared catalog support."""
    maximum, count = max(cutoffs), len(popularity)
    threshold = float(np.median(popularity))
    sums = {method: dict.fromkeys(item_metrics(None, cutoffs), 0.0) for method in methods}
    tails = {method: dict(values) for method, values in sums.items()}
    coverage = {method: {k: set() for k in cutoffs} for method in methods}
    lengths = dict.fromkeys(methods, 0)
    tail_count, rows = 0, 0
    for index, row in enumerate(read_examples(path)):
        require(
            index < len(examples) and row["example_index"] == index,
            "Unexpected ranking row position.",
        )
        require(context(row) == context(examples[index]), "Ranking context differs from source.")
        require(list(row["methods"]) == methods, "Ranking method order differs.")
        seen, target = set(row["seen_items"]), row["target"]
        require(target not in seen and set(row["history"]) <= seen, "Invalid ranking exclusions.")
        is_tail = popularity[target] <= threshold
        tail_count += int(is_tail)
        for method in methods:
            record = row["methods"][method]
            ranked = record["ranked_items"]
            support = warm_items if method in ("generative", "sasrec", "popularity") else count
            require(
                len(ranked) <= min(maximum, support - sum(i < support for i in seen)),
                "Ranking exceeds its eligible catalog.",
            )
            require(
                len(set(ranked)) == len(ranked) and not seen.intersection(ranked),
                "Duplicate or previously seen prediction.",
            )
            require(
                all(
                    isinstance(i, int) and not isinstance(i, bool) and 0 <= i < support
                    for i in ranked
                ),
                "Out-of-support recommendation.",
            )
            rank = ranked.index(target) + 1 if target in ranked else None
            close(record["target_rank"], rank, f"{path}/{index}/{method}/rank")
            metrics = item_metrics(rank, cutoffs)
            compare_tree(record["metrics"], metrics, f"{path}/{index}/{method}/metrics")
            for key, value in metrics.items():
                sums[method][key] += value
                if is_tail:
                    tails[method][key] += value
            lengths[method] += len(ranked)
            for k in cutoffs:
                coverage[method][k].update(ranked[:k])
        rows += 1
    require(rows == len(examples), "Missing per-user rankings.")
    for key, value in {
        "examples": rows,
        "users": len({row["user_id"] for row in examples}),
        "catalog_items": count,
        "cutoffs": cutoffs,
        "methods": methods,
        "exclude_seen": True,
        "contexts_sha256": context_fingerprint(examples),
        "training_popularity_sha256": hashlib.sha256(
            popularity.astype("<f8").tobytes()
        ).hexdigest(),
        "tail_training_count_threshold": threshold,
    }.items():
        close(summary[key], value, f"{path}/{key}")
    for method in methods:
        values = {key: value / rows for key, value in sums[method].items()}
        values.update(
            {
                f"tail_{key}": value / tail_count if tail_count else None
                for key, value in tails[method].items()
            }
        )
        values.update({f"catalog_coverage@{k}": len(coverage[method][k]) / count for k in cutoffs})
        values.update(
            tail_examples=tail_count,
            mean_returned_items=lengths[method] / rows,
            catalog_coverage=values[f"catalog_coverage@{maximum}"],
        )
        for key, value in values.items():
            close(summary[method][key], value, f"{path}/{method}/{key}")
    return {"rows": rows}


def verify_temporal_contexts(processed, config):
    """Reconstruct contexts from complete cohort events, without the prep kernel."""
    warm = read_json(processed / "catalog.json")
    available = read_json(processed / "available_catalog.json")
    require(available[: len(warm)] == warm, "Warm catalog is not the available prefix.")
    warm_ids = {row["asin"]: row["item_id"] for row in warm}
    available_ids = {row["asin"]: row["item_id"] for row in available}
    start, test_start, end = (config[k] for k in ("train_end_ms", "test_start_ms", "test_end_ms"))
    for item in warm:
        require(item["available_from_ms"] < start, "Warm item first observed after training.")
    for item in available[len(warm) :]:
        require(
            start <= item["available_from_ms"] < test_start,
            "New item must first appear in 2022, before the test request.",
        )
    expected = {split: [] for split in ("train", "validation", "test", "mixed_test")}
    popularity = np.zeros(len(warm), dtype=np.int64)
    users = set()
    for sequence in read_examples(processed / "source_cohort.jsonl"):
        user, events = sequence["user_id"], sequence["events"]
        require(user not in users, "Duplicate cohort user.")
        users.add(user)
        require(
            events == sorted(events, key=lambda pair: (pair[0], pair[1])),
            "Unordered cohort source events.",
        )
        require(len({asin for _, asin in events}) == len(events), "Repeated source user-item pair.")
        past_train = [(t, warm_ids[asin]) for t, asin in events if t < start and asin in warm_ids]
        require(
            len(past_train) >= config["min_train_interactions"], "Invalid training-qualified user."
        )
        for timestamp, target in past_train:
            popularity[target] += 1
            earlier = [item for t, item in past_train if t < timestamp]
            if earlier:
                expected["train"].append(
                    {
                        "user_id": user,
                        "history": earlier[-config["max_history"] :],
                        "seen_items": earlier,
                        "target": target,
                    }
                )
        for split, lower, upper in (("validation", start, test_start), ("test", test_start, end)):
            future = [(t, asin) for t, asin in events if lower <= t < upper]
            if not future:
                continue
            _, target_asin = future[0]  # Eligibility is applied only after this choice.
            earlier = [(t, asin) for t, asin in events if t < lower]
            warm_seen = [warm_ids[asin] for _, asin in earlier if asin in warm_ids]
            base = {
                "user_id": user,
                "history": warm_seen[-config["max_history"] :],
                "seen_items": warm_seen,
            }
            if target_asin in warm_ids:
                expected[split].append({**base, "target": warm_ids[target_asin]})
            if split == "test" and target_asin in available_ids:
                seen = [available_ids[asin] for _, asin in earlier if asin in available_ids]
                expected["mixed_test"].append(
                    {
                        "user_id": user,
                        "history": seen[-config["max_history"] :],
                        "seen_items": seen,
                        "target": available_ids[target_asin],
                    }
                )
    rows = {}
    for split in expected:
        rows[split] = read_examples(processed / f"{split}.jsonl")
        require(
            [context(row) for row in rows[split]] == expected[split],
            f"Temporal context reconstruction differs: {split}",
        )
        for row in rows[split]:
            require(row["history_end_timestamp"] < row["timestamp"], "Future history leakage.")
            if split != "train":
                require(
                    row["history_end_timestamp"] < row["request_timestamp"] <= row["timestamp"],
                    "Invalid frozen request timestamp.",
                )
    require(
        np.array_equal(popularity, np.load(processed / "item_popularity.npy")),
        "Training event popularity differs from source histories.",
    )
    stats = read_json(processed / "statistics.json")
    require(users == set(stats["selected_user_ids"]), "Selected cohort differs from saved source.")
    return rows


def verify(run, processed, features, allow_diagnostic=False):
    protocol = read_json(run / "protocol.json")
    payload = {k: v for k, v in protocol.items() if k != "fingerprint"}
    require(_json_hash(payload) == protocol["fingerprint"], "Protocol fingerprint differs.")
    result = verify_completion(run, "completed.json", protocol["fingerprint"])
    require(allow_diagnostic or result["m5_complete"], "Diagnostic run is not a full M5 study.")
    verify_file_map(ROOT / "src/beautyrec", protocol["source_python"])
    verify_file_map(processed, protocol["processed_files"])
    verify_file_map(features, protocol["feature_files"])
    config = protocol["config"]
    contexts = verify_temporal_contexts(processed, config["data"])
    selection = read_json(run / "selection_manifest.json")
    tokenizers = read_json(run / "tokenizer_manifest.json")
    require(
        selection["protocol_fingerprint"] == protocol["fingerprint"], "Selection protocol differs."
    )
    require(
        tokenizers["protocol_fingerprint"] == protocol["fingerprint"], "Tokenizer protocol differs."
    )
    require(
        _json_hash({k: v for k, v in tokenizers.items() if k != "fingerprint"})
        == tokenizers["fingerprint"],
        "Tokenizer fingerprint differs.",
    )
    counts = np.load(processed / "item_popularity.npy", allow_pickle=False)
    extended = np.load(processed / "available_popularity.npy", allow_pickle=False)
    require(
        np.array_equal(extended[: len(counts)], counts) and not extended[len(counts) :].any(),
        "Available-catalog popularity must pad training counts with zeros.",
    )
    total_rows, seeds = 0, []
    for seed in config["seeds"]:
        directory = run / "seeds" / f"seed_{seed}"
        saved = verify_completion(directory, "seed_complete.json", tokenizers["fingerprint"])
        artifact = run / "arms" / f"seed_{seed}"
        verify_file_map(artifact, tokenizers["seeds"][str(seed)]["files"])
        for method in ("generative", "sasrec"):
            branch = directory / method
            require(
                sha256(branch / "best.pt") == selection["selected"][str(seed)][method]["sha256"],
                "Selected checkpoint changed.",
            )
            model, _, checkpoint = load_experiment_checkpoint(branch / "best.pt", artifact, "cpu")
            check_dataset(processed, checkpoint)
            require(checkpoint["initial_checkpoint"] is None, "M5 must use fresh model weights.")
            verify_guard(branch, checkpoint, contexts["validation"], counts, config, seed)
            del model
        split_contexts = {
            split: sample_examples(
                contexts[split],
                config["evaluation"][
                    "max_validation_examples" if split == "validation" else "max_test_examples"
                ],
                config["evaluation"]["sampling_seed"],
            )
            for split in ("validation", "test", "mixed_test")
        }
        for group in ("warm", "new"):
            split_contexts[f"mixed_{group}"] = [
                row for row in split_contexts["mixed_test"] if row["target_type"] == group
            ]
        rankings = {}
        for split, summary in saved["summaries"].items():
            if summary is None:
                require(not split_contexts[split], "Nonempty group has null metrics.")
                continue
            path = directory / "evaluation" / split / "per_user_rankings.jsonl"
            verified = verify_rankings(
                path,
                summary,
                split_contexts[split],
                extended if split.startswith("mixed") else counts,
                summary["methods"],
                config["cutoffs"],
                len(counts),
            )
            total_rows += verified["rows"]
            rankings[split] = {name: [] for name in summary["methods"]}
            for row in read_examples(path):
                for name in summary["methods"]:
                    ranking = row["methods"][name]["ranked_items"]
                    if name in ("generative", "sasrec", "popularity"):
                        require(
                            all(item < len(counts) for item in ranking),
                            "Closed-catalog method emitted an untrained new item.",
                        )
                    rankings[split][name].append(ranking)
        for name, expected in saved["comparisons"].items():
            split, contrast = name.split("/")
            candidate, reference = contrast.split("_minus_")
            actual = bootstrap_comparison(
                split_contexts[split],
                rankings[split][reference],
                rankings[split][candidate],
                cutoffs=config["cutoffs"],
                resamples=config["evaluation"]["bootstrap_resamples"],
                seed=config["evaluation"]["sampling_seed"],
            )
            compare_tree(actual, expected, f"bootstrap/{seed}/{name}")
        seeds.append(saved)
    for split, expected in result["aggregate"].items():
        actual = aggregate_seed_summaries([seed["summaries"][split] for seed in seeds])
        compare_tree(actual, expected, f"aggregate/{split}")
    verify_bundle(run / "bundle")
    return {
        "verified": True,
        "m5_complete": result["m5_complete"],
        "seeds": len(seeds),
        "checkpoint_count": 2 * len(seeds),
        "ranking_context_rows": total_rows,
        "source_training_contexts": len(contexts["train"]),
        "protocol_fingerprint": protocol["fingerprint"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/m5")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/amazon2023/processed")
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "artifacts/amazon2023")
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.run_dir, args.processed_dir, args.feature_dir, args.allow_diagnostic),
            indent=2,
        )
    )
