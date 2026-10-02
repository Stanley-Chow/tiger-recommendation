"""Run the frozen M2 comparison, including controlled optimizer continuations.

The protocol is saved before training or test retrieval. Seeds share data,
validation sampling and evaluation contexts. Complete, unchanged runs are reused;
partial runs reuse only branches whose completion markers verify their weights.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from beautyrec.evaluation import (
    aggregate_seed_summaries,
    bootstrap_comparison,
    evaluate_rankings,
)
from beautyrec.metrics import validate_cutoffs
from beautyrec.retrieval import measure_request_latency, popularity_rankings, retrieve_rankings
from beautyrec.training import check_dataset, read_examples, resolve_device, sample_examples

EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_CUTOFFS = [10, 20, 50, 100, 200]


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    """Keep completion and protocol markers intact if a process is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _positive_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _configuration(config: dict, device_override: str | None) -> dict:
    """Validate orchestration choices before creating any experiment artifacts."""
    result = deepcopy(config)
    for section in ("training", "models", "continuations", "evaluation"):
        if not isinstance(result.get(section), dict):
            raise ValueError(f"The benchmark config needs a {section} object.")
    seeds = result.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be distinct nonnegative integers.")
    result["cutoffs"] = list(validate_cutoffs(result.get("cutoffs", EXPECTED_CUTOFFS)))
    if device_override is not None:
        result["training"]["device"] = device_override
    training, evaluation = result["training"], result["evaluation"]
    for field in ("epochs", "batch_size", "max_history", "patience", "torch_threads"):
        _positive_integer(training[field], f"training.{field}")
    for field in (
        "beam_size",
        "generative_batch_size",
        "sasrec_batch_size",
        "request_latency_requests",
        "bootstrap_resamples",
    ):
        _positive_integer(evaluation[field], f"evaluation.{field}")
    if evaluation["beam_size"] < max(200, max(result["cutoffs"])):
        raise ValueError("Evaluation beam_size must cover top-200 and the maximum metric cutoff.")
    for field in ("max_validation_examples", "max_test_examples"):
        value = evaluation.get(field)
        if value is not None:
            _positive_integer(value, f"evaluation.{field}")
    if training.get("max_train_examples") is not None:
        _positive_integer(training["max_train_examples"], "training.max_train_examples")
    if training.get("validation_examples") is not None:
        _positive_integer(training["validation_examples"], "training.validation_examples")
    sampling_seed = evaluation.get("sampling_seed", 2026)
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int) or sampling_seed < 0:
        raise ValueError("evaluation.sampling_seed must be a nonnegative integer.")
    evaluation["sampling_seed"] = sampling_seed
    continuation = result["continuations"]
    _positive_integer(continuation["epochs"], "continuations.epochs")
    control_epochs = continuation.get("adamw_compute_control_epochs")
    if control_epochs is not None:
        _positive_integer(control_epochs, "continuations.adamw_compute_control_epochs")
    for value, name in (
        (training["learning_rate"], "training.learning_rate"),
        (continuation["learning_rate"], "continuations.learning_rate"),
        (continuation["sam_rho"], "continuations.sam_rho"),
    ):
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not {"generative", "sasrec"} <= set(result["models"]):
        raise ValueError("models must contain generative and sasrec options.")
    result["diagnostic_only"] = bool(result.get("diagnostic_only", False))
    return result


def _diagnostic_reasons(config: dict, statistics: dict) -> list[str]:
    """Keep a limited smoke run distinct from a completed Amazon Beauty M2 run."""
    reasons = []
    if config["diagnostic_only"]:
        reasons.append("The configuration explicitly declares a diagnostic run.")
    if config["seeds"] != EXPECTED_SEEDS:
        reasons.append("The required three training seeds 42, 43 and 44 were not all requested.")
    if config["cutoffs"] != EXPECTED_CUTOFFS:
        reasons.append("The complete 10/20/50/100/200 evaluation cutoffs were not requested.")
    if (
        statistics.get("raw_reviews") != 198_502
        or statistics.get("source_files", {}).get("reviews", {}).get("name")
        != "reviews_Beauty_5.json.gz"
    ):
        reasons.append("The prepared data is not the complete original 2014 Beauty 5-core file.")
    if statistics.get("max_users") is not None or statistics.get("users") != statistics.get(
        "eligible_users_before_subsampling"
    ):
        reasons.append("Prepared users were subsampled or a user limit was declared.")
    training, evaluation = config["training"], config["evaluation"]
    if training.get("max_train_examples") is not None:
        reasons.append("Training examples were capped.")
    if training["epochs"] < 20:
        reasons.append("The base training budget is shorter than the agreed 20-epoch budget.")
    if (
        evaluation.get("max_validation_examples") is not None
        or evaluation.get("max_test_examples") is not None
    ):
        reasons.append("Final validation or test examples were capped.")
    if config["continuations"]["epochs"] != 3:
        reasons.append("The AdamW/SAM continuation budget differs from three epochs.")
    control_epochs = config["continuations"].get("adamw_compute_control_epochs")
    if control_epochs not in (None, 6):
        reasons.append("The optional AdamW compute control differs from six epochs.")
    return reasons


def _protocol(processed_dir: Path, artifact_dir: Path, config: dict) -> dict:
    statistics = json.loads((processed_dir / "statistics.json").read_text(encoding="utf-8"))
    processed_files = (
        "catalog.json",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "item_popularity.npy",
        "statistics.json",
    )
    artifact_files = ("semantic_ids.json", "embeddings.npy", "embedding_config.json")
    protocol = {
        "schema_version": 1,
        "config": config,
        "processed_files": {name: _file_hash(processed_dir / name) for name in processed_files},
        "artifact_files": {name: _file_hash(artifact_dir / name) for name in artifact_files},
        "diagnostic_reasons": _diagnostic_reasons(config, statistics),
        "rules": {
            "selection": "fixed-seed validation sample only; full validation reported separately",
            "test": "one frozen maximum-cutoff ranking; no test-based tuning",
            "continuations": "same base generative weights; fresh optimizer states for each branch",
            "uncertainty": "paired user bootstrap separately per training seed; no seed pooling",
            "latency": "single-history requests after warmup; batched throughput is separate",
        },
    }
    embedding_config = json.loads(
        (artifact_dir / "embedding_config.json").read_text(encoding="utf-8")
    )
    if embedding_config.get("diagnostic_only", False):
        protocol["diagnostic_reasons"].append("The embedding backend is declared diagnostic only.")
    if statistics.get("max_history", 0) < config["training"]["max_history"]:
        protocol["diagnostic_reasons"].append(
            "Prepared histories are shorter than the configured model context window."
        )
    protocol["fingerprint"] = _json_hash(protocol)
    return protocol


def _train_branch(
    processed_dir: Path,
    artifact_dir: Path,
    branch_dir: Path,
    method: str,
    training_options: dict,
    model_options: dict,
    protocol_fingerprint: str,
) -> dict:
    """Reuse an actual completed training branch after verifying its checkpoint."""
    from beautyrec.experiment_training import ExperimentTrainingConfig, train_experiment

    marker_path = branch_dir / "benchmark_training_complete.json"
    settings = {
        "method": method,
        "training": training_options,
        "model": model_options,
        "protocol_fingerprint": protocol_fingerprint,
    }
    fingerprint = _json_hash(settings)
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        checkpoint = Path(marker["checkpoint"])
        if (
            marker["fingerprint"] != fingerprint
            or _file_hash(checkpoint) != marker["checkpoint_sha256"]
        ):
            raise ValueError(
                "A completed benchmark training branch changed; use a new run directory."
            )
        return marker["result"]
    result = train_experiment(
        processed_dir,
        artifact_dir,
        branch_dir,
        method,
        training=ExperimentTrainingConfig(**training_options),
        model_options=model_options,
    )
    if (
        training_options["initial_checkpoint"] is not None
        and result.get("epochs_completed") != training_options["epochs"]
    ):
        raise RuntimeError(
            "An optimizer continuation did not finish its matched epoch budget; "
            "the benchmark cannot be marked complete."
        )
    checkpoint = Path(result.get("checkpoint", branch_dir / "best.pt")).resolve()
    if not checkpoint.is_file():
        raise RuntimeError("Training returned without a saved selected checkpoint.")
    result["checkpoint"] = str(checkpoint)
    _atomic_json(
        marker_path,
        {
            "fingerprint": fingerprint,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_hash(checkpoint),
            "result": result,
        },
    )
    return result


def _branch_options(config: dict, seed: int, method: str) -> dict:
    """Separate model initialization seeds from the shared validation sampling seed."""
    evaluation = config["evaluation"]
    return {
        **config["training"],
        "seed": seed,
        "validation_seed": evaluation["sampling_seed"],
        "evaluation_batch_size": evaluation[
            "generative_batch_size" if method == "generative" else "sasrec_batch_size"
        ],
        "beam_size": evaluation["beam_size"],
        "optimizer": "adamw",
        "initial_checkpoint": None,
        "sam_rho": config["continuations"]["sam_rho"],
    }


def _train_seed(
    processed_dir: Path,
    artifact_dir: Path,
    seed_dir: Path,
    seed: int,
    config: dict,
    protocol_fingerprint: str,
) -> dict[str, dict]:
    branches = {}
    for method in ("generative", "sasrec"):
        branches[method] = _train_branch(
            processed_dir,
            artifact_dir,
            seed_dir / method,
            method,
            _branch_options(config, seed, method),
            config["models"][method],
            protocol_fingerprint,
        )
    continuation = config["continuations"]
    recipes = [
        ("generative_adamw3", "adamw", continuation["epochs"]),
        ("generative_sam3", "sam", continuation["epochs"]),
    ]
    if continuation.get("adamw_compute_control_epochs") is not None:
        recipes.append(("generative_adamw6", "adamw", continuation["adamw_compute_control_epochs"]))
    for name, optimizer, epochs in recipes:
        options = {
            **_branch_options(config, seed, "generative"),
            "epochs": epochs,
            "patience": epochs,
            "learning_rate": continuation["learning_rate"],
            "optimizer": optimizer,
            "initial_checkpoint": branches["generative"]["checkpoint"],
        }
        branches[name] = _train_branch(
            processed_dir,
            artifact_dir,
            seed_dir / name,
            "generative",
            options,
            config["models"]["generative"],
            protocol_fingerprint,
        )
    return branches


def _evaluate_seed(
    processed_dir: Path,
    artifact_dir: Path,
    seed_dir: Path,
    config: dict,
    branches: dict[str, dict],
    validation: list[dict],
    test: list[dict],
    popularity: np.ndarray,
) -> dict:
    from beautyrec.experiment_training import load_experiment_checkpoint

    evaluation = config["evaluation"]
    device = resolve_device(config["training"]["device"])
    max_cutoff = max(config["cutoffs"])
    test_rankings = {"popularity": popularity_rankings(test, popularity, max_cutoff)}
    timing = {}
    checkpoint_hashes = {
        name: _file_hash(Path(result["checkpoint"])) for name, result in branches.items()
    }
    # This marker records the selected weights before any test list is generated.
    _atomic_json(
        seed_dir / "selection_manifest.json",
        {
            "checkpoint_sha256": checkpoint_hashes,
            "cutoffs": config["cutoffs"],
            "beam_size": evaluation["beam_size"],
            "test_tuning": False,
        },
    )
    for name, result in branches.items():
        model, codebook, checkpoint = load_experiment_checkpoint(
            Path(result["checkpoint"]),
            artifact_dir,
            device,
        )
        check_dataset(processed_dir, checkpoint)
        batch_size = evaluation[
            "sasrec_batch_size" if name == "sasrec" else "generative_batch_size"
        ]
        retrieval_options = {
            "max_history": config["training"]["max_history"],
            "top_k": max_cutoff,
            "beam_size": evaluation["beam_size"],
            "batch_size": batch_size,
        }
        validation_rankings, validation_timing = retrieve_rankings(
            model,
            validation,
            codebook,
            **retrieval_options,
        )
        validation_summary = evaluate_rankings(
            validation,
            {name: validation_rankings},
            popularity,
            cutoffs=config["cutoffs"],
            output_dir=seed_dir / name / "evaluation" / "validation",
        )
        validation_summary.update(
            {
                "split": "validation",
                "used_for_selection": False,
                "retrieval_throughput": validation_timing,
            }
        )
        _atomic_json(
            seed_dir / name / "evaluation" / "validation" / "summary.json", validation_summary
        )
        ranked, throughput = retrieve_rankings(model, test, codebook, **retrieval_options)
        latency = measure_request_latency(
            model,
            test,
            codebook,
            max_history=config["training"]["max_history"],
            top_k=max_cutoff,
            beam_size=evaluation["beam_size"],
            requests=evaluation["request_latency_requests"],
        )
        test_rankings[name] = ranked
        timing[name] = {"throughput": throughput, "request_latency": latency}
        individual = evaluate_rankings(
            test,
            {name: ranked},
            popularity,
            cutoffs=config["cutoffs"],
            output_dir=seed_dir / name / "evaluation" / "test",
        )
        individual.update({"split": "test", "timing": timing[name], "test_tuning": False})
        _atomic_json(seed_dir / name / "evaluation" / "test" / "summary.json", individual)
        del model

    summary = evaluate_rankings(
        test,
        test_rankings,
        popularity,
        cutoffs=config["cutoffs"],
        output_dir=seed_dir / "test",
    )
    for name, measurements in timing.items():
        summary[name]["mean_request_latency_ms"] = measurements["request_latency"]["mean_ms"]
        summary[name]["p95_request_latency_ms"] = measurements["request_latency"]["p95_ms"]
        summary[name]["histories_per_second"] = measurements["throughput"]["histories_per_second"]
    summary.update({"split": "test", "test_tuning": False, "timing": timing})
    _atomic_json(seed_dir / "test" / "summary.json", summary)
    comparisons = {}
    pairs = [
        ("sasrec", "generative"),
        ("popularity", "generative"),
        ("generative_adamw3", "generative_sam3"),
    ]
    if "generative_adamw6" in test_rankings:
        pairs.append(("generative_adamw6", "generative_sam3"))
    for reference, candidate in pairs:
        comparison_name = f"{candidate}_minus_{reference}"
        comparisons[comparison_name] = bootstrap_comparison(
            test,
            test_rankings[reference],
            test_rankings[candidate],
            cutoffs=config["cutoffs"],
            resamples=evaluation["bootstrap_resamples"],
            seed=evaluation["sampling_seed"],
        )
    _atomic_json(seed_dir / "paired_bootstrap.json", comparisons)
    return {"test": summary, "comparisons": comparisons, "training": branches}


def _write_report(run_dir: Path, result: dict) -> None:
    aggregate = result["aggregate_test"]

    def value(method: str, metric: str) -> str:
        item = aggregate[method].get(metric)
        if item is None:
            return "—"
        return (
            f"{item['mean']:.4f} ± {item['std']:.4f}"
            if item["std"] is not None
            else f"{item['mean']:.4f}"
        )

    lines = [
        "# M2 results",
        "",
        "Status: "
        + ("complete on full Amazon Beauty" if result["m2_complete"] else "diagnostic run"),
        "",
        f"Training runs: {aggregate['runs']}. Values are mean ± sample standard deviation "
        "across seeds.",
        "",
        "| Method | Recall@100 | MAP@20 | NDCG@20 | Recall@10 | Request ms | Coverage@20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in aggregate["methods"]:
        cells = [
            value(method, metric)
            for metric in (
                "recall@100",
                "map@20",
                "ndcg@20",
                "recall@10",
                "mean_request_latency_ms",
                "catalog_coverage@20",
            )
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Single-positive MAP equals MRR. Tail groups use training frequencies only. "
            f"Every cutoff uses a prefix of one saved top-{max(result['config']['cutoffs'])} "
            "list, with all earlier warm items excluded.",
            "",
            "Generative retrieval uses approximate constrained beam search; SASRec scores the full "
            "catalog. Single-request latency follows warmup and is separate from "
            "batched throughput. "
            "Batch sizes differ by model and are recorded in each seed's timing artifacts.",
            "",
            "Checkpoint selection uses the fixed validation sample. Full validation scores are "
            "reported separately; test scores do not select models or optimizer branches.",
            "",
            "Paired user-bootstrap intervals are saved separately for each seed in "
            "paired_bootstrap.json. They describe user-sampling uncertainty; "
            "seed standard deviation "
            "describes training variation. Negative or inconclusive results remain valid findings.",
        ]
    )
    if result["diagnostic_reasons"]:
        lines.extend(["", "This run does not complete M2:", ""])
        lines.extend(f"- {reason}" for reason in result["diagnostic_reasons"])
    if not result["config"]["continuations"].get("adamw_compute_control_epochs"):
        lines.extend(
            [
                "",
                "The optional six-epoch AdamW compute control was disabled. "
                "SAM versus three-epoch AdamW alone does not establish an equal-compute advantage.",
            ]
        )
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _completed_result(directory: Path, marker_name: str, protocol_fingerprint: str) -> dict | None:
    """Verify every saved artifact before reusing a completed seed or whole run."""
    marker_path = directory / marker_name
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker["protocol_fingerprint"] != protocol_fingerprint:
        raise ValueError("The completion marker belongs to a different benchmark protocol.")
    for relative, digest in marker["result_files"].items():
        if _file_hash(directory / relative) != digest:
            raise ValueError("A completed result artifact changed; use a new run directory.")
    return json.loads((directory / "results.json").read_text(encoding="utf-8"))


def _mark_completed(directory: Path, marker_name: str, protocol_fingerprint: str) -> None:
    result_files = {
        str(path.relative_to(directory)): _file_hash(path)
        for path in directory.rglob("*")
        if path.is_file() and path.name != marker_name and path.suffix != ".tmp"
    }
    _atomic_json(
        directory / marker_name,
        {
            "protocol_fingerprint": protocol_fingerprint,
            "result_files": result_files,
        },
    )


def run_benchmark(
    processed_dir: Path,
    artifact_dir: Path,
    run_dir: Path,
    config: dict,
    device_override: str | None = None,
) -> dict:
    """Train, evaluate and report the frozen multi-seed M2 experiment."""
    processed_dir, artifact_dir, run_dir = map(Path, (processed_dir, artifact_dir, run_dir))
    config = _configuration(config, device_override)
    protocol = _protocol(processed_dir, artifact_dir, config)
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing["fingerprint"] != protocol["fingerprint"]:
            raise ValueError(
                "Benchmark configuration or input fingerprints changed; use a new run directory."
            )
    else:
        _atomic_json(protocol_path, protocol)
    cached = _completed_result(run_dir, "completed.json", protocol["fingerprint"])
    if cached is not None:
        return cached

    evaluation = config["evaluation"]
    validation_all = read_examples(processed_dir / "validation.jsonl")
    test_all = read_examples(processed_dir / "test.jsonl")
    validation = sample_examples(
        validation_all, evaluation.get("max_validation_examples"), evaluation["sampling_seed"]
    )
    test = sample_examples(
        test_all, evaluation.get("max_test_examples"), evaluation["sampling_seed"]
    )
    if not validation or not test:
        raise ValueError("Final validation and test contexts must both be nonempty.")
    popularity = np.load(processed_dir / "item_popularity.npy", allow_pickle=False)
    seed_results = []
    for seed in config["seeds"]:
        print(f"M2 benchmark seed {seed}", flush=True)
        seed_dir = run_dir / "seeds" / f"seed_{seed}"
        cached_seed = _completed_result(seed_dir, "seed_completed.json", protocol["fingerprint"])
        if cached_seed is not None:
            seed_results.append(cached_seed)
            continue
        branches = _train_seed(
            processed_dir, artifact_dir, seed_dir, seed, config, protocol["fingerprint"]
        )
        seed_result = _evaluate_seed(
            processed_dir,
            artifact_dir,
            seed_dir,
            config,
            branches,
            validation,
            test,
            popularity,
        )
        seed_result["seed"] = seed
        seed_results.append(seed_result)
        _atomic_json(seed_dir / "results.json", seed_result)
        _mark_completed(seed_dir, "seed_completed.json", protocol["fingerprint"])
    aggregate = aggregate_seed_summaries([record["test"] for record in seed_results])
    result = {
        "protocol_fingerprint": protocol["fingerprint"],
        "config": config,
        "diagnostic_reasons": protocol["diagnostic_reasons"],
        "diagnostic_only": bool(protocol["diagnostic_reasons"]),
        "m2_complete": not protocol["diagnostic_reasons"],
        "validation_examples": len(validation),
        "validation_examples_available": len(validation_all),
        "test_examples": len(test),
        "test_examples_available": len(test_all),
        "aggregate_test": aggregate,
        "seed_results": seed_results,
    }
    _atomic_json(run_dir / "results.json", result)
    _write_report(run_dir, result)
    # Save the final marker last. A failed branch can never look like a complete run.
    _mark_completed(run_dir, "completed.json", protocol["fingerprint"])
    return result
