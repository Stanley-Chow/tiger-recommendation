"""Freeze, train and compare semantic tokenizers with matched permutation controls.

The stages are deliberately separate: fit all RQ tokenizers, freeze a common
vocabulary and all codebooks, train every recommender, then freeze all selected
weights before generating any test rankings. No stage chooses settings from test
metrics. Metadata-only tokenizer selection is followed by full-catalog refitting.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from beautyrec.benchmark import (
    _atomic_json,
    _completed_result,
    _file_hash,
    _json_hash,
    _mark_completed,
    _positive_integer,
    _train_branch,
)
from beautyrec.embeddings import read_catalog
from beautyrec.evaluation import aggregate_seed_summaries, bootstrap_comparison, evaluate_rankings
from beautyrec.experiment_training import ExperimentTrainingConfig, load_experiment_checkpoint
from beautyrec.metrics import validate_cutoffs
from beautyrec.model import ModelConfig
from beautyrec.retrieval import measure_request_latency, popularity_rankings, retrieve_rankings
from beautyrec.rqvae import RQVAEConfig, RQVAETrainingConfig, train_rqvae_tokenizer
from beautyrec.semantic_ids import SemanticCodebook, load_semantic_ids
from beautyrec.tokenizer_diagnostics import permute_codebook, tokenizer_diagnostics
from beautyrec.training import check_dataset, read_examples, resolve_device, sample_examples

ARMS = ("hkm", "hkm_permuted", "rqvae", "rqvae_permuted")
CUTOFFS = [10, 20, 50, 100, 200]


def _training_options(config: dict, seed: int) -> dict:
    evaluation = config["evaluation"]
    return {
        **config["training"],
        "seed": seed,
        "validation_seed": evaluation["sampling_seed"],
        "evaluation_batch_size": evaluation["generative_batch_size"],
        "beam_size": evaluation["beam_size"],
        "optimizer": "adamw",
        "sam_rho": 0.05,
        "initial_checkpoint": None,
    }


def _validate(processed: Path, artifacts: Path, config: dict, device_override: str | None):
    """Check finite dimensions, matching inputs and controls before making outputs."""
    config = deepcopy(config)
    for name in ("training", "model", "tokenizer", "evaluation", "diagnostics"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"M3 configuration needs a {name} object.")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Seeds must be distinct integers in [0, 2**32).")
    config["cutoffs"] = list(validate_cutoffs(config.get("cutoffs", CUTOFFS)))
    if device_override is not None:
        config["training"]["device"] = device_override
    evaluation = config["evaluation"]
    for name in (
        "beam_size",
        "generative_batch_size",
        "request_latency_requests",
        "bootstrap_resamples",
    ):
        _positive_integer(evaluation[name], f"evaluation.{name}")
    if evaluation["beam_size"] < max(200, max(config["cutoffs"])):
        raise ValueError("Beam size must cover top-200 and every requested cutoff.")
    for name in ("max_validation_examples", "max_test_examples"):
        if evaluation.get(name) is not None:
            _positive_integer(evaluation[name], f"evaluation.{name}")
    for name in ("sample_size", "neighbors", "batch_size"):
        _positive_integer(config["diagnostics"][name], f"diagnostics.{name}")
    for value in (
        evaluation["sampling_seed"],
        config["diagnostics"]["seed"],
        config["tokenizer"]["permutation_seed_offset"],
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
            raise ValueError(
                "Sampling, diagnostics and permutation seeds must be nonnegative integers."
            )
    if not isinstance(config.get("diagnostic_only", False), bool):
        raise ValueError("diagnostic_only must be a boolean.")
    config["diagnostic_only"] = config.get("diagnostic_only", False)
    for name in (
        "d_model",
        "nhead",
        "num_encoder_layers",
        "num_decoder_layers",
        "dim_feedforward",
    ):
        _positive_integer(config["model"][name], f"model.{name}")
    for seed in seeds:
        ExperimentTrainingConfig(**_training_options(config, seed))
    resolve_device(config["training"]["device"])
    catalog, catalog_hash = read_catalog(processed)
    original = load_semantic_ids(artifacts)
    if original.catalog_sha256 != catalog_hash or len(original.item_codes) != len(catalog):
        raise ValueError("Original HKM codes must match the prepared catalog.")
    if original.diagnostics.get("method") != "hierarchical-kmeans":
        raise ValueError("The source artifact must be the original hierarchical k-means tokenizer.")
    vectors = np.load(artifacts / "embeddings.npy", allow_pickle=False)
    embedding_config = json.loads((artifacts / "embedding_config.json").read_text("utf-8"))
    if embedding_config.get("catalog_sha256") != catalog_hash:
        raise ValueError("Frozen embeddings belong to another catalog.")
    if embedding_config.get("frozen") is not True:
        raise ValueError("The embedding manifest must declare frozen vectors.")
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(catalog)
        or vectors.shape[1] < 1
        or not np.issubdtype(vectors.dtype, np.floating)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Frozen vectors must be finite, real, nonzero and catalog aligned.")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Frozen vectors must be finite, real, nonzero and catalog aligned.")
    rq_model = RQVAEConfig(**config["tokenizer"]["model"])
    if vectors.shape[1] != rq_model.input_dim:
        raise ValueError("RQ input_dim must equal the frozen embedding dimension.")
    if (
        original.code_length != rq_model.num_levels + 1
        or original.branching_factor != rq_model.codebook_size
    ):
        raise ValueError("HKM and RQ need matching semantic depth and per-level code capacity.")
    RQVAETrainingConfig(**config["tokenizer"]["training"])
    ModelConfig(
        vocab_size=original.vocab_size,
        max_source_length=config["training"]["max_history"] * original.code_length + 2,
        max_target_length=original.code_length + 1,
        **config["model"],
    )
    statistics = json.loads((processed / "statistics.json").read_text("utf-8"))
    if config["training"]["max_history"] > statistics["max_history"]:
        raise ValueError("The prepared history window is shorter than the model context.")
    popularity = np.load(processed / "item_popularity.npy", allow_pickle=False)
    if (
        popularity.shape != (len(catalog),)
        or not np.issubdtype(popularity.dtype, np.number)
        or np.issubdtype(popularity.dtype, np.complexfloating)
        or not np.isfinite(popularity).all()
        or np.any(popularity < 0)
        or np.any(popularity != np.floor(popularity))
        or not np.isfinite(popularity.astype(np.float64).sum())
        or popularity.astype(np.float64).sum() <= 0
    ):
        raise ValueError("Training popularity needs catalog-aligned nonnegative integer counts.")
    return config, original, vectors, popularity, statistics, embedding_config


def _protocol(
    processed: Path, artifacts: Path, config: dict, statistics: dict, embedding_config: dict
):
    reasons = []
    if config["diagnostic_only"]:
        reasons.append("The configuration declares a diagnostic run.")
    if config["seeds"] != [42, 43, 44] or config["cutoffs"] != CUTOFFS:
        reasons.append("The full three-seed, five-cutoff protocol was not requested.")
    if (
        statistics.get("raw_reviews") != 198502
        or statistics.get("source_files", {}).get("reviews", {}).get("name")
        != "reviews_Beauty_5.json.gz"
    ):
        reasons.append("The input is not the complete original 2014 Beauty 5-core dataset.")
    if (
        statistics.get("max_users") is not None
        or statistics["users"] != statistics["eligible_users_before_subsampling"]
    ):
        reasons.append("The prepared user cohort was limited.")
    if (
        config["training"].get("max_train_examples") is not None
        or config["training"]["epochs"] < 20
    ):
        reasons.append("Recommender training uses a limited diagnostic budget.")
    if config["tokenizer"]["training"]["epochs"] < 100:
        reasons.append("Tokenizer selection uses a limited diagnostic budget.")
    if any(
        config["evaluation"].get(key) is not None
        for key in ("max_validation_examples", "max_test_examples")
    ):
        reasons.append("Final evaluation contexts were capped.")
    if (
        embedding_config.get("diagnostic_only", False)
        or config["tokenizer"]["model"]["input_dim"] != 384
        or embedding_config.get("embedding_dim") != 384
        or embedding_config.get("model_name") != "sentence-transformers/all-MiniLM-L6-v2"
        or embedding_config.get("normalized") is not True
    ):
        reasons.append(
            "The frozen vectors are diagnostic rather than full 384-dimensional text embeddings."
        )
    package_dir = Path(__file__).parent
    protocol = {
        "schema_version": 1,
        "config": config,
        "diagnostic_reasons": reasons,
        "source_python": {path.name: _file_hash(path) for path in sorted(package_dir.glob("*.py"))},
        "processed_files": {
            name: _file_hash(processed / name)
            for name in (
                "catalog.json",
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
                "item_popularity.npy",
                "statistics.json",
            )
        },
        "artifact_files": {
            name: _file_hash(artifacts / name)
            for name in (
                "semantic_ids.json",
                "embeddings.npy",
                "embedding_config.json",
            )
        },
        "rules": {
            "order": "tokenizers; frozen codebooks; recommenders; selected weights; test",
            "tokenizer_selection": "metadata holdout only, followed by full-catalog refit",
            "capacity": "same padded vocabulary and recommender dimensions for all arms",
            "permutation": "whole codes; seeded uniform permutation; fixed points reported",
            "sampling": "same fixed validation and test contexts across models and seeds",
            "test_tuning": False,
            "uncertainty": "paired user bootstrap separately within each seed",
        },
    }
    protocol["fingerprint"] = _json_hash(protocol)
    return protocol


def _metadata_tokenizers(
    processed: Path, artifacts: Path, run_dir: Path, config: dict, fingerprint: str
):
    books, summaries = {}, {}
    catalog, catalog_hash = read_catalog(processed)
    for seed in config["seeds"]:
        directory = run_dir / "tokenizers" / f"seed_{seed}" / "rqvae"
        cached = _completed_result(directory, "tokenizer_complete.json", fingerprint)
        if cached is None:
            options = {
                **config["tokenizer"]["training"],
                "seed": seed,
                "device": config["training"]["device"],
            }
            cached = train_rqvae_tokenizer(
                processed,
                artifacts,
                directory,
                model_config=config["tokenizer"]["model"],
                training_config=options,
            )
        book = load_semantic_ids(directory)
        if (
            book.catalog_sha256 != catalog_hash
            or len(book.item_codes) != len(catalog)
            or book.diagnostics.get("final_fit_items") != len(catalog)
        ):
            raise ValueError("RQ tokenizer must be refitted on all warm-catalog metadata.")
        if not (directory / "tokenizer_complete.json").exists():
            _atomic_json(directory / "results.json", cached)
            _mark_completed(directory, "tokenizer_complete.json", fingerprint)
        books[seed], summaries[str(seed)] = book, cached
    return books, summaries


def _padded(book: SemanticCodebook, vocab_size: int) -> SemanticCodebook:
    payload = book.to_dict()
    payload["vocab_size"] = vocab_size
    payload["diagnostics"].update(
        {
            "natural_vocab_size": book.vocab_size,
            "padded_vocab_size": vocab_size,
            "suffix_capacity": vocab_size - (3 + book.depth * book.branching_factor),
        }
    )
    return SemanticCodebook(**payload)


def _frozen_arms(
    run_dir: Path,
    original: SemanticCodebook,
    rq_books: dict,
    vectors: np.ndarray,
    popularity: np.ndarray,
    config: dict,
    fingerprint: str,
    embedding_config: dict,
):
    from beautyrec.tokenizer_diagnostics import interaction_weighted_diagnostics

    manifest_path = run_dir / "tokenizer_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text("utf-8"))
        if manifest["protocol_fingerprint"] != fingerprint:
            raise ValueError("Frozen tokenizer manifest belongs to another protocol.")
        content = {key: value for key, value in manifest.items() if key != "fingerprint"}
        if _json_hash(content) != manifest.get("fingerprint"):
            raise ValueError(
                "Frozen tokenizer manifest fingerprint changed; use a new run directory."
            )
        for seed_records in manifest["arms"].values():
            for record in seed_records.values():
                for name, digest in record["files"].items():
                    if _file_hash(Path(record["artifact_dir"]) / name) != digest:
                        raise ValueError(
                            "A frozen tokenizer artifact changed; use a new run directory."
                        )
        return manifest
    common_v = max(original.vocab_size, *(book.vocab_size for book in rq_books.values()))
    manifest = {"protocol_fingerprint": fingerprint, "common_vocab_size": common_v, "arms": {}}
    for seed in config["seeds"]:
        hkm, rq = _padded(original, common_v), _padded(rq_books[seed], common_v)
        permutation_seed = config["tokenizer"]["permutation_seed_offset"] + seed
        hkm_random, hkm_mapping = permute_codebook(hkm, seed=permutation_seed)
        rq_random, rq_mapping = permute_codebook(rq, seed=permutation_seed)
        books = dict(zip(ARMS, (hkm, hkm_random, rq, rq_random)))
        manifest["arms"][str(seed)] = {}
        for arm, book in books.items():
            directory = run_dir / "arms" / f"seed_{seed}" / arm
            directory.mkdir(parents=True, exist_ok=True)
            # Training records the frozen encoder metadata, without needing its matrix.
            _atomic_json(directory / "semantic_ids.json", book.to_dict())
            _atomic_json(directory / "embedding_config.json", embedding_config)
            if arm.endswith("permuted"):
                _atomic_json(
                    directory / "permutation_manifest.json",
                    hkm_mapping if arm.startswith("hkm") else rq_mapping,
                )
            _atomic_json(
                directory / "tokenizer_diagnostics.json",
                tokenizer_diagnostics(book, vectors, **config["diagnostics"]),
            )
            _atomic_json(
                directory / "frequency_weighted_diagnostics.json",
                interaction_weighted_diagnostics(book, popularity),
            )
            files = {path.name: _file_hash(path) for path in directory.iterdir() if path.is_file()}
            manifest["arms"][str(seed)][arm] = {
                "artifact_dir": str(directory.resolve()),
                "files": files,
            }
    manifest["fingerprint"] = _json_hash(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest


def _selected_weights(processed: Path, run_dir: Path, config: dict, tokenizer_manifest: dict):
    branches, selected = {}, {}
    for seed in config["seeds"]:
        branches[seed], selected[str(seed)] = {}, {}
        for arm in ARMS:
            artifacts = Path(tokenizer_manifest["arms"][str(seed)][arm]["artifact_dir"])
            print(f"[M3] Seed {seed}: {arm} recommender", flush=True)
            result = _train_branch(
                processed,
                artifacts,
                run_dir / "seeds" / f"seed_{seed}" / arm,
                "generative",
                _training_options(config, seed),
                config["model"],
                tokenizer_manifest["fingerprint"],
            )
            branches[seed][arm] = result
            selected[str(seed)][arm] = {
                "checkpoint": result["checkpoint"],
                "sha256": _file_hash(Path(result["checkpoint"])),
                "artifact_dir": str(artifacts),
                "parameter_count": result["parameter_count"],
            }
    counts = {
        record["parameter_count"]
        for seed_records in selected.values()
        for record in seed_records.values()
    }
    if len(counts) != 1:
        raise ValueError("All M3 recommender arms must have identical parameter counts.")
    manifest = {
        "tokenizer_manifest_fingerprint": tokenizer_manifest["fingerprint"],
        "selected": selected,
        "cutoffs": config["cutoffs"],
        "beam_size": config["evaluation"]["beam_size"],
        "parameter_count": counts.pop(),
        "test_tuning": False,
    }
    path = run_dir / "selection_manifest.json"
    if path.exists() and json.loads(path.read_text("utf-8")) != manifest:
        raise ValueError("Frozen selected weights changed; use a new run directory.")
    _atomic_json(path, manifest)
    return branches


def _evaluate_seed(
    processed: Path,
    run_dir: Path,
    seed: int,
    config: dict,
    manifest: dict,
    branches: dict,
    validation: list,
    test: list,
    popularity: np.ndarray,
):
    directory = run_dir / "seeds" / f"seed_{seed}"
    cached = _completed_result(directory, "seed_complete.json", manifest["fingerprint"])
    if cached is not None:
        return cached
    evaluation = config["evaluation"]
    max_cutoff = max(config["cutoffs"])
    device = resolve_device(config["training"]["device"])
    rankings = {"popularity": popularity_rankings(test, popularity, max_cutoff)}
    timings = {}
    for arm in ARMS:
        artifacts = Path(manifest["arms"][str(seed)][arm]["artifact_dir"])
        model, codebook, checkpoint = load_experiment_checkpoint(
            Path(branches[arm]["checkpoint"]), artifacts, device
        )
        check_dataset(processed, checkpoint)
        options = {
            "max_history": config["training"]["max_history"],
            "top_k": max_cutoff,
            "beam_size": evaluation["beam_size"],
            "batch_size": evaluation["generative_batch_size"],
        }
        val_rankings, val_timing = retrieve_rankings(model, validation, codebook, **options)
        val_summary = evaluate_rankings(
            validation,
            {arm: val_rankings},
            popularity,
            cutoffs=config["cutoffs"],
            output_dir=directory / arm / "evaluation" / "validation",
        )
        val_summary.update(
            {
                "split": "validation",
                "used_for_selection": False,
                "sample_limit": evaluation.get("max_validation_examples"),
                "selection_validation_sample_limit": config["training"]["validation_examples"],
                "retrieval_timing": val_timing,
            }
        )
        _atomic_json(directory / arm / "evaluation" / "validation" / "summary.json", val_summary)
        rankings[arm], throughput = retrieve_rankings(model, test, codebook, **options)
        latency = measure_request_latency(
            model,
            test,
            codebook,
            max_history=options["max_history"],
            top_k=max_cutoff,
            beam_size=options["beam_size"],
            requests=evaluation["request_latency_requests"],
        )
        timings[arm] = {"throughput": throughput, "request_latency": latency}
        del model
    summary = evaluate_rankings(
        test, rankings, popularity, cutoffs=config["cutoffs"], output_dir=directory / "test"
    )
    for arm in ARMS:
        summary[arm].update(
            {
                "mean_request_latency_ms": timings[arm]["request_latency"]["mean_ms"],
                "p95_request_latency_ms": timings[arm]["request_latency"]["p95_ms"],
                "histories_per_second": timings[arm]["throughput"]["histories_per_second"],
            }
        )
    summary.update({"split": "test", "test_tuning": False, "timing": timings})
    _atomic_json(directory / "test" / "summary.json", summary)
    comparisons = {}
    for reference, candidate in (
        ("hkm_permuted", "hkm"),
        ("rqvae_permuted", "rqvae"),
        ("hkm", "rqvae"),
        ("popularity", "hkm"),
        ("popularity", "rqvae"),
    ):
        comparisons[f"{candidate}_minus_{reference}"] = bootstrap_comparison(
            test,
            rankings[reference],
            rankings[candidate],
            cutoffs=config["cutoffs"],
            resamples=evaluation["bootstrap_resamples"],
            seed=evaluation["sampling_seed"],
        )
    _atomic_json(directory / "paired_bootstrap.json", comparisons)
    result = {"seed": seed, "test": summary, "comparisons": comparisons, "training": branches}
    _atomic_json(directory / "results.json", result)
    _mark_completed(directory, "seed_complete.json", manifest["fingerprint"])
    return result


def _report(run_dir: Path, result: dict) -> None:
    aggregate = result["aggregate_test"]
    lines = [
        "# M3 tokenizer ablation",
        "",
        "Status: " + ("complete" if result["m3_complete"] else "diagnostic"),
        "",
        "Values are mean ± sample standard deviation across pipeline seeds. "
        "RQ arms vary tokenizer and recommender initialization; shuffled arms also vary "
        "the code permutation.",
        "",
        "| Arm | Recall@100 | MAP@20 | NDCG@20 | Coverage@20 | Request ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in aggregate["methods"]:
        cells = []
        for metric in (
            "recall@100",
            "map@20",
            "ndcg@20",
            "catalog_coverage@20",
            "mean_request_latency_ms",
        ):
            item = aggregate[arm].get(metric)
            cells.append(
                "—"
                if item is None
                else f"{item['mean']:.5f}"
                + (f" ± {item['std']:.5f}" if item["std"] is not None else "")
            )
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Primary contrasts compare each tokenizer with its own whole-code permutation. "
            "These preserve token counts, vocabulary and prefix occupancy. Frequency-weighted "
            "usage can change because interactions remain attached to their original items.",
            "",
            "RQ versus HKM is secondary: common vocabulary and recommender capacity are matched, "
            "but prefix occupancy, code distribution and training differ. Metadata locality "
            "and reconstruction are diagnostics, not recommendation accuracy.",
            "",
            "Tokenizers use metadata-only holdout epoch selection and full-catalog refit. "
            "Recommenders use the same fixed validation cohort. All selected weights were "
            "frozen before any test retrieval. Per-seed paired bootstrap intervals are not pooled "
            "across seeds, and guard failures are reported rather than suppressing runs.",
        ]
    )
    if result["diagnostic_reasons"]:
        lines.extend(["", "This run does not complete M3:", ""])
        lines.extend(f"- {reason}" for reason in result["diagnostic_reasons"])
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_m3(
    processed_dir: Path,
    artifact_dir: Path,
    run_dir: Path,
    config: dict,
    device_override: str | None = None,
) -> dict:
    """Run the staged M3 tokenizer ablation, reusing only verified completed work."""
    processed, artifacts, run_dir = map(Path, (processed_dir, artifact_dir, run_dir))
    config, original, vectors, popularity, statistics, embedding_config = _validate(
        processed, artifacts, config, device_override
    )
    protocol = _protocol(processed, artifacts, config, statistics, embedding_config)
    path = run_dir / "protocol.json"
    if not path.exists() and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("The run directory contains unrelated files; use a new run directory.")
    if path.exists() and json.loads(path.read_text("utf-8")) != protocol:
        raise ValueError(
            "M3 source, input or configuration fingerprints changed; use a new run directory."
        )
    _atomic_json(path, protocol)
    cached = _completed_result(run_dir, "completed.json", protocol["fingerprint"])
    if cached is not None:
        return cached
    evaluation = config["evaluation"]
    all_validation = read_examples(processed / "validation.jsonl")
    all_test = read_examples(processed / "test.jsonl")
    validation = sample_examples(
        all_validation, evaluation.get("max_validation_examples"), evaluation["sampling_seed"]
    )
    test = sample_examples(
        all_test, evaluation.get("max_test_examples"), evaluation["sampling_seed"]
    )
    if not validation or not test:
        raise ValueError("Final validation and test cohorts must be nonempty.")
    rq_books, tokenizer_results = _metadata_tokenizers(
        processed, artifacts, run_dir, config, protocol["fingerprint"]
    )
    frozen = _frozen_arms(
        run_dir,
        original,
        rq_books,
        vectors,
        popularity,
        config,
        protocol["fingerprint"],
        embedding_config,
    )
    branches = _selected_weights(processed, run_dir, config, frozen)
    seed_results = [
        _evaluate_seed(
            processed, run_dir, seed, config, frozen, branches[seed], validation, test, popularity
        )
        for seed in config["seeds"]
    ]
    result = {
        "protocol_fingerprint": protocol["fingerprint"],
        "tokenizer_manifest_fingerprint": frozen["fingerprint"],
        "config": config,
        "common_vocab_size": frozen["common_vocab_size"],
        "diagnostic_reasons": protocol["diagnostic_reasons"],
        "diagnostic_only": bool(protocol["diagnostic_reasons"]),
        "m3_complete": not protocol["diagnostic_reasons"],
        "validation_examples": len(validation),
        "validation_examples_available": len(all_validation),
        "test_examples": len(test),
        "test_examples_available": len(all_test),
        "aggregate_test": aggregate_seed_summaries([record["test"] for record in seed_results]),
        "seed_results": seed_results,
        "tokenizer_results": tokenizer_results,
    }
    _atomic_json(run_dir / "results.json", result)
    _report(run_dir, result)
    _mark_completed(run_dir, "completed.json", protocol["fingerprint"])
    return result
