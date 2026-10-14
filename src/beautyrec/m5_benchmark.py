"""Run the frozen temporal M5 study, retaining auditable per-user rankings."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from beautyrec.amazon2023 import TemporalCohortConfig
from beautyrec.benchmark import (
    _atomic_json,
    _completed_result,
    _file_hash,
    _json_hash,
    _mark_completed,
    _positive_integer,
    _train_branch,
)
from beautyrec.content_retrieval import ContentRetriever, reciprocal_rank_fusion
from beautyrec.embeddings import read_catalog
from beautyrec.evaluation import aggregate_seed_summaries, bootstrap_comparison, evaluate_rankings
from beautyrec.experiment_training import ExperimentTrainingConfig, load_experiment_checkpoint
from beautyrec.m4_benchmark import canonical_codebook
from beautyrec.metrics import validate_cutoffs
from beautyrec.model import ModelConfig
from beautyrec.retrieval import measure_request_latency, popularity_rankings, retrieve_rankings
from beautyrec.rqvae import RQVAEConfig, RQVAETrainingConfig, train_rqvae_tokenizer
from beautyrec.sasrec import SASRecConfig
from beautyrec.semantic_ids import load_semantic_ids
from beautyrec.training import check_dataset, read_examples, resolve_device, sample_examples

METHODS = ("generative", "sasrec")
MIXED_CONTRASTS = (
    ("generative", "generative_content"),
    ("sasrec", "sasrec_content"),
    ("sasrec_content", "generative_content"),
)


def _training(config, seed, method):
    return {
        **config["training"],
        "seed": seed,
        "optimizer": "adamw",
        "sam_rho": 0.05,
        "initial_checkpoint": None,
        "validation_seed": config["evaluation"]["sampling_seed"],
        "evaluation_batch_size": config["evaluation"][f"{method}_batch_size"],
        "beam_size": config["evaluation"]["beam_size"],
    }


def _inputs(processed, features, config, device_override):
    config = deepcopy(config)
    seeds = config["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("M5 needs distinct uint32 seeds.")
    if not isinstance(config["diagnostic_only"], bool):
        raise ValueError("diagnostic_only must be a boolean.")
    TemporalCohortConfig(**config["data"])
    config["cutoffs"] = list(validate_cutoffs(config["cutoffs"]))
    if device_override:
        config["training"]["device"] = device_override
    resolve_device(config["training"]["device"])
    for field in (
        "beam_size",
        "generative_batch_size",
        "sasrec_batch_size",
        "content_batch_size",
        "request_latency_requests",
        "bootstrap_resamples",
    ):
        _positive_integer(config["evaluation"][field], field)
    if config["evaluation"]["beam_size"] < max(200, max(config["cutoffs"])):
        raise ValueError("Beam size must cover top-200 and every metric cutoff.")
    for field in ("max_validation_examples", "max_test_examples"):
        if config["evaluation"].get(field) is not None:
            _positive_integer(config["evaluation"][field], field)
    reciprocal_rank_fusion(
        [],
        [],
        top_k=config["fusion"]["candidate_count"],
        content_weight=config["fusion"]["content_weight"],
        rank_constant=config["fusion"]["rank_constant"],
    )
    if config["fusion"]["candidate_count"] != max(config["cutoffs"]):
        raise ValueError("Fusion candidate count must equal the largest saved cutoff.")
    warm, catalog_hash = read_catalog(processed)
    stats = json.loads((processed / "statistics.json").read_text("utf-8"))
    if stats["config"] != config["data"]:
        raise ValueError("Prepared data does not match the declared temporal cohort.")
    if config["training"]["max_history"] > stats["max_history"]:
        raise ValueError("Model context exceeds the prepared history window.")
    for marker, directory, field in (
        ("preprocessing_complete.json", processed, "files"),
        ("features_complete.json", features, "files"),
    ):
        saved = json.loads((directory / marker).read_text("utf-8"))
        for name, sha in saved[field].items():
            if _file_hash(directory / name) != sha:
                raise ValueError(f"Input artifact changed: {name}")
    embedding = json.loads((features / "embedding_config.json").read_text("utf-8"))
    available = json.loads((processed / "available_catalog.json").read_text("utf-8"))
    vectors = np.load(features / "available_embeddings.npy", allow_pickle=False)
    warm_vectors = np.load(features / "embeddings.npy", allow_pickle=False)
    if (
        available[: len(warm)] != warm
        or embedding["catalog_sha256"] != catalog_hash
        or embedding["available_catalog_sha256"] != _file_hash(processed / "available_catalog.json")
        or vectors.ndim != 2
        or vectors.shape[0] != len(available)
        or not np.array_equal(vectors[: len(warm)], warm_vectors)
        or not np.isfinite(vectors).all()
        or embedding.get("frozen") is not True
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Frozen metadata must align with warm and available catalogs.")
    rq = RQVAEConfig(**config["tokenizer"]["model"])
    RQVAETrainingConfig(**config["tokenizer"]["training"])
    if rq.input_dim != vectors.shape[1]:
        raise ValueError("Tokenizer input dimension differs from metadata vectors.")
    for seed in seeds:
        for method in METHODS:
            ExperimentTrainingConfig(**_training(config, seed, method))
    ModelConfig(
        vocab_size=100, max_source_length=82, max_target_length=5, **config["models"]["generative"]
    )
    SASRecConfig(
        num_items=len(warm),
        max_history=config["training"]["max_history"],
        **config["models"]["sasrec"],
    )
    popularity = np.load(processed / "item_popularity.npy", allow_pickle=False)
    if popularity.shape != (len(warm),) or np.any(popularity <= 0):
        raise ValueError("Every warm item must have positive cohort training frequency.")
    return config, stats, embedding, vectors, popularity


def _freeze_tokenizers(processed, features, run, config, fingerprint):
    books = {}
    for seed in config["seeds"]:
        path = run / "tokenizers" / f"seed_{seed}"
        cached = _completed_result(path, "tokenizer_complete.json", fingerprint)
        if cached is None:
            print(f"[M5] Fitting warm-only RQ tokenizer, seed {seed}", flush=True)
            result = train_rqvae_tokenizer(
                processed,
                features,
                path,
                config["tokenizer"]["model"],
                {
                    **config["tokenizer"]["training"],
                    "seed": seed,
                    "device": config["training"]["device"],
                },
            )
            _atomic_json(path / "results.json", result)
            _mark_completed(path, "tokenizer_complete.json", fingerprint)
        books[seed] = load_semantic_ids(path)
    capacity = config["tokenizer"]["model"]["codebook_size"]
    suffix = max(book.vocab_size - (3 + book.depth * capacity) for book in books.values())
    records = {}
    for seed, book in books.items():
        path = run / "arms" / f"seed_{seed}"
        path.mkdir(parents=True, exist_ok=True)
        padded = canonical_codebook(book, capacity, suffix)
        ids = path / "semantic_ids.json"
        payload = padded.to_dict()
        if ids.exists() and json.loads(ids.read_text("utf-8")) != payload:
            raise ValueError("Frozen padded semantic IDs changed.")
        _atomic_json(ids, payload)
        embedding = json.loads((features / "embedding_config.json").read_text("utf-8"))
        _atomic_json(path / "embedding_config.json", embedding)
        records[str(seed)] = {
            "artifact_dir": str(path.resolve()),
            "vocab_size": padded.vocab_size,
            "collision_diagnostics": book.diagnostics,
            "files": {
                name: _file_hash(path / name)
                for name in ("semantic_ids.json", "embedding_config.json")
            },
        }
    manifest = {
        "protocol_fingerprint": fingerprint,
        "seeds": records,
        "common_vocab_size": next(iter(records.values()))["vocab_size"],
    }
    manifest["fingerprint"] = _json_hash(manifest)
    path = run / "tokenizer_manifest.json"
    if path.exists() and json.loads(path.read_text("utf-8")) != manifest:
        raise ValueError("Frozen tokenizer manifest changed.")
    _atomic_json(path, manifest)
    return manifest


def _evaluate_seed(processed, run, seed, config, tokenizers, trained, vectors, popularity):
    directory = run / "seeds" / f"seed_{seed}"
    fingerprint = tokenizers["fingerprint"]
    cached = _completed_result(directory, "seed_complete.json", fingerprint)
    if cached is not None:
        return cached
    e = config["evaluation"]
    contexts = {
        split: sample_examples(
            read_examples(processed / f"{split}.jsonl"),
            e["max_validation_examples" if split == "validation" else "max_test_examples"],
            e["sampling_seed"],
        )
        for split in ("validation", "test", "mixed_test")
    }
    lists = {
        split: {"popularity": popularity_rankings(rows, popularity, max(config["cutoffs"]))}
        for split, rows in contexts.items()
    }
    timing = {}
    artifact = Path(tokenizers["seeds"][str(seed)]["artifact_dir"])
    device = resolve_device(config["training"]["device"])
    for method in METHODS:
        model, book, checkpoint = load_experiment_checkpoint(
            Path(trained[method]["checkpoint"]), artifact, device
        )
        check_dataset(processed, checkpoint)
        options = {
            "max_history": config["training"]["max_history"],
            "top_k": max(config["cutoffs"]),
            "beam_size": e["beam_size"],
        }
        for split, rows in contexts.items():
            # The closed branch receives the same full exclusions, projected to its catalog.
            warm_rows = [
                {
                    **row,
                    "history": row.get("warm_history", row["history"]),
                    "seen_items": row.get("warm_seen_items", row["seen_items"]),
                }
                for row in rows
            ]
            if any(not row["history"] for row in warm_rows):
                raise ValueError("Study cohort must have nonempty warm histories.")
            ranked, observed = retrieve_rankings(
                model,
                warm_rows,
                book,
                batch_size=e[f"{method}_batch_size"],
                progress=True,
                **options,
            )
            lists[split][method] = ranked
            timing[f"{split}/{method}"] = observed
        timing[f"request/{method}"] = measure_request_latency(
            model, contexts["test"], book, requests=e["request_latency_requests"], **options
        )
        del model
    content = ContentRetriever(vectors, device)
    mixed_rows = contexts["mixed_test"]
    lists["mixed_test"]["content"], timing["mixed_test/content"] = content.retrieve(
        mixed_rows,
        top_k=config["fusion"]["candidate_count"],
        max_history=config["training"]["max_history"],
        batch_size=e["content_batch_size"],
    )
    for method in METHODS:
        lists["mixed_test"][f"{method}_content"] = [
            reciprocal_rank_fusion(
                a,
                b,
                top_k=config["fusion"]["candidate_count"],
                content_weight=config["fusion"]["content_weight"],
                rank_constant=config["fusion"]["rank_constant"],
            )
            for a, b in zip(lists["mixed_test"][method], lists["mixed_test"]["content"])
        ]
    available_pop = np.load(processed / "available_popularity.npy", allow_pickle=False)
    summaries, comparisons = {}, {}
    for split, rows in contexts.items():
        summaries[split] = evaluate_rankings(
            rows,
            lists[split],
            available_pop if split == "mixed_test" else popularity,
            cutoffs=config["cutoffs"],
            output_dir=directory / "evaluation" / split,
        )
    for group in ("warm", "new"):
        indices = [i for i, row in enumerate(mixed_rows) if row["target_type"] == group]
        if indices:
            summaries[f"mixed_{group}"] = evaluate_rankings(
                [mixed_rows[i] for i in indices],
                {name: [rows[i] for i in indices] for name, rows in lists["mixed_test"].items()},
                available_pop,
                cutoffs=config["cutoffs"],
                output_dir=directory / "evaluation" / f"mixed_{group}",
            )
        else:
            summaries[f"mixed_{group}"] = None
    for split, contrasts in (
        ("test", (("sasrec", "generative"),)),
        ("mixed_test", MIXED_CONTRASTS),
    ):
        for reference, candidate in contrasts:
            name = f"{split}/{candidate}_minus_{reference}"
            comparisons[name] = bootstrap_comparison(
                contexts[split],
                lists[split][reference],
                lists[split][candidate],
                cutoffs=config["cutoffs"],
                resamples=e["bootstrap_resamples"],
                seed=e["sampling_seed"],
            )
    result = {
        "seed": seed,
        "summaries": summaries,
        "comparisons": comparisons,
        "timing": timing,
        "training": trained,
    }
    _atomic_json(directory / "results.json", result)
    _mark_completed(directory, "seed_complete.json", fingerprint)
    return result


def run_m5(processed: Path, features: Path, run: Path, config: dict, device_override=None):
    """Fit all tokenizers, freeze all weights, evaluate, then export inference."""
    processed, features, run = map(Path, (processed, features, run))
    config, stats, embedding, vectors, popularity = _inputs(
        processed, features, config, device_override
    )
    reasons = []
    if config["diagnostic_only"] or embedding.get("diagnostic_only"):
        reasons.append("Explicit diagnostic configuration or offline metadata features.")
    if config["seeds"] != [42, 43, 44] or config["training"]["epochs"] < 20:
        reasons.append("Incomplete training-seed or epoch budget.")
    if config["tokenizer"]["training"]["epochs"] < 100:
        reasons.append("Reduced tokenizer budget.")
    if any(
        config["evaluation"].get(f) is not None
        for f in ("max_validation_examples", "max_test_examples")
    ):
        reasons.append("Final evaluation contexts capped.")
    if config["training"].get("max_train_examples") is not None:
        reasons.append("Training contexts capped.")
    protocol = {
        "schema_version": 1,
        "milestone": "M5",
        "config": config,
        "diagnostic_reasons": reasons,
        "source_python": {
            p.name: _file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "processed_files": {
            p.name: _file_hash(p) for p in sorted(processed.iterdir()) if p.is_file()
        },
        "feature_files": {p.name: _file_hash(p) for p in sorted(features.iterdir()) if p.is_file()},
        "primary_contrast": "generative minus sasrec, warm test recall@100",
        "mixed_contrasts": [list(pair) for pair in MIXED_CONTRASTS],
        "order": "all warm tokenizers; all selected weights frozen; final evaluation; bundle",
        "serving_seed": config["seeds"][0],
        "test_tuning": False,
    }
    fingerprint = _json_hash(protocol)
    protocol["fingerprint"] = fingerprint
    path = run / "protocol.json"
    if path.exists() and json.loads(path.read_text("utf-8")) != protocol:
        raise ValueError("M5 protocol or source inputs changed; use a new run directory.")
    _atomic_json(path, protocol)
    cached = _completed_result(run, "completed.json", fingerprint)
    if cached is not None:
        return cached
    tokenizers = _freeze_tokenizers(processed, features, run, config, fingerprint)
    trained, selection = {}, {}
    for seed in config["seeds"]:
        artifact = Path(tokenizers["seeds"][str(seed)]["artifact_dir"])
        trained[seed], selection[str(seed)] = {}, {}
        for method in METHODS:
            print(f"[M5] Train {method}, seed {seed}", flush=True)
            result = _train_branch(
                processed,
                artifact,
                run / "seeds" / f"seed_{seed}" / method,
                method,
                _training(config, seed, method),
                config["models"][method],
                fingerprint,
            )
            trained[seed][method] = result
            selection[str(seed)][method] = {
                "checkpoint": result["checkpoint"],
                "sha256": _file_hash(Path(result["checkpoint"])),
                "artifact_dir": str(artifact),
                "guard_passed": result["guard_passed"],
            }
    manifest = {"protocol_fingerprint": fingerprint, "selected": selection, "test_tuning": False}
    path = run / "selection_manifest.json"
    if path.exists() and json.loads(path.read_text("utf-8")) != manifest:
        raise ValueError("Selected model weights changed after freezing.")
    _atomic_json(path, manifest)
    seeds = [
        _evaluate_seed(processed, run, seed, config, tokenizers, trained[seed], vectors, popularity)
        for seed in config["seeds"]
    ]
    aggregate = {
        split: aggregate_seed_summaries([s["summaries"][split] for s in seeds])
        for split in seeds[0]["summaries"]
        if seeds[0]["summaries"][split] is not None
    }
    from beautyrec.serving import export_bundle

    bundle = export_bundle(
        processed, features, run, selection[str(config["seeds"][0])], config, fingerprint
    )
    result = {
        "m5_complete": not reasons,
        "diagnostic_only": bool(reasons),
        "diagnostic_reasons": reasons,
        "protocol_fingerprint": fingerprint,
        "data": stats,
        "common_vocab_size": tokenizers["common_vocab_size"],
        "aggregate": aggregate,
        "seeds": seeds,
        "serving_bundle": str(bundle),
        "limitations": [
            "Bounded training-selected cohort; conditional eligible-target metrics.",
            "2023 metadata snapshot has unverified historical availability.",
            "Single-positive MAP equals MRR; models are not parameter matched.",
            "New-item reach belongs to content retrieval, not collision suffix generation.",
        ],
    }
    _atomic_json(run / "results.json", result)
    _mark_completed(run, "completed.json", fingerprint)
    return result
