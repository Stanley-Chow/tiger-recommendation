"""Compare RQ codebook capacity with fresh, capacity-matched recommenders.

The two arms differ only in tokenizer codebook cardinality. Semantic levels
reserve the same token ranges in both arms; collision suffixes share a range.
All metadata fits precede recommender training, and all selected weights are
frozen before report-only validation diagnostics or test retrieval.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from beautyrec.benchmark import (
    _atomic_json,
    _completed_result,
    _file_hash,
    _json_hash,
    _mark_completed,
    _positive_integer,
    _train_branch,
)
from beautyrec.evaluation import aggregate_seed_summaries, bootstrap_comparison, evaluate_rankings
from beautyrec.experiment_training import load_experiment_checkpoint
from beautyrec.m3_benchmark import _protocol, _training_options, _validate
from beautyrec.m4_diagnostics import diagnose_search
from beautyrec.retrieval import measure_request_latency, popularity_rankings, retrieve_rankings
from beautyrec.rqvae import train_rqvae_tokenizer
from beautyrec.semantic_ids import SemanticCodebook, load_semantic_ids
from beautyrec.tokenizer_diagnostics import interaction_weighted_diagnostics, tokenizer_diagnostics
from beautyrec.training import check_dataset, read_examples, resolve_device, sample_examples


def canonical_codebook(
    book: SemanticCodebook, capacity: int, suffix_slots: int
) -> SemanticCodebook:
    """Reserve identical semantic-token ranges without changing any item partition."""
    if capacity < book.branching_factor or suffix_slots < book.vocab_size - (
        3 + book.depth * book.branching_factor
    ):
        raise ValueError("Canonical token ranges cannot shrink a tokenizer.")
    payload = book.to_dict()
    old_suffix = 3 + book.depth * book.branching_factor
    new_suffix = 3 + book.depth * capacity
    payload["item_codes"] = [
        [
            3 + level * capacity + code[level] - (3 + level * book.branching_factor)
            for level in range(book.depth)
        ]
        + [new_suffix + code[-1] - old_suffix]
        for code in book.item_codes
    ]
    payload["branching_factor"] = capacity
    payload["vocab_size"] = new_suffix + suffix_slots
    payload["diagnostics"] = {
        **payload["diagnostics"],
        "quantizer_codebook_size": book.branching_factor,
        "natural_vocab_size": book.vocab_size,
        "semantic_slots_per_level": capacity,
        "canonical_suffix_start": new_suffix,
        "suffix_offset": new_suffix,
        "semantic_vocab_sizes": [capacity] * book.depth,
        "suffix_capacity": suffix_slots,
        "padded_vocab_size": payload["vocab_size"],
    }
    return SemanticCodebook(**payload)


def _fit_tokenizers(processed, artifacts, run, config, fingerprint):
    books, summaries = {}, {}
    for seed in config["seeds"]:
        books[seed], summaries[str(seed)] = {}, {}
        for size in config["codebook_sizes"]:
            arm = f"rq{size}"
            directory = run / "tokenizers" / f"seed_{seed}" / arm
            cached = _completed_result(directory, "tokenizer_complete.json", fingerprint)
            if cached is None:
                model = {**config["tokenizer"]["model"], "codebook_size": size}
                options = {
                    **config["tokenizer"]["training"],
                    "seed": seed,
                    "device": config["training"]["device"],
                }
                cached = train_rqvae_tokenizer(processed, artifacts, directory, model, options)
                _atomic_json(directory / "results.json", cached)
                _mark_completed(directory, "tokenizer_complete.json", fingerprint)
            book = load_semantic_ids(directory)
            if book.branching_factor != size or book.diagnostics.get("final_fit_items") != len(
                book.item_codes
            ):
                raise ValueError("Tokenizer cardinality or full-catalog refit is inconsistent.")
            books[seed][arm], summaries[str(seed)][arm] = book, cached
    return books, summaries


def _freeze_tokenizers(run, books, vectors, popularity, config, embedding_config, fingerprint):
    path = run / "tokenizer_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text("utf-8"))
        content = {k: v for k, v in manifest.items() if k != "fingerprint"}
        if (
            manifest["protocol_fingerprint"] != fingerprint
            or _json_hash(content) != manifest["fingerprint"]
        ):
            raise ValueError("Frozen tokenizer protocol changed; use a new run directory.")
        for records in manifest["arms"].values():
            for record in records.values():
                for name, expected in record["files"].items():
                    if _file_hash(Path(record["artifact_dir"]) / name) != expected:
                        raise ValueError("A frozen tokenizer artifact changed.")
        return manifest
    capacity = max(config["codebook_sizes"])
    suffix_slots = max(
        book.vocab_size - (3 + book.depth * book.branching_factor)
        for seed_books in books.values()
        for book in seed_books.values()
    )
    manifest = {
        "protocol_fingerprint": fingerprint,
        "semantic_slots_per_level": capacity,
        "suffix_slots": suffix_slots,
        "arms": {},
    }
    for seed, seed_books in books.items():
        manifest["arms"][str(seed)] = {}
        for arm, natural in seed_books.items():
            book = canonical_codebook(natural, capacity, suffix_slots)
            directory = run / "arms" / f"seed_{seed}" / arm
            _atomic_json(directory / "semantic_ids.json", book.to_dict())
            _atomic_json(directory / "embedding_config.json", embedding_config)
            _atomic_json(
                directory / "tokenizer_diagnostics.json",
                tokenizer_diagnostics(book, vectors, **config["diagnostics"]),
            )
            _atomic_json(
                directory / "frequency_weighted_diagnostics.json",
                interaction_weighted_diagnostics(book, popularity),
            )
            manifest["arms"][str(seed)][arm] = {
                "artifact_dir": str(directory.resolve()),
                "files": {p.name: _file_hash(p) for p in directory.iterdir() if p.is_file()},
            }
            manifest["common_vocab_size"] = book.vocab_size
    manifest["fingerprint"] = _json_hash(manifest)
    _atomic_json(path, manifest)
    return manifest


def _train_all(processed, run, config, frozen):
    branches, selected = {}, {}
    for seed in config["seeds"]:
        branches[seed], selected[str(seed)] = {}, {}
        for arm, record in frozen["arms"][str(seed)].items():
            print(f"[M4] Seed {seed}: {arm} recommender", flush=True)
            artifacts = Path(record["artifact_dir"])
            branch = _train_branch(
                processed,
                artifacts,
                run / "seeds" / f"seed_{seed}" / arm,
                "generative",
                _training_options(config, seed),
                config["model"],
                frozen["fingerprint"],
            )
            branches[seed][arm] = branch
            selected[str(seed)][arm] = {
                "checkpoint": branch["checkpoint"],
                "sha256": _file_hash(Path(branch["checkpoint"])),
                "artifact_dir": str(artifacts),
                "parameter_count": branch["parameter_count"],
            }
    counts = {r["parameter_count"] for records in selected.values() for r in records.values()}
    if len(counts) != 1:
        raise ValueError("All M4 recommenders must have identical parameter counts.")
    manifest = {
        "selected": selected,
        "parameter_count": counts.pop(),
        "test_tuning": False,
        "tokenizer_manifest_fingerprint": frozen["fingerprint"],
        "cutoffs": config["cutoffs"],
        "beam_size": config["evaluation"]["beam_size"],
    }
    path = run / "selection_manifest.json"
    if path.exists() and json.loads(path.read_text("utf-8")) != manifest:
        raise ValueError("Selected checkpoint hashes changed; use a new directory.")
    _atomic_json(path, manifest)
    return branches


def _evaluate(processed, run, seed, config, frozen, branches, validation, test, popularity):
    directory = run / "seeds" / f"seed_{seed}"
    cached = _completed_result(directory, "seed_complete.json", frozen["fingerprint"])
    if cached is not None:
        return cached
    evaluation = config["evaluation"]
    options = {
        "max_history": config["training"]["max_history"],
        "top_k": max(config["cutoffs"]),
        "beam_size": evaluation["beam_size"],
        "batch_size": evaluation["generative_batch_size"],
    }
    rankings = {"popularity": popularity_rankings(test, popularity, options["top_k"])}
    timings, diagnostics = {}, {}
    for arm, branch in branches.items():
        print(f"[M4] Seed {seed}: {arm} full validation/test and search diagnostics", flush=True)
        artifacts = Path(frozen["arms"][str(seed)][arm]["artifact_dir"])
        model, book, checkpoint = load_experiment_checkpoint(
            Path(branch["checkpoint"]), artifacts, resolve_device(config["training"]["device"])
        )
        check_dataset(processed, checkpoint)
        val_rankings, val_timing = retrieve_rankings(model, validation, book, **options)
        val_dir = directory / arm / "evaluation" / "validation"
        val_summary = evaluate_rankings(
            validation,
            {arm: val_rankings},
            popularity,
            cutoffs=config["cutoffs"],
            output_dir=val_dir,
        )
        val_summary.update(
            split="validation", used_for_selection=False, retrieval_timing=val_timing
        )
        _atomic_json(val_dir / "summary.json", val_summary)
        diagnostic_rows = sample_examples(
            validation,
            config["search_diagnostics"]["examples"],
            config["search_diagnostics"]["sampling_seed"],
        )
        diagnostics[arm] = diagnose_search(
            model,
            book,
            diagnostic_rows,
            config["training"]["max_history"],
            config["search_diagnostics"],
            directory / arm / "search_diagnostics",
        )
        rankings[arm], throughput = retrieve_rankings(model, test, book, **options)
        latency = measure_request_latency(
            model,
            test,
            book,
            max_history=options["max_history"],
            top_k=options["top_k"],
            beam_size=options["beam_size"],
            requests=evaluation["request_latency_requests"],
        )
        timings[arm] = {"throughput": throughput, "request_latency": latency}
        del model
    summary = evaluate_rankings(
        test, rankings, popularity, cutoffs=config["cutoffs"], output_dir=directory / "test"
    )
    for arm in branches:
        summary[arm].update(
            mean_request_latency_ms=timings[arm]["request_latency"]["mean_ms"],
            p95_request_latency_ms=timings[arm]["request_latency"]["p95_ms"],
            histories_per_second=timings[arm]["throughput"]["histories_per_second"],
        )
    summary.update(split="test", test_tuning=False, timing=timings)
    _atomic_json(directory / "test" / "summary.json", summary)
    arms = list(branches)
    comparisons = {
        f"{arms[1]}_minus_{arms[0]}": bootstrap_comparison(
            test,
            rankings[arms[0]],
            rankings[arms[1]],
            cutoffs=config["cutoffs"],
            resamples=evaluation["bootstrap_resamples"],
            seed=evaluation["sampling_seed"],
        )
    }
    _atomic_json(directory / "paired_bootstrap.json", comparisons)
    result = {
        "seed": seed,
        "test": summary,
        "comparisons": comparisons,
        "training": branches,
        "search_diagnostics": diagnostics,
    }
    _atomic_json(directory / "results.json", result)
    _mark_completed(directory, "seed_complete.json", frozen["fingerprint"])
    return result


def _report(run, result):
    aggregate = result["aggregate_test"]
    lines = [
        "# M4: RQ codebook capacity",
        "",
        "Status: " + ("complete" if result["m4_complete"] else "diagnostic"),
        "",
        "Fractions; mean ± sample SD across three pipeline seeds. No seed/user pooling.",
        "",
        "| Arm | Recall@100 | MAP@20 | NDCG@20 | Coverage@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in aggregate["methods"]:
        cells = [
            f"{aggregate[arm][key]['mean']:.6f} ± {aggregate[arm][key]['std'] or 0:.6f}"
            for key in ("recall@100", "map@20", "ndcg@20", "catalog_coverage@20")
        ]
        lines.append("| " + arm + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The primary contrast is RQ-32 minus RQ-16 on Recall@100. MAP@20 is the selection guard.",
        "Validation search diagnostics do not change the frozen test beam 200.",
        "The test users were evaluated in M2/M3. This follow-up is not a new blind test.",
        "Tokenizer capacity, collisions and stopped epochs can differ despite matched generators.",
    ]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (run / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "metric", "mean", "sample_std"])
        for arm in aggregate["methods"]:
            for key, value in aggregate[arm].items():
                writer.writerow([arm, key, value["mean"], value["std"]])


def run_m4(
    processed_dir: Path,
    artifact_dir: Path,
    run_dir: Path,
    config: dict,
    device_override: str | None = None,
) -> dict:
    """Execute one fixed capacity comparison; resume only matching completed artifacts."""
    processed, artifacts, run = map(Path, (processed_dir, artifact_dir, run_dir))
    config, _, vectors, popularity, statistics, embedding_config = _validate(
        processed, artifacts, config, device_override
    )
    sizes = config.get("codebook_sizes")
    if not isinstance(sizes, list) or len(sizes) != 2 or sizes != sorted(set(sizes)):
        raise ValueError("M4 needs two distinct ascending codebook sizes.")
    for size in sizes:
        _positive_integer(size, "codebook size")
    if sizes[0] != config["tokenizer"]["model"]["codebook_size"]:
        raise ValueError("The base tokenizer configuration must use the smaller codebook size.")
    search = config.get("search_diagnostics", {})
    for key in ("examples", "batch_size"):
        _positive_integer(search.get(key), f"search_diagnostics.{key}")
    if search.get("beam_sizes") != [200, 400] or not isinstance(search.get("sampling_seed"), int):
        raise ValueError("Freeze diagnostic beam widths at 200/400 and specify a sampling seed.")
    protocol = _protocol(processed, artifacts, config, statistics, embedding_config)
    if sizes != [16, 32]:
        protocol["diagnostic_reasons"].append("Not the full pre-specified 16 versus 32 comparison.")
    protocol["milestone"] = "M4"
    protocol["rules"].update(
        primary="rq32 minus rq16 Recall@100",
        capacity="canonical semantic/suffix ranges; identical generator parameters",
        prior_test_exposure="M2/M3 test set already examined",
        search_diagnostics="validation only; widths 200/400; report-only, no test beam tuning",
    )
    protocol.pop("fingerprint")
    protocol["fingerprint"] = _json_hash(protocol)
    path = run / "protocol.json"
    if not path.exists() and run.exists() and any(run.iterdir()):
        raise ValueError("Run directory contains unrelated files.")
    if path.exists() and json.loads(path.read_text("utf-8")) != protocol:
        raise ValueError("M4 source/input/config changed; use a new run directory.")
    _atomic_json(path, protocol)
    cached = _completed_result(run, "completed.json", protocol["fingerprint"])
    if cached is not None:
        return cached
    validation = sample_examples(
        read_examples(processed / "validation.jsonl"),
        config["evaluation"]["max_validation_examples"],
        config["evaluation"]["sampling_seed"],
    )
    test = sample_examples(
        read_examples(processed / "test.jsonl"),
        config["evaluation"]["max_test_examples"],
        config["evaluation"]["sampling_seed"],
    )
    if not validation or not test:
        raise ValueError("Evaluation cohorts must be nonempty.")
    books, tokenizer_results = _fit_tokenizers(
        processed, artifacts, run, config, protocol["fingerprint"]
    )
    frozen = _freeze_tokenizers(
        run, books, vectors, popularity, config, embedding_config, protocol["fingerprint"]
    )
    branches = _train_all(processed, run, config, frozen)
    seed_results = [
        _evaluate(
            processed, run, seed, config, frozen, branches[seed], validation, test, popularity
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
        "m4_complete": not protocol["diagnostic_reasons"],
        "validation_examples": len(validation),
        "test_examples": len(test),
        "seed_results": seed_results,
        "tokenizer_results": tokenizer_results,
        "aggregate_test": aggregate_seed_summaries([record["test"] for record in seed_results]),
    }
    _atomic_json(run / "results.json", result)
    _report(run, result)
    _mark_completed(run, "completed.json", protocol["fingerprint"])
    return result
