"""Independently audit M4 artifacts and every saved ranking without training.

Remote absolute paths are mapped in memory. Original files remain unchanged.
Shared scalar metric reconstruction comes from the established M3 verifier.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from verify_m3_results import (
    PathMapper,
    VerificationError,
    close,
    context,
    read_json,
    require,
    sha256,
    source_artifacts,
    verify_completion,
    verify_file_map,
    verify_guard,
    verify_rankings,
)

from beautyrec.benchmark import _json_hash
from beautyrec.evaluation import context_fingerprint
from beautyrec.experiment_training import ExperimentTrainingConfig, load_experiment_checkpoint
from beautyrec.m3_benchmark import _training_options
from beautyrec.rqvae import ResidualQuantizationVAE, RQVAEConfig, RQVAETrainingConfig
from beautyrec.semantic_ids import load_semantic_ids
from beautyrec.training import check_dataset, read_examples, sample_examples

ROOT = Path(__file__).resolve().parents[1]


def tokenizer_audit(directory, config, seed, size, item_count):
    """Check metadata-only schedule selection and the fresh full-catalog refit."""
    saved = read_json(directory / "rqvae_config.json")
    model_config = RQVAEConfig(**{**config["tokenizer"]["model"], "codebook_size": size})
    training = RQVAETrainingConfig(
        **{
            **config["tokenizer"]["training"],
            "seed": seed,
            "device": config["training"]["device"],
        }
    )
    require(
        saved["model"] == json.loads(json.dumps(asdict(model_config)))
        and saved["training"] == asdict(training),
        "Executed tokenizer configuration differs.",
    )
    summary = read_json(directory / "rqvae_summary.json")
    require(summary["selection_uses_recommendation_labels"] is False, "Tokenizer used labels.")
    selection = read_json(directory / "rqvae_selection_config.json")
    refit = read_json(directory / "rqvae_refit_config.json")
    permutation = np.random.default_rng(training.split_seed).permutation(item_count)
    heldout = min(item_count - 1, max(1, round(item_count * training.validation_fraction)))
    require(
        selection["fit_item_ids"] == sorted(permutation[heldout:].tolist())
        and selection["monitor_item_ids"] == sorted(permutation[:heldout].tolist())
        and refit["fit_item_ids"] == list(range(item_count)),
        "Metadata holdout or all-catalog refit indices differ.",
    )
    with (directory / "rqvae_selection_metrics.csv").open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["phase"] == "quantized"]
    chosen = int(min(rows, key=lambda row: float(row["monitor_quantized_reconstruction"]))["epoch"])
    require(
        chosen
        == saved["selected_quantized_epochs"]
        == refit["selected_quantized_epochs"]
        == summary["selected_quantized_epochs"]
        and summary["refit"]["quantized_epochs_completed"] == chosen
        and summary["refit"]["warmup_epochs"] == training.warmup_epochs,
        "Metadata epoch choice or fixed refit schedule differs.",
    )
    payload = torch.load(directory / "rqvae.pt", map_location="cpu", weights_only=True)
    model = ResidualQuantizationVAE(model_config)
    model.load_state_dict(payload["model_state"], strict=True)
    require(
        payload["model_config"] == asdict(model_config)
        and payload["training_config"] == asdict(training)
        and payload["selected_quantized_epochs"] == chosen
        and payload["final_fit_items"] == item_count
        and sum(parameter.numel() for parameter in model.parameters())
        == summary["parameter_count"],
        "Tokenizer checkpoint, refit or parameter count differs.",
    )


def search_audit(directory, book, examples):
    """Reconstruct prefix coverage, search certificates and token summary statistics."""
    summary = read_json(directory / "summary.json")
    rows = [
        json.loads(line) for line in (directory / "per_user.jsonl").read_text("utf-8").splitlines()
    ]
    require(len(rows) == len(examples), "Search diagnostic cohort size differs.")
    require(
        summary["split"] == "validation" and summary["used_for_selection"] is False,
        "Search diagnostics must remain report-only validation.",
    )
    require(summary["contexts_sha256"] == context_fingerprint(examples), "Search contexts differ.")
    token_logp, correct = [], []
    for row, example in zip(rows, examples):
        require(context(row) == context(example), "Search diagnostic context differs.")
        logs = np.asarray(row["target_token_log_probabilities"])
        require(
            logs.shape == (5,) and np.isfinite(logs).all() and (logs <= 1e-6).all(),
            "Invalid target token likelihoods.",
        )
        require(
            abs(row["target_log_probability"] - float(logs.sum())) <= summary["score_tolerance"],
            "Target sequence likelihood differs beyond float32 tolerance.",
        )
        token_logp.append(logs)
        correct.append(row["teacher_forced_token_argmax_correct"])
        for width in ("200", "400"):
            record = row["beam"][width]
            ids, scores = record["candidates"], record["scores"]
            require(
                len(ids)
                == len(scores)
                == min(200, len(book.item_codes) - len(set(row["seen_items"]))),
                "Search candidate count differs.",
            )
            require(
                len(set(ids)) == len(ids) and not set(ids).intersection(row["seen_items"]),
                "Repeated or previously seen search candidates.",
            )
            require(
                all(isinstance(i, int) and 0 <= i < len(book.item_codes) for i in ids),
                "Out-of-catalog search candidate.",
            )
            require(
                np.isfinite(scores).all() and all(a >= b for a, b in zip(scores, scores[1:])),
                "Search scores are invalid or unsorted.",
            )
            rank = ids.index(row["target"]) + 1 if row["target"] in ids else None
            close(record["rank"], rank, "Search target rank")
            hits = [
                any(book.item_codes[i][:d] == book.item_codes[row["target"]][:d] for i in ids)
                for d in (1, 2, 3)
            ]
            require(record["prefix_hits"] == hits, "Search prefix coverage differs.")
            certificate = (
                rank is None
                and len(ids) == 200
                and row["target_log_probability"] > scores[-1] + summary["score_tolerance"]
            )
            close(record["certified_search_miss"], certificate, "Search-miss certificate")
            if rank is not None:
                require(
                    abs(scores[rank - 1] - row["target_log_probability"])
                    <= summary["score_tolerance"],
                    "Target teacher-forced and decoded scores differ.",
                )
    close(
        summary["mean_target_token_nll"],
        (-np.asarray(token_logp).mean(axis=0)).tolist(),
        "Token NLL",
    )
    close(
        summary["teacher_forced_token_argmax_accuracy"],
        np.asarray(correct).mean(axis=0).tolist(),
        "Token accuracy",
    )
    for width in ("200", "400"):
        records = [row["beam"][width] for row in rows]
        close(
            summary[width]["recall@100"],
            float(np.mean([r["rank"] is not None and r["rank"] <= 100 for r in records])),
            "Search R100",
        )
        close(
            summary[width]["recall@200"],
            float(np.mean([r["rank"] is not None for r in records])),
            "Search R200",
        )
        close(
            summary[width]["certified_search_misses"],
            sum(r["certified_search_miss"] for r in records),
            "Search misses",
        )
        close(
            summary[width]["prefix_coverage@200"],
            np.asarray([r["prefix_hits"] for r in records]).mean(axis=0).tolist(),
            "Prefix coverage",
        )
    return len(rows)


def verify(run: Path, project: Path, allow_diagnostic=False):
    protocol = read_json(run / "protocol.json")
    config, fingerprint = protocol["config"], protocol["fingerprint"]
    require(protocol["milestone"] == "M4", "Wrong milestone.")
    require(
        _json_hash({k: v for k, v in protocol.items() if k != "fingerprint"}) == fingerprint,
        "Protocol fingerprint differs.",
    )
    result = verify_completion(run, "completed.json", fingerprint)
    diagnostic = bool(protocol["diagnostic_reasons"])
    require(not diagnostic or allow_diagnostic, "Diagnostic run cannot establish M4 completion.")
    require(
        result["m4_complete"] == (not diagnostic)
        and result["diagnostic_only"] == diagnostic
        and result["config"] == config
        and result["protocol_fingerprint"] == fingerprint,
        "Result protocol or completion flag differs.",
    )
    if not diagnostic:
        expected = read_json(project / "configs/m4.json")
        expected["training"]["device"] = config["training"]["device"]
        require(
            expected == config
            and config["seeds"] == [42, 43, 44]
            and config["codebook_sizes"] == [16, 32],
            "Full configuration differs.",
        )
    selection, frozen = (
        read_json(run / "selection_manifest.json"),
        read_json(run / "tokenizer_manifest.json"),
    )
    mapper = PathMapper(run, project, selection)
    require(
        _json_hash({k: v for k, v in frozen.items() if k != "fingerprint"})
        == frozen["fingerprint"],
        "Tokenizer manifest fingerprint differs.",
    )
    require(
        frozen["protocol_fingerprint"] == fingerprint
        and selection["test_tuning"] is False
        and selection["tokenizer_manifest_fingerprint"] == frozen["fingerprint"]
        and result["tokenizer_manifest_fingerprint"] == frozen["fingerprint"]
        and result["common_vocab_size"] == frozen["common_vocab_size"]
        and selection["cutoffs"] == config["cutoffs"]
        and selection["beam_size"] == config["evaluation"]["beam_size"],
        "Tokenizer/selection protocol differs.",
    )
    verify_file_map(project / "src/beautyrec", protocol["source_python"])
    first_seed = str(config["seeds"][0])
    arms = [f"rq{k}" for k in config["codebook_sizes"]]
    training_config = read_json(run / "seeds" / f"seed_{first_seed}" / arms[0] / "config.json")
    processed = mapper(training_config["processed_dir"])
    verify_file_map(processed, protocol["processed_files"])
    source_artifacts(project, processed, protocol["artifact_files"])
    validation_all, test_all = (
        read_examples(processed / "validation.jsonl"),
        read_examples(processed / "test.jsonl"),
    )
    validation = sample_examples(
        validation_all,
        config["evaluation"]["max_validation_examples"],
        config["evaluation"]["sampling_seed"],
    )
    test = sample_examples(
        test_all, config["evaluation"]["max_test_examples"], config["evaluation"]["sampling_seed"]
    )
    require(
        result["validation_examples"] == len(validation) and result["test_examples"] == len(test),
        "Result cohort sizes differ.",
    )
    if not diagnostic:
        require(
            len(validation) == 22312
            and len(test) == 22225
            and config["evaluation"]["beam_size"] == 200,
            "Full evaluation cohort or beam differs.",
        )
    popularity = np.load(processed / "item_popularity.npy", allow_pickle=False)
    ranking_rows = search_rows = checkpoints = 0
    natural_suffix = []
    for seed in config["seeds"]:
        seed_dir = run / "seeds" / f"seed_{seed}"
        seed_result = verify_completion(seed_dir, "seed_complete.json", frozen["fingerprint"])
        embedded = next(r for r in result["seed_results"] if r["seed"] == seed)
        require(embedded == seed_result, "Embedded seed summary differs.")
        for size, arm in zip(config["codebook_sizes"], arms):
            tokenizer = run / "tokenizers" / f"seed_{seed}" / arm
            verify_completion(tokenizer, "tokenizer_complete.json", fingerprint)
            raw = load_semantic_ids(tokenizer)
            require(raw.branching_factor == size, "Natural quantizer cardinality differs.")
            tokenizer_audit(tokenizer, config, seed, size, len(popularity))
            slots = raw.vocab_size - 3 - raw.depth * size
            natural_suffix.append(slots)
            artifacts = run / "arms" / f"seed_{seed}" / arm
            verify_file_map(artifacts, frozen["arms"][str(seed)][arm]["files"])
            book = load_semantic_ids(artifacts)
            require(
                len(raw.item_codes) == len(book.item_codes) == len(popularity)
                and raw.catalog_sha256 == book.catalog_sha256,
                "Tokenizer catalog or item count differs.",
            )
            require(
                book.vocab_size == frozen["common_vocab_size"]
                and book.branching_factor == max(config["codebook_sizes"]),
                "Canonical generator vocabulary differs.",
            )
            for before, after in zip(raw.item_codes, book.item_codes):
                require(
                    all(
                        before[level] - 3 - level * size
                        == after[level] - 3 - level * max(config["codebook_sizes"])
                        for level in range(3)
                    ),
                    "Remapping changed a semantic index.",
                )
                require(
                    before[-1] - (3 + 3 * size)
                    == after[-1] - (3 + 3 * max(config["codebook_sizes"])),
                    "Remapping changed a suffix index.",
                )
            branch = seed_dir / arm
            record = selection["selected"][str(seed)][arm]
            require(
                sha256(branch / "best.pt") == record["sha256"], "Selected checkpoint SHA differs."
            )
            model, loaded_book, checkpoint = load_experiment_checkpoint(
                branch / "best.pt", artifacts, "cpu"
            )
            expected_model = {
                "vocab_size": frozen["common_vocab_size"],
                "max_source_length": config["training"]["max_history"] * book.code_length + 2,
                "max_target_length": book.code_length + 1,
                "pad_token": 0,
                **config["model"],
            }
            require(asdict(model.config) == expected_model, "Transformer architecture differs.")
            check_dataset(processed, checkpoint)
            require(
                loaded_book.item_codes == book.item_codes, "Loaded checkpoint uses different codes."
            )
            require(
                checkpoint["training_config"]
                == asdict(ExperimentTrainingConfig(**_training_options(config, seed))),
                "Executed training configuration differs.",
            )
            require(
                checkpoint["training_config"]["initial_checkpoint"] is None,
                "Model was not trained fresh.",
            )
            require(
                sum(p.numel() for p in model.parameters())
                == record["parameter_count"]
                == selection["parameter_count"],
                "Generator capacity differs.",
            )
            verify_guard(branch, checkpoint, validation_all, popularity, config, seed)
            checkpoints += 1
            del model
            val_dir = branch / "evaluation/validation"
            val_summary = read_json(val_dir / "summary.json")
            require(val_summary["used_for_selection"] is False, "Full validation selected weights.")
            ranking_rows += verify_rankings(
                val_dir / "per_user_rankings.jsonl",
                val_summary,
                validation,
                popularity,
                [arm],
                config["cutoffs"],
            )["rows"]
            cohort = sample_examples(
                validation,
                config["search_diagnostics"]["examples"],
                config["search_diagnostics"]["sampling_seed"],
            )
            search_rows += search_audit(branch / "search_diagnostics", book, cohort)
        summary = read_json(seed_dir / "test/summary.json")
        ranking_rows += verify_rankings(
            seed_dir / "test/per_user_rankings.jsonl",
            summary,
            test,
            popularity,
            ["popularity", *arms],
            config["cutoffs"],
        )["rows"]
        comparison = seed_result["comparisons"][f"{arms[1]}_minus_{arms[0]}"]
        require(
            comparison["examples"] == len(test)
            and comparison["users"] == len({row["user_id"] for row in test})
            and comparison["contexts_sha256"] == context_fingerprint(test)
            and comparison["resamples"] == config["evaluation"]["bootstrap_resamples"]
            and comparison["seed"] == config["evaluation"]["sampling_seed"]
            and comparison["confidence"] == 0.95
            and comparison["cutoffs"] == config["cutoffs"]
            and comparison["interval_method"] == "paired percentile bootstrap"
            and comparison["difference"] == "candidate minus reference",
            "Bootstrap cohort or paired-resampling metadata differs.",
        )
        for metric, value in comparison["delta"].items():
            close(
                value["mean"],
                summary[arms[1]][metric] - summary[arms[0]][metric],
                "Paired contrast mean",
            )
            require(
                np.isfinite([value["mean"], value["lower"], value["upper"]]).all()
                and value["lower"] <= value["upper"],
                "Invalid bootstrap interval.",
            )
    require(
        frozen["common_vocab_size"] == 3 + 3 * max(config["codebook_sizes"]) + max(natural_suffix),
        "Common suffix capacity is not the largest observed group.",
    )
    aggregate = result["aggregate_test"]
    aggregate_pairs = 0
    for arm in aggregate["methods"]:
        for key, values in aggregate[arm].items():
            samples = [r["test"][arm][key] for r in result["seed_results"]]
            close(values["mean"], float(np.mean(samples)), "Aggregate mean")
            close(
                values["std"],
                float(np.std(samples, ddof=1)) if len(samples) > 1 else None,
                "Aggregate sample SD",
            )
            aggregate_pairs += 1
    return {
        "status": "verified_diagnostic" if diagnostic else "verified_full_m4",
        "selected_checkpoints": checkpoints,
        "generator_parameter_count": selection["parameter_count"],
        "ranking_context_rows_checked": ranking_rows,
        "search_diagnostic_rows_checked": search_rows,
        "aggregate_metric_pairs_checked": aggregate_pairs,
        "common_vocab_size": frozen["common_vocab_size"],
        "test_users_per_seed": len(test),
        "validation_users_per_arm": len(validation),
        "completion_sha256": sha256(run / "completed.json"),
        "results_sha256": sha256(run / "results.json"),
        "bootstrap_draws_recreated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/m4")
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        report = verify(args.run_dir.resolve(), args.project_dir.resolve(), args.allow_diagnostic)
    except (VerificationError, ValueError, KeyError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
