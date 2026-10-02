"""Read-only audit of downloaded M3 artifacts, checkpoints, and saved rankings.

Run with the project's Python environment:
    python scripts/verify_m3_results.py --run-dir results/m3 --project-dir .

Recorded remote absolute paths are mapped in memory; no artifact is rewritten.
Every saved validation/test ranking is checked and its metrics are reconstructed
from the source contexts. No training or recommendation decoding is performed.
Use --allow-diagnostic solely for a declared smoke run. JSON goes to stdout and
any failed check returns exit status 1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from beautyrec.benchmark import _completed_result, _json_hash  # noqa: E402
from beautyrec.evaluation import context_fingerprint  # noqa: E402
from beautyrec.experiment_training import (  # noqa: E402
    ExperimentTrainingConfig,
    load_experiment_checkpoint,
)
from beautyrec.m3_benchmark import ARMS, _training_options  # noqa: E402
from beautyrec.semantic_ids import load_semantic_ids  # noqa: E402
from beautyrec.training import check_dataset, read_examples, sample_examples  # noqa: E402


class VerificationError(ValueError):
    """An artifact contradicts its recorded or frozen protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_parts(value: str | Path) -> tuple[str, ...]:
    return PurePosixPath(str(value).replace("\\", "/")).parts


def contained(base: Path, parts: tuple[str, ...]) -> Path:
    require(not any(part in {"..", "."} for part in parts), "Unsafe artifact path.")
    target = base.joinpath(*parts).resolve()
    require(target.is_relative_to(base.resolve()), "Artifact path escapes its mapped directory.")
    return target


class PathMapper:
    """Map the selected checkpoint's recorded run/project prefix to this checkout."""

    def __init__(self, run_dir: Path, project_dir: Path, selection: dict):
        self.run_dir, self.project_dir = run_dir.resolve(), project_dir.resolve()
        first_seed = next(iter(selection["selected"]))
        first_arm = next(iter(selection["selected"][first_seed]))
        checkpoint = selection["selected"][first_seed][first_arm]["checkpoint"]
        parts = path_parts(checkpoint)
        suffix = ("seeds", f"seed_{first_seed}", first_arm, "best.pt")
        require(parts[-4:] == suffix, "Cannot identify recorded run prefix from checkpoint.")
        self.remote_run = parts[:-4]
        # Normal runs live in <project>/results/<run name>, even if renamed locally.
        results_positions = [i for i, part in enumerate(self.remote_run) if part == "results"]
        require(bool(results_positions), "Recorded run path has no project/results prefix.")
        self.remote_project = self.remote_run[: results_positions[-1]]

    def __call__(self, value: str | Path) -> Path:
        parts = path_parts(value)
        if parts[: len(self.remote_run)] == self.remote_run:
            return contained(self.run_dir, parts[len(self.remote_run) :])
        if parts[: len(self.remote_project)] == self.remote_project:
            return contained(self.project_dir, parts[len(self.remote_project) :])
        # Explicit suffix mapping also supports moving a remote result directory.
        for i in range(len(parts) - 1):
            if parts[i : i + 2] == ("results", self.remote_run[-1]):
                return contained(self.run_dir, parts[i + 2 :])
        raise VerificationError(f"Cannot map recorded absolute path: {value}")


def verify_completion(directory: Path, marker_name: str, fingerprint: str) -> dict:
    marker = read_json(directory / marker_name)
    require(bool(marker.get("result_files")), f"Empty completion marker: {directory}")
    for relative in marker["result_files"]:
        parts = path_parts(relative)
        require(
            not PurePosixPath(str(relative).replace("\\", "/")).is_absolute(),
            f"Completion path must be relative: {relative}",
        )
        contained(directory, parts)
    result = _completed_result(directory, marker_name, fingerprint)
    require(result is not None, f"Missing completion evidence: {directory}")
    return result


def verify_file_map(directory: Path, records: dict[str, str]) -> None:
    for relative, expected in records.items():
        target = contained(directory, path_parts(relative))
        require(sha256(target) == expected, f"SHA-256 mismatch: {target}")


def source_artifacts(project: Path, processed: Path, records: dict[str, str]) -> Path:
    candidates = [project / "artifacts" / "beauty", processed.parent / "artifacts"]
    if (project / "artifacts").is_dir():
        candidates.extend(path for path in (project / "artifacts").iterdir() if path.is_dir())
    for candidate in dict.fromkeys(candidates):
        if all((candidate / name).is_file() for name in records):
            if all(sha256(candidate / name) == expected for name, expected in records.items()):
                return candidate
    raise VerificationError(
        "No local original tokenizer/embedding directory matches protocol hashes."
    )


def close(actual: Any, expected: Any, label: str) -> None:
    if expected is None:
        require(actual is None, f"Expected null at {label}.")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10),
            f"Numeric mismatch at {label}: saved={actual}, reconstructed={expected}",
        )
    else:
        require(actual == expected, f"Mismatch at {label}: {actual!r} != {expected!r}")


def context(row: dict) -> dict:
    return {
        "user_id": row["user_id"],
        "history": row["history"],
        "seen_items": row.get("seen_items", row["history"]),
        "target": row["target"],
    }


def item_metrics(rank: int | None, cutoffs: list[int]) -> dict[str, float]:
    values = {}
    for cutoff in cutoffs:
        hit = rank is not None and rank <= cutoff
        values[f"recall@{cutoff}"] = float(hit)
        values[f"ndcg@{cutoff}"] = 1 / math.log2(rank + 1) if hit else 0.0
        values[f"map@{cutoff}"] = 1 / rank if hit else 0.0
        values[f"mrr@{cutoff}"] = values[f"map@{cutoff}"]
    return values


def verify_rankings(
    path: Path,
    summary: dict,
    examples: list[dict],
    popularity: np.ndarray,
    methods: list[str],
    cutoffs: list[int],
) -> dict:
    """Stream all rows and reconstruct metrics with independent scalar formulas."""
    n_items, maximum = len(popularity), max(cutoffs)
    threshold = float(np.median(popularity))
    sums = {method: {key: 0.0 for key in item_metrics(None, cutoffs)} for method in methods}
    tails = {method: dict(values) for method, values in sums.items()}
    coverage = {method: {cutoff: set() for cutoff in cutoffs} for method in methods}
    lengths = dict.fromkeys(methods, 0)
    tail_examples, rows = 0, 0
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            row = json.loads(line)
            require(index < len(examples), f"Extra ranking row in {path}")
            require(
                row.get("example_index") == index and context(row) == context(examples[index]),
                f"Source context mismatch at {path}:{index + 1}",
            )
            require(list(row["methods"]) == methods, f"Methods differ at {path}:{index + 1}")
            seen = set(row["seen_items"])
            target = row["target"]
            require(target not in seen, f"Target was previously seen at {path}:{index + 1}")
            require(set(row["history"]) <= seen, f"Incomplete exclusions at {path}:{index + 1}")
            is_tail = popularity[target] <= threshold
            tail_examples += int(is_tail)
            for method in methods:
                record = row["methods"][method]
                ranked = record["ranked_items"]
                label = f"{path}:{index + 1}/{method}"
                require(
                    all(
                        isinstance(item, int) and not isinstance(item, bool) and 0 <= item < n_items
                        for item in ranked
                    ),
                    f"Padding or out-of-catalog item at {label}",
                )
                require(len(set(ranked)) == len(ranked), f"Repeated recommendation at {label}")
                require(not (seen & set(ranked)), f"Previously seen recommendation at {label}")
                require(
                    len(ranked) == min(maximum, n_items - len(seen)),
                    f"Unexpected returned length at {label}: {len(ranked)}",
                )
                rank = ranked.index(target) + 1 if target in ranked else None
                close(record["target_rank"], rank, f"{label}/target_rank")
                values = item_metrics(rank, cutoffs)
                require(set(record["metrics"]) == set(values), f"Metric keys differ at {label}")
                for key, value in values.items():
                    close(record["metrics"][key], value, f"{label}/{key}")
                    sums[method][key] += value
                    if is_tail:
                        tails[method][key] += value
                lengths[method] += len(ranked)
                for cutoff in cutoffs:
                    coverage[method][cutoff].update(ranked[:cutoff])
            rows += 1
    require(rows == len(examples), f"Missing ranking rows in {path}: {rows}/{len(examples)}")
    for key, expected in {
        "examples": rows,
        "users": len({row["user_id"] for row in examples}),
        "catalog_items": n_items,
        "cutoffs": cutoffs,
        "methods": methods,
        "exclude_seen": True,
        "contexts_sha256": context_fingerprint(examples),
        "training_popularity_sha256": hashlib.sha256(
            popularity.astype("<f8").tobytes()
        ).hexdigest(),
        "tail_training_count_threshold": threshold,
    }.items():
        close(summary.get(key), expected, f"{path}/{key}")
    reconstructed = {}
    for method in methods:
        values = {key: value / rows for key, value in sums[method].items()}
        values.update(
            {
                f"tail_{key}": value / tail_examples if tail_examples else None
                for key, value in tails[method].items()
            }
        )
        values.update(
            {
                f"catalog_coverage@{cutoff}": len(coverage[method][cutoff]) / n_items
                for cutoff in cutoffs
            }
        )
        values.update(
            {
                "tail_examples": tail_examples,
                "mean_returned_items": lengths[method] / rows,
                "catalog_coverage": values[f"catalog_coverage@{maximum}"],
            }
        )
        for key, value in values.items():
            close(summary[method].get(key), value, f"{path}/{method}/{key}")
        reconstructed[method] = values
    return {"rows": rows, "methods": methods, "metrics": reconstructed}


def verify_guard(
    branch: Path,
    checkpoint: dict,
    validation: list[dict],
    popularity: np.ndarray,
    config: dict,
    seed: int,
) -> None:
    cohort = sample_examples(
        validation, config["training"]["validation_examples"], config["evaluation"]["sampling_seed"]
    )
    require(
        checkpoint["validation_contexts_sha256"] == context_fingerprint(cohort),
        f"Selection cohort differs for seed {seed}/{branch.name}.",
    )
    order = np.lexsort((np.arange(len(popularity)), -popularity))
    total = 0.0
    cutoff = int(config["training"]["guard_metric"].split("@")[1])
    for row in cohort:
        seen = set(row["seen_items"])
        eligible = []
        for item in order:
            if item not in seen:
                eligible.append(int(item))
                if len(eligible) == cutoff:
                    break
        if row["target"] in eligible:
            total += 1 / (eligible.index(row["target"]) + 1)
    threshold = total / len(cohort) - config["training"]["map_guard_tolerance"]
    close(checkpoint["guard_threshold"], threshold, f"{branch}/guard_threshold")
    with (branch / "metrics.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    require(bool(rows), f"Missing epoch metrics for {branch}")
    metric, guard = config["training"]["selection_metric"], config["training"]["guard_metric"]
    ranked = []
    for row in rows:
        guarded = float(row[f"validation_{guard}"])
        close(row["guard_passed"] == "True", guarded >= threshold, f"{branch}/epoch guard")
        ranked.append((guarded >= threshold, float(row[f"validation_{metric}"]), guarded))
    best = max(range(len(rows)), key=ranked.__getitem__)
    require(
        int(rows[best]["epoch"]) == checkpoint["epoch"], f"Selected epoch is not best at {branch}"
    )
    close(checkpoint["guard_passed"], ranked[best][0], f"{branch}/selected guard")


def verify(
    run_dir: Path, project: Path, allow_diagnostic: bool, expected_config: Path | None
) -> dict:
    report: dict[str, Any] = {"status": "running", "run_dir": str(run_dir), "checks": []}
    protocol = read_json(run_dir / "protocol.json")
    config = protocol["config"]
    fingerprint = protocol["fingerprint"]
    require(
        _json_hash({key: value for key, value in protocol.items() if key != "fingerprint"})
        == fingerprint,
        "Protocol content fingerprint does not match.",
    )
    result = verify_completion(run_dir, "completed.json", fingerprint)
    selection = read_json(run_dir / "selection_manifest.json")
    mapper = PathMapper(run_dir, project, selection)
    diagnostic = bool(protocol["diagnostic_reasons"])
    require(
        not diagnostic or allow_diagnostic,
        "Diagnostic run cannot verify full M3; use --allow-diagnostic.",
    )
    close(result["m3_complete"], not diagnostic, "results/m3_complete")
    require(
        result["protocol_fingerprint"] == fingerprint and result["config"] == config,
        "Result/protocol configuration differs.",
    )
    if not diagnostic:
        require(config["seeds"] == [42, 43, 44], "Full M3 requires seeds 42/43/44.")
        require(
            selection["beam_size"] == config["evaluation"]["beam_size"] == 200,
            "Full M3 requires frozen beam width 200.",
        )
        expected_config = expected_config or project / "configs" / "m3.json"
    if expected_config is not None:
        expected = read_json(expected_config)
        # Device is the documented runtime override; scientific settings stay exact.
        expected["training"]["device"] = config["training"]["device"]
        require(expected == config, f"Executed configuration differs from {expected_config}.")
    report["diagnostic_only"] = diagnostic
    report["checks"].append("complete relative-file hashes and frozen protocol/config")
    first_seed = str(config["seeds"][0])
    first_config = read_json(run_dir / "seeds" / f"seed_{first_seed}" / "hkm" / "config.json")
    processed = mapper(first_config["processed_dir"])
    verify_file_map(processed, protocol["processed_files"])
    originals = source_artifacts(project, processed, protocol["artifact_files"])
    require(
        set(protocol["source_python"])
        == {path.name for path in (project / "src" / "beautyrec").glob("*.py")},
        "Local Python source inventory differs from the frozen protocol.",
    )
    verify_file_map(project / "src" / "beautyrec", protocol["source_python"])
    report["processed_dir"], report["original_artifact_dir"] = str(processed), str(originals)
    report["checks"].append("current local source and unchanged input SHA-256 hashes")
    statistics = read_json(processed / "statistics.json")
    catalog = read_json(processed / "catalog.json")
    popularity = np.load(processed / "item_popularity.npy", allow_pickle=False)
    validation_all, test_all = (
        read_examples(processed / "validation.jsonl"),
        read_examples(processed / "test.jsonl"),
    )
    with (processed / "train.jsonl").open(encoding="utf-8") as source:
        train_count = sum(bool(line.strip()) for line in source)
    require(
        train_count == statistics["examples"]["train"]
        and len(catalog) == statistics["items"]
        and len(validation_all) == statistics["examples"]["validation"]
        and len(test_all) == statistics["examples"]["test"]
        and len(popularity) == len(catalog),
        "Source cohort counts differ from prepared statistics.",
    )
    if not diagnostic:
        require(
            (
                statistics["raw_reviews"],
                statistics["users"],
                len(catalog),
                train_count,
                len(validation_all),
                len(test_all),
            )
            == (198502, 22363, 12068, 131413, 22312, 22225),
            "Full Beauty counts do not match the frozen M3 protocol.",
        )
        require(
            config["evaluation"]["max_validation_examples"] is None
            and config["evaluation"]["max_test_examples"] is None,
            "Full evaluation was capped.",
        )
    validation = sample_examples(
        validation_all,
        config["evaluation"]["max_validation_examples"],
        config["evaluation"]["sampling_seed"],
    )
    test = sample_examples(
        test_all, config["evaluation"]["max_test_examples"], config["evaluation"]["sampling_seed"]
    )
    close(result["validation_examples"], len(validation), "result validation count")
    close(result["test_examples"], len(test), "result test count")
    original = load_semantic_ids(originals)
    require(
        original.catalog_sha256 == sha256(processed / "catalog.json"),
        "Original HKM catalog fingerprint differs from the source catalog.",
    )
    frozen = read_json(run_dir / "tokenizer_manifest.json")
    require(
        _json_hash({key: value for key, value in frozen.items() if key != "fingerprint"})
        == frozen["fingerprint"],
        "Tokenizer manifest content fingerprint differs.",
    )
    require(
        frozen["protocol_fingerprint"] == fingerprint
        and selection["tokenizer_manifest_fingerprint"] == frozen["fingerprint"]
        and result["tokenizer_manifest_fingerprint"] == frozen["fingerprint"],
        "Frozen tokenizer fingerprints disagree.",
    )
    seeds = [str(seed) for seed in config["seeds"]]
    require(
        set(frozen["arms"]) == set(selection["selected"]) == set(seeds), "Seed manifest differs."
    )
    natural_vocab = [original.vocab_size]
    for seed in seeds:
        raw_dir = run_dir / "tokenizers" / f"seed_{seed}" / "rqvae"
        verify_completion(raw_dir, "tokenizer_complete.json", fingerprint)
        verify_file_map(raw_dir, read_json(raw_dir / "rqvae_completed.json")["artifact_sha256"])
        raw_book = load_semantic_ids(raw_dir)
        require(
            raw_book.catalog_sha256 == original.catalog_sha256
            and raw_book.diagnostics["final_fit_items"] == len(catalog),
            "RQ refit catalog differs.",
        )
        natural_vocab.append(raw_book.vocab_size)
        require(
            set(frozen["arms"][seed]) == set(selection["selected"][seed]) == set(ARMS),
            f"Missing arm for seed {seed}.",
        )
        books = {}
        for arm in ARMS:
            record = frozen["arms"][seed][arm]
            directory = mapper(record["artifact_dir"])
            require(
                directory == (run_dir / "arms" / f"seed_{seed}" / arm).resolve(),
                f"Unexpected mapped artifact path for {seed}/{arm}.",
            )
            verify_file_map(directory, record["files"])
            book = load_semantic_ids(directory)
            require(
                book.catalog_sha256 == original.catalog_sha256
                and len(book.item_codes) == len(catalog)
                and book.code_length == original.code_length
                and book.vocab_size == frozen["common_vocab_size"],
                f"Code capacity differs: {seed}/{arm}",
            )
            books[arm] = book
        require(
            books["hkm"].item_codes == original.item_codes, f"HKM codes changed for seed {seed}."
        )
        require(
            books["rqvae"].item_codes == raw_book.item_codes,
            f"RQ codes changed after refit: {seed}",
        )
        mappings = []
        for semantic, permuted in (("hkm", "hkm_permuted"), ("rqvae", "rqvae_permuted")):
            mapping = read_json(
                run_dir / "arms" / f"seed_{seed}" / permuted / "permutation_manifest.json"
            )
            values = mapping["item_to_source_item"]
            require(
                sorted(values) == list(range(len(catalog))),
                f"Non-bijective permutation: {seed}/{permuted}",
            )
            require(
                mapping["seed"] == config["tokenizer"]["permutation_seed_offset"] + int(seed),
                f"Wrong permutation seed: {seed}/{permuted}",
            )
            require(
                books[permuted].item_codes == [books[semantic].item_codes[item] for item in values],
                f"Whole-code permutation mismatch: {seed}/{permuted}",
            )
            require(
                sorted(books[permuted].item_codes) == sorted(books[semantic].item_codes),
                f"Codeword set changed: {seed}/{permuted}",
            )
            mappings.append(values)
        require(mappings[0] == mappings[1], f"HKM/RQ item permutations differ for seed {seed}.")
    require(frozen["common_vocab_size"] == max(natural_vocab), "Common V is not maximum natural V.")
    report["checks"].append(
        "all tokenizer hashes, full catalog, unchanged HKM, common V and bijective controls"
    )
    parameter_counts, checkpoint_count = set(), 0
    for seed in seeds:
        for arm in ARMS:
            record = selection["selected"][seed][arm]
            checkpoint_path = mapper(record["checkpoint"])
            branch = run_dir / "seeds" / f"seed_{seed}" / arm
            require(
                checkpoint_path == (branch / "best.pt").resolve(),
                f"Unexpected selected checkpoint: {seed}/{arm}",
            )
            require(
                sha256(checkpoint_path) == record["sha256"],
                f"Selected checkpoint hash differs: {seed}/{arm}",
            )
            artifacts = mapper(record["artifact_dir"])
            require(
                artifacts == (run_dir / "arms" / f"seed_{seed}" / arm).resolve(),
                f"Selected checkpoint refers to the wrong tokenizer: {seed}/{arm}",
            )
            model, book, checkpoint = load_experiment_checkpoint(checkpoint_path, artifacts, "cpu")
            check_dataset(processed, checkpoint)
            expected_training = asdict(
                ExperimentTrainingConfig(**_training_options(config, int(seed)))
            )
            require(
                checkpoint["training_config"] == expected_training,
                f"Training settings differ: {seed}/{arm}",
            )
            actual = sum(
                checkpoint["model_state"][name].numel() for name, _ in model.named_parameters()
            )
            require(
                actual
                == sum(parameter.numel() for parameter in model.parameters())
                == record["parameter_count"]
                == selection["parameter_count"],
                f"Tensor parameter count differs: {seed}/{arm}",
            )
            require(
                model.config.vocab_size == book.vocab_size == frozen["common_vocab_size"],
                f"Checkpoint vocabulary differs: {seed}/{arm}",
            )
            for name, value in config["model"].items():
                require(
                    getattr(model.config, name) == value,
                    f"Model dimension differs: {seed}/{arm}/{name}",
                )
            marker = read_json(branch / "benchmark_training_complete.json")
            require(
                marker["checkpoint_sha256"] == record["sha256"]
                and mapper(marker["checkpoint"]) == checkpoint_path,
                f"Training marker checkpoint differs: {seed}/{arm}",
            )
            require(
                marker["fingerprint"]
                == _json_hash(
                    {
                        "method": "generative",
                        "training": _training_options(config, int(seed)),
                        "model": config["model"],
                        "protocol_fingerprint": frozen["fingerprint"],
                    }
                ),
                f"Training marker protocol differs: {seed}/{arm}",
            )
            verify_guard(branch, checkpoint, validation_all, popularity, config, int(seed))
            parameter_counts.add(actual)
            checkpoint_count += 1
            del model, checkpoint
    require(
        len(parameter_counts) == 1 and checkpoint_count == len(seeds) * 4,
        "Incomplete matched checkpoint set.",
    )
    require(
        selection["test_tuning"] is False and selection["cutoffs"] == config["cutoffs"],
        "Selection/test settings differ.",
    )
    report["selected_checkpoints"], report["parameter_count"] = (
        checkpoint_count,
        parameter_counts.pop(),
    )
    report["checks"].append(
        "checkpoint tensor counts/shapes, data, fresh training and guard selection"
    )
    ranking_rows = 0
    for seed in seeds:
        directory = run_dir / "seeds" / f"seed_{seed}"
        verify_completion(directory, "seed_complete.json", frozen["fingerprint"])
        summary = read_json(directory / "test" / "summary.json")
        checked = verify_rankings(
            directory / "test" / "per_user_rankings.jsonl",
            summary,
            test,
            popularity,
            ["popularity", *ARMS],
            config["cutoffs"],
        )
        ranking_rows += checked["rows"]
        for arm in ARMS:
            val_dir = directory / arm / "evaluation" / "validation"
            val_summary = read_json(val_dir / "summary.json")
            require(
                val_summary["used_for_selection"] is False,
                f"Full validation selected weights: {seed}/{arm}",
            )
            checked = verify_rankings(
                val_dir / "per_user_rankings.jsonl",
                val_summary,
                validation,
                popularity,
                [arm],
                config["cutoffs"],
            )
            ranking_rows += checked["rows"]
    report["ranking_context_rows_checked"] = ranking_rows
    report["test_users_per_seed"], report["validation_users_per_arm"] = len(test), len(validation)
    report["common_vocab_size"] = frozen["common_vocab_size"]
    report["checks"].append(
        "all saved validation/test rankings and reconstructed metrics, coverage and tail metrics"
    )
    report["status"] = "verified_diagnostic" if diagnostic else "verified_full_m3"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results" / "m3")
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional expected config; full runs default to configs/m3.json.",
    )
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        report = verify(
            args.run_dir.resolve(),
            args.project_dir.resolve(),
            args.allow_diagnostic,
            args.config.resolve() if args.config else None,
        )
    except (VerificationError, ValueError, KeyError, OSError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": "failed", "run_dir": str(args.run_dir.resolve()), "error": str(error)},
                indent=2,
            )
        )
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
