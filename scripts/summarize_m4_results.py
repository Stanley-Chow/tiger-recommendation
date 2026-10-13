"""Export readable numeric tables from completed, independently audited M4 results.

This script reads saved summaries; it never trains, decodes or selects a model.
Run verify_m4_results.py first. Fractions become percentages only for display.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("rq16", "rq32")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(lines, title, headings, rows):
    lines.extend([f"## {title}", "", "| " + " | ".join(headings) + " |"])
    lines.append("|" + "---|" * len(headings))
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def summarize(run, output):
    result = read(run / "results.json")
    if not result["m4_complete"] or result["diagnostic_only"]:
        raise ValueError("Numeric publication requires the full M4 study.")
    if result["config"]["seeds"] != [42, 43, 44] or result["test_examples"] != 22225:
        raise ValueError("Unexpected M4 seed or test cohort.")
    aggregate = result["aggregate_test"]
    lines = [
        "# M4 complete numeric tables",
        "",
        "Quality values are percentages; contrasts and their intervals are percentage points.",
        "Mean ± SD uses three pipeline fits and sample SD (ddof=1). User intervals condition",
        "on each fitted pair. Repeated users across seeds are not pooled.",
        "",
    ]
    quality = (
        "recall@20",
        "recall@100",
        "recall@200",
        "map@20",
        "ndcg@20",
        "tail_recall@100",
        "catalog_coverage@20",
    )
    table(
        lines,
        "Full test quality",
        ["Arm", *quality],
        [
            [
                arm,
                *[
                    f"{aggregate[arm][key]['mean'] * 100:.3f} ± "
                    f"{aggregate[arm][key]['std'] * 100:.3f}"
                    for key in quality
                ],
            ]
            for arm in aggregate["methods"]
        ],
    )
    rows, deltas = [], []
    for record in result["seed_results"]:
        for arm in ARMS:
            rows.append(
                [
                    record["seed"],
                    arm,
                    f"{record['test'][arm]['recall@100'] * 100:.3f}",
                    f"{record['test'][arm]['map@20'] * 100:.3f}",
                ]
            )
        comparison = record["comparisons"]["rq32_minus_rq16"]["delta"]
        for metric in ("recall@100", "map@20"):
            value = comparison[metric]
            deltas.append(
                [
                    record["seed"],
                    metric,
                    f"{value['mean'] * 100:+.3f}",
                    f"[{value['lower'] * 100:+.3f}, {value['upper'] * 100:+.3f}]",
                ]
            )
    table(lines, "Individual fitted pairs", ["Seed", "Arm", "Recall@100", "MAP@20"], rows)
    table(
        lines,
        "RQ-32 minus RQ-16: 1,000 paired user-bootstrap resamples",
        ["Seed", "Metric", "Delta", "Conditional 95% interval"],
        deltas,
    )
    training, metadata, search, tokens = [], [], [], []
    for record in result["seed_results"]:
        seed = record["seed"]
        for arm in ARMS:
            branch = run / "seeds" / f"seed_{seed}" / arm
            fit = read(branch / "training_summary.json")
            training.append(
                [
                    seed,
                    arm,
                    fit["epochs_completed"],
                    fit["best_epoch"],
                    fit["guard_passed"],
                    f"{fit['best_validation_metrics']['map@20'] * 100:.3f}",
                    f"{fit['guard_threshold'] * 100:.3f}",
                    f"{fit['train_seconds']:.2f}",
                    f"{fit['validation_seconds']:.2f}",
                    fit["optimizer_updates"],
                ]
            )
            rq = read(run / "tokenizers" / f"seed_{seed}" / arm / "rqvae_summary.json")
            code = rq["semantic_code_diagnostics"]
            metadata.append(
                [
                    seed,
                    arm,
                    rq["selected_quantized_epochs"],
                    rq["parameter_count"],
                    "/".join(map(str, rq["final_diagnostics"]["used_codes_by_level"])),
                    code["leaf_count"],
                    code["max_leaf_size"],
                    f"{code['colliding_item_fraction'] * 100:.3f}",
                    f"{rq['final_diagnostics']['quantized_reconstruction_mse']:.6f}",
                    f"{rq['selection_seconds'] + rq['refit_seconds']:.2f}",
                ]
            )
            diagnostic = record["search_diagnostics"][arm]
            for width in ("200", "400"):
                metrics = diagnostic[width]
                search.append(
                    [
                        seed,
                        arm,
                        width,
                        f"{metrics['recall@100'] * 100:.2f}",
                        f"{metrics['recall@200'] * 100:.2f}",
                        "/".join(f"{v * 100:.1f}" for v in metrics["prefix_coverage@200"]),
                        metrics["certified_search_misses"],
                    ]
                )
            for position, (nll, accuracy) in enumerate(
                zip(
                    diagnostic["mean_target_token_nll"],
                    diagnostic["teacher_forced_token_argmax_accuracy"],
                ),
                start=1,
            ):
                tokens.append(
                    [
                        seed,
                        arm,
                        ["L1", "L2", "L3", "suffix", "EOS"][position - 1],
                        f"{nll:.4f}",
                        f"{accuracy * 100:.2f}",
                    ]
                )
    table(
        lines,
        "Checkpoint selection and training",
        [
            "Seed",
            "Arm",
            "Epochs",
            "Selected",
            "Guard passed",
            "Selection MAP@20",
            "Guard threshold",
            "Train s",
            "Validation s",
            "Updates",
        ],
        training,
    )
    table(
        lines,
        "Metadata tokenizers",
        [
            "Seed",
            "Arm",
            "Chosen quantized epochs",
            "Parameters",
            "Used entries L1/L2/L3",
            "Occupied L3 prefixes",
            "Largest collision group",
            "Items in collision groups (%)",
            "Refit reconstruction MSE",
            "Selection + refit s",
        ],
        metadata,
    )
    lines.extend(
        [
            "RQ-16 intentionally reserves 16 unused semantic slots per level in the common",
            "generator vocabulary. Collision fractions refer to products in groups of size >1,",
            "not the fraction of unoccupied possible prefixes.",
            "",
        ]
    )
    table(
        lines,
        "Report-only validation search: 1,000 users per arm",
        [
            "Seed",
            "Arm",
            "Beam",
            "Recall@100",
            "Recall@200",
            "Prefix coverage L1/L2/L3 (%)",
            "Ranking certificates",
        ],
        search,
    )
    lines.extend(
        [
            "A certificate shows an omitted target outranks the returned 200th candidate.",
            "It proves a search-ranking violation, not that the target is in the oracle top 200.",
            "The test beam remains 200. Wider validation beams do not select new settings.",
            "",
        ]
    )
    table(
        lines,
        "Teacher-forced validation token prediction",
        ["Seed", "Arm", "Position", "Mean NLL (nats)", "Full-vocabulary argmax accuracy (%)"],
        tokens,
    )
    table(
        lines,
        "Test retrieval timing: mean ± pipeline SD",
        ["Arm", "Mean request ms", "p95 request ms", "Histories/s"],
        [
            [
                arm,
                *[
                    f"{aggregate[arm][key]['mean']:.2f} ± {aggregate[arm][key]['std']:.2f}"
                    for key in (
                        "mean_request_latency_ms",
                        "p95_request_latency_ms",
                        "histories_per_second",
                    )
                ],
            ]
            for arm in ARMS
        ],
    )
    lines.extend(
        [
            "Request latency uses 100 requests after one warmup; batched throughput uses",
            "batch 16 and includes its first call. Neither includes metadata encoding.",
            "",
        ]
    )
    for metric in ("recall@100", "map@20"):
        values = [
            record["comparisons"]["rq32_minus_rq16"]["delta"][metric]["mean"] * 100
            for record in result["seed_results"]
        ]
        lines.append(
            f"Pipeline delta {metric}: {np.mean(values):+.3f} ± "
            f"{np.std(values, ddof=1):.3f} points."
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "m4_numeric_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "m4_results.json").write_bytes((run / "results.json").read_bytes())
    print("Exported M4 numeric tables and the unchanged full summary JSON.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/m4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs")
    arguments = parser.parse_args()
    summarize(arguments.run_dir, arguments.output_dir)
