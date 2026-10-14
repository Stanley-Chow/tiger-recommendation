"""Render verified M5 results as publication figures and a concise Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "popularity": "Popularity",
    "generative": "Generative",
    "sasrec": "SASRec",
    "content": "Content",
    "generative_content": "Gen + content",
    "sasrec_content": "SAS + content",
}
COLORS = {
    "popularity": "#CFCECE",
    "generative": "#3775BA",
    "sasrec": "#E9A6A1",
    "content": "#42949E",
    "generative_content": "#AADCA9",
    "sasrec_content": "#DDF3DE",
}


def percentage(summary, method, metric):
    value = summary[method][metric]
    mean, spread = 100 * value["mean"], value["std"]
    return f"{mean:.3f}%" if spread is None else f"{mean:.3f} ± {100 * spread:.3f}%"


def draw(result, output):
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    warm, new = result["aggregate"]["test"], result["aggregate"].get("mixed_new")

    def bars(ax, summary, metric, methods, title):
        means = np.array([summary[method][metric]["mean"] * 100 for method in methods])
        spread = np.array([(summary[method][metric]["std"] or 0) * 100 for method in methods])
        ax.bar(
            np.arange(len(methods)),
            means,
            yerr=spread,
            capsize=4,
            color=[COLORS[m] for m in methods],
            edgecolor="#272727",
            linewidth=1,
        )
        ax.set_xticks(
            np.arange(len(methods)),
            [LABELS[m] for m in methods],
            rotation=25 if len(methods) > 3 else 0,
            ha="right" if len(methods) > 3 else "center",
        )
        ax.set_ylabel("Percent")
        ax.set_title(title, loc="left", weight="bold", fontsize=12)
        ax.set_ylim(0, max(1.0, float(np.max(means + spread)) * 1.28))
        for x, mean, std in zip(range(len(methods)), means, spread):
            ax.text(
                x, mean + std + ax.get_ylim()[1] * 0.025, f"{mean:.2f}", ha="center", fontsize=10
            )

    bars(axes[0, 0], warm, "recall@100", warm["methods"], "(a) Warm test • Recall@100")
    bars(axes[0, 1], warm, "map@20", warm["methods"], "(b) Warm test • MAP@20")
    if new is not None:
        bars(axes[1, 0], new, "recall@100", new["methods"], "(c) New-item targets • Recall@100")
    else:
        axes[1, 0].text(0.5, 0.5, "No eligible new-item targets", ha="center")
    ax = axes[1, 1]
    intervals = [
        seed["comparisons"]["test/generative_minus_sasrec"]["delta"]["recall@100"]
        for seed in result["seeds"]
    ]
    for index, interval in enumerate(intervals):
        ax.hlines(
            index, 100 * interval["lower"], 100 * interval["upper"], color="#0F4D92", linewidth=2
        )
        ax.plot(100 * interval["mean"], index, "o", color="#0F4D92")
    ax.axvline(0, color="#767676", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(intervals)), [f"Seed {s['seed']}" for s in result["seeds"]])
    ax.set_ylim(-0.6, len(intervals) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Generative − SASRec • percentage points")
    ax.set_title(
        "(d) Paired user 95% intervals • Recall@100", loc="left", weight="bold", fontsize=12
    )
    fig.tight_layout(pad=2.0, h_pad=3.2, w_pad=2.5)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        path = output / f"m5_results.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text("utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def write_report(result, audit, runtime, session, path):
    aggregate, data = result["aggregate"], result["data"]
    warm = aggregate["test"]
    eligibility = data["eligibility"]["test"]
    coverage = (
        100
        * eligibility["available_catalog_eligible"]
        / eligibility["users_with_first_future_target"]
    )
    gen_recall_delta = 100 * (
        warm["generative"]["recall@100"]["mean"] - warm["sasrec"]["recall@100"]["mean"]
    )
    gen_map_delta = 100 * (warm["generative"]["map@20"]["mean"] - warm["sasrec"]["map@20"]["mean"])
    below_popularity = [
        LABELS[m]
        for m in ("generative", "sasrec")
        if warm[m]["map@20"]["mean"] < warm["popularity"]["map@20"]["mean"]
    ]
    guard_count = sum(
        int(training["guard_passed"])
        for seed in result["seeds"]
        for training in seed["training"].values()
    )
    fusion_changes = []
    for method in ("generative", "sasrec"):
        mixed = aggregate["mixed_test"]
        delta_r = 100 * (
            mixed[f"{method}_content"]["recall@100"]["mean"] - mixed[method]["recall@100"]["mean"]
        )
        delta_m = 100 * (
            mixed[f"{method}_content"]["map@20"]["mean"] - mixed[method]["map@20"]["mean"]
        )
        fusion_changes.append(
            f"{LABELS[method]} + content changes mixed Recall@100 by "
            f"{delta_r:+.3f} points and MAP@20 by {delta_m:+.3f} points."
        )

    def cold_hits(method):
        counts = [
            round(
                seed["summaries"]["mixed_new"][method]["recall@100"]
                * seed["summaries"]["mixed_new"]["examples"]
            )
            for seed in result["seeds"]
        ]
        return str(counts[0]) if len(set(counts)) == 1 else "/".join(map(str, counts))

    lines = [
        "# M5 results: temporal Amazon 2023 recommendation",
        "",
        "Six fresh recommendation models compare the frozen RQ-32 generative recipe with "
        "SASRec across three pipeline seeds. A separate mixed-catalog test evaluates fixed "
        "content retrieval and rank fusion. All selected checkpoint hashes were frozen "
        "before final evaluation.",
        "",
        "![M5 results](figures/m5_results.svg)",
        "",
        "## Warm eligible test",
        "",
        "Values are mean ± sample SD across three seeds. SD describes pipeline variation, "
        "not a confidence interval. Popularity and pure content are deterministic controls.",
        "",
        "| Method | Recall@20 | Recall@100 | Recall@200 | MAP@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in warm["methods"]:
        cells = [
            percentage(warm, method, metric)
            for metric in ("recall@20", "recall@100", "recall@200", "map@20")
        ]
        lines.append("| " + " | ".join([LABELS[method], *cells]) + " |")
    deltas = [
        s["comparisons"]["test/generative_minus_sasrec"]["delta"]["recall@100"]["mean"]
        for s in result["seeds"]
    ]
    lines += [
        "",
        f"The primary mean Recall@100 difference (generative minus SASRec) is "
        f"**{100 * np.mean(deltas):+.3f} percentage points**. The paired seed-difference "
        f"sample SD is {100 * np.std(deltas, ddof=1):.3f} points.",
        "",
        "| Pipeline seed | Recall@100 difference (points) | Conditional user 95% interval |",
        "|---|---:|---:|",
    ]
    for seed in result["seeds"]:
        interval = seed["comparisons"]["test/generative_minus_sasrec"]["delta"]["recall@100"]
        lines.append(
            f"| {seed['seed']} | {100 * interval['mean']:+.3f} | "
            f"[{100 * interval['lower']:+.3f}, {100 * interval['upper']:+.3f}] |"
        )
    lines += [
        "",
        "Intervals use 1,000 paired user-bootstrap resamples separately within each seed. "
        "They condition on trained weights, selected catalog and eligible contexts. Repeated "
        "seeds are not pooled as additional users; intervals are not simultaneous across metrics.",
        "",
        "## Warm/new available catalog",
        "",
        "The closed models emit only warm items. Their new-target recall is structurally zero. "
        "New-item reach belongs to the frozen content route. Equal-weight reciprocal-rank "
        "fusion uses constant 60 and 200 candidates per branch, without test tuning.",
        "",
        "| Method | Mixed Recall@100 | Mixed MAP@20 | Warm Recall@100 | New Recall@100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in aggregate["mixed_test"]["methods"]:
        cells = [
            percentage(aggregate[split], method, metric)
            for split, metric in (
                ("mixed_test", "recall@100"),
                ("mixed_test", "map@20"),
                ("mixed_warm", "recall@100"),
                ("mixed_new", "recall@100"),
            )
        ]
        lines.append("| " + " | ".join([LABELS[method], *cells]) + " |")
    lines += [
        "",
        f"There are {data['examples']['test']:,} warm eligible test users and "
        f"{data['examples']['mixed_test']:,} mixed eligible users, including "
        f"{data['eligibility']['test'].get('new_eligible', 0):,} new-item targets. "
        f"The expanded catalog covers "
        f"{coverage:.2f}% "
        "of active cohort users' first test targets. "
        "Quality is conditional on eligibility. See the [dataset card](m5_data_card.md) for "
        "unavailable targets, frozen histories, and the historical metadata limitation.",
        "",
        "## Interpretation and next work",
        "",
        f"Among learned branches, generative minus SASRec changes mean warm Recall@100 "
        f"by {gen_recall_delta:+.3f} points and MAP@20 by {gen_map_delta:+.3f} points. "
        f"{guard_count} of {2 * len(result['seeds'])} selected checkpoints pass the validation "
        f"guard. Models with mean test MAP@20 below popularity: "
        f"{', '.join(below_popularity) or 'none'}. A validation constraint does not guarantee "
        "quality under a later temporal distribution.",
        "",
        " ".join(fusion_changes) + f" At Recall@100, new-item hit counts are "
        f"{cold_hits('content')} for pure content, {cold_hits('generative_content')} for "
        f"generative + content, and {cold_hits('sasrec_content')} for SASRec + content "
        f"out of {eligibility.get('new_eligible', 0)} targets per seed. Repeated identical "
        "cold hit counts do not create more independent evidence. New-item reach is "
        "implemented; these small observed hit counts limit claims about its quality.",
        "",
        "This is a complete research and inference prototype with measured limitations. "
        "The preselected hybrid API default demonstrates a capability; the results do "
        "not validate it as a production policy. Keep the dataset and diagnose ranking "
        "and new-item retrieval before changing many components. See the "
        "[next-study options](after_m5.md) after the code-review checkpoint.",
        "",
        "## Training and inference resources",
        "",
        f"Hardware: {runtime['gpu']}. PyTorch {runtime['cuda_smoke']['torch']}; "
        f"CUDA {runtime['cuda_smoke']['cuda']}. Learned recommendation parameter counts "
        "are not matched between architectures.",
        "",
        "| Seed | Model | Parameters | Selected epoch / completed | Train minutes | MAP guard |",
        "|---|---|---:|---:|---:|---|",
    ]
    for seed in result["seeds"]:
        for method, training in seed["training"].items():
            lines.append(
                f"| {seed['seed']} | {LABELS[method]} | {training['parameter_count']:,} | "
                f"{training['best_epoch']} / {training['epochs_completed']} | "
                f"{training['train_seconds'] / 60:.2f} | "
                f"{'pass' if training['guard_passed'] else 'FAIL: fallback retained'} |"
            )
    lines += [
        "",
        "Checkpoint selection maximizes validation Recall@100 subject to MAP@20 "
        "at least popularity, on the same fixed 1,000 users. Single-positive MAP equals MRR. "
        "The serving seed is prespecified as 42, independently of test performance.",
        "",
        "| Model | Warm mean ms | Mean of seed p95 ms | Test batch histories/s |",
        "|---|---:|---:|---:|",
    ]
    for method in ("generative", "sasrec"):
        means = [s["timing"][f"request/{method}"]["mean_ms"] for s in result["seeds"]]
        p95 = [s["timing"][f"request/{method}"]["p95_ms"] for s in result["seeds"]]
        throughput = [
            s["timing"][f"test/{method}"]["histories_per_second"] for s in result["seeds"]
        ]
        lines.append(
            f"| {LABELS[method]} | {np.mean(means):.2f} | {np.mean(p95):.2f} | "
            f"{np.mean(throughput):.1f} |"
        )
    lines += [
        "",
        "Latency columns average per-seed observations after warmup, using 100 "
        "single-history requests per seed; throughput uses the configured method-specific "
        "batch sizes. These are model retrieval timings including CPU result transfer, "
        "excluding model loading, index construction, text encoding, and HTTP overhead. "
        "The generator uses approximate constrained beam search; SASRec scores the warm "
        "catalog exactly. Full-catalog evaluation uses no sampled negatives.",
        "",
        "## Verification and delivery",
        "",
        f"The independent audit passed **{audit['checkpoint_count']} checkpoints**, "
        f"**{audit['ranking_context_rows']:,} ranking context rows**, and "
        f"**{audit['source_training_contexts']:,} reconstructed training contexts**. "
        "Checks include hashes, strict-past histories, first-target eligibility, MAP-guard "
        "selection, scalar metrics, bootstrap intervals, seed aggregation and the portable bundle.",
        "",
        f"Executed source commit: `{runtime['source_commit']}`. "
        f"Protocol fingerprint: `{result['protocol_fingerprint']}`.",
        "",
        "Large artifacts remain local under `results/m5/` and `artifacts/amazon2023/`. "
        "The bundle needs no raw reviews or text-model download at inference time. "
        "Use the [reproduction guide](m5_reproduction.md) and then the "
        "[learning/code-review checkpoint](learning_checkpoint_m5.md).",
        "",
    ]
    if session.get("rental_cost_rmb") is not None:
        lines += [
            f"Settled M5 rental: **¥{session['rental_cost_rmb']:.2f}**. "
            f"Cumulative project spend: **¥{session['cumulative_cost_rmb']:.2f}**. "
            "The experiment GPU is stopped and temporary SSH access removed.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/m5")
    parser.add_argument("--audit", type=Path, default=ROOT / "results/m5-verification.json")
    parser.add_argument("--runtime", type=Path, default=ROOT / "results/m5-source-runtime.json")
    parser.add_argument("--session", type=Path, default=ROOT / ".remote/m5_session.json")
    args = parser.parse_args()
    result = json.loads((args.run_dir / "results.json").read_text("utf-8"))
    audit = json.loads(args.audit.read_text("utf-8-sig"))
    if args.runtime.exists():
        runtime = json.loads(args.runtime.read_text("utf-8"))
    else:
        branch = args.run_dir / "seeds" / "seed_42" / "generative" / "config.json"
        manifest = json.loads(branch.read_text("utf-8"))
        runtime = {
            "gpu": f"physical model unrecorded; device {manifest['device_used']}",
            "source_commit": "not recorded; consult protocol source hashes",
            "cuda_smoke": {"torch": manifest["torch_version"], "cuda": "unrecorded"},
        }
    session = json.loads(args.session.read_text("utf-8-sig")) if args.session.exists() else {}
    if not result["m5_complete"] or not audit["verified"]:
        raise ValueError("Publish a full results report only after the independent M5 audit.")
    if audit["protocol_fingerprint"] != result["protocol_fingerprint"]:
        raise ValueError("Audit belongs to another M5 run.")
    draw(result, ROOT / "docs/figures")
    write_report(result, audit, runtime, session, ROOT / "docs/m5_results.md")
    selection = json.loads((args.run_dir / "selection_manifest.json").read_text("utf-8"))
    data_fields = (
        "dataset",
        "source_rows",
        "users",
        "eligible_users_before_subsampling",
        "items",
        "new_items",
        "available_test_items",
        "training_interactions",
        "examples",
        "eligibility",
        "max_history",
        "items_missing_metadata",
    )
    summary = {
        "milestone": "M5",
        "complete": True,
        "executed_source_commit": runtime["source_commit"],
        "protocol_fingerprint": result["protocol_fingerprint"],
        "data": {key: result["data"][key] for key in data_fields},
        "metrics": result["aggregate"],
        "paired_comparisons": [
            {"seed": seed["seed"], "comparisons": seed["comparisons"]} for seed in result["seeds"]
        ],
        "trained_models": [
            {
                "seed": seed["seed"],
                "method": method,
                "checkpoint_sha256": selection["selected"][str(seed["seed"])][method]["sha256"],
                **{
                    key: training[key]
                    for key in (
                        "parameter_count",
                        "best_epoch",
                        "epochs_completed",
                        "guard_passed",
                        "train_seconds",
                        "validation_seconds",
                        "peak_cuda_allocated_mb",
                    )
                },
            }
            for seed in result["seeds"]
            for method, training in seed["training"].items()
        ],
        "hardware": runtime["cuda_smoke"],
        "audit": audit,
        "settled_cost_rmb": session.get("rental_cost_rmb"),
        "cumulative_project_cost_rmb": session.get("cumulative_cost_rmb"),
        "limitations": result["limitations"],
    }
    (ROOT / "docs/m5_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "docs/m5_verification.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote docs/m5_results.md and docs/figures/m5_results.{svg,pdf,png}")


if __name__ == "__main__":
    main()
