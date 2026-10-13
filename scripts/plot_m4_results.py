"""Export the completed M4 comparison with pipeline SD and conditional user CIs."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]


def plot(run: Path, output: Path):
    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    if (
        not result["m4_complete"]
        or result["diagnostic_only"]
        or result["config"]["seeds"] != [42, 43, 44]
    ):
        raise ValueError("Only the complete three-seed M4 study may produce the final figure.")
    if result["test_examples"] != 22225 or not (run / "completed.json").is_file():
        raise ValueError("Expected the completed full test cohort.")
    arms, markers = ["rq16", "rq32"], ["o", "s", "^"]
    seeds = result["seed_results"]
    contexts = seeds[0]["test"]["contexts_sha256"]
    if any(row["test"]["contexts_sha256"] != contexts for row in seeds):
        raise ValueError("Test cohorts differ across seeds.")
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    colors = ["#3775BA", "#42949E"]
    for index, metric in enumerate(["recall@100", "map@20"]):
        ax = axes[index]
        values = np.asarray([[row["test"][arm][metric] for arm in arms] for row in seeds])
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        for col, arm in enumerate(arms):
            saved = result["aggregate_test"][arm][metric]
            if not np.isclose(saved["mean"], mean[col], atol=1e-12) or not np.isclose(
                saved["std"], sd[col], atol=1e-12
            ):
                raise ValueError("Figure statistics differ from saved aggregates.")
            ax.errorbar(
                col,
                mean[col],
                yerr=sd[col],
                fmt="D",
                color=colors[col],
                capsize=5,
                markersize=8,
                elinewidth=2,
                zorder=3,
            )
        for row, marker in enumerate(markers):
            offset = (row - 1) * 0.06
            ax.plot(np.arange(2) + offset, values[row], color="#BBBBBB", linewidth=0.8, zorder=1)
            ax.scatter(
                np.arange(2) + offset,
                values[row],
                marker=marker,
                facecolor="white",
                edgecolor="#272727",
                s=35,
                zorder=4,
            )
        ax.set_xticks([0, 1], ["RQ-16", "RQ-32"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(0, float(max(values.max(), (mean + sd).max())) * 1.15)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.set_title(
            f"({'ab'[index]}) {metric.replace('recall', 'Recall').replace('map', 'MAP')}",
            loc="left",
        )
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    ax = axes[2]
    for index, row in enumerate(seeds):
        interval = row["comparisons"]["rq32_minus_rq16"]["delta"]["recall@100"]
        center, lower, upper = (
            interval["mean"] * 100,
            interval["lower"] * 100,
            interval["upper"] * 100,
        )
        # Percentile intervals need not contain the observed point estimate.
        ax.hlines(index, lower, upper, color="#3775BA", linewidth=1.8)
        ax.vlines([lower, upper], index - 0.07, index + 0.07, color="#3775BA", linewidth=1.8)
        ax.scatter(center, index, marker=markers[index], color="#3775BA", s=49, zorder=3)
    ax.axvline(0, color="#767676", linestyle="--", linewidth=1)
    ax.set_yticks(range(3), [f"Seed {row['seed']}" for row in seeds])
    ax.invert_yaxis()
    ax.set_ylim(2.6, -0.6)
    ax.set_xlabel("RQ-32 − RQ-16 (percentage points)")
    ax.set_title("(c) Paired Δ Recall@100", loc="left")
    handles = [
        Line2D([], [], marker="D", linestyle="none", color="#3775BA", label="Mean ± pipeline SD")
    ]
    handles += [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#272727",
            label=f"Seed {seed}",
        )
        for seed, marker in zip([42, 43, 44], markers)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=4, fontsize=11)
    fig.suptitle(
        "Amazon Beauty 2014 — M4 RQ codebook capacity", fontsize=20, fontweight="bold", y=0.99
    )
    fig.text(
        0.5,
        0.90,
        "22,225 test users · 3 paired pipeline seeds · matched generator capacity",
        ha="center",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.025,
        "(a–b): pipeline sample SD. (c): conditional 95% paired user bootstrap. "
        "Users are not pooled.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.76), w_pad=3)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        path = output / f"m4_capacity_comparison.{extension}"
        fig.savefig(path, dpi=300)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                + "\n",
                encoding="utf-8",
            )
    plt.close(fig)
    caption = (
        "**M4 RQ codebook capacity.** Three independent pipeline seeds compare 16 and 32 entries "
        "per residual codebook under matched generator capacity. Panels (a–b) show full-test "
        "Recall@100 and MAP@20; small connected points are paired seeds and large diamonds "
        "show mean ± sample SD. Panel (c) reports RQ-32 minus RQ-16 Recall@100 in percentage "
        "points with per-seed 95% paired user-bootstrap intervals (1,000 resamples), conditional "
        "on fitted models. All arms use the same 22,225 eligible warm-item test users, full "
        "seen-item exclusions and beam 200. Wider beams are validation diagnostics only. "
        "Pipeline SD and user intervals measure different uncertainty; users across seeds "
        "are not pooled. These test users were previously evaluated in M2/M3.\n"
    )
    (output / "m4_capacity_comparison.caption.md").write_text(caption, encoding="utf-8")
    (output / "m4_capacity_comparison.source.json").write_text(
        json.dumps(
            {
                "results_sha256": hashlib.sha256((run / "results.json").read_bytes()).hexdigest(),
                "protocol_fingerprint": result["protocol_fingerprint"],
                "contexts_sha256": contexts,
                "pipeline_seeds": [42, 43, 44],
                "test_examples": result["test_examples"],
                "common_vocab_size": result["common_vocab_size"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Exported M4 SVG, PDF, 300-dpi PNG, caption and provenance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/m4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    plot(args.run_dir, args.output_dir)
