"""Plot saved M3 test scores; this script never fits models or changes results.

Full experiment:
    python scripts/plot_m3_results.py --run-dir results/m3 --output-dir docs/figures

Diagnostic preview (kept inside its own run folder):
    python scripts/plot_m3_results.py --run-dir results/m3_cpu_smoke_v2 \
        --output-dir results/m3_cpu_smoke_v2/figures --smoke-preview

Install the project's ``figures`` extra to obtain matplotlib. The plot shows
pipeline-seed variation, while paired per-user bootstrap intervals remain in the
saved experiment results. Repeated users are never pooled as independent samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("hkm", "hkm_permuted", "rqvae", "rqvae_permuted")
LABELS = ("HKM", "HKM\npermuted", "RQ-VAE", "RQ-VAE\npermuted")
METRICS = (("recall@100", "Recall@100"), ("map@20", "MAP@20"))
# Blue and teal follow the skill palette; controls also have distinct mean shapes.
COLORS = ("#0F4D92", "#767676", "#42949E", "#767676")
MEAN_MARKERS = ("D", "d", "D", "d")
SEED_MARKERS = ("o", "s", "^")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _load_scores(run_dir: Path) -> tuple[dict, list[int], dict[str, np.ndarray]]:
    """Check matched cohorts and reconstruct the saved seed mean/sample SD."""
    result = _read_json(run_dir / "results.json")
    records = result["seed_results"]
    if not isinstance(records, list) or not records:
        raise ValueError("Results must include at least one completed pipeline seed.")
    seeds = [record["seed"] for record in records]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("Pipeline seeds must be integer identifiers.")
    if len(set(seeds)) != len(seeds) or seeds != result["config"]["seeds"]:
        raise ValueError("Saved pipeline seeds do not match the experiment configuration.")
    aggregate = result["aggregate_test"]
    cohorts = {record["test"]["contexts_sha256"] for record in records}
    if len(cohorts) != 1 or cohorts != {aggregate["contexts_sha256"]}:
        raise ValueError("All pipeline seeds must use exactly the same frozen test contexts.")
    if any(record["test"]["examples"] != result["test_examples"] for record in records):
        raise ValueError("Saved test context counts are inconsistent.")
    if result.get("m3_complete") is True and (
        result.get("diagnostic_only") or result.get("diagnostic_reasons") or seeds != [42, 43, 44]
    ):
        raise ValueError("A full M3 result must contain the three complete pipeline seeds.")

    scores = {}
    for metric, _ in METRICS:
        values = np.array([[record["test"][arm][metric] for arm in ARMS] for record in records])
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError(f"{metric} must contain real scores.")
        values = values.astype(np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{metric} scores must lie in [0, 1].")
        for column, arm in enumerate(ARMS):
            saved = aggregate[arm][metric]
            mean = float(values[:, column].mean())
            std = float(values[:, column].std(ddof=1)) if len(seeds) > 1 else None
            if not math.isclose(saved["mean"], mean, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Saved {arm} {metric} mean disagrees with its pipeline seeds.")
            if (saved["std"] is None) != (std is None) or (
                std is not None
                and not math.isclose(saved["std"], std, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError(f"Saved {arm} {metric} SD disagrees with its pipeline seeds.")
        scores[metric] = values
    return result, seeds, scores


def _diagnostic(result: dict) -> bool:
    return result.get("m3_complete") is not True or bool(result.get("diagnostic_only"))


def _caption(result: dict, seeds: list[int], title: str, smoke_preview: bool) -> str:
    count = len(seeds)
    spread = (
        f"Large diamonds show the mean and whiskers show ±1 sample standard deviation "
        f"across {count} pipeline seeds ({', '.join(map(str, seeds))})."
        if count > 1
        else f"One pipeline seed ({seeds[0]}) is shown; sample SD is undefined "
        "and no whisker is drawn."
    )
    diagnostic = (
        " This explicitly labeled CPU smoke preview does not establish recommendation "
        "quality on the full Amazon dataset."
        if smoke_preview
        else " This diagnostic run does not complete the full M3 experiment."
        if _diagnostic(result)
        else ""
    )
    if result["config"]["evaluation"].get("max_test_examples") is not None:
        diagnostic += " Final test evaluation was capped by the saved run configuration."
    cohort_note = (
        "The same users recur across seeds; SD describes pipeline variation rather than "
        "user-level uncertainty or a confidence interval."
        if count > 1
        else "This one-seed preview cannot estimate variation across fitted pipelines."
    )
    return (
        f"**{title}.** Test Recall@100 and MAP@20 for HKM, RQ-VAE and their full-code "
        f"permutation controls. {spread} Small points show the individual seeds. Each arm "
        f"uses the same {result['test_examples']:,} frozen test contexts and excludes the "
        f"entire previously seen item history. {cohort_note} "
        "Per-seed paired user bootstrap comparisons are saved separately; "
        "users across seeds are not pooled as independent observations. Whole-code "
        "permutation preserves the code set, vocabulary and catalog prefix occupancy, "
        "while interaction-weighted exposure can change. Single-positive MAP@20 equals "
        "reciprocal rank when the held-out item appears within the first 20 results, "
        f"and zero otherwise.{diagnostic}\n"
    )


def plot_results(run_dir: Path, output_dir: Path, *, smoke_preview: bool = False) -> list[Path]:
    """Export the same validated, provenance-recorded figure as SVG, PDF and PNG."""
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    result, seeds, scores = _load_scores(run_dir)
    diagnostic = _diagnostic(result)
    if diagnostic and not output_dir.is_relative_to(run_dir / "figures"):
        raise ValueError("Diagnostic plots must stay inside the source run's figures directory.")
    if smoke_preview and (not diagnostic or result["config"]["training"]["device"] != "cpu"):
        raise ValueError("--smoke-preview requires an incomplete diagnostic CPU run.")
    title = (
        "CPU smoke — diagnostic only"
        if smoke_preview
        else "M3 diagnostic run — incomplete"
        if diagnostic
        else "Amazon Beauty 2014 — M3 tokenizer ablation"
    )
    first_test = result["seed_results"][0]["test"]
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 13,
            "axes.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3))
    fig.suptitle(title, fontsize=16, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.91,
        f"{result['test_examples']:,} test contexts · "
        f"{first_test['catalog_items']:,} catalog items · "
        f"{len(seeds)} pipeline seed{'s' if len(seeds) > 1 else ''}",
        ha="center",
        fontsize=11,
        color="#4D4D4D",
    )
    offsets = np.linspace(-0.13, 0.13, len(seeds)) if len(seeds) > 1 else np.array([0.10])
    for panel, (ax, (metric, label)) in enumerate(zip(axes, METRICS)):
        values = scores[metric]
        means = values.mean(axis=0)
        standard_deviations = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(ARMS))
        for column, (color, marker) in enumerate(zip(COLORS, MEAN_MARKERS)):
            ax.errorbar(
                column,
                means[column],
                yerr=standard_deviations[column] if len(seeds) > 1 else None,
                fmt=marker,
                markersize=8,
                color=color,
                markeredgecolor="#272727",
                markeredgewidth=0.8,
                elinewidth=1.8,
                capsize=5,
                capthick=1.5,
                zorder=4,
            )
            for row, offset in enumerate(offsets):
                ax.scatter(
                    column + offset,
                    values[row, column],
                    marker=SEED_MARKERS[row % len(SEED_MARKERS)],
                    s=34,
                    facecolor="white",
                    edgecolor="#272727",
                    linewidth=0.9,
                    zorder=5,
                )
        upper = max(float(values.max()), float((means + standard_deviations).max()))
        lower = min(0, float((means - standard_deviations).min()) * 1.10)
        ax.set_ylim(lower, max(0.01, upper * 1.15))
        ax.set_xlim(-0.45, 3.45)
        ax.set_xticks(range(len(ARMS)), LABELS)
        ax.tick_params(axis="x", length=0, pad=10, labelsize=12)
        ax.tick_params(axis="y", length=4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.set_ylabel(label)
        ax.set_title(f"({'ab'[panel]}) {label}", loc="left", fontsize=14, pad=12)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.7, zorder=0)

    handles = [Line2D([], [], marker="D", markersize=7, color="#272727", linestyle="none")]
    labels = ["Mean ± sample SD" if len(seeds) > 1 else "Mean (SD undefined)"]
    for row, seed in enumerate(seeds):
        handles.append(
            Line2D(
                [],
                [],
                marker=SEED_MARKERS[row % len(SEED_MARKERS)],
                markerfacecolor="white",
                markeredgecolor="#272727",
                markersize=6,
                linestyle="none",
            )
        )
        labels.append(f"Seed {seed}")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=len(labels),
        fontsize=10,
    )
    footer = (
        "Diagnostic preview only; one seed gives no estimate of sample SD. "
        "Test contexts are shared across all four arms."
        if diagnostic and len(seeds) == 1
        else "Whiskers: pipeline sample SD, not a user-level confidence interval. "
        "Users are shared across seeds, not pooled."
    )
    fig.text(0.5, 0.025, footer, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.075, 1, 0.80), pad=1.2, w_pad=3)
    fig.subplots_adjust(top=0.74, bottom=0.22)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "m3_tokenizer_ablation"
    outputs = []
    for extension in ("svg", "pdf", "png"):
        path = output_dir / f"{prefix}.{extension}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(fig)
    caption_path = output_dir / f"{prefix}.caption.md"
    caption_path.write_text(_caption(result, seeds, title, smoke_preview), encoding="utf-8")
    outputs.append(caption_path)
    provenance = {
        "results_file": str(run_dir / "results.json"),
        "results_sha256": hashlib.sha256((run_dir / "results.json").read_bytes()).hexdigest(),
        "protocol_fingerprint": result["protocol_fingerprint"],
        "tokenizer_manifest_fingerprint": result["tokenizer_manifest_fingerprint"],
        "m3_complete": result["m3_complete"],
        "diagnostic_only": diagnostic,
        "explicit_smoke_preview": smoke_preview,
        "pipeline_seeds": seeds,
        "arms": list(ARMS),
        "seed_values": {metric: values.tolist() for metric, values in scores.items()},
        "spread": "sample standard deviation across pipeline seeds; undefined for one seed",
        "contexts_sha256": first_test["contexts_sha256"],
        "test_examples": result["test_examples"],
    }
    provenance_path = output_dir / f"{prefix}.source.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    outputs.append(provenance_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results" / "m3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    parser.add_argument(
        "--smoke-preview", action="store_true", help="Explicitly label a diagnostic CPU smoke run."
    )
    args = parser.parse_args()
    try:
        paths = plot_results(args.run_dir, args.output_dir, smoke_preview=args.smoke_preview)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
