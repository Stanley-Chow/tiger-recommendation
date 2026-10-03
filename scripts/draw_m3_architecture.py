"""Draw the frozen M3 experiment structure as editable SVG, PDF and 300-dpi PNG.

The schematic documents docs/m3_protocol.md and docs/learning_checkpoint_m3.md.
It reports design, without performance numbers, and needs no model or dataset.
Run after installing the project's figures extra:
    python scripts/draw_m3_architecture.py --output-dir docs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as DrawingPath  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK = "#243244"
GRAY = "#5B6675"
BLUE = "#0F4D92"
TEAL = "#42949E"
PURPLE = "#77518F"
EDGE = "#C9D2DC"


def draw_m3_architecture(output_dir: Path) -> None:
    """Compose the tokenizer, recommendation and evaluation stages, then export."""
    sources = [ROOT / "docs" / name for name in ("m3_protocol.md", "learning_checkpoint_m3.md")]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"The figure's protocol source is missing: {path}")
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "text.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(16, 11.3))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 16), ylim=(0, 11.3))
    ax.axis("off")
    text_bounds: list[tuple[Text, tuple[float, float, float, float]]] = []

    def label(x: float, y: float, value: str, **kwargs) -> Text:
        return ax.text(x, y, value, va="center", **kwargs)

    def panel(x: float, y: float, width: float, height: float, title: str, color: str):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.015,rounding_size=0.09",
                facecolor="white",
                edgecolor=EDGE,
                linewidth=1,
            )
        )
        label(x + 0.22, y + height - 0.26, title, fontsize=13, weight="bold", color=color)

    def block(x, y, width, height, title, body, color=BLUE, fill="#EEF4FA", body_size=10.3):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.015,rounding_size=0.055",
                facecolor=fill,
                edgecolor=color,
                linewidth=1.1,
            )
        )
        heading = label(
            x + width / 2,
            y + height - 0.23,
            title,
            fontsize=11,
            weight="bold",
            ha="center",
            color=color,
        )
        detail = label(
            x + width / 2,
            y + (height - 0.35) / 2,
            body,
            fontsize=body_size,
            ha="center",
            linespacing=1.35,
        )
        text_bounds.extend((text, (x, y, width, height)) for text in (heading, detail))

    def arrow(points, color=GRAY, dashed=False):
        path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1))
        ax.add_patch(
            FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.25,
                color=color,
                linestyle="--" if dashed else "-",
                zorder=2,
            )
        )

    def note(x, y, value, color=GRAY):
        return label(
            x,
            y,
            value,
            ha="center",
            fontsize=10.1,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
        )

    label(
        0.35,
        10.97,
        "M3: semantic item codes and matched permutation controls",
        fontsize=21,
        weight="bold",
    )
    label(
        0.36,
        10.60,
        "Amazon Beauty 2014 · metadata-only tokenization → next-item prediction "
        "→ frozen test evaluation",
        fontsize=12,
        color=GRAY,
    )

    # A. RQ learns only metadata reconstruction; every item receives a unique full code.
    panel(0.35, 6.0, 15.30, 4.25, "A  METADATA-ONLY TOKENIZATION", TEAL)
    block(
        0.60,
        8.20,
        2.15,
        1.0,
        "Product metadata",
        "Title · brand\nCategories · description",
        TEAL,
        "#EFF7F7",
    )
    block(
        3.25,
        8.20,
        2.45,
        1.0,
        "Frozen MiniLM",
        "384-d normalized vectors\nTraining-catalog items only",
        TEAL,
        "#EFF7F7",
    )
    arrow([(2.78, 8.70), (3.22, 8.70)], TEAL)
    block(
        6.20,
        8.80,
        3.25,
        1.0,
        "Fixed hierarchical k-means",
        "3 levels · up to 16 branches\nDeterministic within-prefix suffix",
    )
    block(
        6.20,
        6.40,
        3.25,
        1.65,
        "Learned residual quantizer (RQ)",
        "3 × 16 codebooks · 32-d latent\nReconstruction + codebook + commitment\n"
        "90/10 metadata epoch selection\nFresh full-catalog refit",
        TEAL,
        "#EFF7F7",
        body_size=10,
    )
    arrow([(5.72, 8.70), (5.94, 8.70), (5.94, 9.30), (6.17, 9.30)], TEAL)
    arrow([(5.94, 8.70), (5.94, 7.23), (6.17, 7.23)], TEAL)
    ax.plot([5.94], [8.70], "o", color=TEAL, markersize=3)
    block(
        0.60,
        6.40,
        5.10,
        1.20,
        "Tokenizer fitting objective: metadata structure",
        "RQ reconstructs frozen metadata vectors.\n"
        "No recommendation-validation or test labels fit tokenizers.\n"
        "Tokenizer fitting is separate from next-item training.",
        TEAL,
        "#F7FBFB",
        body_size=10,
    )

    # Whole-code controls retain the suffix, exact code set and trie topology.
    for y, name, body, color, fill in (
        (9.05, "HKM", "Semantic item-to-code assignment", BLUE, "#EEF4FA"),
        (8.05, "HKM-permuted", "Whole codeword permutation", GRAY, "#F3F4F5"),
        (7.05, "RQ", "Semantic item-to-code assignment", TEAL, "#EFF7F7"),
        (6.05, "RQ-permuted", "Whole codeword permutation", GRAY, "#F3F4F5"),
    ):
        block(10.00, y, 2.45, 0.72, name, body, color, fill, body_size=9.8)
    arrow([(9.48, 9.30), (9.72, 9.30), (9.72, 9.41), (9.97, 9.41)], BLUE)
    arrow([(9.48, 7.23), (9.72, 7.23), (9.72, 7.41), (9.97, 7.41)], TEAL)
    arrow([(11.225, 9.02), (11.225, 8.80)], GRAY)
    arrow([(11.225, 7.02), (11.225, 6.80)], GRAY)
    block(
        13.30,
        7.45,
        2.10,
        1.60,
        "Freeze all code mappings",
        "4 arms per seed\n3 tokens + unique suffix\nCommon vocabulary\nMatched model capacity",
        PURPLE,
        "#F5EFF8",
        body_size=10,
    )
    # A common bus merges the four arms; only the bus exit has an arrowhead.
    ax.plot([12.85, 12.85], [6.41, 9.41], color=GRAY, linewidth=1.25, zorder=2)
    for y in (9.41, 8.41, 7.41, 6.41):
        ax.plot([12.48, 12.85], [y, y], color=GRAY, linewidth=1.25, zorder=2)
    arrow([(12.85, 8.25), (13.27, 8.25)], PURPLE)
    label(
        6.24,
        6.16,
        "Fit all three RQ tokenizers before recommendation training.",
        fontsize=9.8,
        color=TEAL,
    )

    # B. Fixed code tables map interaction histories/targets to four-token codewords.
    panel(0.35, 3.45, 15.30, 2.15, "B  NEXT-ITEM TRAINING AND VALIDATION SELECTION", BLUE)
    arrow(
        [(14.35, 7.42), (14.35, 5.84), (0.15, 5.84), (0.15, 4.55), (0.57, 4.55)],
        PURPLE,
        dashed=True,
    )
    note(10.50, 5.84, "frozen item-to-code tables", PURPLE)
    block(
        0.60,
        3.95,
        3.00,
        1.20,
        "Chronological histories",
        "Last 20 warm items → code tokens\nPredict the next reviewed item",
    )
    block(
        4.10,
        3.95,
        4.00,
        1.20,
        "4 arms × 3 paired seeds = 12 runs",
        "Same encoder–decoder Transformer\nFresh weights; common vocabulary\n"
        "Teacher-forced token cross entropy",
    )
    block(
        8.60,
        3.95,
        3.00,
        1.20,
        "Validation checkpoint selection",
        "Fixed 1,000 users; maximize Recall@100\nMAP@20 ≥ validation popularity\n"
        "Save fallback if no epoch passes",
        PURPLE,
        "#F5EFF8",
        body_size=9.7,
    )
    block(
        12.10,
        3.95,
        3.25,
        1.20,
        "Freeze all 12 selected checkpoints",
        "Record hashes before any test retrieval\nNo test-driven epoch or setting changes",
        PURPLE,
        "#F5EFF8",
        body_size=9.8,
    )
    for left, right in ((3.63, 4.07), (8.13, 8.57), (11.63, 12.07)):
        arrow([(left, 4.55), (right, 4.55)], BLUE)
    label(
        0.62,
        3.68,
        "Next-item objective: token cross entropy. "
        "MiniLM, tokenizers and item-code mappings stay frozen.",
        fontsize=10.3,
        color=BLUE,
    )

    # C. The selected weights are frozen globally before any final test list is built.
    panel(0.35, 0.45, 15.30, 2.58, "C  FROZEN RETRIEVAL AND HELD-OUT COMPARISONS", PURPLE)
    arrow(
        [(13.72, 3.92), (13.72, 3.24), (0.15, 3.24), (0.15, 1.95), (0.57, 1.95)],
        PURPLE,
        dashed=True,
    )
    note(8.0, 3.24, "selected weights frozen for inference", PURPLE)
    block(
        0.60,
        1.40,
        3.30,
        1.10,
        "Catalog-trie beam search",
        "Beam width 200 → top-200 list\nExclude ALL earlier warm seen items\n"
        "Approximate autoregressive retrieval",
        body_size=10,
    )
    block(
        4.40,
        1.40,
        3.00,
        1.10,
        "Full eligible warm-target test cohort",
        "Same contexts across arms and seeds\nRecall@100 · MAP@20\n"
        "One saved list → all planned cutoffs",
        body_size=10,
    )
    block(
        7.90,
        1.15,
        7.45,
        1.35,
        "Planned paired contrasts",
        "Primary (Recall@100): HKM − HKM-permuted; RQ − RQ-permuted\n"
        "Secondary: RQ − HKM\n"
        "Per-seed paired user bootstrap; pipeline mean ± sample SD",
        PURPLE,
        "#F5EFF8",
        body_size=11,
    )
    arrow([(3.93, 1.95), (4.37, 1.95)], BLUE)
    arrow([(7.43, 1.95), (7.87, 1.95)], PURPLE)
    label(
        0.62,
        0.95,
        "Retain finite poor outcomes and failed guards; "
        "do not pool users across seeds as independent observations.",
        fontsize=10.3,
        color=GRAY,
    )
    label(
        0.37,
        0.22,
        "Solid arrows: data/model flow. Dashed arrows: frozen artifacts.",
        fontsize=10.1,
        color=GRAY,
    )
    label(
        15.63,
        0.22,
        "Protocol schematic · no performance results",
        ha="right",
        fontsize=10.1,
        color=GRAY,
    )

    # Check real rendered text against every box before writing the deliverables.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text, (x, y, width, height) in text_bounds:
        bounds = text.get_window_extent(renderer).transformed(ax.transData.inverted())
        if bounds.x0 < x + 0.025 or bounds.x1 > x + width - 0.025:
            raise ValueError(f"Label exceeds its box width: {text.get_text()!r}")
        if bounds.y0 < y + 0.02 or bounds.y1 > y + height - 0.02:
            raise ValueError(f"Label exceeds its box height: {text.get_text()!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        target = output_dir / f"m3_architecture.{extension}"
        fig.savefig(target, dpi=300, metadata={"Creator": "beautyrec M3 protocol schematic"})
        print(target)
    plt.close(fig)
    caption = output_dir / "m3_architecture.caption.md"
    caption.write_text(
        "**M3 tokenizer experiment structure.** Product metadata for the warm training catalog "
        "is encoded once with frozen 384-dimensional MiniLM vectors. Fixed hierarchical k-means "
        "(HKM) and learned residual quantization (RQ; three 16-entry codebooks) produce three "
        "semantic tokens plus a deterministic collision suffix. RQ fits a metadata-only "
        "reconstruction/codebook/commitment objective, selecting an epoch count on a 90/10 "
        "metadata split before fresh full-catalog refitting. Each tokenizer is paired with a "
        "permutation of complete codewords, including suffixes; each pair preserves its code "
        "set and trie topology while changing item-content alignment; interaction-weighted "
        "token exposure can also change. All three RQ tokenizers "
        "are fitted before code mappings and a common vocabulary are frozen. Four arms use "
        "three paired pipeline seeds (42, 43, 44), fresh Transformer weights and identical model "
        "dimensions. Their separate next-item objective is teacher-forced full-vocabulary "
        "token cross entropy. Epoch selection uses the same 1,000-user recommendation-validation "
        "cohort: maximize Recall@100 subject to MAP@20 at least validation popularity; retain a "
        "reported fallback if no epoch passes. All 12 selected checkpoint hashes are saved "
        "before any test retrieval. Trie-constrained beam search uses width 200 and excludes "
        "the full warm seen-item history. Final testing includes all eligible warm-target "
        "users, with cold targets excluded by the frozen preprocessing protocol. "
        "Primary contrasts use Recall@100; MAP@20 and other "
        "cutoffs are descriptive. Final full-test metrics and planned paired contrasts "
        "retain negative or inconclusive outcomes. Per-seed user-bootstrap intervals condition "
        "on fitted models; sample SD across pipeline seeds describes fitting variability and "
        "does not treat repeated users as independent samples. This figure depicts the design, "
        "without performance claims.\n\n"
        "Sources: [frozen protocol](../m3_protocol.md), "
        "[M3 learning checkpoint](../learning_checkpoint_m3.md).\n",
        encoding="utf-8",
    )
    print(caption)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    draw_m3_architecture(args.output_dir)


if __name__ == "__main__":
    main()
