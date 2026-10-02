"""Draw the implemented M2 architecture as editable SVG, PDF, and 300-dpi PNG.

Run from the repository root after installing ``pip install -e ".[figures]"``:
    python scripts/draw_architecture.py

This documentation figure needs no dataset, checkpoints, or GPU. Dimensions come
from the checked-in M2 config; catalog/tokenizer counts describe the completed M2
artifacts. Update these annotations if the tokenizer or model architecture changes.
"""

from __future__ import annotations

import argparse
import json
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
GREEN = "#32725B"
PURPLE = "#77518F"
EDGE = "#C9D2DC"


def draw_architecture(output_dir: Path) -> None:
    """Compose three readable stages, then export the same figure in each format."""
    config = json.loads((ROOT / "configs/m2.json").read_text(encoding="utf-8"))
    model = config["models"]["generative"]
    max_history = config["training"]["max_history"]
    beam_size = config["evaluation"]["beam_size"]
    code_length = 4
    max_source_tokens = max_history * code_length + 2

    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "text.color": INK,
            "svg.fonttype": "none",  # Keep labels editable in vector applications.
            "pdf.fonttype": 42,  # Embed TrueType fonts for a portable paper figure.
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(14, 10.1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 14), ylim=(0, 10.1))
    ax.axis("off")
    text_bounds: list[tuple[Text, tuple[float, float, float, float]]] = []

    def label(x: float, y: float, text: str, **kwargs: object) -> Text:
        return ax.text(x, y, text, va="center", **kwargs)

    def panel(
        x: float, y: float, width: float, height: float, title: str, color: str
    ) -> None:
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
        label(x + 0.22, y + height - 0.30, title, fontsize=13, weight="bold", color=color)

    def block(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        body: str,
        color: str = BLUE,
        fill: str = "#EEF4FA",
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.015,rounding_size=0.055",
                facecolor=fill,
                edgecolor=color,
                linewidth=1.15,
            )
        )
        heading = label(
            x + width / 2,
            y + height - 0.24,
            title,
            fontsize=12.5,
            weight="bold",
            ha="center",
            color=color,
        )
        detail = label(
            x + width / 2,
            y + height / 2 - 0.13,
            body,
            fontsize=10.8,
            ha="center",
            linespacing=1.45,
        )
        text_bounds.extend((text, (x, y, width, height)) for text in (heading, detail))

    def arrow(points: list[tuple[float, float]], color: str = GRAY, dashed=False) -> None:
        path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1))
        ax.add_patch(
            FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color=color,
                linestyle="--" if dashed else "-",
                zorder=2,
            )
        )

    label(0.35, 9.80, "Semantic-ID generative recommendation", fontsize=21, weight="bold")
    label(
        0.36,
        9.45,
        "Implemented M2 model · Amazon Reviews 2014 Beauty · simplified TIGER-style",
        fontsize=12,
        color=GRAY,
    )

    # A: metadata encoding and clustering happen once, before model training.
    panel(0.35, 7.72, 13.30, 1.43, "A  OFFLINE ITEM TOKENIZER", GREEN)
    item_y = 7.93
    block(
        0.58,
        item_y,
        2.65,
        0.75,
        "Product metadata",
        "Title · brand\nCategories · description",
        GREEN,
        "#EDF6F0",
    )
    block(
        3.69,
        item_y,
        2.75,
        0.75,
        "Frozen MiniLM",
        "all-MiniLM-L6-v2\n384-d normalized vectors",
        GREEN,
        "#EDF6F0",
    )
    block(
        6.90,
        item_y,
        2.75,
        0.75,
        "Hierarchical k-means",
        "3 levels · branching ≤16\n+ unique within-leaf suffix",
        GREEN,
        "#EDF6F0",
    )
    block(
        10.11,
        item_y,
        3.30,
        0.75,
        "Fixed item ↔ code table",
        "12,068 items · [c1, c2, c3, u]\n4 tokens per item · vocabulary 94",
        GREEN,
        "#EDF6F0",
    )
    for left, right in ((3.25, 3.66), (6.46, 6.87), (9.67, 10.08)):
        arrow([(left, 8.30), (right, 8.30)], GREEN)

    # Route the fixed code table outside panel B so its heading stays unobscured.
    arrow(
        [(11.76, 7.91), (11.76, 7.53), (0.17, 7.53), (0.17, 6.20), (0.57, 6.20)], GREEN, dashed=True
    )
    label(
        6.8,
        7.53,
        "fixed item-to-code lookup",
        ha="center",
        fontsize=10.5,
        color=GREEN,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
    )

    # B: one shared token embedding serves encoder and decoder inputs.
    panel(0.35, 4.47, 13.30, 2.73, "B  TRAINABLE ENCODER–DECODER TRANSFORMER", BLUE)
    model_y = 5.59
    block(
        0.58,
        model_y,
        2.65,
        1.07,
        "History → code tokens",
        f"Last {max_history} items, in time order\nBOS + codes + EOS · ≤{max_source_tokens} tokens",
    )
    block(
        3.69,
        model_y,
        2.75,
        1.07,
        "Transformer encoder",
        "Shared embedding + positions\n"
        f"{model['num_encoder_layers']} layers · bidirectional attention",
    )
    block(
        6.90,
        model_y,
        2.75,
        1.07,
        "Transformer decoder",
        "Shared embedding + positions\n"
        f"{model['num_decoder_layers']} layers · causal + cross-attention",
    )
    block(
        10.11,
        model_y,
        3.30,
        1.07,
        "Next-token logits",
        f"Separate linear head · {model['d_model']} → 94\nFull-vocabulary scores",
    )
    for left, right in ((3.25, 3.66), (6.46, 6.87), (9.67, 10.08)):
        arrow([(left, 6.13), (right, 6.13)], BLUE)

    # The decoder sees a true prefix in training and its generated prefix at inference.
    block(6.90, 4.70, 2.75, 0.60, "BOS + previous code tokens", "True prefix / generated prefix")
    arrow([(8.275, 5.32), (8.275, 5.56)], BLUE)
    label(
        0.60,
        5.16,
        f"M2: d = {model['d_model']} · {model['nhead']} heads · FFN = {model['dim_feedforward']}"
        f" · dropout = {model['dropout']}",
        fontsize=10.7,
        color=GRAY,
    )
    label(
        0.60,
        4.85,
        "Input embeddings are shared; positions are sinusoidal.",
        fontsize=10.7,
        color=GRAY,
    )

    # C: the common logits feed a training loss or an inference search procedure.
    panel(0.35, 0.62, 6.29, 3.42, "C1  TRAINING", PURPLE)
    panel(6.91, 0.62, 6.74, 3.42, "C2  RECOMMENDATION", BLUE)
    score_y = 4.28
    arrow([(11.76, 5.57), (11.76, score_y), (5.04, score_y), (5.04, 3.45)], PURPLE)
    arrow([(11.76, score_y), (11.76, 3.45)], BLUE)
    ax.plot([11.76], [score_y], "o", color=GRAY, markersize=3, zorder=3)

    block(
        0.64,
        2.39,
        2.48,
        1.04,
        "Next-item labels",
        "[c1, c2, c3, u, EOS]\nUsing A's fixed code table",
        PURPLE,
        "#F5EFF8",
    )
    block(
        3.81,
        2.39,
        2.45,
        1.04,
        "Token cross entropy",
        "Full vocabulary · ignore PAD\nBackpropagate to Transformer",
        PURPLE,
        "#F5EFF8",
    )
    arrow([(3.14, 2.91), (3.78, 2.91)], PURPLE)
    label(0.68, 1.97, "Teacher forcing: decoder receives the true prefix.", fontsize=11.1)
    label(0.68, 1.59, "Decoder input: [BOS, c1, c2, c3, u]", fontsize=11.1)
    label(0.68, 1.13, "MiniLM and the item codes stay frozen.", fontsize=11.1, color=PURPLE)

    block(
        7.19,
        2.39,
        2.42,
        1.04,
        "Catalog constraints",
        "Valid-code prefix trie from A\nExclude ALL earlier seen items",
    )
    block(
        10.11,
        2.39,
        3.30,
        1.04,
        "Autoregressive beam search",
        f"Log-softmax → mask · width {beam_size}\nGenerate 4 code tokens + EOS",
    )
    arrow([(9.63, 2.91), (10.08, 2.91)], BLUE)
    block(
        8.12,
        0.93,
        4.97,
        0.80,
        "Code → catalog item lookup",
        "Ranked unseen products · return top K",
    )
    arrow([(11.76, 2.37), (11.76, 1.75)], BLUE)
    label(
        7.20,
        2.05,
        "Rank by Σ log p, including EOS; no masked renormalization.",
        fontsize=10.2,
        color=GRAY,
    )

    label(
        0.37,
        0.27,
        "Solid arrows: model/data flow.  Dashed arrow: fixed tokenizer lookup.",
        fontsize=10.4,
        color=GRAY,
    )
    label(
        13.63,
        0.27,
        "Warm-item retrieval over the training catalog",
        ha="right",
        fontsize=10.4,
        color=GRAY,
    )

    # Catch overflowing box labels before writing any deliverables.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text, (x, y, width, height) in text_bounds:
        bounds = text.get_window_extent(renderer).transformed(ax.transData.inverted())
        if bounds.x0 < x + 0.03 or bounds.x1 > x + width - 0.03:
            raise ValueError(f"Label exceeds its box width: {text.get_text()!r}")
        if bounds.y0 < y + 0.02 or bounds.y1 > y + height - 0.02:
            raise ValueError(f"Label exceeds its box height: {text.get_text()!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        target = output_dir / f"model_architecture.{extension}"
        fig.savefig(target, dpi=300, metadata={"Creator": "beautyrec architecture figure"})
        print(target)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    draw_architecture(args.output_dir)


if __name__ == "__main__":
    main()
