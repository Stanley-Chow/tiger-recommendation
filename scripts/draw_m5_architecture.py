"""Export the M5 temporal recommendation architecture as SVG, PDF, and PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as DrawingPath  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, TEAL, INK = "#0F4D92", "#42949E", "#243244"


def draw(output: Path):
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "text.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(16, 9.5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 16), ylim=(0, 9.5))
    ax.axis("off")
    ax.text(
        0.5,
        9.05,
        "M5  |  Temporal recommendation with an explicit new-item route",
        fontsize=19,
        weight="bold",
    )
    ax.text(
        0.5,
        8.6,
        "Amazon Reviews 2023 Beauty & Personal Care  •  train < 2022"
        "  •  validate 2022  •  test 2023",
        fontsize=12,
        color="#5B6675",
    )
    bounds = []

    def block(x, y, width, title, body, color=BLUE):
        height = 1.25
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=.015,rounding_size=.06",
                linewidth=1.2,
                edgecolor=color,
                facecolor="#EEF4FA" if color == BLUE else "#EFF7F6",
            )
        )
        for yy, value, size, weight in (
            (y + 0.96, title, 11.5, "bold"),
            (y + 0.42, body, 10.2, "normal"),
        ):
            text = ax.text(
                x + width / 2,
                yy,
                value,
                ha="center",
                va="center",
                fontsize=size,
                weight=weight,
                linespacing=1.35,
            )
            bounds.append((text, (x, y, width, height)))

    def arrow(points, color=BLUE, dashed=False):
        path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1))
        ax.add_patch(
            FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color=color,
                linestyle="--" if dashed else "-",
                zorder=0,
            )
        )

    block(
        0.5,
        7.0,
        3.0,
        "Static product metadata",
        "Title, store, categories,\nfeatures and description",
    )
    block(
        4.1,
        7.0,
        3.0,
        "Frozen MiniLM encoder",
        "Pinned pretrained weights\n384-dimensional unit vectors",
        TEAL,
    )
    block(
        7.7,
        7.0,
        3.4,
        "Warm-only RQ tokenizer",
        "Three 32-entry residual codebooks\nSeparate metadata objective",
    )
    block(
        11.7,
        7.0,
        3.8,
        "Warm semantic IDs",
        "Three semantic tokens + suffix\nUnique, fixed training-item codes",
    )
    block(
        0.5,
        4.3,
        3.0,
        "Chronological user history",
        "Frozen at request cutoff\nFull seen set; latest 20 inputs",
    )
    block(4.1, 4.3, 3.0, "Closed sequence branch", "Generative Transformer\nor SASRec baseline")
    block(
        7.7,
        4.3,
        3.4,
        "Warm candidate retrieval",
        "Generator: trie + beam 200\nSASRec: full-catalog top 200",
    )
    block(
        11.7,
        4.3,
        3.8,
        "Fixed rank fusion",
        "Equal-weight reciprocal ranks\nConstant 60; no test tuning",
        TEAL,
    )
    block(
        4.1,
        1.9,
        3.0,
        "Mean-history prototype",
        "Latest 20 available item vectors\nNormalize before cosine scoring",
        TEAL,
    )
    block(
        7.7,
        1.9,
        3.4,
        "Available-catalog content",
        "Warm + new items; exact top 200\nExclude all earlier seen products",
        TEAL,
    )
    block(
        11.7,
        1.9,
        3.8,
        "Verified inference bundle",
        "Prespecified seed 42\nASIN-based CLI + local HTTP API",
        TEAL,
    )
    arrow([(3.52, 7.62), (4.08, 7.62)])
    arrow([(7.12, 7.62), (7.68, 7.62)])
    arrow([(11.12, 7.62), (11.68, 7.62)])
    arrow([(13.6, 6.98), (13.6, 6.2), (5.6, 6.2), (5.6, 5.57)])
    ax.text(
        10.4, 6.36, "Generator uses fixed warm codes", fontsize=10, ha="center"
    )
    arrow([(3.52, 4.92), (4.08, 4.92)])
    arrow([(7.12, 4.92), (7.68, 4.92)])
    arrow([(11.12, 4.92), (11.68, 4.92)])
    arrow([(3.52, 4.65), (3.8, 4.65), (3.8, 2.52), (4.08, 2.52)], TEAL)
    arrow([(7.12, 2.52), (7.68, 2.52)], TEAL)
    arrow([(7.12, 7.35), (7.4, 7.35), (7.4, 3.55), (9.4, 3.55), (9.4, 3.17)], TEAL, True)
    ax.text(9.4, 3.7, "Frozen warm + new vectors", fontsize=10, ha="center", color=TEAL)
    arrow([(11.12, 2.52), (11.4, 2.52), (11.4, 4.65), (11.68, 4.65)], TEAL)
    arrow([(13.6, 4.28), (13.6, 3.17)], TEAL)
    ax.text(
        0.5,
        0.85,
        "New items are reached through content retrieval; "
        "collision suffixes do not provide zero-shot generation.",
        fontsize=11,
        weight="bold",
    )
    ax.text(
        0.5,
        0.43,
        "Three pipeline seeds • Recall/NDCG/MAP/MRR @ 10/20/50/100/200"
        " • Paired user bootstrap • Independent artifact audit",
        fontsize=10.5,
        color="#5B6675",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text, (x, y, width, height) in bounds:
        box = text.get_window_extent(renderer).transformed(ax.transData.inverted())
        if box.x0 < x or box.x1 > x + width or box.y0 < y or box.y1 > y + height:
            raise RuntimeError(f"Label exceeds its architecture block: {text.get_text()}")
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(output / f"m5_architecture.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw(ROOT / "docs/figures")
