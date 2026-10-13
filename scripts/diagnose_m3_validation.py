"""Inspect a fixed validation cohort before M4, using only saved M3 rankings.

Prefix coverage is descriptive: finding a related code does not prove that a
target was lost specifically at the suffix. No recommendation model is trained.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beautyrec.benchmark import _file_hash  # noqa: E402
from beautyrec.m4_diagnostics import prefix_hits  # noqa: E402
from beautyrec.semantic_ids import load_semantic_ids  # noqa: E402
from beautyrec.training import read_examples, sample_examples  # noqa: E402


def main():
    run = ROOT / "results/m3"
    marker = json.loads((run / "completed.json").read_text())
    contexts = sample_examples(read_examples(ROOT / "data/processed/validation.jsonl"), 1000, 2027)
    selected = {row["user_id"]: row for row in contexts}
    records = []
    for seed in (42, 43, 44):
        for arm in ("hkm", "rqvae"):
            book_path = run / "arms" / f"seed_{seed}" / arm / "semantic_ids.json"
            rankings = (
                run
                / "seeds"
                / f"seed_{seed}"
                / arm
                / "evaluation/validation/per_user_rankings.jsonl"
            )
            for path in (book_path, rankings):
                relative = path.relative_to(run).as_posix()
                assert _file_hash(path) == marker["result_files"][relative]
            book = load_semantic_ids(book_path)
            counts, hits, n = [0, 0, 0], 0, 0
            with rankings.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["user_id"] not in selected:
                        continue
                    assert row["target"] == selected[row["user_id"]]["target"]
                    candidates = row["methods"][arm]["ranked_items"]
                    values = prefix_hits(
                        book.item_codes[row["target"]], candidates, book.item_codes
                    )
                    counts = [left + int(right) for left, right in zip(counts, values)]
                    hits += row["target"] in candidates
                    n += 1
            assert n == len(contexts)
            records.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "examples": n,
                    "recall@200": hits / n,
                    "prefix_coverage@200": [value / n for value in counts],
                }
            )
    result = {
        "source": "verified saved M3 validation rankings only",
        "sampling_seed": 2027,
        "examples": len(contexts),
        "used_for_selection": False,
        "records": records,
        "caveat": "Prefix coverage is descriptive; it does not attribute errors causally.",
    }
    output = ROOT / "results/m4_preflight"
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_prefix_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
