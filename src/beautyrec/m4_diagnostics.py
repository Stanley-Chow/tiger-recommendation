"""Report token prediction and approximate-search errors on validation users.

Teacher-forced target likelihood is used only for diagnosis, never to insert an
item into retrieval. An absent eligible target that scores above the last kept
candidate certifies a beam-search miss (up to an explicit numerical tolerance).
Wider beams are diagnostic and do not change the frozen test retrieval setting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from beautyrec.batched_decoding import batch_constrained_beam_search
from beautyrec.benchmark import _atomic_json
from beautyrec.decoding import SemanticTrie
from beautyrec.evaluation import context_fingerprint
from beautyrec.model import build_source_tokens


def prefix_hits(target_code, ranked_items, item_codes):
    """Descriptive prefix coverage; it does not identify a causal failure stage."""
    return [
        any(item_codes[item][:depth] == target_code[:depth] for item in ranked_items)
        for depth in (1, 2, 3)
    ]


@torch.inference_mode()
def diagnose_search(model, book, examples, max_history: int, config: dict, output: Path):
    model.eval()
    device = next(model.parameters()).device
    trie = SemanticTrie(book.item_codes, book.eos_token)
    records = []
    tolerance = 1e-4
    for start in range(0, len(examples), config["batch_size"]):
        batch = examples[start : start + config["batch_size"]]
        source = pad_sequence(
            [
                torch.tensor(build_source_tokens(row["history"], book.item_codes, max_history))
                for row in batch
            ],
            batch_first=True,
            padding_value=book.pad_token,
        ).to(device)
        labels = torch.tensor(
            [book.item_codes[row["target"]] + [book.eos_token] for row in batch], device=device
        )
        decoder = torch.cat(
            (torch.full((len(batch), 1), book.bos_token, device=device), labels[:, :-1]), dim=1
        )
        logits = model(source, decoder)
        log_probs = logits.log_softmax(-1)
        target_token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).cpu()
        correct = logits.argmax(-1).eq(labels).cpu()
        exclusions = [set(row.get("seen_items", row["history"])) for row in batch]
        predictions = {
            str(width): batch_constrained_beam_search(
                model, source, trie, top_k=200, beam_size=width, exclude_items=exclusions
            )
            for width in config["beam_sizes"]
        }
        for index, row in enumerate(batch):
            if row["target"] in exclusions[index]:
                raise ValueError("Diagnostic target is excluded by the source context.")
            target_score = float(target_token_logp[index].sum())
            item = {
                "user_id": row["user_id"],
                "history": row["history"],
                "seen_items": row.get("seen_items", row["history"]),
                "target": row["target"],
                "target_token_log_probabilities": target_token_logp[index].tolist(),
                "teacher_forced_token_argmax_correct": correct[index].tolist(),
                "target_log_probability": target_score,
                "beam": {},
            }
            for width, rows in predictions.items():
                ranked = rows[index]
                ids = [candidate for candidate, _ in ranked]
                rank = ids.index(row["target"]) + 1 if row["target"] in ids else None
                cutoff_score = ranked[-1][1] if len(ranked) == 200 else None
                certified = (
                    rank is None
                    and cutoff_score is not None
                    and target_score > cutoff_score + tolerance
                )
                if rank is not None and abs(ranked[rank - 1][1] - target_score) > tolerance:
                    raise ValueError("Teacher-forced and decoded target likelihood disagree.")
                item["beam"][width] = {
                    "rank": rank,
                    "candidates": ids,
                    "scores": [s for _, s in ranked],
                    "prefix_hits": prefix_hits(
                        book.item_codes[row["target"]], ids, book.item_codes
                    ),
                    "certified_search_miss": certified,
                }
            records.append(item)
    summary = {
        "split": "validation",
        "used_for_selection": False,
        "examples": len(examples),
        "contexts_sha256": context_fingerprint(examples),
        "beam_sizes": config["beam_sizes"],
        "score_tolerance": tolerance,
        "candidate_cutoff": 200,
        "mean_target_token_nll": (
            -np.asarray([r["target_token_log_probabilities"] for r in records]).mean(axis=0)
        ).tolist(),
        "teacher_forced_token_argmax_accuracy": np.asarray(
            [r["teacher_forced_token_argmax_correct"] for r in records]
        )
        .mean(axis=0)
        .tolist(),
        "interpretation": (
            "Token metrics condition on true earlier target tokens. Prefix coverage and wider-beam "
            "differences are descriptive; certified misses are a lower bound, "
            "not exhaustive search error."
        ),
    }
    for width in config["beam_sizes"]:
        rows = [r["beam"][str(width)] for r in records]
        summary[str(width)] = {
            "recall@100": float(
                np.mean([r["rank"] is not None and r["rank"] <= 100 for r in rows])
            ),
            "recall@200": float(np.mean([r["rank"] is not None for r in rows])),
            "certified_search_misses": sum(r["certified_search_miss"] for r in rows),
            "prefix_coverage@200": np.asarray([r["prefix_hits"] for r in rows])
            .mean(axis=0)
            .tolist(),
        }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_user.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    _atomic_json(output / "summary.json", summary)
    return summary
