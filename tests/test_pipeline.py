"""A real checkpoint round trip through all offline pipeline stages."""

import io
import json
import sys

import pytest
import torch

from beautyrec import cli
from beautyrec.cli import run_demo
from beautyrec.training import (
    HistoryLengthBatchSampler,
    RecommendationDataset,
    check_dataset,
    load_checkpoint,
    resolve_device,
)


def test_length_batches_cover_every_example_without_wasting_padding():
    examples = [{"history": [0] * length} for length in [8, 1, 7, 2, 6, 3, 5, 4]]
    dataset = RecommendationDataset(examples, codebook=None, max_history=10)
    sampler = HistoryLengthBatchSampler(
        dataset, batch_size=2, generator=torch.Generator().manual_seed(7)
    )
    batches = list(sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(8))
    assert all(
        max(len(examples[i]["history"]) for i in batch)
        - min(len(examples[i]["history"]) for i in batch)
        == 1
        for batch in batches
    )


def test_cli_emits_readable_utf8_even_with_a_legacy_console(monkeypatch):
    raw_output = io.BytesIO()
    legacy_output = io.TextIOWrapper(raw_output, encoding="ascii")
    monkeypatch.setattr(sys, "stdout", legacy_output)
    monkeypatch.setattr(sys, "argv", ["beautyrec", "recommend", "--history", "A"])
    monkeypatch.setattr(cli, "run_recommendation", lambda _: [{"title": "Crème®"}])
    cli.main()
    legacy_output.flush()
    assert json.loads(raw_output.getvalue().decode("utf-8"))[0]["title"] == "Crème®"


def test_offline_pipeline_trains_evaluates_and_reloads(tmp_path):
    result = run_demo(tmp_path, epochs=1)
    assert result["evaluation"]["diagnostic_only"] is True
    assert result["evaluation"]["examples"] > 0
    assert (tmp_path / "training/metrics.csv").is_file()
    _, codebook, checkpoint = load_checkpoint(
        tmp_path / "training/best.pt",
        tmp_path / "artifacts",
        resolve_device("cpu"),
    )
    assert checkpoint["epoch"] == 1
    assert len({tuple(code) for code in codebook.item_codes}) == len(codebook.item_codes)
    summary = json.loads((tmp_path / "evaluation/summary.json").read_text())
    assert 0 <= summary["generative"]["recall@5"] <= 1

    # The catalog can stay identical while labels change; catalog hashes alone miss this.
    test_path = tmp_path / "processed/test.jsonl"
    rows = [json.loads(line) for line in test_path.read_text().splitlines()]
    rows[0]["target"] = (rows[0]["target"] + 1) % len(codebook.item_codes)
    test_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="splits or popularity changed"):
        check_dataset(tmp_path / "processed", checkpoint)

    codes_path = tmp_path / "artifacts/semantic_ids.json"
    payload = json.loads(codes_path.read_text())
    payload["diagnostics"]["seed"] += 1
    codes_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Semantic IDs do not match"):
        load_checkpoint(
            tmp_path / "training/best.pt", tmp_path / "artifacts", resolve_device("cpu")
        )
