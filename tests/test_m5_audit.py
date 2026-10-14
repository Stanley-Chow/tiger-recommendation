"""Check that the independent M5 audit catches coherent-looking metric errors."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from beautyrec.amazon2023 import prepare_temporal_cohort
from beautyrec.evaluation import evaluate_rankings
from beautyrec.m5_demo import create_demo_source

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("m5_audit", ROOT / "scripts/verify_m5_results.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_audit_recalculates_metrics_instead_of_trusting_saved_values(tmp_path):
    examples = [{"user_id": "user", "history": [0], "seen_items": [0], "target": 2}]
    popularity = np.array([2, 1, 1, 1])
    summary = evaluate_rankings(examples, {"content": [[1, 2, 3]]}, popularity, output_dir=tmp_path)
    path = tmp_path / "per_user_rankings.jsonl"
    audit.verify_rankings(path, summary, examples, popularity, ["content"], summary["cutoffs"], 4)
    record = json.loads(path.read_text("utf-8"))
    record["methods"]["content"]["metrics"]["map@20"] = 0.6
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Numeric mismatch"):
        audit.verify_rankings(
            path, summary, examples, popularity, ["content"], summary["cutoffs"], 4
        )


@pytest.mark.parametrize("field", ["history", "warm_history", "target_type", "timestamp"])
def test_audit_reconstructs_contexts_from_untruncated_source(tmp_path, field):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    config = create_demo_source(raw)
    prepare_temporal_cohort(raw, processed, config)
    options = json.loads((processed / "statistics.json").read_text("utf-8"))["config"]
    audit.verify_temporal_contexts(processed, options)
    path = processed / ("test.jsonl" if field == "history" else "mixed_test.jsonl")
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    if field in ("history", "warm_history"):
        records[0][field] = records[0][field][:-1]
    elif field == "target_type":
        records[0][field] = "warm" if records[0][field] == "new" else "new"
    else:
        records[0][field] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    with pytest.raises(
        ValueError, match="Temporal context|Warm projection|target label|event time"
    ):
        audit.verify_temporal_contexts(processed, options)
