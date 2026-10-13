"""Protect capacity remapping and the real M4 train/evaluate/resume boundary."""

import json
from pathlib import Path

import numpy as np
import pytest

from beautyrec.data import create_demo_data, prepare_dataset
from beautyrec.embeddings import build_embeddings
from beautyrec.m4_benchmark import canonical_codebook, run_m4
from beautyrec.semantic_ids import SemanticCodebook, build_semantic_ids


def test_canonical_ranges_preserve_partitions_and_suffixes():
    original = SemanticCodebook(
        item_codes=[[3, 19, 35, 51], [3, 19, 35, 52], [18, 34, 50, 51]],
        vocab_size=53,
        code_length=4,
        catalog_sha256="0" * 64,
        depth=3,
        branching_factor=16,
        diagnostics={"method": "rqvae"},
    )
    transformed = canonical_codebook(original, 32, 4)
    assert transformed.item_codes == [[3, 35, 67, 99], [3, 35, 67, 100], [18, 50, 82, 99]]
    assert transformed.vocab_size == 103 and transformed.branching_factor == 32
    assert original.item_codes[1] == [3, 19, 35, 52]
    assert len({tuple(code) for code in transformed.item_codes}) == 3
    with pytest.raises(ValueError, match="cannot shrink"):
        canonical_codebook(original, 32, 1)


@pytest.fixture
def small_project(tmp_path):
    reviews, metadata = create_demo_data(tmp_path / "raw")
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    prepare_dataset(reviews, metadata, processed, max_history=5)
    build_embeddings(processed, artifacts, backend="tfidf")
    vectors = np.random.default_rng(3).normal(size=(16, 128)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(artifacts / "embeddings.npy", vectors)
    manifest = json.loads((artifacts / "embedding_config.json").read_text())
    manifest["embedding_dim"] = 128
    (artifacts / "embedding_config.json").write_text(json.dumps(manifest))
    build_semantic_ids(processed, artifacts, branching_factor=16, depth=3)
    config = json.loads((Path(__file__).parents[1] / "configs/m3_smoke.json").read_text())
    config["tokenizer"]["model"]["input_dim"] = 128
    config["codebook_sizes"] = [16, 32]
    config["search_diagnostics"] = {
        "examples": 4,
        "batch_size": 2,
        "sampling_seed": 2027,
        "beam_sizes": [200, 400],
    }
    return processed, artifacts, tmp_path / "run", config


def test_real_two_arm_smoke_freezes_shapes_scores_and_rejects_mutated_cache(small_project):
    processed, artifacts, run, config = small_project
    result = run_m4(processed, artifacts, run, config)
    assert result["diagnostic_only"] and not result["m4_complete"]
    assert result["aggregate_test"]["methods"] == ["popularity", "rq16", "rq32"]
    selected = json.loads((run / "selection_manifest.json").read_text())
    assert len({r["parameter_count"] for r in selected["selected"]["42"].values()}) == 1
    diagnostics = result["seed_results"][0]["search_diagnostics"]
    assert diagnostics["rq16"]["contexts_sha256"] == diagnostics["rq32"]["contexts_sha256"]
    assert len(diagnostics["rq16"]["mean_target_token_nll"]) == 5
    assert run_m4(processed, artifacts, run, config) == result
    # A changed completed ranking must never silently reuse a previous result.
    ranking = run / "seeds/seed_42/test/per_user_rankings.jsonl"
    with ranking.open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        run_m4(processed, artifacts, run, config)


def test_invalid_capacity_order_writes_no_protocol(small_project):
    processed, artifacts, run, config = small_project
    config["codebook_sizes"] = [32, 16]
    with pytest.raises(ValueError, match="ascending"):
        run_m4(processed, artifacts, run, config)
    assert not run.exists()
