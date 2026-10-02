"""Checks for item alignment, collision handling, and offline reproducibility."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from beautyrec.embeddings import build_embeddings, product_text
from beautyrec.semantic_ids import build_semantic_ids, load_semantic_ids


def write_catalog(directory: Path, titles: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    catalog = [
        {"item_id": index, "asin": f"PRODUCT{index}", "title": title}
        for index, title in enumerate(titles)
    ]
    path = directory / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def write_embeddings(directory: Path, catalog: Path, vectors: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "embeddings.npy", vectors.astype(np.float32))
    (directory / "embedding_config.json").write_text(
        json.dumps({"catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )


def test_identical_embeddings_get_unique_leaf_suffixes(tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    catalog = write_catalog(processed, ["Same cream"] * 5)
    write_embeddings(artifacts, catalog, np.ones((5, 3)))

    payload = build_semantic_ids(processed, artifacts, branching_factor=2, depth=3)
    codebook = load_semantic_ids(artifacts)

    assert codebook.item_codes == [[3, 5, 7, suffix] for suffix in range(9, 14)]
    assert codebook.vocab_size == 14
    assert len({tuple(code) for code in codebook.item_codes}) == 5
    assert payload["diagnostics"]["leaf_count"] == 1
    assert payload["diagnostics"]["unique_embedding_count"] == 1


def test_tiny_nodes_use_level_tokens_then_suffix(tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    catalog = write_catalog(processed, ["A", "B", "C"])
    write_embeddings(artifacts, catalog, np.eye(3))

    build_semantic_ids(processed, artifacts, branching_factor=16, depth=3)
    codebook = load_semantic_ids(artifacts / "semantic_ids.json")

    assert codebook.item_codes == [[3, 19, 35, 51], [4, 19, 35, 51], [5, 19, 35, 51]]
    assert codebook.code_length == 4
    assert codebook.vocab_size == 52
    assert codebook.to_dict()["diagnostics"]["max_leaf_size"] == 1


def test_clustering_is_reproducible_and_keeps_neighboring_items_together(tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    catalog = write_catalog(processed, ["A", "B", "C", "D", "E"])
    vectors = np.array([[0, 0], [0.01, 0], [10, 10], [10.01, 10], [10.01, 10]])
    write_embeddings(artifacts, catalog, vectors)

    first = build_semantic_ids(processed, artifacts, branching_factor=2, depth=1, seed=7)
    first_contents = (artifacts / "semantic_ids.json").read_bytes()
    second = build_semantic_ids(processed, artifacts, branching_factor=2, depth=1, seed=7)

    assert first == second
    assert first_contents == (artifacts / "semantic_ids.json").read_bytes()
    codes = first["item_codes"]
    assert codes[0][0] == codes[1][0]
    assert codes[2][0] == codes[3][0] == codes[4][0]
    assert codes[0][0] != codes[2][0]
    assert codes[3] != codes[4]


def test_catalog_changes_invalidate_embeddings(tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    catalog = write_catalog(processed, ["Old title", "Another title"])
    write_embeddings(artifacts, catalog, np.eye(2))
    write_catalog(processed, ["Updated title", "Another title"])

    with pytest.raises(ValueError, match="different catalog"):
        build_semantic_ids(processed, artifacts)


@pytest.mark.parametrize("vectors", [np.ones((1, 3)), np.array([[0, np.nan], [1, 0]])])
def test_invalid_embedding_artifacts_are_rejected(tmp_path, vectors):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    catalog = write_catalog(processed, ["A", "B"])
    write_embeddings(artifacts, catalog, vectors)

    with pytest.raises(ValueError, match="Embedding|finite"):
        build_semantic_ids(processed, artifacts)


@pytest.mark.parametrize(
    "arguments",
    [
        {"branching_factor": 0},
        {"branching_factor": 1},
        {"branching_factor": True},
        {"depth": 0},
        {"depth": 1.5},
        {"seed": -1},
    ],
)
def test_invalid_tokenizer_arguments_are_rejected(tmp_path, arguments):
    with pytest.raises(ValueError):
        build_semantic_ids(tmp_path, tmp_path, **arguments)


def test_loaded_codebook_rejects_colliding_item_codes(tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    catalog = write_catalog(processed, ["A", "B"])
    write_embeddings(artifacts, catalog, np.eye(2))
    payload = build_semantic_ids(processed, artifacts)
    payload["item_codes"][1] = payload["item_codes"][0]
    path = artifacts / "semantic_ids.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_semantic_ids(path)


def test_tfidf_backend_is_normalized_reproducible_and_marked_diagnostic(tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    write_catalog(processed, ["Red berry lip balm", "Rose face cream", "Foam face cleanser"])

    first = build_embeddings(processed, artifacts, backend="tfidf", seed=11)
    first_vectors = np.load(artifacts / "embeddings.npy")
    second = build_embeddings(processed, artifacts, backend="tfidf", seed=11)
    second_vectors = np.load(artifacts / "embeddings.npy")

    assert first == second
    assert first["diagnostic_only"] is True
    assert first["frozen"] is True
    assert first_vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(first_vectors, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(first_vectors, second_vectors, atol=1e-6)
    codes = build_semantic_ids(processed, artifacts, branching_factor=2, depth=2)
    assert codes["diagnostics"]["diagnostic_only"] is True


@pytest.mark.parametrize("titles", [["Only product"], ["Same product"] * 4, ["!!!", "???"]])
def test_tfidf_handles_tiny_and_identical_metadata(tmp_path, titles):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    write_catalog(processed, titles)

    build_embeddings(processed, artifacts, backend="tfidf")
    vectors = np.load(artifacts / "embeddings.npy")
    build_semantic_ids(processed, artifacts, branching_factor=2, depth=3)
    codebook = load_semantic_ids(artifacts)

    assert vectors.shape == (len(titles), 1)
    assert np.isfinite(vectors).all()
    assert len({tuple(code) for code in codebook.item_codes}) == len(titles)


def test_product_text_uses_metadata_and_flattens_categories():
    text = product_text(
        {
            "asin": "PRIVATE_ITEM_ID",
            "title": "Rose cream",
            "brand": "Example",
            "categories": [["Beauty", "Skin Care"]],
            "description": ["Soft texture", "Light scent"],
        }
    )
    assert text == "Rose cream. Example. Beauty. Skin Care. Soft texture. Light scent"
    assert "PRIVATE_ITEM_ID" not in text
