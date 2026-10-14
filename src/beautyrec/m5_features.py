"""Encode available metadata once, then expose its warm prefix to the tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from beautyrec.benchmark import _atomic_json, _file_hash, _json_hash
from beautyrec.embeddings import DEFAULT_TEXT_MODEL, _offline_embeddings, product_text, read_catalog

MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_FILES = {
    "config_sentence_transformers.json": (
        "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3"
    ),
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "sentence_bert_config.json": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
}


def build_m5_features(
    processed: Path,
    output: Path,
    *,
    model_path=None,
    device="cpu",
    batch_size=128,
    max_text_length=256,
    offline=False,
):
    """Pin the public encoder revision; offline features are explicitly diagnostic."""
    processed, output = Path(processed), Path(output)
    warm, warm_hash = read_catalog(processed)
    available_path = processed / "available_catalog.json"
    available = json.loads(available_path.read_text("utf-8"))
    if available[: len(warm)] != warm:
        raise ValueError("The available catalog must begin with the exact warm catalog.")
    fingerprint = _json_hash(
        {
            "warm": warm_hash,
            "available": _file_hash(available_path),
            "revision": MINILM_REVISION,
            "offline": offline,
            "batch_size": batch_size,
            "max_text_length": max_text_length,
        }
    )
    marker = output / "features_complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text("utf-8"))
        if saved["fingerprint"] != fingerprint or any(
            _file_hash(output / name) != sha for name, sha in saved["files"].items()
        ):
            raise ValueError("Completed feature inputs or files changed; use a new directory.")
        return json.loads((output / "embedding_config.json").read_text("utf-8"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Incomplete feature directory; use a new output directory.")
    texts = [product_text(item) for item in available]
    if offline:
        vectors, extra = _offline_embeddings(texts, seed=2026)
    else:
        from sentence_transformers import SentenceTransformer

        if model_path is None:
            encoder = SentenceTransformer(
                DEFAULT_TEXT_MODEL, revision=MINILM_REVISION, device=device
            )
            model_files = {}
        else:
            model_path = Path(model_path)
            if any(_file_hash(model_path / name) != sha for name, sha in MINILM_FILES.items()):
                raise ValueError("Local encoder files differ from the pinned MiniLM revision.")
            encoder = SentenceTransformer(str(model_path), device=device, local_files_only=True)
            model_files = {
                str(path.relative_to(model_path)): _file_hash(path)
                for path in sorted(model_path.rglob("*"))
                if path.is_file()
            }
        encoder.max_seq_length = max_text_length
        vectors = encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        extra = {
            "diagnostic_only": False,
            "revision": MINILM_REVISION,
            "local_encoder_files": model_files,
        }
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[0] != len(available) or not np.isfinite(vectors).all():
        raise ValueError("The encoder returned invalid catalog vectors.")
    manifest = {
        "model_name": "tfidf-svd-diagnostic" if offline else DEFAULT_TEXT_MODEL,
        "embedding_dim": vectors.shape[1],
        "catalog_sha256": warm_hash,
        "available_catalog_sha256": _file_hash(available_path),
        "normalized": True,
        "frozen": True,
        "max_text_length": max_text_length,
        "warm_items": len(warm),
        "available_items": len(available),
        "fields": ["title", "store", "categories", "features", "description"],
        "historical_metadata_caveat": "2023 snapshot; historical field availability unverified",
        **extra,
    }
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "available_embeddings.npy", vectors, allow_pickle=False)
    np.save(output / "embeddings.npy", vectors[: len(warm)], allow_pickle=False)
    _atomic_json(output / "embedding_config.json", manifest)
    _atomic_json(
        marker,
        {
            "fingerprint": fingerprint,
            "files": {
                name: _file_hash(output / name)
                for name in ("embeddings.npy", "available_embeddings.npy", "embedding_config.json")
            },
        },
    )
    return manifest
