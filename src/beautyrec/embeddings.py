"""Turn product metadata into frozen, normalized feature vectors.

The sentence-transformer encoder is the intended experiment setting. TF-IDF
and truncated SVD provide an explicit offline option for testing the pipeline;
they are not a reproduction of TIGER's text embeddings.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MISSING_METADATA_TEXT = "Beauty product with unavailable metadata."


def read_catalog(processed_dir: Path) -> tuple[list[dict[str, Any]], str]:
    """Read the item ordering shared by every stage and hash its exact contents."""
    catalog_path = Path(processed_dir) / "catalog.json"
    raw_catalog = catalog_path.read_bytes()
    catalog = json.loads(raw_catalog)
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("catalog.json must contain a nonempty list of products.")
    for index, item in enumerate(catalog):
        if not isinstance(item, dict) or item.get("item_id") != index:
            raise ValueError("Catalog item_id values must equal their contiguous list positions.")
    return catalog, hashlib.sha256(raw_catalog).hexdigest()


def _flatten_text(value: Any) -> list[str]:
    """Handle the nested category and description lists in Amazon metadata."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [text for entry in value for text in _flatten_text(entry)]
    return []


def product_text(item: dict[str, Any]) -> str:
    """Use readable metadata, without leaking item identifiers into its features."""
    existing_text = item.get("text")
    if isinstance(existing_text, str) and existing_text.strip():
        return existing_text.strip()

    parts: list[str] = []
    for field in ("title", "brand", "categories", "description"):
        parts.extend(_flatten_text(item.get(field)))
    return ". ".join(parts) if parts else MISSING_METADATA_TEXT


def _offline_embeddings(texts: list[str], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Build small deterministic features without fetching a pretrained model."""
    # A punctuation-only title has no word tokens. Keep even that case usable.
    texts = [text if re.search(r"\w", text) else MISSING_METADATA_TEXT for text in texts]
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    features = vectorizer.fit_transform(texts)

    # A one-product or identical-text catalog has no meaningful reduced axes.
    # Use a single constant vector instead of triggering SVD's zero-variance warning.
    if len(set(texts)) == 1 or min(features.shape) == 1:
        vectors = np.ones((len(texts), 1), dtype=np.float32)
        component_count = 1
    else:
        component_count = min(128, features.shape[0] - 1, features.shape[1] - 1)
        reducer = TruncatedSVD(n_components=component_count, random_state=seed)
        vectors = reducer.fit_transform(features).astype(np.float32)

    # SVD can place a document at the origin. A constant fallback keeps its code
    # deterministic while preserving the normalized-vector artifact contract.
    zero_rows = np.linalg.norm(vectors, axis=1) == 0
    vectors[zero_rows, 0] = 1.0
    vectors = normalize(vectors, norm="l2", axis=1).astype(np.float32)
    return vectors, {
        "tfidf_feature_count": int(features.shape[1]),
        "svd_component_count": int(component_count),
        "diagnostic_only": True,
    }


def build_embeddings(
    processed_dir: Path,
    output_dir: Path,
    backend: str = "sentence-transformers",
    model_name: str = DEFAULT_TEXT_MODEL,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
    max_text_length: int | None = None,
) -> dict[str, Any]:
    """Save one frozen metadata embedding for each catalog item.

    The row number of ``embeddings.npy`` is the corresponding ``item_id``.
    Its manifest records a catalog checksum so later stages cannot silently use
    embeddings with a different item ordering.
    """
    if backend not in {"sentence-transformers", "tfidf"}:
        raise ValueError("backend must be 'sentence-transformers' or 'tfidf'.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if max_text_length is not None and (
        isinstance(max_text_length, bool)
        or not isinstance(max_text_length, int)
        or max_text_length < 8
    ):
        raise ValueError("max_text_length must be at least 8 tokens or None.")

    catalog, catalog_sha256 = read_catalog(processed_dir)
    texts = [product_text(item) for item in catalog]
    details: dict[str, Any] = {"diagnostic_only": False}

    if backend == "tfidf":
        vectors, details = _offline_embeddings(texts, seed)
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "Sentence embeddings require the text extra: pip install -e '.[text]'. "
                "Use backend='tfidf' for an offline pipeline check."
            ) from error
        # Keep downloaded weights inside the project, rather than a hidden user cache.
        encoder = SentenceTransformer(
            model_name, device=device, cache_folder=str(Path(output_dir) / "text_model_cache")
        )
        native_length = encoder.max_seq_length
        if max_text_length is not None:
            if max_text_length > native_length:
                raise ValueError(
                    f"max_text_length exceeds this encoder's native limit {native_length}."
                )
            encoder.max_seq_length = max_text_length
        details["max_text_length"] = encoder.max_seq_length
        encoder.eval()
        vectors = encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(catalog) or vectors.shape[1] < 1:
        raise ValueError("The encoder must return one nonempty vector per catalog item.")
    if not np.isfinite(vectors).all():
        raise ValueError("The encoder produced nonfinite embedding values.")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("The encoder produced a zero embedding vector.")
    vectors = normalize(vectors, norm="l2", axis=1).astype(np.float32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", vectors, allow_pickle=False)
    config = {
        "backend": backend,
        "model_name": model_name if backend == "sentence-transformers" else "tfidf-svd",
        "catalog_sha256": catalog_sha256,
        "n_items": len(catalog),
        "embedding_dim": int(vectors.shape[1]),
        "frozen": True,
        "normalized": True,
        "seed": seed,
        "text_fields": ["title", "brand", "categories", "description"],
        **details,
    }
    (output_dir / "embedding_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return config
