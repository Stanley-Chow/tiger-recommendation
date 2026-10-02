"""Hierarchical k-means codes for a simplified TIGER-style recommender.

Each clustering level contributes one token, from coarse product groups to
finer groups. A final token distinguishes products that occupy the same leaf.
The resulting fixed-length codes identify every item exactly once.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from beautyrec.embeddings import read_catalog


@dataclass
class SemanticCodebook:
    """Catalog-aligned item codes and the vocabulary needed by the generator."""

    item_codes: list[list[int]]
    vocab_size: int
    code_length: int
    catalog_sha256: str
    pad_token: int = 0
    bos_token: int = 1
    eos_token: int = 2
    branching_factor: int = 0
    depth: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.pad_token, self.bos_token, self.eos_token) != (0, 1, 2):
            raise ValueError("Semantic codes reserve PAD=0, BOS=1, and EOS=2.")
        if self.code_length < 1 or self.vocab_size < 4 or not self.item_codes:
            raise ValueError("A codebook needs a nonempty catalog and positive code length.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.catalog_sha256):
            raise ValueError("catalog_sha256 must be a lowercase SHA-256 checksum.")
        for code in self.item_codes:
            if len(code) != self.code_length:
                raise ValueError("All semantic codes must have the declared fixed length.")
            if any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or not 3 <= token < self.vocab_size
                for token in code
            ):
                raise ValueError("Item codes must contain valid non-special vocabulary tokens.")
        if len({tuple(code) for code in self.item_codes}) != len(self.item_codes):
            raise ValueError("Every catalog item must have a unique semantic code.")

        if self.depth > 0 and self.branching_factor > 0:
            if self.code_length != self.depth + 1:
                raise ValueError("A hierarchical code has depth semantic tokens and one suffix.")
            suffix_offset = 3 + self.depth * self.branching_factor
            for code in self.item_codes:
                for level, token in enumerate(code[:-1]):
                    start = 3 + level * self.branching_factor
                    if not start <= token < start + self.branching_factor:
                        raise ValueError("A semantic token is outside its level's vocabulary.")
                if code[-1] < suffix_offset:
                    raise ValueError("The final code token must belong to the suffix vocabulary.")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible artifact representation."""
        return asdict(self)


def load_semantic_ids(path: Path) -> SemanticCodebook:
    """Load and validate a saved codebook before training or recommending."""
    path = Path(path)
    if path.is_dir():
        path = path / "semantic_ids.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("semantic_ids.json must contain a codebook object.")
    try:
        return SemanticCodebook(**payload)
    except TypeError as error:
        raise ValueError("The semantic-ID artifact has missing or unexpected fields.") from error


def _cluster_node(
    vectors: np.ndarray, item_ids: np.ndarray, branching: int, seed: int
) -> list[np.ndarray]:
    """Split a node, then give children stable labels ordered by their first item."""
    node_vectors = vectors[item_ids]
    unique_vectors, inverse = np.unique(node_vectors, axis=0, return_inverse=True)
    cluster_count = min(branching, len(unique_vectors))
    if cluster_count == 1:
        return [item_ids]

    if cluster_count == len(unique_vectors):
        # With one cluster for each distinct vector, no optimization is needed.
        labels = inverse
    else:
        labels = KMeans(
            n_clusters=cluster_count,
            n_init=10,
            random_state=seed,
            algorithm="lloyd",
        ).fit_predict(node_vectors)

    children = [item_ids[labels == label] for label in np.unique(labels)]
    return sorted(children, key=lambda child: int(child[0]))


def build_semantic_ids(
    processed_dir: Path,
    artifact_dir: Path,
    branching_factor: int = 16,
    depth: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Create deterministic hierarchical codes from the saved product embeddings.

    Clusters smaller than the branching factor use fewer children. When a node
    contains only one distinct vector, remaining semantic levels use index zero.
    Leaf-local suffixes preserve uniqueness even for identical product metadata.
    """
    if (
        isinstance(branching_factor, bool)
        or not isinstance(branching_factor, int)
        or branching_factor < 2
    ):
        raise ValueError("branching_factor must be an integer of at least 2.")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")

    catalog, catalog_sha256 = read_catalog(processed_dir)
    artifact_dir = Path(artifact_dir)
    embedding_config = json.loads(
        (artifact_dir / "embedding_config.json").read_text(encoding="utf-8")
    )
    if embedding_config.get("catalog_sha256") != catalog_sha256:
        raise ValueError("Embeddings belong to a different catalog; rebuild them first.")
    vectors = np.load(artifact_dir / "embeddings.npy", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(catalog) or vectors.shape[1] < 1:
        raise ValueError("Embedding rows must match the catalog and contain nonempty vectors.")
    if not np.issubdtype(vectors.dtype, np.number) or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must contain finite numeric values.")

    n_items = len(catalog)
    level_indices = np.zeros((n_items, depth), dtype=np.int64)
    leaves: list[np.ndarray] = []
    split_counts = [0] * depth

    def visit(item_ids: np.ndarray, level: int) -> None:
        if level == depth:
            leaves.append(item_ids)
            return
        children = _cluster_node(vectors, item_ids, branching_factor, seed)
        if len(children) == 1:
            # The initially zero-filled remaining levels pad this terminal node.
            leaves.append(item_ids)
            return
        split_counts[level] += 1
        for child_index, child_ids in enumerate(children):
            level_indices[child_ids, level] = child_index
            visit(child_ids, level + 1)

    visit(np.arange(n_items), 0)

    level_offsets = 3 + np.arange(depth) * branching_factor
    codes = np.zeros((n_items, depth + 1), dtype=np.int64)
    codes[:, :depth] = level_indices + level_offsets
    suffix_offset = 3 + depth * branching_factor
    max_leaf_size = max(len(leaf) for leaf in leaves)
    for leaf in leaves:
        # Catalog IDs arrive sorted; their rank within the leaf is reproducible.
        codes[leaf, -1] = suffix_offset + np.arange(len(leaf))

    codebook = SemanticCodebook(
        item_codes=codes.tolist(),
        vocab_size=suffix_offset + max_leaf_size,
        code_length=depth + 1,
        catalog_sha256=catalog_sha256,
        branching_factor=branching_factor,
        depth=depth,
        diagnostics={
            "method": "hierarchical-kmeans",
            "seed": seed,
            "n_items": n_items,
            "unique_embedding_count": int(len(np.unique(vectors, axis=0))),
            "leaf_count": len(leaves),
            "max_leaf_size": max_leaf_size,
            "suffix_vocab_size": max_leaf_size,
            "split_counts_by_level": split_counts,
            "embedding_backend": embedding_config.get("backend"),
            "diagnostic_only": embedding_config.get("diagnostic_only", False),
        },
    )
    payload = codebook.to_dict()
    (artifact_dir / "semantic_ids.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
