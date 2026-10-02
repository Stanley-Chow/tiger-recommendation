"""Hand-worked tokenizer usage and cosine geometry checks without training."""

import hashlib
import json
import math

import numpy as np
import pytest

from beautyrec.semantic_ids import SemanticCodebook
from beautyrec.tokenizer_diagnostics import permute_codebook, tokenizer_diagnostics


@pytest.fixture
def codebook():
    # Two semantic groups, two items per group, plus two unused suffix slots.
    return SemanticCodebook(
        item_codes=[[3, 5], [3, 6], [4, 5], [4, 6]],
        vocab_size=9,
        code_length=2,
        catalog_sha256="a" * 64,
        branching_factor=2,
        depth=1,
        diagnostics={"method": "handcrafted", "semantic_vocab_sizes": [2], "suffix_offset": 5},
    )


@pytest.fixture
def vectors():
    # Each item has a unique nearest neighbor in its semantic group, cosine 0.8.
    return np.array([[1, 0], [0.8, 0.6], [-1, 0], [-0.8, -0.6]], dtype=np.float64)


def test_permutation_preserves_capacity_code_set_and_counts_without_mutating_input(
    codebook, vectors
):
    before = codebook.to_dict()
    permuted, manifest = permute_codebook(codebook, seed=0)
    repeated, repeated_manifest = permute_codebook(codebook, seed=0)
    assert permuted.to_dict() == repeated.to_dict()
    assert manifest == repeated_manifest
    assert codebook.to_dict() == before
    assert sorted(permuted.item_codes) == sorted(codebook.item_codes)
    assert permuted.catalog_sha256 == codebook.catalog_sha256
    assert permuted.vocab_size == codebook.vocab_size
    assert permuted.code_length == codebook.code_length
    assert sorted(manifest["item_to_source_item"]) == [0, 1, 2, 3]
    assert manifest["fixed_points"] == 1
    for destination, source in enumerate(manifest["item_to_source_item"]):
        assert permuted.item_codes[destination] == codebook.item_codes[source]

    original_stats = tokenizer_diagnostics(codebook, vectors, neighbors=1)
    permuted_stats = tokenizer_diagnostics(permuted, vectors, neighbors=1)
    for field in ["positions", "cumulative_prefixes", "leaf_before_suffix", "suffix_vocabulary"]:
        assert original_stats[field] == permuted_stats[field]
    assert (
        original_stats["cosine_neighbors"]["query_item_ids"]
        == permuted_stats["cosine_neighbors"]["query_item_ids"]
    )
    assert original_stats["cosine_neighbors"]["prefix_agreement"][0]["observed_match_fraction"] == 1
    assert permuted_stats["cosine_neighbors"]["prefix_agreement"][0]["observed_match_fraction"] == 0
    permuted.item_codes[0][-1] = 8
    assert codebook.to_dict() == before


def test_uniform_permutation_reports_fixed_points_instead_of_repairing_them(codebook):
    # This ordinary seeded draw happens to be the identity; the manifest exposes it.
    permuted, manifest = permute_codebook(codebook, seed=1)
    assert manifest["fixed_points"] == 4
    assert permuted.item_codes == codebook.item_codes
    without_hash = {
        key: value for key, value in manifest.items() if key != "manifest_content_sha256"
    }
    canonical = json.dumps(without_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert manifest["manifest_content_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()


def test_utilization_perplexity_collisions_and_padded_suffix_are_distinct(codebook, vectors):
    report = tokenizer_diagnostics(codebook, vectors, neighbors=1)
    semantic, suffix = report["positions"]
    assert semantic["available_capacity"] == semantic["used_tokens"] == 2
    assert semantic["utilization"] == 1
    assert semantic["entropy_bits"] == 1
    assert semantic["perplexity"] == 2
    assert suffix["available_capacity"] == 4
    assert suffix["used_tokens"] == 2
    assert suffix["utilization"] == 0.5
    assert suffix["perplexity"] == 2
    leaf = report["leaf_before_suffix"]
    assert leaf["occupied_prefixes"] == leaf["available_prefix_capacity"] == 2
    assert leaf["collision_prefixes"] == 2
    assert leaf["items_in_collision_groups"] == 4
    assert leaf["excess_items_without_suffix"] == 2
    assert leaf["unordered_collision_pairs"] == 2
    assert leaf["occupancy_histogram"] == {"2": 2}
    assert leaf["random_distinct_item_pair_match_probability"] == pytest.approx(1 / 3)
    assert report["full_code_collisions"] == 0
    assert report["suffix_vocabulary"]["observed_token_span"] == 2
    assert report["suffix_vocabulary"]["slots_through_maximum"] == 2
    assert report["suffix_vocabulary"]["unused_capacity"] == 2


def test_cosine_neighbors_match_handcrafted_geometry_and_chance(codebook, vectors):
    # Scale rows differently: cosine should depend on direction, not magnitude.
    report = tokenizer_diagnostics(codebook, vectors * np.array([2, 3, 4, 5])[:, None], neighbors=1)
    neighbors = report["cosine_neighbors"]
    assert neighbors["queries"] == neighbors["pairs"] == 4
    assert neighbors["mean_cosine_similarity"] == pytest.approx(0.8)
    agreement = neighbors["prefix_agreement"][0]
    assert agreement["matching_neighbor_pairs"] == 4
    assert agreement["observed_match_fraction"] == 1
    assert agreement["random_other_item_match_probability"] == pytest.approx(1 / 3)
    assert agreement["catalog_random_pair_match_probability"] == pytest.approx(1 / 3)
    assert agreement["observed_minus_chance"] == pytest.approx(2 / 3)
    assert agreement["agreement_enrichment"] == pytest.approx(3)


def test_cumulative_prefixes_distinguish_leaf_collisions_from_level_counts():
    codes = SemanticCodebook(
        item_codes=[[3, 5, 7], [3, 5, 8], [3, 6, 7], [4, 5, 7], [4, 5, 8], [4, 6, 7]],
        vocab_size=10,
        code_length=3,
        catalog_sha256="a" * 64,
        branching_factor=2,
        depth=2,
    )
    report = tokenizer_diagnostics(codes, np.eye(6), neighbors=1, batch_size=2)
    coarse, leaf = report["cumulative_prefixes"]
    assert coarse["occupied_prefixes"] == 2
    assert coarse["unordered_collision_pairs"] == 6
    assert coarse["random_distinct_item_pair_match_probability"] == pytest.approx(0.4)
    assert leaf["occupied_prefixes"] == leaf["available_prefix_capacity"] == 4
    assert leaf["unordered_collision_pairs"] == 2
    assert leaf["items_in_collision_groups"] == 4
    assert leaf["excess_items_without_suffix"] == 2
    assert leaf["random_distinct_item_pair_match_probability"] == pytest.approx(2 / 15)
    # Second-position token counts 4:2 have entropy H(2/3,1/3).
    entropy = -(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3)
    assert report["positions"][1]["perplexity"] == pytest.approx(2**entropy)
    # Orthogonal vectors tie at cosine zero; lower item ID determines neighbors.
    agreements = report["cosine_neighbors"]["prefix_agreement"]
    assert agreements[0]["observed_match_fraction"] == 0.5
    assert agreements[1]["observed_match_fraction"] == pytest.approx(1 / 3)


def test_query_sampling_is_fixed_and_batching_does_not_change_agreement(codebook, vectors):
    first = tokenizer_diagnostics(
        codebook, vectors, sample_size=2, neighbors=1, batch_size=1, seed=9
    )
    second = tokenizer_diagnostics(
        codebook, vectors, sample_size=2, neighbors=1, batch_size=2, seed=9
    )
    assert (
        first["cosine_neighbors"]["query_item_ids"] == second["cosine_neighbors"]["query_item_ids"]
    )
    assert (
        first["cosine_neighbors"]["prefix_agreement"]
        == second["cosine_neighbors"]["prefix_agreement"]
    )
    assert first["cosine_neighbors"]["queries"] == 2
    assert first["cosine_neighbors"]["neighbors"] == 1
    assert (
        first["normalized_embedding_values_sha256"] == second["normalized_embedding_values_sha256"]
    )


def test_undeclared_capacity_stays_unknown_instead_of_using_observed_span(vectors):
    custom = SemanticCodebook(
        item_codes=[[3, 5], [3, 6], [4, 5], [4, 6]],
        vocab_size=9,
        code_length=2,
        catalog_sha256="a" * 64,
    )
    report = tokenizer_diagnostics(custom, vectors, neighbors=1)
    assert report["positions"][0]["used_tokens"] == 2
    assert report["positions"][0]["available_capacity"] is None
    assert report["positions"][0]["utilization"] is None
    assert report["suffix_vocabulary"]["available_capacity"] is None


def test_single_item_has_no_self_neighbor_or_nan():
    single = SemanticCodebook(
        item_codes=[[3, 5]],
        vocab_size=6,
        code_length=2,
        catalog_sha256="a" * 64,
        branching_factor=2,
        depth=1,
    )
    report = tokenizer_diagnostics(single, np.array([[1.0, 0.0]]))
    assert report["cosine_neighbors"]["neighbors"] == 0
    assert report["cosine_neighbors"]["mean_cosine_similarity"] is None
    assert (
        report["cosine_neighbors"]["prefix_agreement"][0]["random_other_item_match_probability"]
        is None
    )
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "vectors",
    [
        np.zeros((4, 2)),
        np.ones((3, 2)),
        np.full((4, 2), np.nan),
        np.ones((4, 2), dtype=complex),
    ],
)
def test_invalid_embedding_geometry_is_rejected(codebook, vectors):
    with pytest.raises(ValueError):
        tokenizer_diagnostics(codebook, vectors)
