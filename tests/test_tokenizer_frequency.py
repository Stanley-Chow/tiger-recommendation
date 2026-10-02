"""Verify the distinction between catalog occupancy and interaction exposure."""

import math

import numpy as np
import pytest

from beautyrec.semantic_ids import SemanticCodebook
from beautyrec.tokenizer_diagnostics import interaction_weighted_diagnostics, permute_codebook


def codes():
    return SemanticCodebook(
        item_codes=[[3, 5], [3, 6], [4, 5], [4, 6]],
        vocab_size=7,
        code_length=2,
        catalog_sha256="a" * 64,
        depth=1,
        branching_factor=2,
    )


def test_popular_items_change_exposure_even_when_permutation_keeps_catalog_occupancy():
    book = codes()
    counts = np.array([5, 3, 1, 1])
    original = interaction_weighted_diagnostics(book, counts)
    shuffled, _ = permute_codebook(book, seed=0)
    control = interaction_weighted_diagnostics(shuffled, counts)
    group = original["cumulative_prefixes"][0]
    expected_entropy = -0.8 * math.log2(0.8) - 0.2 * math.log2(0.2)
    assert original["training_interactions"] == control["training_interactions"] == 10
    assert group["entropy_bits"] == pytest.approx(expected_entropy)
    assert group["perplexity"] == pytest.approx(2**expected_entropy)
    assert group["largest_group_interaction_share"] == pytest.approx(0.8)
    assert group["catalog_occupied_prefixes"] == 2
    assert control["cumulative_prefixes"][0]["catalog_occupied_prefixes"] == 2
    assert control["cumulative_prefixes"][0]["largest_group_interaction_share"] == pytest.approx(
        0.6
    )
    assert original["training_counts_values_sha256"] == control["training_counts_values_sha256"]


def test_zero_frequency_items_do_not_create_nan_entropy():
    result = interaction_weighted_diagnostics(codes(), np.array([4, 0, 0, 0]))
    for position in result["positions"]:
        assert position["groups_with_positive_training_count"] == 1
        assert position["entropy_bits"] == 0
        assert position["perplexity"] == 1


@pytest.mark.parametrize(
    "counts", [np.zeros(4), [1, 2], [1, 1, -1, 1], [1, 2, 0.5, 1], [1, 1, np.nan, 1]]
)
def test_invalid_training_count_weights_are_rejected(counts):
    with pytest.raises(ValueError):
        interaction_weighted_diagnostics(codes(), counts)
