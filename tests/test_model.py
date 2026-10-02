"""Check causality and catalog correctness rather than memorizing implementation details."""

import math

import pytest
import torch

from beautyrec.decoding import SemanticTrie, constrained_beam_search
from beautyrec.metrics import popularity_ranking, ranking_metrics
from beautyrec.model import GenerativeRecommender, ModelConfig, build_source_tokens


@pytest.fixture
def model():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    return GenerativeRecommender(
        ModelConfig(
            vocab_size=12,
            max_source_length=10,
            max_target_length=4,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
    ).eval()


def test_future_target_tokens_do_not_change_earlier_predictions(model):
    source = torch.tensor([[1, 3, 5, 2]])
    first = model(source, torch.tensor([[1, 3, 5]]))
    changed = model(source, torch.tensor([[1, 3, 7]]))
    torch.testing.assert_close(first[:, :2], changed[:, :2])


def test_source_padding_is_ignored(model):
    decoder = torch.tensor([[1, 3]])
    unpadded = model(torch.tensor([[1, 3, 5, 2]]), decoder)
    padded = model(torch.tensor([[1, 3, 5, 2, 0, 0]]), decoder)
    torch.testing.assert_close(unpadded, padded, rtol=1e-5, atol=1e-6)


def test_decoding_only_returns_valid_unseen_unique_items(model):
    codes = [[3, 5], [3, 6], [4, 7], [4, 8]]
    result = constrained_beam_search(
        model,
        torch.tensor([1, 3, 5, 2]),
        SemanticTrie(codes),
        top_k=3,
        beam_size=4,
        exclude_items={0},
    )
    assert {item for item, _ in result} == {1, 2, 3}
    assert all(math.isfinite(score) for _, score in result)
    assert [score for _, score in result] == sorted((score for _, score in result), reverse=True)


def test_all_seen_catalog_returns_empty(model):
    result = constrained_beam_search(
        model,
        torch.tensor([1, 3, 5, 2]),
        SemanticTrie([[3, 5]]),
        top_k=1,
        beam_size=1,
        exclude_items={0},
    )
    assert result == []


def test_wide_beam_matches_exhaustive_code_likelihoods(model):
    """Validate ranking probabilities independently of the beam's traversal."""
    codes = [[3, 5], [3, 6], [4, 7], [4, 8]]
    source = torch.tensor([[1, 3, 5, 2]])
    decoder = torch.tensor([[1, *code] for code in codes])
    labels = torch.tensor([[*code, 2] for code in codes])
    with torch.inference_mode():
        logits = model(source.expand(4, -1), decoder)
        exact_scores = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum(-1)
    expected = sorted(range(4), key=lambda item: -float(exact_scores[item]))
    decoded = constrained_beam_search(model, source, SemanticTrie(codes), top_k=4, beam_size=4)
    assert [item for item, _ in decoded] == expected
    for item, score in decoded:
        assert score == pytest.approx(float(exact_scores[item]), abs=1e-5)


def test_duplicate_codes_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        SemanticTrie([[3, 5], [3, 5]])


def test_history_truncation_preserves_whole_item_codes():
    assert build_source_tokens([0, 1, 2], [[3, 6], [4, 7], [5, 8]], 2) == [1, 4, 7, 5, 8, 2]


def test_single_target_metrics_and_popularity_tie_breaks():
    import numpy as np

    assert ranking_metrics([4, 2], target=2, top_k=2) == (1.0, 1 / math.log2(3))
    assert ranking_metrics([4, 2], target=2, top_k=1) == (0.0, 0.0)
    assert popularity_ranking(np.array([5, 5, 3]), excluded={0}, top_k=2) == [1, 2]
