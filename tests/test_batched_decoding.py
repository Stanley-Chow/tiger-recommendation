"""Compare batched retrieval against the independent single-history decoder."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import beautyrec.batched_decoding as batched
from beautyrec.batched_decoding import batch_constrained_beam_search
from beautyrec.decoding import SemanticTrie, constrained_beam_search
from beautyrec.model import GenerativeRecommender, ModelConfig


@pytest.fixture
def catalog():
    return SemanticTrie([[3, 5, 9], [3, 5, 10], [3, 6, 9], [4, 7, 9], [4, 7, 10], [4, 8, 9]])


@pytest.fixture
def model():
    torch.manual_seed(31)
    torch.set_num_threads(1)
    return GenerativeRecommender(
        ModelConfig(
            vocab_size=14,
            max_source_length=8,
            max_target_length=4,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
    ).eval()


def example_sources():
    return torch.tensor(
        [[1, 3, 5, 9, 2, 0, 0, 0], [1, 4, 7, 9, 3, 6, 9, 2], [1, 4, 8, 9, 2, 0, 0, 0]]
    )


@pytest.mark.parametrize("beam_size", [2, 4, 8])
def test_batch_matches_reference_ranks_scores_and_seen_exclusions(model, catalog, beam_size):
    sources = example_sources()
    exclusions = [{0}, {1, 2, -1, 100}, set(range(6))]
    top_k = min(beam_size, 4)
    actual = batch_constrained_beam_search(
        model, sources, catalog, top_k=top_k, beam_size=beam_size, exclude_items=exclusions
    )
    expected = [
        constrained_beam_search(
            model, source, catalog, top_k=top_k, beam_size=beam_size, exclude_items=seen
        )
        for source, seen in zip(sources, exclusions)
    ]
    for user_actual, user_expected, seen in zip(actual, expected, exclusions):
        assert [item for item, _ in user_actual] == [item for item, _ in user_expected]
        assert len({item for item, _ in user_actual}) == len(user_actual)
        assert all(item not in seen for item, _ in user_actual)
        for (item, score), (expected_item, expected_score) in zip(user_actual, user_expected):
            assert item == expected_item
            assert score == pytest.approx(expected_score, abs=1e-5)
    assert actual[-1] == []


def test_batch_padding_does_not_change_the_first_users_predictions(model, catalog):
    source = example_sources()[0, :5]
    alone = batch_constrained_beam_search(model, source.unsqueeze(0), catalog, top_k=6, beam_size=8)
    mixed = batch_constrained_beam_search(model, example_sources(), catalog, top_k=6, beam_size=8)
    assert [item for item, _ in alone[0]] == [item for item, _ in mixed[0]]
    torch.testing.assert_close(
        torch.tensor([score for _, score in alone[0]]),
        torch.tensor([score for _, score in mixed[0]]),
        rtol=1e-5,
        atol=1e-6,
    )


class TiedModel(nn.Module):
    """Every token has equal probability, making cutoff tie rules observable."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=14)
        self.encode_calls = 0
        self.decoder_batch_sizes = []

    def encode(self, sources):
        self.encode_calls += 1
        return torch.zeros((*sources.shape, 1), device=sources.device), sources.eq(0)

    def decode(self, tokens, memory, padding):
        self.decoder_batch_sizes.append(tokens.size(0))
        return torch.zeros((*tokens.shape, self.config.vocab_size), device=tokens.device)


@pytest.mark.parametrize("beam_size", [2, 4, 20])
def test_equal_scores_match_lexicographic_reference_and_prune_seen_branches(catalog, beam_size):
    model = TiedModel()
    sources = example_sources()
    exclusions = [set(), {0, 1}, set(range(6))]
    top_k = min(beam_size, 4)
    actual = batch_constrained_beam_search(
        model, sources, catalog, top_k=top_k, beam_size=beam_size, exclude_items=exclusions
    )
    for source, seen, user_actual in zip(sources, exclusions, actual):
        expected = constrained_beam_search(
            model, source, catalog, top_k=top_k, beam_size=beam_size, exclude_items=seen
        )
        assert [item for item, _ in user_actual] == [item for item, _ in expected]
        assert [score for _, score in user_actual] == pytest.approx(
            [score for _, score in expected], abs=1e-5
        )


def test_encodes_once_and_grows_beams_only_as_the_trie_branches(catalog):
    model = TiedModel()
    batch_constrained_beam_search(model, example_sources(), catalog, top_k=6, beam_size=200)
    assert model.encode_calls == 1
    # One root, two first-level nodes, four second-level nodes, six leaves.
    assert model.decoder_batch_sizes == [3, 6, 12, 18]


def test_ties_follow_codes_rather_than_catalog_item_order():
    codes = [[4, 8, 9], [4, 7, 10], [4, 7, 9], [3, 6, 9], [3, 5, 10], [3, 5, 9]]
    predictions = batch_constrained_beam_search(
        TiedModel(), example_sources()[:1], SemanticTrie(codes), top_k=4, beam_size=4
    )
    assert [item for item, _ in predictions[0]] == [5, 4, 3, 2]


def test_static_index_is_cached_across_inference_batches(monkeypatch, catalog):
    calls = 0
    original = batched._build_tensor_index

    def counted_build(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(batched, "_build_tensor_index", counted_build)
    model = TiedModel()
    for _ in range(2):
        batch_constrained_beam_search(model, example_sources(), catalog, top_k=2, beam_size=4)
    assert calls == 1


def test_large_cutoff_returns_every_available_item_without_inactive_beams(catalog):
    predictions = batch_constrained_beam_search(
        TiedModel(), example_sources(), catalog, top_k=200, beam_size=200,
        exclude_items=[{0, 1, 2, 3, 4}, set(), set(range(6))],
    )
    assert [item for item, _ in predictions[0]] == [5]
    assert [item for item, _ in predictions[1]] == list(range(6))
    assert predictions[2] == []


@pytest.mark.parametrize(
    "arguments",
    [{"top_k": 0}, {"top_k": 4, "beam_size": 2}, {"beam_size": True}, {"exclude_items": []}],
)
def test_invalid_search_arguments_are_rejected(catalog, arguments):
    with pytest.raises(ValueError):
        batch_constrained_beam_search(TiedModel(), example_sources(), catalog, **arguments)


def test_single_history_vector_needs_an_explicit_batch_dimension(catalog):
    with pytest.raises(ValueError, match="batch-by-length"):
        batch_constrained_beam_search(TiedModel(), example_sources()[0], catalog)
