"""Verify causal histories, catalog indexing, and full-softmax learning."""

import pytest
import torch

from beautyrec.sasrec import (
    SASRecConfig,
    SASRecDataset,
    SASRecRecommender,
    build_history_tokens,
    collate_sasrec_examples,
    prepare_history_batch,
)


@pytest.fixture
def model():
    torch.manual_seed(23)
    torch.set_num_threads(1)
    return SASRecRecommender(
        SASRecConfig(
            num_items=6,
            max_history=8,
            d_model=16,
            nhead=2,
            num_layers=2,
            dim_feedforward=32,
            dropout=0.0,
        )
    ).eval()


def test_future_history_items_do_not_change_earlier_states(model):
    first = model.encode(torch.tensor([[1, 2, 3, 4]]))
    changed = model.encode(torch.tensor([[1, 2, 5, 6]]))
    torch.testing.assert_close(first[:, :2], changed[:, :2])


def test_right_padding_and_longer_batch_neighbors_do_not_change_scores(model):
    alone = model(torch.tensor([[1, 2, 3]]))
    padded = model(torch.tensor([[1, 2, 3, 0, 0]]))
    batch = model(torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 2, 1]]))
    torch.testing.assert_close(alone, padded, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(alone[0], batch[0], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(batch).all()


def test_full_catalog_logits_have_no_pad_output_class(model):
    scores = model(torch.tensor([[1, 2, 0], [6, 3, 4]]))
    assert scores.shape == (2, 6)
    assert torch.isfinite(scores).all()
    loss = torch.nn.functional.cross_entropy(scores, torch.tensor([0, 5]))
    assert torch.isfinite(loss)


def test_batch_inference_accepts_unshifted_catalog_histories(model):
    histories = [[0, 1, 2], [5]]
    tokens = prepare_history_batch(histories, num_items=6, max_history=8)
    assert tokens.tolist() == [[1, 2, 3], [6, 0, 0]]
    torch.testing.assert_close(model.score_histories(histories), model(tokens))


def test_history_truncation_preserves_recent_items_and_shifts_only_inputs():
    assert build_history_tokens([0, 1, 2, 3], num_items=4, max_history=2) == [3, 4]
    dataset = SASRecDataset(
        [{"history": [0, 1, 2], "target": 0}, {"history": [3], "target": 3}],
        num_items=4,
        max_history=2,
    )
    tokens, targets = collate_sasrec_examples([dataset[0], dataset[1]])
    assert tokens.tolist() == [[2, 3], [4, 0]]
    assert targets.tolist() == [0, 3]
    assert tokens.dtype == targets.dtype == torch.long


@pytest.mark.parametrize("history", [[], [-1], [6], [True], [1.5], [6, 0, 1]])
def test_invalid_raw_histories_are_rejected_even_before_truncation(history):
    with pytest.raises(ValueError, match="history|History"):
        build_history_tokens(history, num_items=6, max_history=2)


@pytest.mark.parametrize(
    "tokens",
    [
        torch.tensor([[0, 0]]),
        torch.tensor([[0, 1]]),
        torch.tensor([[1, 0, 2]]),
        torch.tensor([[-1, 1]]),
        torch.tensor([[7]]),
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([1, 2]),
        torch.ones((1, 9), dtype=torch.long),
    ],
)
def test_invalid_or_left_padded_token_batches_are_rejected(model, tokens):
    with pytest.raises(ValueError):
        model(tokens)


@pytest.mark.parametrize("target", [-1, 4, True, 1.5])
def test_dataset_rejects_invalid_target_classes(target):
    dataset = SASRecDataset([{"history": [0, 1], "target": target}], num_items=4)
    with pytest.raises(ValueError, match="Targets"):
        dataset[0]


@pytest.mark.parametrize(
    "options",
    [
        {"num_items": 0},
        {"max_history": 0},
        {"num_layers": -1},
        {"d_model": 15, "nhead": 2},
        {"nhead": True},
        {"dropout": 1.0},
    ],
)
def test_invalid_model_dimensions_are_rejected(options):
    with pytest.raises(ValueError):
        SASRecConfig(**({"num_items": 4} | options))


def test_empty_inference_or_training_batches_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        prepare_history_batch([], num_items=4)
    with pytest.raises(ValueError, match="empty"):
        collate_sasrec_examples([])


def test_full_softmax_can_learn_distinct_next_items_without_training_pad(model):
    dataset = SASRecDataset(
        [
            {"history": [0], "target": 1},
            {"history": [0, 4], "target": 1},
            {"history": [2], "target": 3},
            {"history": [2, 5], "target": 3},
        ],
        num_items=6,
        max_history=8,
    )
    tokens, targets = collate_sasrec_examples([dataset[index] for index in range(len(dataset))])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    initial_loss = float(torch.nn.functional.cross_entropy(model(tokens), targets).detach())
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(tokens), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        scores = model(tokens)
        final_loss = float(torch.nn.functional.cross_entropy(scores, targets))
    assert final_loss < initial_loss * 0.2
    assert scores.argmax(dim=1).tolist() == targets.tolist()
    torch.testing.assert_close(model.item_embedding.weight[0], torch.zeros(16))
