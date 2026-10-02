"""Hand-worked ranking examples distinguish AP denominators and cutoffs."""

import math

import pytest

from beautyrec.metrics import average_precision_at_k, next_item_metrics, ranking_metrics


def test_single_positive_map_and_mrr_are_reciprocal_rank_with_cutoff():
    metrics = next_item_metrics([4, 8, 2], target=2, cutoffs=[1, 2, 3, 5])
    for cutoff in [1, 2]:
        assert metrics[f"recall@{cutoff}"] == 0
        assert metrics[f"ndcg@{cutoff}"] == 0
        assert metrics[f"map@{cutoff}"] == 0
        assert metrics[f"mrr@{cutoff}"] == 0
    for cutoff in [3, 5]:
        assert metrics[f"recall@{cutoff}"] == 1
        assert metrics[f"ndcg@{cutoff}"] == 0.5
        assert metrics[f"map@{cutoff}"] == pytest.approx(1 / 3)
        assert metrics[f"mrr@{cutoff}"] == pytest.approx(1 / 3)
    assert ranking_metrics([4, 8, 2], target=2, top_k=3) == (1, 0.5)


def test_multi_positive_ap_denominator_counts_all_relevant_items():
    # Positives at ranks 1 and 3 contribute precisions 1 and 2/3.
    assert average_precision_at_k([1, 9, 2, 3], {1, 2, 3}, top_k=3) == pytest.approx(5 / 9)
    # One positive inside a one-item cutoff, but three positives in total.
    assert average_precision_at_k([1, 9, 2, 3], {1, 2, 3}, top_k=1) == pytest.approx(1 / 3)
    assert average_precision_at_k([1, 9, 2, 3], {1, 2, 3}, top_k=4) == pytest.approx(29 / 36)
    assert average_precision_at_k([1, 9], set(), top_k=2) == 0


def test_missing_target_and_short_list_receive_zero_without_padding():
    assert ranking_metrics([], target=2, top_k=20) == (0, 0)
    assert ranking_metrics([1], target=2, top_k=20) == (0, 0)
    assert ranking_metrics([1, 2], target=2, top_k=20) == (1, 1 / math.log2(3))


def test_duplicate_predictions_are_rejected_even_beyond_requested_cutoff():
    with pytest.raises(ValueError, match="duplicate"):
        ranking_metrics([1, 2, 1], target=1, top_k=1)
    with pytest.raises(ValueError, match="duplicate"):
        average_precision_at_k([1, 1], {1, 2}, top_k=2)


@pytest.mark.parametrize("cutoffs", [[], [0], [-1], [1.5], [True]])
def test_invalid_cutoffs_are_rejected(cutoffs):
    with pytest.raises(ValueError, match="positive integers"):
        next_item_metrics([1], target=1, cutoffs=cutoffs)


def test_cutoffs_are_normalized_deterministically():
    metrics = next_item_metrics([1], target=1, cutoffs=[20, 10, 20])
    assert list(metrics)[:4] == ["recall@10", "ndcg@10", "map@10", "mrr@10"]
    assert len(metrics) == 8
