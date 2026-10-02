"""Check saved-list metrics and paired uncertainty on small known rankings."""

import json
import math

import numpy as np
import pytest

from beautyrec.evaluation import (
    aggregate_seed_summaries,
    bootstrap_comparison,
    evaluate_rankings,
)


@pytest.fixture
def examples():
    return [
        {"user_id": "alice", "history": [0], "seen_items": [0], "target": 1},
        {"user_id": "bob", "history": [0], "seen_items": [0], "target": 2},
        {"user_id": "chris", "history": [5], "seen_items": [0, 5], "target": 3},
        {"user_id": "dana", "history": [0], "seen_items": [0], "target": 4},
    ]


@pytest.fixture
def rankings():
    return [[1, 2, 3], [3, 2, 4], [1, 3, 2], [2, 1, 3]]


def test_metrics_tail_coverage_and_latency_match_hand_calculation(examples, rankings):
    report = evaluate_rankings(
        examples,
        {"model": rankings},
        np.array([10, 8, 1, 2, 9, 3]),
        cutoffs=[1, 2, 3],
        latency_ms={"model": [1, 2, 3, 4]},
    )
    model = report["model"]
    assert model["recall@1"] == 0.25
    assert model["recall@2"] == 0.75
    assert model["recall@3"] == 0.75
    assert model["ndcg@2"] == pytest.approx((1 + 2 / math.log2(3)) / 4)
    assert model["map@2"] == model["mrr@2"] == 0.5
    assert model["tail_examples"] == 2
    assert model["tail_recall@2"] == 1
    assert model["tail_map@2"] == 0.5
    assert model["catalog_coverage@1"] == 0.5
    assert model["catalog_coverage@2"] == 0.5
    assert model["catalog_coverage@3"] == pytest.approx(4 / 6)
    assert model["mean_latency_ms"] == 2.5
    assert model["p95_latency_ms"] == pytest.approx(3.85)
    assert report["cutoffs"] == [1, 2, 3]
    assert report["metric_protocol"]["map_equals_mrr"] is True


def test_saved_rows_retain_same_context_rank_and_maximum_cutoff_list(tmp_path, examples, rankings):
    report = evaluate_rankings(
        examples,
        {"model": rankings},
        np.ones(6),
        cutoffs=[1, 2],
        output_dir=tmp_path,
    )
    rows = [
        json.loads(line) for line in (tmp_path / "per_user_rankings.jsonl").read_text().splitlines()
    ]
    assert rows[0]["methods"]["model"]["ranked_items"] == [1, 2]
    assert rows[0]["methods"]["model"]["target_rank"] == 1
    assert rows[1]["methods"]["model"]["target_rank"] == 2
    assert rows[3]["methods"]["model"]["target_rank"] is None
    assert rows[2]["history"] == [5]
    assert rows[2]["seen_items"] == [0, 5]
    assert report == json.loads((tmp_path / "summary.json").read_text())


@pytest.mark.parametrize(
    "bad_ranking, message",
    [
        ([1, 1], "duplicate"),
        ([0, 1], "excluded"),
        ([6, 1], "outside"),
        ([-1, 1], "outside"),
    ],
)
def test_invalid_predictions_cannot_enter_metrics(examples, rankings, bad_ranking, message):
    changed = [bad_ranking, *rankings[1:]]
    with pytest.raises(ValueError, match=message):
        evaluate_rankings(examples, {"model": changed}, np.ones(6), cutoffs=[2])


def test_full_seen_exclusion_rejects_item_outside_source_history(examples, rankings):
    changed = [*rankings[:2], [0, 3, 2], rankings[3]]
    assert 0 not in examples[2]["history"]
    with pytest.raises(ValueError, match="excluded"):
        evaluate_rankings(examples, {"model": changed}, np.ones(6), cutoffs=[2])


def test_target_cannot_be_excluded_and_method_lengths_must_align(examples, rankings):
    with pytest.raises(ValueError, match="one-to-one"):
        evaluate_rankings(examples, {"model": rankings[:-1]}, np.ones(6), cutoffs=[2])
    bad_examples = [{**examples[0], "seen_items": [0, 1]}, *examples[1:]]
    with pytest.raises(ValueError, match="target cannot"):
        evaluate_rankings(bad_examples, {"model": rankings}, np.ones(6), cutoffs=[2])


def test_paired_bootstrap_identical_methods_have_exact_zero_difference(examples, rankings):
    first = bootstrap_comparison(
        examples, rankings, rankings, cutoffs=[1, 2], resamples=200, seed=9
    )
    second = bootstrap_comparison(
        examples, rankings, rankings, cutoffs=[1, 2], resamples=200, seed=9
    )
    assert first == second
    assert first["reference"]["recall@2"]["mean"] == 0.75
    assert first["reference"]["map@2"]["mean"] == 0.5
    for interval in first["delta"].values():
        assert interval == {"mean": 0, "lower": 0, "upper": 0}


def test_paired_bootstrap_improvement_direction_and_bounds(examples, rankings):
    perfect = [[example["target"]] for example in examples]
    report = bootstrap_comparison(
        examples, rankings, perfect, cutoffs=[1, 2], resamples=200, seed=9
    )
    assert report["delta"]["recall@1"]["mean"] == 0.75
    assert report["delta"]["map@2"]["mean"] == 0.5
    assert report["candidate"]["recall@1"] == {"mean": 1, "lower": 1, "upper": 1}
    assert report["delta"]["recall@1"]["lower"] >= 0
    assert report["delta"]["recall@1"]["upper"] <= 1


def test_paired_bootstrap_resamples_repeated_user_examples_together():
    contexts = [
        {"user_id": "alice", "history": [0], "target": 1},
        {"user_id": "alice", "history": [0], "target": 2},
    ]
    # One user's two outcomes always remain one miss and one hit after resampling.
    report = bootstrap_comparison(contexts, [[], [2]], [[1], []], cutoffs=[1], resamples=30)
    assert report["users"] == 1
    assert report["reference"]["recall@1"] == {"mean": 0.5, "lower": 0.5, "upper": 0.5}
    assert report["delta"]["recall@1"] == {"mean": 0, "lower": 0, "upper": 0}


def test_paired_comparison_rejects_context_changes_even_with_same_user_ids(examples, rankings):
    candidate_examples = [{**examples[0], "seen_items": [0, 5]}, *examples[1:]]
    with pytest.raises(ValueError, match="identical ordered contexts"):
        bootstrap_comparison(
            examples,
            rankings,
            rankings,
            cutoffs=[2],
            candidate_examples=candidate_examples,
        )
    with pytest.raises(ValueError, match="identical ordered contexts"):
        bootstrap_comparison(
            examples,
            rankings,
            rankings,
            cutoffs=[2],
            candidate_examples=list(reversed(examples)),
        )


def test_seed_aggregation_reports_sample_standard_deviation(examples, rankings):
    first = evaluate_rankings(examples, {"model": rankings}, np.ones(6), cutoffs=[1])
    perfect = [[example["target"]] for example in examples]
    second = evaluate_rankings(examples, {"model": perfect}, np.ones(6), cutoffs=[1])
    report = aggregate_seed_summaries([first, second])
    assert report["model"]["recall@1"]["mean"] == 0.625
    assert report["model"]["recall@1"]["std"] == pytest.approx(0.75 / math.sqrt(2))
    assert aggregate_seed_summaries([first])["model"]["recall@1"]["std"] is None
    with pytest.raises(ValueError, match="same contexts"):
        aggregate_seed_summaries([first, {**second, "contexts_sha256": "different"}])
