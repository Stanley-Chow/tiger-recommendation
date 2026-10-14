"""Protect global time boundaries, cohort sampling and new-item eligibility."""

import csv
import gzip
import json
from dataclasses import replace

import numpy as np
import pytest

from beautyrec.amazon2023 import TemporalCohortConfig, prepare_temporal_cohort, read_interactions
from beautyrec.training import read_examples


def create_temporal_data(directory, reverse=False):
    directory.mkdir(parents=True, exist_ok=True)
    config = TemporalCohortConfig(max_users=8, max_warm_items=8, max_new_items=2)
    rows = []
    for user in range(8):
        # Rotation puts every product in training while leaving unseen warm targets.
        for position in range(5):
            rows.append(
                [
                    f"user{user}",
                    f"W{(user + position) % 8}",
                    5,
                    config.train_end_ms - (6 - position) * 100000,
                ]
            )
        rows.append([f"user{user}", f"W{(user + 5) % 8}", 4, config.train_end_ms + 10000])
        target = f"W{(user + 6) % 8}" if user % 2 == 0 else f"N{user % 2}"
        rows.append([f"user{user}", target, 5, config.test_start_ms + 20000])
    # Establish new-item availability in 2022 without selecting these inactive users.
    rows += [["observer", f"N{i}", 5, config.train_end_ms + 1000] for i in range(2)]
    # A first future target outside the available catalog must not be skipped.
    rows.append(["user0", "UNAVAILABLE", 5, config.test_start_ms + 10000])
    # Tied-time training products must never become each other's history.
    rows[1][-1] = rows[0][-1]
    if reverse:
        rows.reverse()
    reviews = directory / "Beauty_and_Personal_Care.csv.gz"
    with gzip.open(reviews, "wt", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
        writer.writerows(rows)
    metadata = directory / "meta_Beauty_and_Personal_Care.jsonl.gz"
    with gzip.open(metadata, "wt", encoding="utf-8") as file:
        for asin in [*[f"W{i}" for i in range(8)], "N0", "N1"]:
            file.write(
                json.dumps(
                    {
                        "parent_asin": asin,
                        "title": f"Gentle skincare {asin[-1]}",
                        "store": "Example",
                        "categories": ["Beauty", "Skincare"],
                        "description": ["Hydrating cleanser"],
                        "average_rating": "DO_NOT_EMBED",
                        "rating_number": "DO_NOT_EMBED",
                        "price": "DO_NOT_EMBED",
                        "text": "DO_NOT_EMBED",
                    }
                )
                + "\n"
            )
    return config


def test_temporal_pipeline_preserves_past_and_reports_unavailable_first_target(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    config = create_temporal_data(raw)
    result = prepare_temporal_cohort(raw, processed, config)
    assert result["users"] == 8 and result["items"] == 8 and result["new_items"] == 2
    assert result["examples"]["test"] == 3 and result["examples"]["mixed_test"] == 7
    assert result["eligibility"]["test"]["target_outside_available_test_catalog"] == 1
    assert result["eligibility"]["test"]["new_eligible"] == 4
    assert np.load(processed / "item_popularity.npy").sum() == 40
    for split in ("train", "validation", "test"):
        for row in read_examples(processed / f"{split}.jsonl"):
            assert row["history_end_timestamp"] < row["timestamp"]
            assert row["target"] not in row["seen_items"]
            if split != "train":
                assert row["history_end_timestamp"] < row["request_timestamp"] <= row["timestamp"]
    mixed = read_examples(processed / "mixed_test.jsonl")
    assert not any(row["user_id"] == "user0" for row in mixed)
    assert all(row["target"] >= 8 for row in mixed if row["target_type"] == "new")
    catalog_text = (processed / "available_catalog.json").read_text(encoding="utf-8")
    assert "DO_NOT_EMBED" not in catalog_text
    assert prepare_temporal_cohort(raw, processed, config) == result
    with (processed / "test.jsonl").open("a", encoding="utf-8") as file:
        file.write("{}\n")
    with pytest.raises(ValueError, match="preprocessing changed"):
        prepare_temporal_cohort(raw, processed, config)


def test_cohort_and_examples_are_invariant_to_source_row_order(tmp_path):
    outputs = []
    for index, reverse in enumerate((False, True)):
        raw, output = tmp_path / f"raw{index}", tmp_path / f"output{index}"
        config = replace(create_temporal_data(raw, reverse), max_users=5)
        result = prepare_temporal_cohort(raw, output, config)
        outputs.append((result, output))
    assert outputs[0][0]["selected_user_ids"] == outputs[1][0]["selected_user_ids"]
    for name in (
        "catalog.json",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "mixed_test.jsonl",
    ):
        assert (outputs[0][1] / name).read_bytes() == (outputs[1][1] / name).read_bytes()


def test_wrong_schema_and_second_timestamps_fail_clearly(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("user_id,parent_asin,rating,timestamp\nu,i,5,1600000000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid interaction"):
        list(read_interactions(path))
    path.write_text("user_id,asin,timestamp\nu,i,1600000000000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        list(read_interactions(path))


@pytest.mark.parametrize(
    "change",
    [{"seed": True}, {"max_users": 0}, {"min_train_interactions": 1}, {"test_start_ms": 1}],
)
def test_invalid_temporal_config(change):
    with pytest.raises(ValueError):
        TemporalCohortConfig(**change)
