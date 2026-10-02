"""Check that preprocessing preserves chronological eligibility and safe input."""

import gzip
import io
import json

import numpy as np
import pytest

from beautyrec.data import create_demo_data, download_beauty, prepare_dataset, read_records


def write_records(path, records, *, python_literals=False):
    """Write small realistic raw files without depending on network access."""
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for record in records:
            output.write((repr(record) if python_literals else json.dumps(record)) + "\n")
    return path


def make_inputs(tmp_path, sequences):
    reviews = [
        {"reviewerID": user, "asin": asin, "unixReviewTime": timestamp}
        for user, events in sequences.items()
        for timestamp, asin in events
    ]
    metadata = [
        {
            "asin": asin,
            "title": f"Gentle {asin}",
            "categories": [["Beauty", "Skincare"], ["Beauty", "Creams"]],
        }
        for asin in sorted({record["asin"] for record in reviews})
    ]
    return (
        write_records(tmp_path / "reviews.json.gz", reviews),
        write_records(tmp_path / "metadata.json.gz", metadata, python_literals=True),
    )


def load_examples(path):
    return list(read_records(path))


def test_last_two_split_drops_cold_targets_without_catalog_leakage(tmp_path):
    reviews, metadata = make_inputs(
        tmp_path,
        {
            "alice": list(enumerate(["A", "B", "C", "X", "Y"])),
            "bob": list(enumerate(["D", "E", "F", "A", "Z"])),
        },
    )
    output = tmp_path / "prepared"
    stats = prepare_dataset(reviews, metadata, output)
    catalog = json.loads((output / "catalog.json").read_text("utf-8"))
    item_mapping = {row["asin"]: row["item_id"] for row in catalog}

    assert sorted(item_mapping) == ["A", "B", "C", "D", "E", "F"]
    assert stats["cold_target_examples_dropped"] == {"validation": 1, "test": 2}
    assert stats["cold_history_events_removed"] == {"validation": 0, "test": 1}
    assert stats["examples"] == {"train": 4, "validation": 1, "test": 0}
    assert load_examples(output / "train.jsonl")[:2] == [
        {
            "user_id": "alice",
            "history": [item_mapping["A"]],
            "seen_items": [item_mapping["A"]],
            "target": item_mapping["B"],
        },
        {
            "user_id": "alice",
            "history": [item_mapping["A"], item_mapping["B"]],
            "seen_items": [item_mapping["A"], item_mapping["B"]],
            "target": item_mapping["C"],
        },
    ]
    assert load_examples(output / "validation.jsonl") == [
        {
            "user_id": "bob",
            "history": [item_mapping[item] for item in ["D", "E", "F"]],
            "seen_items": [item_mapping[item] for item in ["D", "E", "F"]],
            "target": item_mapping["A"],
        }
    ]
    # Prefix expansion must not multiply counts of early interactions.
    np.testing.assert_array_equal(np.load(output / "item_popularity.npy"), np.ones(6))


def test_test_history_contains_only_past_warm_items_and_clips_history(tmp_path):
    reviews, metadata = make_inputs(
        tmp_path,
        {
            "alice": list(enumerate(["A", "B", "C", "X", "D"])),
            "bob": list(enumerate(["D", "B", "C", "A", "X"])),
        },
    )
    output = tmp_path / "prepared"
    prepare_dataset(reviews, metadata, output, max_history=2)
    mapping = json.loads((output / "item_mapping.json").read_text("utf-8"))
    assert "X" not in mapping
    assert load_examples(output / "test.jsonl") == [
        {
            "user_id": "alice",
            "history": [mapping["B"], mapping["C"]],
            "seen_items": [mapping["A"], mapping["B"], mapping["C"]],
            "target": mapping["D"],
        }
    ]
    bob_validation = load_examples(output / "validation.jsonl")[0]
    assert bob_validation["history"] == [mapping["B"], mapping["C"]]
    assert bob_validation["target"] == mapping["A"]


def test_duplicate_reviews_keep_earliest_timestamp_and_ties_use_asin(tmp_path):
    reviews, metadata = make_inputs(
        tmp_path,
        {
            "alice": [(20, "B"), (1, "B"), (1, "A"), (3, "C"), (4, "D"), (5, "E")],
            "bob": list(enumerate(["D", "E", "A", "B", "C"])),
        },
    )
    output = tmp_path / "prepared"
    stats = prepare_dataset(reviews, metadata, output)
    mapping = json.loads((output / "item_mapping.json").read_text("utf-8"))
    first = load_examples(output / "train.jsonl")[0]
    assert first == {
        "user_id": "alice",
        "history": [mapping["A"]],
        "seen_items": [mapping["A"]],
        "target": mapping["B"],
    }
    assert stats["duplicate_user_item_reviews_removed"] == 1
    assert stats["unique_user_item_interactions"] == 10
    assert stats["training_interactions"] == 6
    assert np.load(output / "item_popularity.npy").sum() == 6


def test_missing_metadata_uses_fallback_without_dropping_interactions(tmp_path):
    reviews, _ = make_inputs(tmp_path, {"alice": list(enumerate(["A", "B", "C", "D", "E"]))})
    metadata = write_records(tmp_path / "empty_meta.json.gz", [{"asin": "A", "title": "A &amp; B"}])
    output = tmp_path / "prepared"
    stats = prepare_dataset(reviews, metadata, output)
    catalog = json.loads((output / "catalog.json").read_text("utf-8"))
    assert stats["items_missing_metadata"] == 2
    assert catalog[0]["title"] == "A & B"
    assert catalog[1]["title"] == "Beauty product B"
    assert catalog[1]["text"] == "Title: Beauty product B"
    assert stats["training_interactions"] == 3


def test_subsampling_is_reproducible_and_catalog_comes_from_subset(tmp_path):
    raw_reviews, raw_metadata = create_demo_data(tmp_path / "raw")
    first = prepare_dataset(raw_reviews, raw_metadata, tmp_path / "one", max_users=3, seed=7)
    second = prepare_dataset(raw_reviews, raw_metadata, tmp_path / "two", max_users=3, seed=7)
    assert first == second
    assert first["eligible_users_before_subsampling"] == 32
    assert first["users"] == 3
    assert first["training_interactions"] == 18
    assert first["examples"]["train"] == 15
    for filename in ["catalog.json", "train.jsonl", "validation.jsonl", "test.jsonl"]:
        assert (tmp_path / "one" / filename).read_bytes() == (
            tmp_path / "two" / filename
        ).read_bytes()


def test_demo_has_nonempty_warm_splits_and_unique_item_histories(tmp_path):
    reviews, metadata = create_demo_data(tmp_path / "raw")
    output = tmp_path / "prepared"
    stats = prepare_dataset(reviews, metadata, output)
    assert stats["items"] == 16
    assert stats["examples"] == {"train": 160, "validation": 32, "test": 32}
    assert stats["cold_target_examples_dropped"] == {"validation": 0, "test": 0}
    for split in ["train", "validation", "test"]:
        for row in load_examples(output / f"{split}.jsonl"):
            assert row["target"] not in row["history"]
            assert row["target"] not in row["seen_items"]
            assert len(set(row["history"])) == len(row["history"])


def test_seen_items_preserve_old_products_outside_model_context(tmp_path):
    reviews, metadata = make_inputs(
        tmp_path,
        {
            "alice": list(enumerate(["A", "B", "C", "D", "E", "F", "G", "H"])),
            "bob": list(enumerate(["G", "H", "A", "B", "C", "D", "E", "F"])),
        },
    )
    output = tmp_path / "prepared"
    prepare_dataset(reviews, metadata, output, max_history=2)
    mapping = json.loads((output / "item_mapping.json").read_text("utf-8"))
    train_example = load_examples(output / "train.jsonl")[4]
    validation_example = load_examples(output / "validation.jsonl")[0]
    test_example = load_examples(output / "test.jsonl")[0]

    assert train_example["history"] == [mapping["D"], mapping["E"]]
    assert train_example["seen_items"] == [mapping[item] for item in "ABCDE"]
    assert validation_example["history"] == [mapping["E"], mapping["F"]]
    assert validation_example["seen_items"] == [mapping[item] for item in "ABCDEF"]
    assert test_example["history"] == [mapping["F"], mapping["G"]]
    assert test_example["seen_items"] == [mapping[item] for item in "ABCDEFG"]
    for example in [train_example, validation_example, test_example]:
        assert mapping["A"] not in example["history"]
        assert mapping["A"] in example["seen_items"]
        assert example["target"] not in example["seen_items"]


def test_literal_metadata_parser_never_executes_code(tmp_path):
    marker = tmp_path / "should_not_exist"
    source = tmp_path / "malicious.jsonl"
    source.write_text(f"__import__('pathlib').Path({str(marker)!r}).touch()\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        list(read_records(source))
    assert not marker.exists()


def test_download_rejects_html_and_cleans_partial_file(tmp_path, monkeypatch):
    response = io.BytesIO(b"<html>Unavailable</html>")
    response.headers = {"Content-Length": "24"}
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="original host may be unavailable"):
        download_beauty(tmp_path)
    assert not (tmp_path / "reviews_Beauty_5.json.gz").exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_records_hash_and_reuses_verified_cache(tmp_path, monkeypatch):
    calls = []
    payload = gzip.compress(b'{"asin": "A"}\n')

    def fake_urlopen(request, **kwargs):
        calls.append(request.full_url)
        response = io.BytesIO(payload)
        response.headers = {"Content-Length": str(len(payload))}
        return response

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    files = download_beauty(tmp_path)
    assert len(calls) == 2
    assert all("snap.stanford.edu/data/amazon/productGraph/categoryFiles/" in url for url in calls)
    manifest = json.loads((tmp_path / "download_manifest.json").read_text("utf-8"))
    assert manifest["files"]["reviews"]["bytes"] == len(payload)
    assert len(manifest["files"]["metadata"]["sha256"]) == 64
    assert download_beauty(tmp_path) == files
    assert len(calls) == 2
    files["reviews"].write_bytes(gzip.compress(b"different data"))
    with pytest.raises(ValueError, match="Cached file changed"):
        download_beauty(tmp_path)


@pytest.mark.parametrize(
    "options",
    [
        {"max_history": 0},
        {"min_interactions": 2},
        {"max_users": 0},
    ],
)
def test_invalid_configuration_fails_early(tmp_path, options):
    with pytest.raises(ValueError):
        prepare_dataset(tmp_path / "missing", tmp_path / "missing", tmp_path / "out", **options)
