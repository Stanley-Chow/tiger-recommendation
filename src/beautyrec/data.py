"""Download and prepare chronological next-item examples from Amazon Beauty.

Only training interactions define the item catalog and popularity counts. A user's
penultimate interaction is the validation target; their final interaction is the
test target. This is a per-user chronological split, not a global time cutoff.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import html
import json
import os
import random
import re
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

DATASET_PAGE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/index_2014.html"
# Use the dataset authors' original HTTP file host. Its HTTPS response can stall
# during large transfers; local checksums and gzip validation detect file changes.
DATASET_BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
DOWNLOAD_FILES = {
    "reviews": "reviews_Beauty_5.json.gz",
    "metadata": "meta_Beauty.json.gz",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_gzip(path: Path) -> None:
    """Check the full gzip stream, including its checksum, in bounded memory."""
    with gzip.open(path, "rb") as source:
        for _ in iter(lambda: source.read(1024 * 1024), b""):
            pass


def download_beauty(raw_dir: Path) -> dict[str, Path]:
    """Download the original 2014 Beauty files, reusing valid cached downloads.

    Downloads use a temporary file and replace the destination only after gzip
    integrity checks. ``download_manifest.json`` records size, SHA-256 and origin;
    its hash detects later changes but is not a publisher-provided checksum.
    A server error fails clearly instead of silently switching dataset versions.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "download_manifest.json"
    previous = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    manifest: dict[str, Any] = {"dataset_page": DATASET_PAGE, "files": {}}
    result = {}

    for name, filename in DOWNLOAD_FILES.items():
        destination = raw_dir / filename
        url = DATASET_BASE_URL + filename
        print(
            f"Downloading {filename}" if not destination.exists() else f"Using {filename}",
            flush=True,
        )
        cached = previous.get("files", {}).get(name, {})
        if destination.exists():
            _validate_gzip(destination)
            digest = _sha256(destination)
            if cached.get("sha256") and cached["sha256"] != digest:
                raise ValueError(
                    f"Cached file changed: {destination}. Remove it and download again."
                )
        else:
            temporary = destination.with_suffix(destination.suffix + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "beautyrec/0.1"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected_size = response.headers.get("Content-Length")
                    with temporary.open("wb") as target:
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            target.write(chunk)
                if expected_size and temporary.stat().st_size != int(expected_size):
                    raise ValueError("The server returned an incomplete download.")
                _validate_gzip(temporary)
                digest = _sha256(temporary)
                os.replace(temporary, destination)
            except Exception as error:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not download {url}. The original host may be unavailable. "
                    f"Download the 2014 file from {DATASET_PAGE} and place it at "
                    f"{destination}; do not substitute Amazon 2023 All_Beauty."
                ) from error
        result[name] = destination
        manifest["files"][name] = {
            "filename": filename,
            "url": url,
            "bytes": destination.stat().st_size,
            "sha256": digest,
        }

    _write_json(manifest_path, manifest)
    return result


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Read JSON lines or the old dataset's Python dictionary literal format.

    ``ast.literal_eval`` only reads literal data; unlike ``eval`` it cannot run
    code contained in a metadata line. Malformed records fail with a line number
    so corrupted input is not silently omitted.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = ast.literal_eval(line)
                if not isinstance(record, dict):
                    raise ValueError("Expected a dictionary record.")
            except (ValueError, SyntaxError) as error:
                raise ValueError(f"Invalid record in {path}, line {line_number}.") from error
            yield record


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_examples(path: Path, examples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for example in examples:
            target.write(json.dumps(example, ensure_ascii=False) + "\n")


def _clean_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(filter(None, (_clean_text(part) for part in value)))
    if not isinstance(value, str):
        return ""
    text = html.unescape(value)
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def _categories(value: Any) -> list[str]:
    """Flatten category paths, preserving their readable order without repeats."""
    if isinstance(value, str):
        return [cleaned] if (cleaned := _clean_text(value)) else []
    if isinstance(value, list):
        return list(dict.fromkeys(part for child in value for part in _categories(child)))
    return []


def _catalog_record(item_id: int, asin: str, metadata: dict[str, Any]) -> dict[str, Any]:
    title = _clean_text(metadata.get("title")) or f"Beauty product {asin}"
    brand = _clean_text(metadata.get("brand"))
    categories = _categories(metadata.get("categories", []))
    description = _clean_text(metadata.get("description"))
    fields = [f"Title: {title}"]
    if brand:
        fields.append(f"Brand: {brand}")
    if categories:
        fields.append(f"Categories: {'; '.join(categories)}")
    if description:
        fields.append(f"Description: {description}")
    return {
        "item_id": item_id,
        "asin": asin,
        "title": title,
        "brand": brand,
        "categories": categories,
        "description": description,
        "text": ". ".join(fields),
    }


def prepare_dataset(
    review_path: Path,
    metadata_path: Path,
    output_dir: Path,
    max_history: int = 20,
    min_interactions: int = 5,
    max_users: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Prepare deterministic train/validation/test files and a warm item catalog.

    Repeated (user, item) reviews keep the earliest timestamp. After ordering by
    timestamp and ASIN, users with too few distinct items are discarded. Optional
    user subsampling happens before catalog construction, so all saved artifacts
    describe the same subset. Metadata and popularity use training items only.

    Training examples use expanding prefixes, clipped to ``max_history`` items.
    ``seen_items`` retains all earlier warm items for recommendation exclusion,
    including products outside the model's shorter ``history`` context window.
    Test history includes the earlier validation interaction when it is warm.
    Held-out targets absent from the training catalog are counted and omitted;
    a cold item is also omitted from later histories. Metadata-only items never
    enter the catalog. Missing metadata gets explicit fallback product text.
    """
    if max_history < 1:
        raise ValueError("max_history must be at least 1.")
    if min_interactions < 3:
        raise ValueError("min_interactions must be at least 3 for a last-two split.")
    if max_users is not None and max_users < 1:
        raise ValueError("max_users must be positive when provided.")

    earliest: dict[tuple[str, str], int] = {}
    raw_reviews = invalid_reviews = duplicate_reviews = 0
    for record in read_records(review_path):
        raw_reviews += 1
        user, asin = record.get("reviewerID"), record.get("asin")
        timestamp = record.get("unixReviewTime")
        if not isinstance(user, str) or not user or not isinstance(asin, str) or not asin:
            invalid_reviews += 1
            continue
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            invalid_reviews += 1
            continue
        key = (user, asin)
        if key in earliest:
            duplicate_reviews += 1
            earliest[key] = min(earliest[key], timestamp)
        else:
            earliest[key] = timestamp

    events_by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for (user, asin), timestamp in earliest.items():
        events_by_user[user].append((timestamp, asin))
    eligible_users = sorted(
        user for user, events in events_by_user.items() if len(events) >= min_interactions
    )
    if not eligible_users:
        raise ValueError("No users remain. Check the input fields and min_interactions.")
    selected_users = eligible_users
    if max_users is not None and len(eligible_users) > max_users:
        selected_users = sorted(random.Random(seed).sample(eligible_users, max_users))
    sequences = {
        user: [asin for _, asin in sorted(events_by_user[user])] for user in selected_users
    }

    # Every original training event is counted once, including the first event.
    # Counting overlapping prefix examples would overstate frequent early items.
    popularity = Counter(asin for sequence in sequences.values() for asin in sequence[:-2])
    item_to_id = {asin: index for index, asin in enumerate(sorted(popularity))}
    metadata_by_asin = {}
    for record in read_records(metadata_path):
        asin = record.get("asin")
        if isinstance(asin, str) and asin in item_to_id:
            metadata_by_asin[asin] = record
    catalog = [
        _catalog_record(index, asin, metadata_by_asin.get(asin, {}))
        for asin, index in item_to_id.items()
    ]

    examples: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    cold_targets = {"validation": 0, "test": 0}
    cold_history_events = {"validation": 0, "test": 0}
    for user, sequence in sequences.items():
        training_sequence = sequence[:-2]
        for position in range(1, len(training_sequence)):
            seen_items = [item_to_id[asin] for asin in training_sequence[:position]]
            examples["train"].append(
                {
                    "user_id": user,
                    "history": seen_items[-max_history:],
                    "seen_items": seen_items,
                    "target": item_to_id[training_sequence[position]],
                }
            )
        for split, position in (("validation", len(sequence) - 2), ("test", len(sequence) - 1)):
            history = sequence[:position]
            cold_history_events[split] += sum(asin not in item_to_id for asin in history)
            if sequence[position] not in item_to_id:
                cold_targets[split] += 1
                continue
            warm_history = [item_to_id[asin] for asin in history if asin in item_to_id]
            examples[split].append(
                {
                    "user_id": user,
                    "history": warm_history[-max_history:],
                    "seen_items": warm_history,
                    "target": item_to_id[sequence[position]],
                }
            )
    if not examples["train"]:
        raise ValueError("No training examples remain. Use users with at least 4 distinct items.")

    stats = {
        "split_protocol": "per-user chronological leave-last-two-out; warm training catalog",
        "raw_reviews": raw_reviews,
        "invalid_reviews": invalid_reviews,
        "duplicate_user_item_reviews_removed": duplicate_reviews,
        "unique_user_item_interactions": len(earliest),
        "users_before_filtering": len(events_by_user),
        "eligible_users_before_subsampling": len(eligible_users),
        "users_removed_by_min_interactions": len(events_by_user) - len(eligible_users),
        "users": len(selected_users),
        "items": len(catalog),
        "interactions_in_selected_users": sum(map(len, sequences.values())),
        "training_interactions": sum(popularity.values()),
        "examples": {split: len(rows) for split, rows in examples.items()},
        "cold_target_examples_dropped": cold_targets,
        "cold_history_events_removed": cold_history_events,
        "items_missing_metadata": len(catalog) - len(metadata_by_asin),
        "items_missing_title": sum(
            not _clean_text(record.get("title")) for record in metadata_by_asin.values()
        ),
        "max_history": max_history,
        "min_interactions": min_interactions,
        "max_users": max_users,
        "seed": seed,
        "source_files": {
            "reviews": {"name": Path(review_path).name, "sha256": _sha256(Path(review_path))},
            "metadata": {"name": Path(metadata_path).name, "sha256": _sha256(Path(metadata_path))},
        },
        "selected_user_ids": selected_users,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "catalog.json", catalog)
    _write_json(output_dir / "item_mapping.json", item_to_id)
    _write_json(output_dir / "statistics.json", stats)
    for split, rows in examples.items():
        _write_examples(output_dir / f"{split}.jsonl", rows)
    np.save(
        output_dir / "item_popularity.npy",
        np.asarray([popularity[record["asin"]] for record in catalog], dtype=np.int64),
    )
    return stats


def create_demo_data(raw_dir: Path) -> tuple[Path, Path]:
    """Write small synthetic data for a CPU smoke run; this is not Amazon data.

    Thirty-two users each visit eight distinct products. Rotated histories ensure
    all sixteen products appear in training, with thematic metadata suitable for
    exercising text embeddings and hierarchical semantic IDs offline.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    review_path = raw_dir / "demo_reviews.json.gz"
    metadata_path = raw_dir / "demo_metadata.json.gz"
    themes = [
        ("Skincare", "gentle hydrating facial cleanser moisturizer serum"),
        ("Haircare", "nourishing shampoo conditioner hair treatment shine"),
        ("Makeup", "cosmetic lipstick blush foundation soft pigment"),
        ("Fragrance", "fresh floral perfume body mist scent"),
    ]
    with gzip.open(metadata_path, "wt", encoding="utf-8") as target:
        for item in range(16):
            category, description = themes[item // 4]
            target.write(
                json.dumps(
                    {
                        "asin": f"DEMO{item:06d}",
                        "title": f"{category} product {item % 4 + 1}",
                        "brand": f"Demo {category}",
                        "categories": [["Beauty", category]],
                        "description": description,
                    }
                )
                + "\n"
            )
    with gzip.open(review_path, "wt", encoding="utf-8") as target:
        for user in range(32):
            # Each theme is followed by related products, then a second theme.
            start = user % 16
            for position in range(8):
                item = (start + position) % 16
                target.write(
                    json.dumps(
                        {
                            "reviewerID": f"demo_user_{user:03d}",
                            "asin": f"DEMO{item:06d}",
                            "unixReviewTime": 1_400_000_000 + user * 100 + position,
                            "overall": 5.0,
                        }
                    )
                    + "\n"
                )
    return review_path, metadata_path
