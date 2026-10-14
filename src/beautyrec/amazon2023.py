"""Build a bounded Amazon 2023 cohort using global time cutoffs.

Dataset identity: Beauty_and_Personal_Care, official deduplicated 0-core IDs.
The 2014 preparation path remains unchanged. All cohort choices use training
events or product availability before the test window, never test outcomes.
"""

from __future__ import annotations

import concurrent.futures
import csv
import gzip
import hashlib
import heapq
import json
import sqlite3
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path

import numpy as np

from beautyrec.data import _sha256, _validate_gzip, _write_examples, _write_json

DATASET_PAGE = "https://amazon-reviews-2023.github.io/"
SOURCES = {
    "interactions": (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/"
        "0core/rating_only/Beauty_and_Personal_Care.csv.gz"
    ),
    "metadata": (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
        "meta_categories/meta_Beauty_and_Personal_Care.jsonl.gz"
    ),
}


@dataclass(frozen=True)
class TemporalCohortConfig:
    """Millisecond UTC cutoffs and bounded, training-defined cohort sizes."""

    train_end_ms: int = 1640995200000  # 2022-01-01 00:00:00 UTC
    test_start_ms: int = 1672531200000  # 2023-01-01 00:00:00 UTC
    test_end_ms: int = 1696118400000  # 2023-10-01; source ends September 2023
    max_users: int = 50000
    max_warm_items: int = 40000
    max_new_items: int = 20000
    min_train_interactions: int = 5
    max_history: int = 20
    seed: int = 2026

    def __post_init__(self):
        if not 0 < self.train_end_ms < self.test_start_ms < self.test_end_ms:
            raise ValueError("Require increasing positive global time cutoffs.")
        for name in (
            "max_users",
            "max_warm_items",
            "max_new_items",
            "min_train_interactions",
            "max_history",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.min_train_interactions < 2:
            raise ValueError("Need at least two training interactions per selected user.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32).")


def stable_priority(identifier: str, seed: int) -> str:
    """Hash sampling is invariant to input order and Python's randomized hash."""
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


def download_amazon2023(raw_dir: Path) -> dict[str, Path]:
    """Download both official gzip files with bounded retries and resumable parts.

    A completed file is accepted only after full gzip CRC and SHA-256 checks.
    SHA-256 records local provenance; the publisher does not supply these hashes.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "download_manifest.json"
    previous = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    def fetch(name_url):
        name, url = name_url
        destination = raw_dir / url.rsplit("/", 1)[1]
        if not destination.exists():
            part = destination.with_suffix(destination.suffix + ".part")
            for attempt in range(3):
                try:
                    offset = part.stat().st_size if part.exists() else 0
                    headers = {"User-Agent": "beautyrec/0.1", "Accept-Encoding": "identity"}
                    if offset:
                        headers["Range"] = f"bytes={offset}-"
                    with urllib.request.urlopen(
                        urllib.request.Request(url, headers=headers), timeout=60
                    ) as response:
                        resumed = offset > 0 and response.status == 206
                        if resumed and not response.headers.get("Content-Range", "").startswith(
                            f"bytes {offset}-"
                        ):
                            raise ValueError("Unexpected partial-response byte range.")
                        base = offset if resumed else 0
                        expected = response.headers.get("Content-Length")
                        with part.open("ab" if resumed else "wb") as target:
                            started = time.monotonic()
                            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                                target.write(chunk)
                                if target.tell() > 3 * 1024**3:
                                    raise ValueError(
                                        "Dataset download exceeds the 3 GiB file bound."
                                    )
                                if time.monotonic() - started > 20:
                                    print(
                                        f"{name}: {target.tell() / 1024**2:.0f} MiB downloaded",
                                        flush=True,
                                    )
                                    started = time.monotonic()
                    if expected and part.stat().st_size != base + int(expected):
                        raise ValueError("Incomplete dataset response.")
                    _validate_gzip(part)
                    part.replace(destination)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        _validate_gzip(destination)
        digest = _sha256(destination)
        cached = previous.get("files", {}).get(name, {})
        if cached.get("sha256") and digest != cached["sha256"]:
            raise ValueError(f"Completed source file changed: {destination}")
        print(f"{name}: complete, {destination.stat().st_size:,} bytes", flush=True)
        return (
            name,
            destination,
            {
                "url": url,
                "filename": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": digest,
            },
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fetch, SOURCES.items()))
    _write_json(
        manifest_path,
        {"dataset_page": DATASET_PAGE, "files": {name: record for name, _, record in records}},
    )
    return {name: path for name, path, _ in records}


def read_interactions(path: Path):
    """Stream official parent-ASIN interactions, checking the millisecond schema."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source)
        if rows.fieldnames != ["user_id", "parent_asin", "rating", "timestamp"]:
            raise ValueError("Unexpected Amazon 2023 pure-ID schema.")
        for number, row in enumerate(rows, start=2):
            user, asin = row["user_id"], row["parent_asin"]
            try:
                timestamp = int(row["timestamp"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid timestamp at CSV line {number}") from error
            if not user or not asin or not 100_000_000_000 <= timestamp < 4102444800000:
                raise ValueError(f"Invalid interaction at CSV line {number}")
            yield user, asin, timestamp


def _cached_stage(directory, name, fingerprint, build):
    """Reuse only a complete, hash-matching preparation stage."""
    directory.mkdir(parents=True, exist_ok=True)
    path, marker = directory / f"{name}.json", directory / f"{name}_complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text("utf-8"))
        if saved != {"fingerprint": fingerprint, "sha256": _sha256(path)}:
            raise ValueError(f"Preparation cache changed: {name}")
        return json.loads(path.read_text("utf-8"))
    result = build()
    _write_json(path, result)
    _write_json(marker, {"fingerprint": fingerprint, "sha256": _sha256(path)})
    return result


def _source_census(interactions, config):
    counts, first_seen, rows = Counter(), {}, 0
    for _, asin, timestamp in read_interactions(interactions):
        rows += 1
        if timestamp < config.test_start_ms:
            first_seen[asin] = min(first_seen.get(asin, timestamp), timestamp)
        if timestamp < config.train_end_ms:
            counts[asin] += 1
        if rows % 2_000_000 == 0:
            print(f"Source census: {rows:,} interactions", flush=True)
    return {"rows": rows, "train_counts": counts, "first_seen": first_seen}


def _select_users(interactions, warm_items, config, database):
    """Count training activity on disk rather than retaining millions of users."""
    with sqlite3.connect(database) as connection:
        # This database is a rebuildable cache, protected by a separate completion marker.
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("DROP TABLE IF EXISTS user_counts")
        connection.execute(
            "CREATE TABLE user_counts(user TEXT PRIMARY KEY, n INTEGER) WITHOUT ROWID"
        )
        batch = Counter()
        for number, (user, asin, timestamp) in enumerate(read_interactions(interactions), start=1):
            if timestamp < config.train_end_ms and asin in warm_items:
                batch[user] += 1
            if number % 100_000 == 0:
                connection.executemany(
                    "INSERT INTO user_counts VALUES (?,?) "
                    "ON CONFLICT(user) DO UPDATE SET n=n+excluded.n",
                    batch.items(),
                )
                connection.commit()
                batch.clear()
            if number % 2_000_000 == 0:
                print(f"Training-only user counts: {number:,} source rows", flush=True)
        connection.executemany(
            "INSERT INTO user_counts VALUES (?,?) ON CONFLICT(user) DO UPDATE SET n=n+excluded.n",
            batch.items(),
        )
        connection.commit()
        eligible = connection.execute(
            "SELECT user FROM user_counts WHERE n>=?", (config.min_train_interactions,)
        )
        users = heapq.nsmallest(
            config.max_users,
            (row[0] for row in eligible),
            key=lambda user: (stable_priority(user, config.seed), user),
        )
        count = connection.execute(
            "SELECT COUNT(*) FROM user_counts WHERE n>=?", (config.min_train_interactions,)
        ).fetchone()[0]
    if not users:
        raise ValueError("No users meet the training-only cohort requirement.")
    return {"eligible_users": count, "selected_users": sorted(users)}


def _selected_sequences(interactions, users, config):
    selected = set(users)
    earliest = defaultdict(dict)
    duplicates = 0
    for number, (user, asin, timestamp) in enumerate(read_interactions(interactions), start=1):
        if user in selected and timestamp < config.test_end_ms:
            if asin in earliest[user]:
                duplicates += 1
            earliest[user][asin] = min(earliest[user].get(asin, timestamp), timestamp)
        if number % 2_000_000 == 0:
            print(f"Collecting complete cohort histories: {number:,} source rows", flush=True)
    return {
        "duplicates_removed": duplicates,
        "users": [
            {"user_id": user, "events": sorted((t, asin) for asin, t in earliest[user].items())}
            for user in users
        ],
    }


def _text_parts(value):
    if isinstance(value, str):
        return [" ".join(value.split())] if value.strip() else []
    if isinstance(value, list):
        return [part for entry in value for part in _text_parts(entry)]
    return []


def _metadata_catalog(metadata_path, warm_asins, new_asins, first_seen):
    wanted = set(warm_asins) | set(new_asins)
    metadata = {}
    opener = gzip.open if str(metadata_path).endswith(".gz") else open
    with opener(metadata_path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            asin = record.get("parent_asin")
            if asin in wanted:
                title = _text_parts(record.get("title"))
                brand = _text_parts(record.get("store"))
                categories = _text_parts(record.get("categories"))
                description = _text_parts(record.get("features")) + _text_parts(
                    record.get("description")
                )
                metadata[asin] = {
                    "title": " ".join(title),
                    "brand": " ".join(brand),
                    "categories": categories,
                    "description": " ".join(description),
                }
    catalog = []
    for index, asin in enumerate([*warm_asins, *new_asins]):
        fields = metadata.get(asin, {"title": "", "brand": "", "categories": [], "description": ""})
        text = ". ".join(
            _text_parts(fields["title"])
            + _text_parts(fields["brand"])
            + fields["categories"]
            + _text_parts(fields["description"])
        )
        catalog.append(
            {
                "item_id": index,
                "asin": asin,
                **fields,
                "text": text or "Beauty product with unavailable metadata.",
                "metadata_missing": asin not in metadata,
                "available_from_ms": first_seen[asin],
                "training_item": index < len(warm_asins),
            }
        )
    return catalog


def _examples_from_sequences(sequences, warm_asins, catalog_asins, config):
    warm = {asin: index for index, asin in enumerate(warm_asins)}
    available = {asin: index for index, asin in enumerate(catalog_asins)}
    rows = {"train": [], "validation": [], "test": [], "mixed_test": []}
    counts = {name: Counter() for name in ("validation", "test")}
    popularity = np.zeros(len(warm), dtype=np.int64)
    for sequence in sequences:
        user, events = sequence["user_id"], sequence["events"]
        train = [(t, warm[asin]) for t, asin in events if t < config.train_end_ms and asin in warm]
        if len(train) < config.min_train_interactions:
            raise ValueError("Selected user no longer meets the deduplicated training requirement.")
        seen = []
        previous_time = None
        for timestamp, group in groupby(train, key=lambda event: event[0]):
            group = list(group)
            for _, target in group:
                popularity[target] += 1
                if seen:
                    rows["train"].append(
                        {
                            "user_id": user,
                            "history": seen[-config.max_history :],
                            "seen_items": list(seen),
                            "target": target,
                            "timestamp": timestamp,
                            "history_end_timestamp": previous_time,
                        }
                    )
            seen.extend(target for _, target in group)
            previous_time = timestamp
        for split, start, end in (
            ("validation", config.train_end_ms, config.test_start_ms),
            ("test", config.test_start_ms, config.test_end_ms),
        ):
            future = next(((t, asin) for t, asin in events if start <= t < end), None)
            if future is None:
                counts[split]["no_future_interaction"] += 1
                continue
            counts[split]["users_with_first_future_target"] += 1
            past = [(t, asin) for t, asin in events if t < start]
            warm_seen = [warm[asin] for _, asin in past if asin in warm]
            if not warm_seen:
                counts[split]["no_warm_history"] += 1
                continue
            timestamp, asin = future
            base = {
                "user_id": user,
                "history": warm_seen[-config.max_history :],
                "seen_items": warm_seen,
                "timestamp": timestamp,
                "request_timestamp": start,
                "history_end_timestamp": max(t for t, _ in past),
            }
            if asin in warm:
                rows[split].append({**base, "target": warm[asin]})
                counts[split]["warm_eligible"] += 1
            else:
                counts[split]["target_outside_warm_catalog"] += 1
            if split == "test":
                if asin not in available:
                    counts[split]["target_outside_available_test_catalog"] += 1
                    continue
                all_seen = [available[item] for _, item in past if item in available]
                rows["mixed_test"].append(
                    {
                        **base,
                        "history": all_seen[-config.max_history :],
                        "seen_items": all_seen,
                        "warm_history": warm_seen[-config.max_history :],
                        "warm_seen_items": warm_seen,
                        "target": available[asin],
                        "target_type": "warm" if asin in warm else "new",
                        "source_history_events": len(past),
                        "available_history_events": len(all_seen),
                    }
                )
                counts[split]["available_catalog_eligible"] += 1
                counts[split]["new_eligible" if asin not in warm else "warm_mixed_eligible"] += 1
    if not rows["train"] or not rows["validation"] or not rows["test"]:
        raise ValueError("Need nonempty training, warm validation and warm test cohorts.")
    return rows, popularity, counts


def prepare_temporal_cohort(raw_dir: Path, output_dir: Path, config: TemporalCohortConfig):
    """Make a training-defined warm catalog and a separately available dynamic catalog."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    interactions = raw_dir / "Beauty_and_Personal_Care.csv.gz"
    metadata = raw_dir / "meta_Beauty_and_Personal_Care.jsonl.gz"
    sources = {
        name: {"filename": path.name, "sha256": _sha256(path)}
        for name, path in (("interactions", interactions), ("metadata", metadata))
    }
    fingerprint = hashlib.sha256(
        json.dumps({"config": asdict(config), "sources": sources}, sort_keys=True).encode()
    ).hexdigest()
    complete = output_dir / "preprocessing_complete.json"
    if complete.exists():
        marker = json.loads(complete.read_text("utf-8"))
        if marker["fingerprint"] != fingerprint or any(
            _sha256(output_dir / name) != digest for name, digest in marker["files"].items()
        ):
            raise ValueError("Completed preprocessing changed; use a new directory.")
        return json.loads((output_dir / "statistics.json").read_text("utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Incomplete output directory; inspect it and use a fresh destination.")
    cache = raw_dir / ".cohort_cache" / fingerprint[:20]
    census = _cached_stage(
        cache, "census", fingerprint, lambda: _source_census(interactions, config)
    )
    warm_candidates = [
        asin
        for asin, _ in heapq.nsmallest(
            config.max_warm_items,
            census["train_counts"].items(),
            key=lambda pair: (-pair[1], pair[0]),
        )
    ]
    warm_set = set(warm_candidates)
    selected = _cached_stage(
        cache,
        "selected_users",
        fingerprint,
        lambda: _select_users(interactions, warm_set, config, cache / "user_counts.sqlite"),
    )
    sequences = _cached_stage(
        cache,
        "cohort_sequences",
        fingerprint,
        lambda: _selected_sequences(interactions, selected["selected_users"], config),
    )
    observed = {
        asin
        for row in sequences["users"]
        for t, asin in row["events"]
        if t < config.train_end_ms and asin in warm_set
    }
    warm_asins = sorted(observed)
    new_asins = sorted(
        heapq.nsmallest(
            config.max_new_items,
            (
                asin
                for asin, t in census["first_seen"].items()
                if config.train_end_ms <= t < config.test_start_ms
            ),
            key=lambda asin: (stable_priority(asin, config.seed), asin),
        )
    )
    print(
        f"Building metadata catalog: {len(warm_asins):,} warm + {len(new_asins):,} new items",
        flush=True,
    )
    catalog = _metadata_catalog(metadata, warm_asins, new_asins, census["first_seen"])
    examples, popularity, counts = _examples_from_sequences(
        sequences["users"], warm_asins, [row["asin"] for row in catalog], config
    )
    statistics = {
        "dataset": "Amazon Reviews 2023 Beauty_and_Personal_Care deduplicated 0-core",
        "split_protocol": (
            "global UTC windows; frozen histories; first future target before eligibility"
        ),
        "config": asdict(config),
        "source_files": sources,
        "source_rows": census["rows"],
        "users": len(selected["selected_users"]),
        "eligible_users_before_subsampling": selected["eligible_users"],
        "selected_user_ids": selected["selected_users"],
        "items": len(warm_asins),
        "new_items": len(new_asins),
        "available_test_items": len(catalog),
        "training_interactions": int(popularity.sum()),
        "max_history": config.max_history,
        "examples": {name: len(rows) for name, rows in examples.items()},
        "eligibility": {name: dict(value) for name, value in counts.items()},
        "cohort_source_interactions": sum(len(row["events"]) for row in sequences["users"]),
        "duplicate_cohort_user_item_events_removed": sequences["duplicates_removed"],
        "items_missing_metadata": sum(row["metadata_missing"] for row in catalog),
        "static_metadata_snapshot_limitation": (
            "Released 2023 text; historical availability is unverified."
        ),
        "availability_proxy": "first observed review before request cutoff",
        "preprocessing_fingerprint": fingerprint,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "catalog.json", catalog[: len(warm_asins)])
    _write_json(output_dir / "available_catalog.json", catalog)
    _write_json(output_dir / "item_mapping.json", {asin: i for i, asin in enumerate(warm_asins)})
    _write_json(output_dir / "statistics.json", statistics)
    _write_examples(output_dir / "source_cohort.jsonl", sequences["users"])
    for name, rows in examples.items():
        _write_examples(output_dir / f"{name}.jsonl", rows)
    np.save(output_dir / "item_popularity.npy", popularity, allow_pickle=False)
    np.save(
        output_dir / "available_popularity.npy",
        np.pad(popularity, (0, len(new_asins))),
        allow_pickle=False,
    )
    _write_json(
        complete,
        {
            "fingerprint": fingerprint,
            "files": {
                path.name: _sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()
            },
        },
    )
    print(
        json.dumps(
            {
                key: statistics[key]
                for key in ("users", "items", "new_items", "examples", "eligibility")
            },
            indent=2,
        ),
        flush=True,
    )
    return statistics
