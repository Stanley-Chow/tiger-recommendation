"""A tiny offline M5 run that exercises data, both learned models, and inference."""

import csv
import gzip
import json
from dataclasses import asdict
from pathlib import Path

from beautyrec.amazon2023 import TemporalCohortConfig, prepare_temporal_cohort
from beautyrec.benchmark import _atomic_json
from beautyrec.m5_benchmark import run_m5
from beautyrec.m5_features import build_m5_features


def create_demo_source(directory: Path):
    """Create synthetic reviews with warm, new, tied-time, and unavailable targets."""
    directory.mkdir(parents=True, exist_ok=True)
    config = TemporalCohortConfig(max_users=8, max_warm_items=8, max_new_items=2)
    rows = []
    for user in range(8):
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
        target = f"W{(user + 6) % 8}" if user % 2 == 0 else "N1"
        rows.append([f"user{user}", target, 5, config.test_start_ms + 20000])
    rows += [["observer", f"N{i}", 5, config.train_end_ms + 1000] for i in range(2)]
    rows.append(["user0", "UNAVAILABLE", 5, config.test_start_ms + 10000])
    rows[1][-1] = rows[0][-1]
    with gzip.open(
        directory / "Beauty_and_Personal_Care.csv.gz", "wt", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
        writer.writerows(rows)
    with gzip.open(
        directory / "meta_Beauty_and_Personal_Care.jsonl.gz", "wt", encoding="utf-8"
    ) as file:
        for asin in [*[f"W{i}" for i in range(8)], "N0", "N1"]:
            file.write(
                json.dumps(
                    {
                        "parent_asin": asin,
                        "title": f"Gentle skincare {asin[-1]}",
                        "store": "Example",
                        "categories": ["Beauty", "Skincare"],
                        "description": ["Hydrating cleanser"],
                    }
                )
                + "\n"
            )
    return config


def run_demo(directory: Path, config_path: Path):
    directory = Path(directory)
    raw, processed, features = (directory / name for name in ("raw", "processed", "features"))
    if not (processed / "preprocessing_complete.json").exists():
        data_config = create_demo_source(raw)
        prepare_temporal_cohort(raw, processed, data_config)
    else:
        saved = json.loads((processed / "statistics.json").read_text("utf-8"))
        data_config = TemporalCohortConfig(**saved["config"])
    embedding = build_m5_features(processed, features, offline=True)
    config = json.loads(Path(config_path).read_text("utf-8"))
    config.update(diagnostic_only=True, seeds=[42], data=asdict(data_config))
    config["training"].update(epochs=1, patience=1, batch_size=8, validation_examples=8)
    config["tokenizer"]["model"].update(
        input_dim=embedding["embedding_dim"], hidden_dims=[16], latent_dim=4, codebook_size=2
    )
    config["tokenizer"]["training"].update(
        epochs=1, warmup_epochs=1, patience=1, batch_size=8, init_samples=8
    )
    config["models"]["generative"].update(
        d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32
    )
    config["models"]["sasrec"].update(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    config["evaluation"].update(
        bootstrap_resamples=20,
        request_latency_requests=2,
        generative_batch_size=4,
        sasrec_batch_size=4,
        content_batch_size=4,
    )
    _atomic_json(directory / "demo_config.json", config)
    return run_m5(processed, features, directory / "run", config, "cpu")
