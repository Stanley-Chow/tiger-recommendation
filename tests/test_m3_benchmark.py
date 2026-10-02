"""Check M3 phase ordering, common capacity and immutable resume without training."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from beautyrec import experiment_training, m3_benchmark
from beautyrec.data import create_demo_data, prepare_dataset
from beautyrec.embeddings import build_embeddings
from beautyrec.experiment_training import ExperimentTrainingConfig, train_experiment
from beautyrec.semantic_ids import SemanticCodebook, build_semantic_ids, load_semantic_ids
from beautyrec.training import dataset_fingerprint


@pytest.fixture
def project(tmp_path):
    reviews, metadata = create_demo_data(tmp_path / "raw")
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    prepare_dataset(reviews, metadata, processed, max_history=5)
    build_embeddings(processed, artifacts, backend="tfidf")
    # The runner checks the frozen vector dimension. No text model download is needed.
    vectors = np.random.default_rng(3).normal(size=(16, 128)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(artifacts / "embeddings.npy", vectors)
    embedding_config = json.loads((artifacts / "embedding_config.json").read_text("utf-8"))
    embedding_config["embedding_dim"] = 128
    (artifacts / "embedding_config.json").write_text(json.dumps(embedding_config), "utf-8")
    build_semantic_ids(processed, artifacts, branching_factor=16, depth=3)
    config = json.loads(
        (Path(__file__).parents[1] / "configs" / "m3_smoke.json").read_text("utf-8")
    )
    config["tokenizer"]["model"]["input_dim"] = 128
    config["seeds"] = [42, 43, 44]
    config["evaluation"]["bootstrap_resamples"] = 20
    return SimpleNamespace(
        processed=processed, artifacts=artifacts, run=tmp_path / "run", config=config
    )


@pytest.fixture
def fake_runs(project, monkeypatch):
    events, trained_books = [], []

    def tokenizer(processed, artifacts, directory, model_config, training_config):
        assert (project.run / "protocol.json").exists()
        assert not any(event[0] == "recommender" for event in events)
        seed = training_config["seed"]
        events.append(("tokenizer", seed))
        source = load_semantic_ids(artifacts)
        group_size = {42: 2, 43: 4, 44: 8}[seed]
        book = SemanticCodebook(
            item_codes=[
                [3 + item // group_size, 19, 35, 51 + item % group_size] for item in range(16)
            ],
            vocab_size=51 + group_size,
            code_length=4,
            catalog_sha256=source.catalog_sha256,
            depth=3,
            branching_factor=16,
            diagnostics={"method": "rqvae", "final_fit_items": 16},
        )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "semantic_ids.json").write_text(json.dumps(book.to_dict()), "utf-8")
        (directory / "rqvae.pt").write_bytes(b"fake metadata weights")
        return {
            "semantic_ids": str(directory / "semantic_ids.json"),
            "selected_quantized_epochs": 1,
        }

    def recommender(processed, artifacts, directory, method, training, model_options=None):
        assert len([event for event in events if event[0] == "tokenizer"]) == 3
        assert (project.run / "tokenizer_manifest.json").exists()
        assert not any(event[0] == "retrieval" for event in events)
        book = load_semantic_ids(artifacts)
        trained_books.append((training.seed, directory.name, book.vocab_size))
        assert training.initial_checkpoint is None and training.optimizer == "adamw"
        assert training.validation_seed == 2026
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "best.pt"
        checkpoint.write_text(json.dumps({"arm": directory.name, "seed": training.seed}), "utf-8")
        events.append(("recommender", training.seed, directory.name))
        return {
            "checkpoint": str(checkpoint.resolve()),
            "parameter_count": 137 + 2 * book.vocab_size,
            "epochs_completed": training.epochs,
            "guard_passed": False,
        }

    def load(path, artifacts, device):
        record = json.loads(path.read_text("utf-8"))
        return (
            SimpleNamespace(**record),
            load_semantic_ids(artifacts),
            {
                "dataset_fingerprint": dataset_fingerprint(project.processed),
            },
        )

    def retrieve(model, examples, book, **options):
        selected = json.loads((project.run / "selection_manifest.json").read_text("utf-8"))
        assert sum(len(arms) for arms in selected["selected"].values()) == 12
        assert len([event for event in events if event[0] == "recommender"]) == 12
        events.append(("retrieval", model.seed, model.arm))
        rows = [
            [item for item in range(16) if item not in example["seen_items"]][: options["top_k"]]
            for example in examples
        ]
        return rows, {"histories_per_second": 100, "batch_size": options["batch_size"]}

    monkeypatch.setattr(m3_benchmark, "train_rqvae_tokenizer", tokenizer)
    monkeypatch.setattr(experiment_training, "train_experiment", recommender)
    monkeypatch.setattr(m3_benchmark, "load_experiment_checkpoint", load)
    monkeypatch.setattr(m3_benchmark, "retrieve_rankings", retrieve)
    monkeypatch.setattr(
        m3_benchmark,
        "measure_request_latency",
        lambda *args, **kwargs: {
            "mean_ms": 2,
            "p95_ms": 3,
            "requests": kwargs["requests"],
        },
    )
    return SimpleNamespace(events=events, books=trained_books)


def run(project):
    return m3_benchmark.run_m3(project.processed, project.artifacts, project.run, project.config)


def test_all_tokenizers_then_all_recommenders_then_test_with_common_vocabulary(project, fake_runs):
    result = run(project)
    phases = [event[0] for event in fake_runs.events]
    assert phases[:3] == ["tokenizer"] * 3
    assert phases[3:15] == ["recommender"] * 12
    assert all(phase == "retrieval" for phase in phases[15:])
    assert {capacity for _, _, capacity in fake_runs.books} == {59}
    assert result["common_vocab_size"] == 59
    assert result["m3_complete"] is False and result["diagnostic_only"] is True
    assert len(result["seed_results"]) == 3
    for record in result["seed_results"]:
        assert record["test"]["methods"] == ["popularity", *m3_benchmark.ARMS]
        assert "rqvae_minus_rqvae_permuted" in record["comparisons"]
        assert "hkm_minus_hkm_permuted" in record["comparisons"]
        assert "rqvae_minus_hkm" in record["comparisons"]
        assert all(not training["guard_passed"] for training in record["training"].values())
    assert (project.run / "completed.json").exists()
    for directory in (project.run / "arms").glob("seed_*/*"):
        assert not (directory / "embeddings.npy").exists()
        assert (directory / "tokenizer_diagnostics.json").exists()
        assert (directory / "frequency_weighted_diagnostics.json").exists()
        assert json.loads((directory / "embedding_config.json").read_text("utf-8")) == json.loads(
            (project.artifacts / "embedding_config.json").read_text("utf-8")
        )
    manifest = json.loads((project.run / "tokenizer_manifest.json").read_text("utf-8"))
    assert all(
        "embedding_config.json" in record["files"]
        for seed_records in manifest["arms"].values()
        for record in seed_records.values()
    )


def test_finalized_arm_supports_real_one_epoch_training_without_embedding_matrix(
    project, fake_runs
):
    run(project)
    artifacts = project.run / "arms" / "seed_42" / "hkm"
    directory = project.run.parent / "real_training"
    assert not (artifacts / "embeddings.npy").exists()
    # This imported function remains real while fake_runs patches the module entry point.
    result = train_experiment(
        project.processed,
        artifacts,
        directory,
        "generative",
        ExperimentTrainingConfig(
            epochs=1,
            batch_size=8,
            max_history=5,
            patience=1,
            device="cpu",
            torch_threads=1,
            max_train_examples=8,
            validation_examples=2,
            evaluation_batch_size=2,
            beam_size=200,
        ),
        model_options={
            "d_model": 8,
            "nhead": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.1,
        },
    )
    assert result["epochs_completed"] == result["best_epoch"] == 1
    assert Path(result["checkpoint"]).is_file()
    saved = json.loads((directory / "config.json").read_text("utf-8"))
    assert saved["embedding_config"] == json.loads(
        (project.artifacts / "embedding_config.json").read_text("utf-8")
    )


def test_permutation_preserves_full_code_set_and_book_capacity_for_each_seed(project, fake_runs):
    run(project)
    for seed in [42, 43, 44]:
        raw_rq = project.run / "tokenizers" / f"seed_{seed}" / "rqvae"
        assert load_semantic_ids(raw_rq).vocab_size == 51 + {42: 2, 43: 4, 44: 8}[seed]
        for semantic, randomized in [("hkm", "hkm_permuted"), ("rqvae", "rqvae_permuted")]:
            base = project.run / "arms" / f"seed_{seed}"
            source, control = (
                load_semantic_ids(base / semantic),
                load_semantic_ids(base / randomized),
            )
            assert source.vocab_size == control.vocab_size == 59
            assert sorted(source.item_codes) == sorted(control.item_codes)
            mapping = json.loads(
                (base / randomized / "permutation_manifest.json").read_text("utf-8")
            )
            assert mapping["seed"] == 10000 + seed
            assert len(mapping["item_to_source_item"]) == 16


def test_completed_run_and_seed_outputs_resume_without_model_calls(project, fake_runs):
    first = run(project)
    events = list(fake_runs.events)
    assert run(project) == first
    assert fake_runs.events == events
    (project.run / "completed.json").unlink()
    (project.run / "results.json").unlink()
    assert run(project) == first
    assert fake_runs.events == events


def test_changed_configuration_or_frozen_codes_cannot_be_reused(project, fake_runs):
    run(project)
    changed = deepcopy(project.config)
    changed["training"]["learning_rate"] = 0.002
    with pytest.raises(ValueError, match="fingerprints changed"):
        m3_benchmark.run_m3(project.processed, project.artifacts, project.run, changed)
    codes = project.run / "arms" / "seed_42" / "hkm" / "semantic_ids.json"
    codes.write_text(codes.read_text("utf-8") + "\n", "utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        run(project)


def test_changed_source_fingerprint_refuses_resume(project, fake_runs, monkeypatch):
    run(project)
    original = m3_benchmark._protocol

    def changed(*args):
        protocol = original(*args)
        protocol["source_python"]["pretend_edit.py"] = "changed"
        return protocol

    monkeypatch.setattr(m3_benchmark, "_protocol", changed)
    with pytest.raises(ValueError, match="fingerprints changed"):
        run(project)


def test_changed_tokenizer_manifest_refuses_partial_run_resume(project, fake_runs):
    run(project)
    (project.run / "completed.json").unlink()
    (project.run / "results.json").unlink()
    path = project.run / "tokenizer_manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    manifest["common_vocab_size"] += 1
    path.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(ValueError, match="manifest fingerprint changed"):
        run(project)


@pytest.mark.parametrize(
    "kind", ["dimension", "beam", "nan_lr", "duplicate_seed", "unhashable_seed", "model_width"]
)
def test_invalid_configuration_fails_before_outputs(project, kind):
    if kind == "dimension":
        project.config["tokenizer"]["model"]["input_dim"] = 384
    elif kind == "beam":
        project.config["evaluation"]["beam_size"] = 100
    elif kind == "nan_lr":
        project.config["training"]["learning_rate"] = float("nan")
    elif kind == "duplicate_seed":
        project.config["seeds"] = [42, 42]
    elif kind == "unhashable_seed":
        project.config["seeds"] = [[42]]
    else:
        project.config["model"]["d_model"] = float("nan")
    with pytest.raises(ValueError):
        run(project)
    assert not project.run.exists()


@pytest.mark.parametrize("kind", ["unfrozen", "zero_vector", "complex_vector", "fractional_count"])
def test_invalid_frozen_inputs_fail_before_outputs(project, kind):
    if kind == "unfrozen":
        path = project.artifacts / "embedding_config.json"
        manifest = json.loads(path.read_text("utf-8"))
        manifest["frozen"] = False
        path.write_text(json.dumps(manifest), "utf-8")
    elif kind == "fractional_count":
        path = project.processed / "item_popularity.npy"
        counts = np.load(path).astype(np.float64)
        counts[0] += 0.5
        np.save(path, counts)
    else:
        path = project.artifacts / "embeddings.npy"
        vectors = np.load(path)
        if kind == "zero_vector":
            vectors[0] = 0
        else:
            vectors = vectors.astype(np.complex64)
        np.save(path, vectors)
    with pytest.raises(ValueError):
        run(project)
    assert not project.run.exists()


def test_interrupted_recommender_does_not_mark_m3_complete(project, fake_runs, monkeypatch):
    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(experiment_training, "train_experiment", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        run(project)
    assert (project.run / "tokenizer_manifest.json").exists()
    assert not (project.run / "selection_manifest.json").exists()
    assert not (project.run / "completed.json").exists()
