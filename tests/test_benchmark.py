"""Exercise protocol, continuation controls and resume behavior without training."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from beautyrec import benchmark
from beautyrec.data import create_demo_data, prepare_dataset
from beautyrec.embeddings import build_embeddings
from beautyrec.semantic_ids import build_semantic_ids, load_semantic_ids
from beautyrec.training import dataset_fingerprint


@pytest.fixture
def project(tmp_path):
    reviews, metadata = create_demo_data(tmp_path / "raw")
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    prepare_dataset(reviews, metadata, processed, max_history=5)
    build_embeddings(processed, artifacts, backend="tfidf")
    build_semantic_ids(processed, artifacts, branching_factor=2, depth=2)
    config = json.loads(
        (Path(__file__).parents[1] / "configs" / "m2_smoke.json").read_text("utf-8")
    )
    config["evaluation"]["bootstrap_resamples"] = 30
    return SimpleNamespace(
        processed=processed, artifacts=artifacts, run=tmp_path / "run", config=config
    )


@pytest.fixture
def fake_experiments(project, monkeypatch):
    from beautyrec import experiment_training

    calls = []
    retrieval_calls = []

    def train(processed_dir, artifact_dir, run_dir, method, training, model_options=None):
        assert (project.run / "protocol.json").exists()
        options = asdict(training)
        calls.append(
            {"branch": run_dir.name, "method": method, "training": options, "model": model_options}
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "best.pt"
        checkpoint.write_text(json.dumps({"method": method, "branch": run_dir.name}), "utf-8")
        return {
            "checkpoint": str(checkpoint.resolve()),
            "epochs_completed": training.epochs,
            "best_epoch": 1,
            "method": method,
            "optimizer": training.optimizer,
        }

    def load(path, artifact_dir, device):
        details = json.loads(path.read_text("utf-8"))
        model = SimpleNamespace(method=details["method"], branch=details["branch"])
        return (
            model,
            load_semantic_ids(artifact_dir),
            {
                "dataset_fingerprint": dataset_fingerprint(project.processed),
            },
        )

    def retrieve(model, examples, codebook, **options):
        assert (project.run / "seeds" / "seed_42" / "selection_manifest.json").exists()
        retrieval_calls.append((model.branch, len(examples), options))
        rows = [
            [item for item in range(len(codebook.item_codes)) if item not in example["seen_items"]][
                : options["top_k"]
            ]
            for example in examples
        ]
        return rows, {"histories_per_second": 100, "batch_size": options["batch_size"]}

    monkeypatch.setattr(experiment_training, "train_experiment", train)
    monkeypatch.setattr(experiment_training, "load_experiment_checkpoint", load)
    monkeypatch.setattr(benchmark, "retrieve_rankings", retrieve)
    monkeypatch.setattr(
        benchmark,
        "measure_request_latency",
        lambda *args, **kwargs: {
            "requests": kwargs["requests"],
            "mean_ms": 2.0,
            "p95_ms": 3.0,
            "protocol": "fake request timing for orchestration test",
        },
    )
    return SimpleNamespace(training=calls, retrieval=retrieval_calls)


def test_smoke_freezes_protocol_and_matches_continuation_controls(project, fake_experiments):
    result = benchmark.run_benchmark(
        project.processed, project.artifacts, project.run, project.config
    )
    assert result["diagnostic_only"] is True
    assert result["m2_complete"] is False
    assert result["test_examples"] == result["validation_examples"] == 8
    assert result["test_examples_available"] == 32
    assert result["aggregate_test"]["runs"] == 1
    assert result["aggregate_test"]["methods"] == [
        "popularity",
        "generative",
        "sasrec",
        "generative_adamw3",
        "generative_sam3",
        "generative_adamw6",
    ]
    calls = {call["branch"]: call for call in fake_experiments.training}
    base = calls["generative"]["training"]
    adamw = calls["generative_adamw3"]["training"]
    sam = calls["generative_sam3"]["training"]
    control = calls["generative_adamw6"]["training"]
    assert base["initial_checkpoint"] is None
    assert adamw["initial_checkpoint"] == sam["initial_checkpoint"] == control["initial_checkpoint"]
    assert adamw["epochs"] == sam["epochs"] == 3
    assert adamw["patience"] == sam["patience"] == 3
    assert control["epochs"] == control["patience"] == 6
    assert adamw["learning_rate"] == sam["learning_rate"] == control["learning_rate"] == 0.0001
    assert adamw["optimizer"] == control["optimizer"] == "adamw"
    assert sam["optimizer"] == "sam"
    assert {call["training"]["validation_seed"] for call in calls.values()} == {2026}
    assert (project.run / "completed.json").exists()
    assert "diagnostic run" in (project.run / "REPORT.md").read_text("utf-8")
    comparison_keys = result["seed_results"][0]["comparisons"]
    assert "generative_sam3_minus_generative_adamw6" in comparison_keys


def test_complete_identical_run_reuses_results_without_any_model_calls(project, fake_experiments):
    first = benchmark.run_benchmark(
        project.processed, project.artifacts, project.run, project.config
    )
    before = (len(fake_experiments.training), len(fake_experiments.retrieval))
    second = benchmark.run_benchmark(
        project.processed, project.artifacts, project.run, project.config
    )
    assert first == second
    assert before == (len(fake_experiments.training), len(fake_experiments.retrieval))


def test_completed_seed_survives_interruption_before_global_completion(project, fake_experiments):
    first = benchmark.run_benchmark(
        project.processed, project.artifacts, project.run, project.config
    )
    (project.run / "completed.json").unlink()
    (project.run / "results.json").unlink()
    before = (len(fake_experiments.training), len(fake_experiments.retrieval))
    second = benchmark.run_benchmark(
        project.processed, project.artifacts, project.run, project.config
    )
    assert first == second
    assert before == (len(fake_experiments.training), len(fake_experiments.retrieval))


def test_reusing_changed_configuration_or_inputs_fails_before_training(project, fake_experiments):
    benchmark.run_benchmark(project.processed, project.artifacts, project.run, project.config)
    count = len(fake_experiments.training)
    changed = deepcopy(project.config)
    changed["training"]["learning_rate"] = 0.002
    with pytest.raises(ValueError, match="fingerprints changed"):
        benchmark.run_benchmark(project.processed, project.artifacts, project.run, changed)
    with (project.processed / "test.jsonl").open("a", encoding="utf-8") as output:
        output.write("\n")
    with pytest.raises(ValueError, match="fingerprints changed"):
        benchmark.run_benchmark(project.processed, project.artifacts, project.run, project.config)
    assert len(fake_experiments.training) == count


def test_completed_checkpoint_mutation_cannot_be_reused(project, fake_experiments):
    benchmark.run_benchmark(project.processed, project.artifacts, project.run, project.config)
    checkpoint = project.run / "seeds" / "seed_42" / "generative" / "best.pt"
    checkpoint.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="artifact changed"):
        benchmark.run_benchmark(project.processed, project.artifacts, project.run, project.config)


def test_failed_training_does_not_leave_a_completion_marker(project, fake_experiments, monkeypatch):
    from beautyrec import experiment_training

    def fail(*args, **kwargs):
        raise RuntimeError("simulated training interruption")

    monkeypatch.setattr(experiment_training, "train_experiment", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        benchmark.run_benchmark(project.processed, project.artifacts, project.run, project.config)
    assert (project.run / "protocol.json").exists()
    assert not (project.run / "completed.json").exists()


def test_shortened_optimizer_continuation_cannot_complete_benchmark(
    project, fake_experiments, monkeypatch
):
    from beautyrec import experiment_training

    original = experiment_training.train_experiment

    def shortened(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["training"].initial_checkpoint is not None:
            result["epochs_completed"] -= 1
        return result

    monkeypatch.setattr(experiment_training, "train_experiment", shortened)
    with pytest.raises(RuntimeError, match="matched epoch budget"):
        benchmark.run_benchmark(project.processed, project.artifacts, project.run, project.config)
    assert not (project.run / "completed.json").exists()
    assert not (project.run / "seeds" / "seed_42" / "seed_completed.json").exists()


def test_optional_compute_control_is_explicitly_recorded_when_disabled(project, fake_experiments):
    project.config["continuations"]["adamw_compute_control_epochs"] = None
    result = benchmark.run_benchmark(
        project.processed, project.artifacts, project.run, project.config
    )
    assert "generative_adamw6" not in result["aggregate_test"]["methods"]
    assert len(fake_experiments.training) == 4
    assert "compute control was disabled" in (project.run / "REPORT.md").read_text("utf-8")


def test_diagnostics_cannot_complete_m2_even_if_three_seeds_are_requested(project):
    config = deepcopy(project.config)
    config["diagnostic_only"] = False
    config["seeds"] = [42, 43, 44]
    config["training"]["epochs"] = 20
    reasons = benchmark._protocol(project.processed, project.artifacts, config)[
        "diagnostic_reasons"
    ]
    assert any("original 2014 Beauty" in reason for reason in reasons)
    assert any("Training examples were capped" in reason for reason in reasons)
    assert any("diagnostic only" in reason for reason in reasons)


@pytest.mark.parametrize("change", ["duplicate_seeds", "too_small_beam", "zero_limit"])
def test_invalid_protocol_fails_before_creating_outputs(project, change):
    if change == "duplicate_seeds":
        project.config["seeds"] = [42, 42]
    elif change == "too_small_beam":
        project.config["evaluation"]["beam_size"] = 100
    else:
        project.config["evaluation"]["max_test_examples"] = 0
    with pytest.raises(ValueError):
        benchmark.run_benchmark(project.processed, project.artifacts, project.run, project.config)
    assert not project.run.exists()
