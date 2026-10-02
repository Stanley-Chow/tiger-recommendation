"""Controlled checkpoint selection, warm starts, and two-pass training invariants."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from beautyrec import experiment_training as experiment
from beautyrec.optimizers import SAM
from beautyrec.semantic_ids import SemanticCodebook
from beautyrec.training import load_checkpoint


@pytest.fixture
def tiny_pipeline(tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    processed.mkdir()
    artifacts.mkdir()
    catalog = [{"item_id": item, "title": f"Item {item}"} for item in range(6)]
    catalog_path = processed / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    codebook = SemanticCodebook(
        item_codes=[[item + 3] for item in range(6)],
        vocab_size=9,
        code_length=1,
        catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
    )
    (artifacts / "semantic_ids.json").write_text(json.dumps(codebook.to_dict()), encoding="utf-8")
    (artifacts / "embedding_config.json").write_text(
        json.dumps({"backend": "synthetic unit-test features"}), encoding="utf-8"
    )
    (processed / "statistics.json").write_text(json.dumps({"max_history": 2}), encoding="utf-8")
    train = [
        {"user_id": f"train{i}", "history": [0] if i % 2 else [0, 4], "target": i % 3 + 1}
        for i in range(6)
    ]
    validation = [
        {"user_id": f"validation{i}", "history": [0], "seen_items": [0], "target": i + 1}
        for i in range(3)
    ]
    for name, rows in (("train", train), ("validation", validation), ("test", validation)):
        (processed / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    np.save(processed / "item_popularity.npy", np.array([10, 9, 8, 7, 6, 5]))
    return processed, artifacts, tmp_path


def config(**overrides):
    return experiment.ExperimentTrainingConfig(
        **{
            "epochs": 1,
            "batch_size": 2,
            "max_history": 2,
            "patience": 5,
            "device": "cpu",
            "torch_threads": 1,
            "validation_examples": None,
            **overrides,
        }
    )


def model_options(method):
    options = {"d_model": 8, "nhead": 2, "dim_feedforward": 16, "dropout": 0.1}
    options.update(
        {"num_encoder_layers": 1, "num_decoder_layers": 1}
        if method == "generative"
        else {"num_layers": 1}
    )
    return options


def scripted_retrieval(monkeypatch, epoch_lists):
    iterator = iter(epoch_lists)

    def retrieve(model, examples, codebook, **options):
        assert options["top_k"] == 200
        assert options["beam_size"] >= 200
        return next(iterator), {"peak_cuda_allocated_mb": None, "protocol": "scripted test ranks"}

    monkeypatch.setattr(experiment, "retrieve_rankings", retrieve)


@pytest.mark.parametrize("method", ["generative", "sasrec"])
def test_real_training_checkpoint_roundtrip_for_both_methods(tiny_pipeline, method):
    processed, artifacts, root = tiny_pipeline
    run = root / method
    summary = experiment.train_experiment(
        processed, artifacts, run, method, config(), model_options(method)
    )
    assert summary["epochs_completed"] == summary["best_epoch"] == 1
    assert summary["optimizer_updates"] == summary["backward_passes"] == 3
    assert summary["train_seconds"] > 0 and summary["validation_seconds"] > 0
    assert summary["parameter_count"] > 0
    assert summary["peak_cuda_allocated_mb"] is None
    model, codebook, checkpoint = experiment.load_experiment_checkpoint(
        run / "best.pt", artifacts, "cpu"
    )
    assert checkpoint["method"] == method
    assert len(codebook.item_codes) == 6
    assert not model.training
    assert checkpoint["validation_metrics"] == summary["best_validation_metrics"]
    assert (run / "metrics.csv").is_file()
    assert (run / "config.json").is_file()
    assert (run / "best_validation.json").is_file()
    if method == "generative":
        legacy_model, _, _ = load_checkpoint(run / "best.pt", artifacts, torch.device("cpu"))
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, legacy_model.state_dict()[name], rtol=0, atol=0)


def test_feasible_checkpoint_beats_higher_recall_guard_failure(tiny_pipeline, monkeypatch):
    processed, artifacts, root = tiny_pipeline
    # Popularity MAP is (1 + 1/2 + 1/3) / 3. All targets at rank 2 fail it;
    # retrieving two targets at rank 1 passes it despite lower Recall@100.
    scripted_retrieval(
        monkeypatch,
        [
            [[4, 1], [4, 2], [4, 3]],
            [[1], [2], []],
            [[4, 1], [4, 2], [4, 3]],
        ],
    )
    summary = experiment.train_experiment(
        processed, artifacts, root / "guard", "sasrec", config(epochs=3), model_options("sasrec")
    )
    assert summary["best_epoch"] == 2
    assert summary["guard_passed"] is True
    assert summary["best_validation_metrics"]["recall@100"] == pytest.approx(2 / 3)
    assert summary["best_validation_metrics"]["map@20"] == pytest.approx(2 / 3)
    _, _, checkpoint = experiment.load_experiment_checkpoint(
        root / "guard/best.pt", artifacts, "cpu"
    )
    assert checkpoint["epoch"] == 2 and checkpoint["guard_passed"] is True


def test_no_feasible_epoch_saves_best_recall_fallback_and_reports_failure(
    tiny_pipeline, monkeypatch
):
    processed, artifacts, root = tiny_pipeline
    scripted_retrieval(monkeypatch, [[[1], [], []], [[4, 1], [4, 2], [4, 3]]])
    summary = experiment.train_experiment(
        processed,
        artifacts,
        root / "fallback",
        "sasrec",
        config(epochs=2),
        model_options("sasrec"),
    )
    assert summary["guard_passed"] is False
    assert summary["best_epoch"] == 2
    assert summary["best_validation_metrics"]["recall@100"] == 1.0
    assert summary["best_validation_metrics"]["map@20"] == 0.5
    _, _, checkpoint = experiment.load_experiment_checkpoint(
        root / "fallback/best.pt", artifacts, "cpu"
    )
    assert checkpoint["guard_passed"] is False


def test_map_tie_breaker_changes_checkpoint_when_recall_is_equal(tiny_pipeline, monkeypatch):
    processed, artifacts, root = tiny_pipeline
    scripted_retrieval(
        monkeypatch,
        [
            [[4, 1], [4, 2], [4, 3]],
            [[1], [2], [3]],
        ],
    )
    summary = experiment.train_experiment(
        processed,
        artifacts,
        root / "tie",
        "sasrec",
        config(epochs=2, map_guard_tolerance=1),
        model_options("sasrec"),
    )
    assert summary["best_epoch"] == 2
    assert summary["best_validation_metrics"]["recall@100"] == 1.0
    assert summary["best_validation_metrics"]["map@20"] == 1.0


def test_validation_contexts_stay_fixed_across_training_seeds_and_test_is_not_evaluated(
    tiny_pipeline, monkeypatch
):
    processed, artifacts, root = tiny_pipeline
    parsed_paths = []
    original_read = experiment.read_examples

    def read(path):
        parsed_paths.append(path.name)
        return original_read(path)

    def retrieve(model, examples, codebook, **options):
        return [[row["target"]] for row in examples], {"peak_cuda_allocated_mb": None}

    monkeypatch.setattr(experiment, "read_examples", read)
    monkeypatch.setattr(experiment, "retrieve_rankings", retrieve)
    fingerprints = []
    for seed in (42, 43):
        summary = experiment.train_experiment(
            processed,
            artifacts,
            root / f"seed{seed}",
            "sasrec",
            config(seed=seed, validation_examples=2),
            model_options("sasrec"),
        )
        assert summary["validation_examples_used"] == 2
        fingerprints.append(summary["validation_contexts_sha256"])
    assert fingerprints[0] == fingerprints[1]
    assert "test.jsonl" not in parsed_paths
    assert parsed_paths.count("validation.jsonl") == 2


def test_warm_start_uses_source_weights_and_resets_adamw_for_both_branches(
    tiny_pipeline, monkeypatch
):
    processed, artifacts, root = tiny_pipeline

    def retrieve(model, examples, codebook, **options):
        return [[row["target"]] for row in examples], {"peak_cuda_allocated_mb": None}

    monkeypatch.setattr(experiment, "retrieve_rankings", retrieve)
    source_run = root / "base"
    experiment.train_experiment(
        processed, artifacts, source_run, "generative", config(), model_options("generative")
    )
    source = source_run / "best.pt"
    checkpoint = torch.load(source, weights_only=True)
    assert checkpoint["optimizer_state"]["state"]  # Source AdamW already has moments.
    original_step = experiment._training_step
    first_calls = []

    def step(model, optimizer, closure):
        if not first_calls:
            base = optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer
            assert not base.state
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, checkpoint["model_state"][name], rtol=0, atol=0)
        first_calls.append(True)
        return original_step(model, optimizer, closure)

    monkeypatch.setattr(experiment, "_training_step", step)
    summaries = []
    for optimizer in ("adamw", "sam"):
        first_calls.clear()
        summary = experiment.train_experiment(
            processed,
            artifacts,
            root / optimizer,
            "generative",
            config(optimizer=optimizer, initial_checkpoint=str(source), learning_rate=0.0001),
            model_options("generative"),
        )
        assert summary["initial_checkpoint_epoch"] == 1
        assert (
            summary["initial_checkpoint_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        )
        assert summary["optimizer_state_reset"] is True
        summaries.append(summary)
    assert summaries[1]["optimizer_updates"] == summaries[0]["optimizer_updates"]
    assert summaries[1]["backward_passes"] == 2 * summaries[0]["backward_passes"]


def test_changed_dataset_cannot_be_used_for_a_continuation(tiny_pipeline, monkeypatch):
    processed, artifacts, root = tiny_pipeline
    scripted_retrieval(monkeypatch, [[[1], [2], [3]]])
    experiment.train_experiment(
        processed, artifacts, root / "base", "sasrec", config(), model_options("sasrec")
    )
    (processed / "test.jsonl").write_text("Changed split bytes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="splits or popularity changed"):
        experiment.train_experiment(
            processed,
            artifacts,
            root / "continued",
            "sasrec",
            config(initial_checkpoint=str(root / "base/best.pt")),
            model_options("sasrec"),
        )


def test_sam_zero_radius_matches_adamw_with_dropout_and_random_stream():
    torch.manual_seed(12)
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 1))
    reference = copy.deepcopy(model)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    base = torch.optim.AdamW(reference.parameters(), lr=0.01)
    sam = SAM(torch.optim.AdamW(model.parameters(), lr=0.01), rho=0)
    torch.manual_seed(101)
    ordinary_loss, _, passes = experiment._training_step(
        reference, base, lambda: reference(x).square().mean()
    )
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(101)
    first_loss, second_loss, sam_passes = experiment._training_step(
        model, sam, lambda: model(x).square().mean()
    )
    assert first_loss == second_loss == ordinary_loss
    assert passes == 1 and sam_passes == 2
    assert torch.equal(torch.get_rng_state(), reference_rng)
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_failed_second_sam_loss_restores_exact_weights_and_clears_state():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1e-8)
    original = model.weight.detach().clone()
    optimizer = SAM(torch.optim.AdamW(model.parameters()))
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        loss = model.weight.square().sum()
        return loss if calls == 1 else loss * float("nan")

    with pytest.raises(FloatingPointError, match="finite scalar"):
        experiment._training_step(model, optimizer, closure)
    assert calls == 2
    assert torch.equal(model.weight, original)
    assert not optimizer.pending and not optimizer.base_optimizer.state
    assert model.weight.grad is None


@pytest.mark.parametrize(
    "options",
    [
        {"learning_rate": float("nan")},
        {"beam_size": 199},
        {"seed": -1},
        {"guard_metric": "map@30"},
        {"optimizer": "unknown"},
    ],
)
def test_invalid_training_controls_are_rejected(options):
    with pytest.raises(ValueError):
        config(**options)
