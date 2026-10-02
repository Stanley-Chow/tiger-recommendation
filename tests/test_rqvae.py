"""Check residual quantization mathematics, gradient routing, and frozen artifacts."""

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from beautyrec.rqvae import (
    ResidualQuantizationVAE,
    RQVAEConfig,
    RQVAETrainingConfig,
    indices_to_codebook,
    load_rqvae_checkpoint,
    tokenizer_diagnostics,
    train_rqvae_tokenizer,
)
from beautyrec.semantic_ids import load_semantic_ids


@pytest.fixture
def known_model():
    torch.set_num_threads(1)
    model = ResidualQuantizationVAE(
        RQVAEConfig(input_dim=2, hidden_dims=(), latent_dim=2, num_levels=2, codebook_size=2)
    )
    with torch.no_grad():
        model.encoder[0].weight.copy_(torch.eye(2))
        model.encoder[0].bias.zero_()
        model.decoder[0].weight.copy_(torch.eye(2))
        model.decoder[0].bias.zero_()
        model.codebooks[0].weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        model.codebooks[1].weight.copy_(torch.tensor([[0.0, 0.5], [0.0, -0.5]]))
    return model


def test_residual_codes_code_sum_and_squared_norm_losses(known_model):
    values = torch.tensor([[1.2, 0.7]])
    output = known_model(values)
    assert output.indices.tolist() == [[0, 0]]
    torch.testing.assert_close(output.quantization.quantized, torch.tensor([[1.0, 0.5]]))
    assert float(output.reconstruction_loss.detach()) == pytest.approx(0.08)
    assert float(output.quantization.codebook_loss.detach()) == pytest.approx(0.61)
    assert float(output.quantization.commitment_loss.detach()) == pytest.approx(0.61)
    assert float(output.loss.detach()) == pytest.approx(0.08 + 0.61 + 0.25 * 0.61)
    repeated = known_model(values.repeat(3, 1))
    torch.testing.assert_close(output.loss, repeated.loss)


def test_reconstruction_uses_one_straight_through_gradient(known_model):
    latent = torch.tensor([[1.2, 0.7]], requires_grad=True)
    quantization = known_model.quantize(latent)
    quantization.quantized.sum().backward()
    torch.testing.assert_close(latent.grad, torch.ones_like(latent))
    assert all(book.weight.grad is None for book in known_model.codebooks)


def test_codebook_loss_updates_selected_centroids_only(known_model):
    latent = torch.tensor([[1.2, 0.7]], requires_grad=True)
    known_model.quantize(latent).codebook_loss.backward()
    assert latent.grad is None
    torch.testing.assert_close(
        known_model.codebooks[0].weight.grad, torch.tensor([[-0.4, -1.4], [0.0, 0.0]])
    )
    torch.testing.assert_close(
        known_model.codebooks[1].weight.grad, torch.tensor([[-0.4, -0.4], [0.0, 0.0]])
    )


def test_commitment_loss_updates_encoder_without_cross_level_codebook_gradients(known_model):
    latent = torch.tensor([[1.2, 0.7]], requires_grad=True)
    known_model.quantize(latent).commitment_loss.backward()
    torch.testing.assert_close(latent.grad, torch.tensor([[0.8, 1.8]]))
    assert all(book.weight.grad is None for book in known_model.codebooks)


def test_full_objective_has_finite_encoder_decoder_and_codebook_gradients(known_model):
    known_model(torch.tensor([[1.2, 0.7]])).loss.backward()
    for component in (known_model.encoder, known_model.decoder, known_model.codebooks):
        gradients = [
            parameter.grad for parameter in component.parameters() if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_identical_vectors_initialize_without_artificial_code_variation(known_model):
    vectors = torch.tensor([[1.2, 0.7]]).repeat(4, 1)
    known_model.initialize_codebooks(vectors, seed=42, max_samples=4)
    quantization = known_model.quantize(known_model.encode(vectors))
    assert quantization.indices.tolist() == [[0, 0]] * 4
    for book in known_model.codebooks:
        torch.testing.assert_close(book.weight[0], book.weight[1])


def test_suffix_uniqueness_and_reserved_vocabulary_are_separate():
    raw = np.zeros((5, 3), dtype=np.int64)
    codebook = indices_to_codebook(raw, "a" * 64, codebook_size=16, padded_vocab_size=100)
    assert codebook.item_codes == [[3, 19, 35, token] for token in range(51, 56)]
    assert codebook.vocab_size == 100
    assert codebook.diagnostics["suffix_vocab_size"] == 5
    assert codebook.diagnostics["suffix_capacity"] == 49
    assert codebook.diagnostics["natural_vocab_size"] == 56
    assert len({tuple(code) for code in codebook.item_codes}) == 5


@pytest.mark.parametrize(
    "raw,options",
    [
        (np.zeros((2, 3)), {}),
        (np.zeros((2, 3), dtype=bool), {}),
        (np.zeros((0, 3), dtype=int), {}),
        (np.array([[0, 16, 0]]), {}),
        (np.array([[-1, 0, 0]]), {}),
        (np.zeros((5, 3), dtype=int), {"padded_vocab_size": 55}),
    ],
)
def test_invalid_raw_codes_and_suffix_capacity_are_rejected(raw, options):
    with pytest.raises(ValueError):
        indices_to_codebook(raw, "a" * 64, **options)


def test_diagnostics_report_quantized_and_continuous_error_separately(known_model):
    metrics, indices = tokenizer_diagnostics(known_model, torch.tensor([[1.2, 0.7], [-1.2, -0.7]]))
    assert indices.shape == (2, 2)
    assert metrics["continuous_reconstruction"] == pytest.approx(0)
    assert metrics["quantized_reconstruction"] == pytest.approx(0.08)
    assert metrics["quantized_reconstruction_mse"] == pytest.approx(0.04)
    assert metrics["used_codes_by_level"] == [2, 2]
    assert metrics["perplexity_by_level"] == pytest.approx([2, 2])
    assert len(metrics["mean_residual_norm_by_level"]) == 3


def write_frozen_artifacts(tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "embeddings"
    processed.mkdir()
    artifacts.mkdir()
    catalog = [{"item_id": item, "asin": f"ASIN{item}", "title": str(item)} for item in range(8)]
    raw = json.dumps(catalog).encode()
    (processed / "catalog.json").write_bytes(raw)
    vectors = np.random.default_rng(10).normal(size=(8, 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(artifacts / "embeddings.npy", vectors)
    (artifacts / "embedding_config.json").write_text(
        json.dumps(
            {
                "catalog_sha256": hashlib.sha256(raw).hexdigest(),
                "frozen": True,
                "normalized": True,
                "backend": "synthetic",
                "diagnostic_only": True,
            }
        ),
        encoding="utf-8",
    )
    return processed, artifacts, vectors


def test_metadata_selection_refit_checkpoint_and_codes_are_reproducible(tmp_path):
    processed, artifacts, vectors = write_frozen_artifacts(tmp_path)
    model_config = RQVAEConfig(input_dim=4, hidden_dims=(8,), latent_dim=2, codebook_size=2)
    training = RQVAETrainingConfig(
        epochs=4,
        warmup_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        init_samples=8,
        patience=4,
        torch_threads=1,
        device="cpu",
        verbose=False,
    )
    first = train_rqvae_tokenizer(processed, artifacts, tmp_path / "first", model_config, training)
    second = train_rqvae_tokenizer(
        processed, artifacts, tmp_path / "second", model_config, training
    )
    first_book = load_semantic_ids(first["semantic_ids"])
    second_book = load_semantic_ids(second["semantic_ids"])
    assert first_book.item_codes == second_book.item_codes
    assert first_book.diagnostics["diagnostic_only"] is True
    assert first_book.diagnostics["final_fit_items"] == 8
    assert first["selected_quantized_epochs"] == second["selected_quantized_epochs"]
    restored, checkpoint = load_rqvae_checkpoint(first["checkpoint"])
    restored_indices = restored(torch.from_numpy(vectors)).indices.detach().numpy()
    rebuilt = indices_to_codebook(restored_indices, first_book.catalog_sha256, codebook_size=2)
    assert rebuilt.item_codes == first_book.item_codes
    assert checkpoint["final_fit_items"] == 8
    summary = json.loads(open(first["summary"], encoding="utf-8").read())
    assert summary["selection"]["fit_items"] == 7
    assert summary["selection"]["monitor_items"] == 1
    assert summary["refit"]["fit_items"] == summary["refit"]["monitor_items"] == 8
    assert summary["refit"]["quantized_epochs_completed"] == first["selected_quantized_epochs"]
    assert summary["selection_uses_recommendation_labels"] is False
    best_selection = summary["selection"]["best_selection_epoch_statistics"]
    assert best_selection["epoch"] == first["selected_quantized_epochs"]
    assert best_selection["heldout_metadata"] == summary["selection"]["best_monitor_diagnostics"]
    assert len(best_selection["fit_metadata"]["code_counts_by_level"]) == 3
    assert sum(best_selection["fit_metadata"]["code_counts_by_level"][0]) == 7
    assert sum(best_selection["heldout_metadata"]["code_counts_by_level"][0]) == 1
    assert summary["refit"]["best_selection_epoch_statistics"] is None
    assert summary["refit"]["final_monitor_diagnostics"] == summary["final_diagnostics"]
    assert summary["selection_seconds"] > 0
    assert summary["refit_seconds"] > 0
    assert (
        0
        < summary["kmeans_initialization_seconds"]
        < (summary["selection_seconds"] + summary["refit_seconds"])
    )
    assert summary["peak_cuda_allocated_bytes"] is None
    marker = json.loads(Path(first["completion_manifest"]).read_text(encoding="utf-8"))
    expected_files = {
        Path(path).name
        for name, path in first.items()
        if name not in {"completion_manifest", "reused", "selected_quantized_epochs"}
    }
    assert set(marker["artifact_sha256"]) == expected_files
    assert all(
        hashlib.sha256((tmp_path / "first" / filename).read_bytes()).hexdigest() == digest
        for filename, digest in marker["artifact_sha256"].items()
    )
    with open(first["refit_metrics"], encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len([row for row in rows if row["phase"] == "warmup"]) == 2
    assert (
        len([row for row in rows if row["phase"] == "quantized"])
        == first["selected_quantized_epochs"]
    )
    reused = train_rqvae_tokenizer(processed, artifacts, tmp_path / "first", model_config, training)
    assert reused["reused"] is True
    with pytest.raises(ValueError, match="configuration differs"):
        train_rqvae_tokenizer(
            processed, artifacts, tmp_path / "first", model_config, replace(training, seed=43)
        )


def test_completed_artifact_mutation_is_rejected_and_missing_marker_refits(tmp_path):
    processed, artifacts, _ = write_frozen_artifacts(tmp_path)
    model_config = RQVAEConfig(input_dim=4, hidden_dims=(4,), latent_dim=2, codebook_size=2)
    training = RQVAETrainingConfig(
        epochs=1, warmup_epochs=0, batch_size=8, torch_threads=1, device="cpu", verbose=False
    )
    output = tmp_path / "output"
    result = train_rqvae_tokenizer(processed, artifacts, output, model_config, training)
    semantic_path = Path(result["semantic_ids"])
    # Even a parseable edit must not be silently treated as the trained codebook.
    semantic_path.write_text(semantic_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity verification: semantic_ids.json"):
        train_rqvae_tokenizer(processed, artifacts, output, model_config, training)

    Path(result["completion_manifest"]).unlink()
    Path(result["refit_metrics"]).unlink()
    refitted = train_rqvae_tokenizer(processed, artifacts, output, model_config, training)
    assert refitted["reused"] is False
    assert Path(refitted["refit_metrics"]).is_file()
    assert Path(refitted["completion_manifest"]).is_file()
    assert (
        train_rqvae_tokenizer(processed, artifacts, output, model_config, training)["reused"]
        is True
    )


def test_metadata_hash_mismatch_is_rejected_before_training(tmp_path):
    processed, artifacts, _ = write_frozen_artifacts(tmp_path)
    payload = json.loads((processed / "catalog.json").read_text(encoding="utf-8"))
    payload[0]["title"] = "Changed"
    (processed / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different catalog"):
        train_rqvae_tokenizer(processed, artifacts, tmp_path / "output")


def test_existing_tokenizer_directory_is_not_overwritten(tmp_path):
    processed, artifacts, _ = write_frozen_artifacts(tmp_path)
    with pytest.raises(ValueError, match="separate output"):
        train_rqvae_tokenizer(processed, artifacts, artifacts)


def test_small_residual_vq_model_can_improve_quantized_reconstruction():
    torch.manual_seed(5)
    torch.set_num_threads(1)
    vectors = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    model = ResidualQuantizationVAE(
        RQVAEConfig(input_dim=2, hidden_dims=(8,), latent_dim=2, num_levels=2, codebook_size=2)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = (model.decode(model.encode(vectors)) - vectors).square().sum(1).mean()
        loss.backward()
        optimizer.step()
    model.initialize_codebooks(vectors, seed=5)
    before = float(model(vectors).reconstruction_loss.detach())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        output = model(vectors)
        output.loss.backward()
        optimizer.step()
    after = float(model(vectors).reconstruction_loss.detach())
    assert math.isfinite(after)
    assert after < before * 0.5


@pytest.mark.parametrize(
    "config_type,options",
    [
        (RQVAEConfig, {"input_dim": 0}),
        (RQVAEConfig, {"input_dim": 4, "commitment_weight": float("nan")}),
        (RQVAEConfig, {"input_dim": 4, "hidden_dims": (0,)}),
        (RQVAETrainingConfig, {"warmup_epochs": -1}),
        (RQVAETrainingConfig, {"validation_fraction": 1}),
        (RQVAETrainingConfig, {"learning_rate": 0}),
        (RQVAETrainingConfig, {"seed": True}),
    ],
)
def test_invalid_configs_are_rejected(config_type, options):
    with pytest.raises(ValueError):
        config_type(**options)
