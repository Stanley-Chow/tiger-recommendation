"""Shared M2 training and validation for generative retrieval and SASRec.

Checkpoint selection uses a fixed validation subset: maximize Recall@100 among
epochs meeting the popularity MAP@20 guard, then break ties with MAP@20. A run
that never meets the guard still saves its best fallback and reports the failure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from beautyrec.evaluation import DEFAULT_CUTOFFS, context_fingerprint, evaluate_rankings
from beautyrec.model import GenerativeRecommender, ModelConfig
from beautyrec.optimizers import SAM
from beautyrec.retrieval import popularity_rankings, retrieve_rankings, synchronize
from beautyrec.sasrec import SASRecConfig, SASRecDataset, SASRecRecommender, collate_sasrec_examples
from beautyrec.semantic_ids import SemanticCodebook, load_semantic_ids
from beautyrec.training import (
    HistoryLengthBatchSampler,
    RecommendationDataset,
    check_catalog,
    check_dataset,
    collate_examples,
    dataset_fingerprint,
    read_examples,
    resolve_device,
    sample_examples,
    seed_everything,
    write_json,
)


@dataclass
class ExperimentTrainingConfig:
    """Shared optimizer and validation controls; validation sampling ignores train seed."""

    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    max_history: int = 20
    patience: int = 5
    seed: int = 42
    device: str = "auto"
    torch_threads: int = 4
    validation_examples: int | None = 1000
    validation_seed: int = 2026
    evaluation_batch_size: int = 16
    beam_size: int = 200
    optimizer: str = "adamw"
    sam_rho: float = 0.05
    max_train_examples: int | None = None
    initial_checkpoint: str | None = None
    selection_metric: str = "recall@100"
    guard_metric: str = "map@20"
    map_guard_tolerance: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "batch_size",
            "max_history",
            "patience",
            "torch_threads",
            "evaluation_batch_size",
            "beam_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("validation_examples", "max_train_examples"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be positive or None.")
        for name in ("seed", "validation_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32).")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        for name in ("weight_decay", "sam_rho", "map_guard_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.beam_size < max(DEFAULT_CUTOFFS):
            raise ValueError("beam_size must be at least 200 for the shared top-200 ranking.")
        if self.optimizer not in {"adamw", "sam"}:
            raise ValueError("optimizer must be 'adamw' or 'sam'.")
        metrics = {
            f"{name}@{cutoff}"
            for name in ("recall", "ndcg", "map", "mrr")
            for cutoff in DEFAULT_CUTOFFS
        }
        if self.selection_metric not in metrics or self.guard_metric not in metrics:
            raise ValueError("Selection and guard metrics must use a supported metric and cutoff.")
        if self.initial_checkpoint is not None:
            self.initial_checkpoint = str(self.initial_checkpoint)


def load_experiment_checkpoint(
    checkpoint_path: Path,
    artifact_dir: Path,
    device: torch.device | str,
) -> tuple[GenerativeRecommender | SASRecRecommender, SemanticCodebook, dict]:
    """Load either method after checking semantic-ID and catalog fingerprints.

    Legacy generative checkpoints omit ``method``. Dataset fingerprints require
    ``check_dataset(processed_dir, checkpoint)`` because this loader has no split
    directory argument. Training performs that check before a continuation.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    codes_path = Path(artifact_dir) / "semantic_ids.json"
    if hashlib.sha256(codes_path.read_bytes()).hexdigest() != checkpoint["semantic_ids_sha256"]:
        raise ValueError("Semantic IDs do not match this checkpoint. Use its original artifacts.")
    codebook = load_semantic_ids(codes_path)
    if codebook.catalog_sha256 != checkpoint["catalog_sha256"]:
        raise ValueError("Catalog fingerprint does not match this checkpoint.")
    method = checkpoint.get("method", "generative")
    if method == "generative":
        config = ModelConfig(**checkpoint["model_config"])
        if config.vocab_size != codebook.vocab_size:
            raise ValueError("Checkpoint vocabulary differs from its semantic-ID artifact.")
        model = GenerativeRecommender(config)
    elif method == "sasrec":
        config = SASRecConfig(**checkpoint["model_config"])
        if config.num_items != len(codebook.item_codes):
            raise ValueError("Checkpoint item count differs from its semantic-ID artifact.")
        model = SASRecRecommender(config)
    else:
        raise ValueError(f"Unknown checkpoint method: {method!r}.")
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, codebook, checkpoint


def _rng_state(device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return torch.get_rng_state(), torch.cuda.get_rng_state_all() if device.type == "cuda" else None


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def _training_step(
    model: nn.Module,
    optimizer: torch.optim.AdamW | SAM,
    loss_closure: Callable[[], torch.Tensor],
) -> tuple[float, float | None, int]:
    """Update once; SAM reuses the first-pass dropout mask on the identical batch."""
    optimizer.zero_grad(set_to_none=True)
    is_sam = isinstance(optimizer, SAM)
    device = next(model.parameters()).device
    before_forward = _rng_state(device) if is_sam else None
    after_first_pass = None

    def checked_loss() -> torch.Tensor:
        loss = loss_closure()
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("Training loss must be a finite scalar.")
        return loss

    try:
        loss = checked_loss()
        loss.backward()
        if not is_sam:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            return float(loss.detach()), None, 1
        after_first_pass = _rng_state(device)
        optimizer.first_step()
        _restore_rng(before_forward)
        perturbed_loss = checked_loss()
        perturbed_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.second_step()
        return float(loss.detach()), float(perturbed_loss.detach()), 2
    except BaseException:
        if is_sam:
            optimizer.restore()
        else:
            optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        # Extra forward/backward work must not advance the dropout random stream.
        if is_sam and after_first_pass is not None:
            _restore_rng(after_first_pass)


def _initialize_model(
    method: str,
    codebook: SemanticCodebook,
    training: ExperimentTrainingConfig,
    model_options: dict,
    device: torch.device,
    processed_dir: Path,
    artifact_dir: Path,
) -> tuple[GenerativeRecommender | SASRecRecommender, dict]:
    initialization = {
        "initial_checkpoint": None,
        "initial_checkpoint_sha256": None,
        "initial_checkpoint_epoch": None,
        "optimizer_state_reset": True,
        "initialization": "random recommendation-model weights; fresh AdamW state",
    }
    if training.initial_checkpoint is not None:
        path = Path(training.initial_checkpoint)
        model, _, checkpoint = load_experiment_checkpoint(path, artifact_dir, device)
        check_dataset(processed_dir, checkpoint)
        if checkpoint.get("method", "generative") != method:
            raise ValueError(
                "Initial checkpoint method differs from the requested training method."
            )
        if checkpoint["training_config"]["max_history"] != training.max_history:
            raise ValueError("Continuation must use the initial checkpoint's history window.")
        saved_config = asdict(model.config)
        if any(saved_config.get(name) != value for name, value in model_options.items()):
            raise ValueError("Continuation model options differ from its checkpoint architecture.")
        initialization.update(
            {
                "initial_checkpoint": str(path.resolve()),
                "initial_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "initial_checkpoint_epoch": checkpoint["epoch"],
                "initialization": "checkpoint weights only; AdamW moment state reset",
            }
        )
        return model, initialization
    if method == "generative":
        config = ModelConfig(
            vocab_size=codebook.vocab_size,
            max_source_length=training.max_history * codebook.code_length + 2,
            max_target_length=codebook.code_length + 1,
            **model_options,
        )
        model = GenerativeRecommender(config)
    else:
        config = SASRecConfig(
            num_items=len(codebook.item_codes), max_history=training.max_history, **model_options
        )
        model = SASRecRecommender(config)
    return model.to(device), initialization


def train_experiment(
    processed_dir: Path,
    artifact_dir: Path,
    run_dir: Path,
    method: str,
    training: ExperimentTrainingConfig,
    model_options: dict | None = None,
) -> dict:
    """Train on expanding prefixes and select a checkpoint using validation only."""
    if method not in {"generative", "sasrec"}:
        raise ValueError("method must be 'generative' or 'sasrec'.")
    processed_dir, artifact_dir, run_dir = map(Path, (processed_dir, artifact_dir, run_dir))
    seed_everything(training.seed)
    torch.set_num_threads(training.torch_threads)
    device = resolve_device(training.device)
    codebook = load_semantic_ids(artifact_dir / "semantic_ids.json")
    check_catalog(processed_dir, codebook)
    statistics = json.loads((processed_dir / "statistics.json").read_text(encoding="utf-8"))
    if training.max_history > statistics["max_history"]:
        raise ValueError("Training history window exceeds the prepared history window.")
    train_examples = sample_examples(
        read_examples(processed_dir / "train.jsonl"),
        training.max_train_examples,
        training.validation_seed,
    )
    validation = sample_examples(
        read_examples(processed_dir / "validation.jsonl"),
        training.validation_examples,
        training.validation_seed,
    )
    if not train_examples or not validation:
        raise ValueError("Training and validation examples must both be nonempty.")
    popularity = np.load(processed_dir / "item_popularity.npy", allow_pickle=False)
    if len(popularity) != len(codebook.item_codes):
        raise ValueError("Popularity counts and semantic-ID catalog sizes differ.")
    popularity_lists = popularity_rankings(validation, popularity, top_k=max(DEFAULT_CUTOFFS))
    popularity_evaluation = evaluate_rankings(
        validation, {"popularity": popularity_lists}, popularity
    )
    popularity_metrics = popularity_evaluation["popularity"]
    guard_threshold = popularity_metrics[training.guard_metric] - training.map_guard_tolerance
    model, initialization = _initialize_model(
        method, codebook, training, model_options or {}, device, processed_dir, artifact_dir
    )
    if method == "generative":
        dataset = RecommendationDataset(train_examples, codebook, training.max_history)
        collate = collate_examples
        loss_function = nn.CrossEntropyLoss(ignore_index=codebook.pad_token)
        objective = "teacher-forced next semantic token cross entropy, excluding PAD"
    else:
        dataset = SASRecDataset(train_examples, len(codebook.item_codes), training.max_history)
        collate = collate_sasrec_examples
        loss_function = nn.CrossEntropyLoss()
        objective = "next-item full-catalog softmax cross entropy, all warm items"
    generator = torch.Generator().manual_seed(training.seed)
    loader = DataLoader(
        dataset,
        batch_sampler=HistoryLengthBatchSampler(dataset, training.batch_size, generator),
        collate_fn=collate,
        num_workers=0,
        generator=generator,
    )
    base_optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    optimizer = (
        SAM(base_optimizer, rho=training.sam_rho)
        if training.optimizer == "sam"
        else (base_optimizer)
    )
    fingerprints = dataset_fingerprint(processed_dir)
    ids_sha256 = hashlib.sha256((artifact_dir / "semantic_ids.json").read_bytes()).hexdigest()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    manifest = {
        "method": method,
        "training": asdict(training),
        "model": asdict(model.config),
        "processed_dir": str(processed_dir.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "train_examples_used": len(train_examples),
        "validation_examples_used": len(validation),
        "validation_contexts_sha256": context_fingerprint(validation),
        "catalog_sha256": codebook.catalog_sha256,
        "semantic_ids_sha256": ids_sha256,
        "dataset_fingerprint": fingerprints,
        "dataset_statistics": statistics,
        "embedding_config": json.loads(
            (artifact_dir / "embedding_config.json").read_text(encoding="utf-8")
        ),
        "torch_version": str(torch.__version__),
        "device_used": str(device),
        "parameter_count": parameter_count,
        "objective": objective,
        "checkpoint_selection": {
            "maximize": training.selection_metric,
            "guard": training.guard_metric,
            "minimum": guard_threshold,
            "tie_breaker": training.guard_metric,
            "fallback": "maximize selection metric if every epoch fails the guard; report failure",
        },
        "popularity_validation_metrics": popularity_metrics,
        "sam_dropout_rng": "reuse first-pass masks; random stream advances once per batch",
        **initialization,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", manifest)
    print(
        f"{method}/{training.optimizer} on {device}: {len(train_examples):,} examples, "
        f"{parameter_count:,} parameters; fixed validation {len(validation):,}",
        flush=True,
    )
    history = []
    best_rank = None
    best_epoch = None
    best_metrics = None
    stale_epochs = 0
    total_backwards = 0
    train_peak = None
    validation_peak = None
    wall_started = time.perf_counter()
    for epoch in range(1, training.epochs + 1):
        model.train()
        loss_sum, perturbed_sum, units_seen = 0.0, 0.0, 0
        epoch_backwards = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        train_started = time.perf_counter()
        for batch_number, raw_batch in enumerate(loader, start=1):
            batch = tuple(tensor.to(device) for tensor in raw_batch)
            if method == "generative":
                source, decoder_input, labels = batch
                units = int(labels.ne(codebook.pad_token).sum())

                def loss_closure():
                    logits = model(source, decoder_input)
                    return loss_function(
                        logits.reshape(-1, codebook.vocab_size), labels.reshape(-1)
                    )
            else:
                source, labels = batch
                units = labels.numel()

                def loss_closure():
                    return loss_function(model(source), labels)

            loss, perturbed, backwards = _training_step(model, optimizer, loss_closure)
            loss_sum += loss * units
            if perturbed is not None:
                perturbed_sum += perturbed * units
            units_seen += units
            epoch_backwards += backwards
            if batch_number % 250 == 0:
                print(f"Epoch {epoch:02d}, batch {batch_number}/{len(loader)}", flush=True)
        synchronize(device)
        train_seconds = time.perf_counter() - train_started
        if device.type == "cuda":
            observed = torch.cuda.max_memory_allocated(device) / 2**20
            train_peak = max(train_peak or 0, observed)
        validation_started = time.perf_counter()
        rankings, retrieval_timing = retrieve_rankings(
            model,
            validation,
            codebook,
            max_history=training.max_history,
            top_k=max(DEFAULT_CUTOFFS),
            beam_size=training.beam_size,
            batch_size=training.evaluation_batch_size,
        )
        evaluation = evaluate_rankings(
            validation, {method: rankings, "popularity": popularity_lists}, popularity
        )
        validation_seconds = time.perf_counter() - validation_started
        metrics = evaluation[method]
        selected = float(metrics[training.selection_metric])
        guarded = float(metrics[training.guard_metric])
        if not math.isfinite(selected) or not math.isfinite(guarded):
            raise FloatingPointError("Checkpoint-selection metrics must be finite.")
        guard_passed = guarded >= guard_threshold
        candidate_rank = (guard_passed, selected, guarded)
        improved = best_rank is None or candidate_rank > best_rank
        record = {
            "epoch": epoch,
            "train_cross_entropy": loss_sum / units_seen,
            "train_perturbed_cross_entropy": (
                perturbed_sum / units_seen if training.optimizer == "sam" else None
            ),
            "train_loss_unit": "semantic_token" if method == "generative" else "next_item",
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "optimizer_updates": len(loader),
            "backward_passes": epoch_backwards,
            "guard_passed": guard_passed,
            "guard_threshold": guard_threshold,
            "checkpoint_selected": improved,
            **{f"validation_{name}": value for name, value in metrics.items()},
        }
        total_backwards += epoch_backwards
        if device.type == "cuda":
            validation_peak = max(validation_peak or 0, retrieval_timing["peak_cuda_allocated_mb"])
        history.append(record)
        with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        print(
            f"Epoch {epoch:02d}: loss={record['train_cross_entropy']:.4f}, "
            f"{training.selection_metric}={selected:.4f}, {training.guard_metric}={guarded:.4f}, "
            f"guard={'pass' if guard_passed else 'fail'}",
            flush=True,
        )
        if improved:
            best_rank, best_epoch, best_metrics = candidate_rank, epoch, dict(metrics)
            stale_epochs = 0
            checkpoint = {
                "method": method,
                "model_config": asdict(model.config),
                "training_config": asdict(training),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "catalog_sha256": codebook.catalog_sha256,
                "semantic_ids_sha256": ids_sha256,
                "dataset_fingerprint": fingerprints,
                "validation_contexts_sha256": manifest["validation_contexts_sha256"],
                "validation_metrics": best_metrics,
                "guard_passed": guard_passed,
                "guard_threshold": guard_threshold,
                **initialization,
            }
            torch.save(checkpoint, run_dir / "best.pt")
            evaluation["retrieval_timing"] = retrieval_timing
            write_json(run_dir / "best_validation.json", evaluation)
        else:
            stale_epochs += 1
            if stale_epochs >= training.patience:
                print(
                    "Early stopping: validation checkpoint selection stopped improving.", flush=True
                )
                break
    result = {
        "method": method,
        "optimizer": training.optimizer,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "guard_passed": bool(best_rank[0]),
        "guard_threshold": guard_threshold,
        "guard_metric": training.guard_metric,
        "selection_metric": training.selection_metric,
        "popularity_validation_metrics": popularity_metrics,
        "validation_examples_used": len(validation),
        "validation_contexts_sha256": manifest["validation_contexts_sha256"],
        "checkpoint": str((run_dir / "best.pt").resolve()),
        "parameter_count": parameter_count,
        "train_seconds": sum(row["train_seconds"] for row in history),
        "validation_seconds": sum(row["validation_seconds"] for row in history),
        "wall_seconds": time.perf_counter() - wall_started,
        "optimizer_updates": sum(row["optimizer_updates"] for row in history),
        "backward_passes": total_backwards,
        "peak_cuda_train_allocated_mb": train_peak,
        "peak_cuda_validation_allocated_mb": validation_peak,
        "peak_cuda_allocated_mb": max(train_peak or 0, validation_peak or 0)
        if device.type == "cuda"
        else None,
        **initialization,
    }
    write_json(run_dir / "training_summary.json", result)
    return result
