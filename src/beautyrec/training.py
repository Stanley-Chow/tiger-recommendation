"""Training, evaluation, checkpoint loading, and saved experiment records."""

import csv
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from beautyrec.decoding import SemanticTrie, constrained_beam_search
from beautyrec.metrics import popularity_ranking, ranking_metrics
from beautyrec.model import GenerativeRecommender, ModelConfig, build_source_tokens
from beautyrec.semantic_ids import SemanticCodebook, load_semantic_ids


@dataclass
class TrainingConfig:
    """Practical single-GPU defaults; validation sampling only selects checkpoints."""

    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    max_history: int = 20
    patience: int = 3
    seed: int = 42
    device: str = "auto"
    torch_threads: int = 4
    top_k: int = 10
    beam_size: int = 20
    eval_max_examples: int | None = 256
    max_train_examples: int | None = None
    bucket_by_history_length: bool = True

    def __post_init__(self) -> None:
        if (
            min(self.epochs, self.batch_size, self.max_history, self.patience, self.torch_threads)
            < 1
        ):
            raise ValueError("Epochs, batch size, history, patience and threads must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Require learning_rate > 0 and weight_decay >= 0.")
        if self.top_k < 1 or self.beam_size < self.top_k:
            raise ValueError("Require beam_size >= top_k >= 1.")
        for limit in (self.eval_max_examples, self.max_train_examples):
            if limit is not None and limit < 1:
                raise ValueError("Example limits must be positive or null.")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or install a CUDA-enabled PyTorch.")
    return device


def seed_everything(seed: int) -> None:
    """Fix sampling and initialization; CUDA reproducibility still depends on hardware."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_examples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def sample_examples(examples: list[dict], limit: int | None, seed: int) -> list[dict]:
    """A fixed random subset, shared by the model and baseline."""
    if limit is None or limit >= len(examples):
        return examples
    indices = sorted(random.Random(seed).sample(range(len(examples)), limit))
    return [examples[index] for index in indices]


class RecommendationDataset(Dataset):
    """Shift target codes by one position for standard teacher-forced cross entropy."""

    def __init__(self, examples: list[dict], codebook: SemanticCodebook, max_history: int) -> None:
        self.examples = examples
        self.codebook = codebook
        self.max_history = max_history

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        codes = self.codebook.item_codes[example["target"]]
        source = build_source_tokens(
            example["history"],
            self.codebook.item_codes,
            self.max_history,
            self.codebook.bos_token,
            self.codebook.eos_token,
        )
        return (
            torch.tensor(source, dtype=torch.long),
            torch.tensor([self.codebook.bos_token, *codes], dtype=torch.long),
            torch.tensor([*codes, self.codebook.eos_token], dtype=torch.long),
        )


def collate_examples(batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    return tuple(
        pad_sequence(list(column), batch_first=True, padding_value=0) for column in zip(*batch)
    )


class HistoryLengthBatchSampler(Sampler):
    """Reduce padding by batching similar history lengths, then shuffle the batches.

    Each example appears once per epoch. Batch order and the order inside each
    batch change reproducibly; labels, histories and dataset splits do not change.
    """

    def __init__(
        self, dataset: RecommendationDataset, batch_size: int, generator: torch.Generator
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.generator = generator
        indices = sorted(
            range(len(dataset)),
            key=lambda index: min(len(dataset.examples[index]["history"]), dataset.max_history),
        )
        self.batches = [
            indices[start : start + batch_size] for start in range(0, len(indices), batch_size)
        ]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        for batch_index in torch.randperm(len(self.batches), generator=self.generator).tolist():
            batch = self.batches[batch_index]
            order = torch.randperm(len(batch), generator=self.generator).tolist()
            yield [batch[index] for index in order]


def check_catalog(processed_dir: Path, codebook: SemanticCodebook) -> None:
    digest = hashlib.sha256((processed_dir / "catalog.json").read_bytes()).hexdigest()
    if digest != codebook.catalog_sha256:
        raise ValueError("Catalog changed after tokenization. Rebuild embeddings and semantic IDs.")


def dataset_fingerprint(processed_dir: Path) -> dict[str, str]:
    """Fingerprint the examples and counts, so a checkpoint cannot use changed splits."""
    names = ("train.jsonl", "validation.jsonl", "test.jsonl", "item_popularity.npy")
    return {name: hashlib.sha256((processed_dir / name).read_bytes()).hexdigest() for name in names}


def check_dataset(processed_dir: Path, checkpoint: dict) -> None:
    if dataset_fingerprint(processed_dir) != checkpoint["dataset_fingerprint"]:
        raise ValueError("Interaction splits or popularity changed. Use this checkpoint's dataset.")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@torch.inference_mode()
def evaluate_model(
    model: GenerativeRecommender,
    examples: list[dict],
    codebook: SemanticCodebook,
    popularity: np.ndarray,
    max_history: int,
    top_k: int,
    beam_size: int,
    output_path: Path | None = None,
) -> dict:
    """Retrieve against the entire warm catalog; never insert the ground-truth item.

    Previously observed items are removed for both methods. Tail items have a
    training count at or below the median catalog count (ties stay in the tail).
    """
    if not examples:
        raise ValueError("Evaluation split is empty after filtering.")
    model.eval()
    device = next(model.parameters()).device
    trie = SemanticTrie(codebook.item_codes, codebook.eos_token)
    tail_threshold = float(np.median(popularity))
    records = []
    coverage = {"generative": set(), "popularity": set()}
    latency_ms = []
    for example in examples:
        seen = set(example.get("seen_items", example["history"]))
        source = torch.tensor(
            build_source_tokens(example["history"], codebook.item_codes, max_history),
            dtype=torch.long,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        predictions = constrained_beam_search(
            model,
            source,
            trie,
            top_k=top_k,
            beam_size=beam_size,
            exclude_items=seen,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms.append((time.perf_counter() - start) * 1000)
        rankings = {
            "generative": [item for item, _ in predictions],
            "popularity": popularity_ranking(popularity, seen, top_k),
        }
        for method, ranked in rankings.items():
            recall, ndcg = ranking_metrics(ranked, example["target"], top_k)
            coverage[method].update(ranked)
            records.append(
                {
                    "user_id": example["user_id"],
                    "method": method,
                    "target": example["target"],
                    "recall": recall,
                    "ndcg": ndcg,
                    "tail_target": int(popularity[example["target"]] <= tail_threshold),
                }
            )
    summary = {}
    for method in coverage:
        method_records = [record for record in records if record["method"] == method]
        tail_records = [record for record in method_records if record["tail_target"]]
        summary[method] = {
            f"recall@{top_k}": float(np.mean([record["recall"] for record in method_records])),
            f"ndcg@{top_k}": float(np.mean([record["ndcg"] for record in method_records])),
            f"tail_recall@{top_k}": (
                float(np.mean([record["recall"] for record in tail_records]))
                if tail_records
                else None
            ),
            "tail_examples": len(tail_records),
            "catalog_coverage": len(coverage[method]) / len(popularity),
        }
    summary.update(
        {
            "examples": len(examples),
            "catalog_items": len(popularity),
            "top_k": top_k,
            "beam_size": beam_size,
            "exclude_seen": True,
            "embedding_backend": codebook.diagnostics.get("embedding_backend"),
            "diagnostic_only": codebook.diagnostics.get("diagnostic_only", False),
            "tail_training_count_threshold": tail_threshold,
            "mean_generative_latency_ms": float(np.mean(latency_ms)),
            "p95_generative_latency_ms": float(np.percentile(latency_ms, 95)),
            "latency_protocol": (
                "single-history decoding; includes initial calls; excludes text embedding"
            ),
        }
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return summary


def train_model(
    processed_dir: Path,
    artifact_dir: Path,
    run_dir: Path,
    training: TrainingConfig,
    model_options: dict | None = None,
) -> dict:
    seed_everything(training.seed)
    torch.set_num_threads(training.torch_threads)
    device = resolve_device(training.device)
    codebook = load_semantic_ids(artifact_dir / "semantic_ids.json")
    check_catalog(processed_dir, codebook)
    dataset_statistics = json.loads((processed_dir / "statistics.json").read_text(encoding="utf-8"))
    if training.max_history > dataset_statistics["max_history"]:
        raise ValueError(
            "Training max_history exceeds the prepared history window. "
            "Re-run prepare with a larger --max-history or reduce the training config."
        )
    train_examples = sample_examples(
        read_examples(processed_dir / "train.jsonl"),
        training.max_train_examples,
        training.seed,
    )
    validation = sample_examples(
        read_examples(processed_dir / "validation.jsonl"),
        training.eval_max_examples,
        training.seed,
    )
    if not train_examples or not validation:
        raise ValueError("Training and validation examples must both be nonempty.")
    popularity = np.load(processed_dir / "item_popularity.npy", allow_pickle=False)
    model_config = ModelConfig(
        vocab_size=codebook.vocab_size,
        max_source_length=training.max_history * codebook.code_length + 2,
        max_target_length=codebook.code_length + 1,
        **(model_options or {}),
    )
    model = GenerativeRecommender(model_config).to(device)
    generator = torch.Generator().manual_seed(training.seed)
    dataset = RecommendationDataset(train_examples, codebook, training.max_history)
    if training.bucket_by_history_length:
        loader = DataLoader(
            dataset,
            batch_sampler=HistoryLengthBatchSampler(dataset, training.batch_size, generator),
            collate_fn=collate_examples,
            num_workers=0,
            generator=generator,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=collate_examples,
            num_workers=0,
            generator=generator,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    loss_function = nn.CrossEntropyLoss(ignore_index=codebook.pad_token)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "training": asdict(training),
        "model": asdict(model_config),
        "processed_dir": str(processed_dir.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "train_examples_used": len(train_examples),
        "validation_examples_used": len(validation),
        "catalog_sha256": codebook.catalog_sha256,
        "dataset_fingerprint": dataset_fingerprint(processed_dir),
        "dataset_statistics": dataset_statistics,
        "embedding_config": json.loads(
            (artifact_dir / "embedding_config.json").read_text(encoding="utf-8")
        ),
        "torch_version": str(torch.__version__),
        "device_used": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_selection": f"validation ndcg@{training.top_k}",
    }
    write_json(run_dir / "config.json", manifest)
    print(
        f"Training on {device}: {len(train_examples):,} examples, "
        f"{manifest['parameter_count']:,} parameters",
        flush=True,
    )
    history = []
    best_ndcg, stale_epochs = -1.0, 0
    for epoch in range(1, training.epochs + 1):
        model.train()
        loss_sum, tokens_seen = 0.0, 0
        started = time.perf_counter()
        for batch_number, (source, decoder_input, labels) in enumerate(loader, start=1):
            source, decoder_input, labels = (
                tensor.to(device) for tensor in (source, decoder_input, labels)
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, decoder_input)
            loss = loss_function(logits.reshape(-1, codebook.vocab_size), labels.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Training loss became non-finite; inspect data and learning rate."
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = int(labels.ne(codebook.pad_token).sum())
            loss_sum += float(loss.detach()) * count
            tokens_seen += count
            if batch_number % 250 == 0:
                print(f"Epoch {epoch:02d}, batch {batch_number}/{len(loader)}", flush=True)
        evaluation = evaluate_model(
            model,
            validation,
            codebook,
            popularity,
            training.max_history,
            training.top_k,
            training.beam_size,
        )
        ndcg = evaluation["generative"][f"ndcg@{training.top_k}"]
        record = {
            "epoch": epoch,
            "train_token_cross_entropy": loss_sum / tokens_seen,
            "validation_recall": evaluation["generative"][f"recall@{training.top_k}"],
            "validation_ndcg": ndcg,
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(record)
        with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        print(
            f"Epoch {epoch:02d}: loss={record['train_token_cross_entropy']:.4f}, "
            f"validation NDCG@{training.top_k}={ndcg:.4f}",
            flush=True,
        )
        if ndcg > best_ndcg:
            best_ndcg, stale_epochs = ndcg, 0
            torch.save(
                {
                    "model_config": asdict(model_config),
                    "training_config": asdict(training),
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "catalog_sha256": codebook.catalog_sha256,
                    "dataset_fingerprint": manifest["dataset_fingerprint"],
                    "semantic_ids_sha256": hashlib.sha256(
                        (artifact_dir / "semantic_ids.json").read_bytes()
                    ).hexdigest(),
                },
                run_dir / "best.pt",
            )
            write_json(run_dir / "best_validation.json", evaluation)
        else:
            stale_epochs += 1
            if stale_epochs >= training.patience:
                print("Early stopping: validation NDCG stopped improving.", flush=True)
                break
    result = {
        "epochs_completed": len(history),
        "best_validation_ndcg": best_ndcg,
        "checkpoint": str((run_dir / "best.pt").resolve()),
    }
    write_json(run_dir / "training_summary.json", result)
    return result


def load_checkpoint(
    checkpoint_path: Path,
    artifact_dir: Path,
    device: torch.device,
) -> tuple[GenerativeRecommender, SemanticCodebook, dict]:
    """Load a weights-only checkpoint and reject mismatched tokenization artifacts."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    codes_path = artifact_dir / "semantic_ids.json"
    if hashlib.sha256(codes_path.read_bytes()).hexdigest() != checkpoint["semantic_ids_sha256"]:
        raise ValueError("Semantic IDs do not match this checkpoint. Use its original artifacts.")
    codebook = load_semantic_ids(codes_path)
    model = GenerativeRecommender(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, codebook, checkpoint
