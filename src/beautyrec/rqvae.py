"""Learn semantic IDs with a deterministic residual vector-quantized autoencoder.

The name RQ-VAE follows TIGER, but this model has no Gaussian posterior or KL
term. Its objective combines squared reconstruction error, codebook losses,
and encoder commitment losses. Reconstruction uses one straight-through
gradient; codebook vectors receive their own quantization-loss gradients.

This compact gradient-based adaptation is not an exact TIGER reproduction:
https://arxiv.org/abs/2305.05065. Defaults retain the project's three 16-way
semantic levels, frozen MiniLM inputs, and deterministic collision suffixes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from beautyrec.embeddings import read_catalog
from beautyrec.semantic_ids import SemanticCodebook
from beautyrec.training import resolve_device, seed_everything


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


@dataclass
class RQVAEConfig:
    """Encoder, decoder, and independent residual-codebook dimensions."""

    input_dim: int
    hidden_dims: tuple[int, ...] = (256, 128)
    latent_dim: int = 32
    num_levels: int = 3
    codebook_size: int = 16
    commitment_weight: float = 0.25

    def __post_init__(self) -> None:
        self.hidden_dims = tuple(self.hidden_dims)
        for name in ("input_dim", "latent_dim", "num_levels", "codebook_size"):
            _positive_integer(getattr(self, name), name)
        for width in self.hidden_dims:
            _positive_integer(width, "hidden layer width")
        if not math.isfinite(self.commitment_weight) or self.commitment_weight < 0:
            raise ValueError("commitment_weight must be finite and nonnegative.")


@dataclass
class RQVAETrainingConfig:
    """Metadata-only epoch selection, followed by full-catalog refitting."""

    epochs: int = 100
    warmup_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    validation_fraction: float = 0.1
    split_seed: int = 2026
    patience: int = 20
    init_samples: int = 4096
    seed: int = 42
    device: str = "auto"
    torch_threads: int = 4
    verbose: bool = True

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience", "init_samples", "torch_threads"):
            _positive_integer(getattr(self, name), name)
        if (
            isinstance(self.warmup_epochs, bool)
            or not isinstance(self.warmup_epochs, Integral)
            or self.warmup_epochs < 0
        ):
            raise ValueError("warmup_epochs must be a nonnegative integer.")
        for name in ("seed", "split_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32).")
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not math.isfinite(self.validation_fraction) or not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be strictly between zero and one.")


@dataclass
class QuantizationOutput:
    """Quantized values, raw level indices, and separately inspectable losses."""

    quantized: torch.Tensor
    indices: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    residual_norms: torch.Tensor


@dataclass
class RQVAEOutput:
    """Forward-pass values; losses average squared vector norms over examples."""

    reconstruction: torch.Tensor
    latent: torch.Tensor
    quantization: QuantizationOutput
    reconstruction_loss: torch.Tensor
    loss: torch.Tensor

    @property
    def indices(self) -> torch.Tensor:
        return self.quantization.indices


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for next_width in hidden_dims:
        layers.extend([nn.Linear(width, next_width), nn.ReLU()])
        width = next_width
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


def _validate_matrix(values: torch.Tensor, width: int, name: str) -> None:
    if values.ndim != 2 or values.size(0) < 1 or values.size(1) != width:
        raise ValueError(f"{name} must be a nonempty batch-by-{width} matrix.")
    if not values.dtype.is_floating_point or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain finite floating-point values.")


class ResidualQuantizationVAE(nn.Module):
    """A deterministic encoder, three-level residual quantizer, and decoder."""

    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _mlp(config.input_dim, config.hidden_dims, config.latent_dim)
        self.decoder = _mlp(
            config.latent_dim, tuple(reversed(config.hidden_dims)), config.input_dim
        )
        self.codebooks = nn.ModuleList(
            [
                nn.Embedding(config.codebook_size, config.latent_dim)
                for _ in range(config.num_levels)
            ]
        )
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1 / config.codebook_size, 1 / config.codebook_size)

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        _validate_matrix(vectors, self.config.input_dim, "Input vectors")
        return self.encoder(vectors)

    def quantize(self, latents: torch.Tensor) -> QuantizationOutput:
        """Approximate a latent with a sum of independently learned code vectors.

        Previous selected vectors are detached when forming later residuals.
        Each commitment term therefore trains the encoder toward a cumulative
        code sum, while each codebook term updates only its selected vectors.
        """
        _validate_matrix(latents, self.config.latent_dim, "Latent vectors")
        residual = latents
        code_sum = torch.zeros_like(latents)
        codebook_loss = latents.new_zeros(())
        commitment_loss = latents.new_zeros(())
        indices = []
        norms = [residual.detach().norm(dim=1)]
        for codebook in self.codebooks:
            # Direct differences avoid cancellation in ||r||²+||e||²-2r·e.
            distances = (residual.unsqueeze(1) - codebook.weight.unsqueeze(0)).square().sum(-1)
            level_indices = distances.argmin(dim=1)
            selected = codebook(level_indices)
            codebook_loss = codebook_loss + (residual.detach() - selected).square().sum(1).mean()
            commitment_loss = (
                commitment_loss + (residual - selected.detach()).square().sum(1).mean()
            )
            code_sum = code_sum + selected
            residual = residual - selected.detach()
            indices.append(level_indices)
            norms.append(residual.detach().norm(dim=1))

        # Exactly the code sum in the forward pass, one identity gradient for z.
        straight_through = code_sum.detach() + (latents - latents.detach())
        return QuantizationOutput(
            straight_through,
            torch.stack(indices, dim=1),
            codebook_loss,
            commitment_loss,
            torch.stack(norms, dim=1),
        )

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        _validate_matrix(quantized, self.config.latent_dim, "Decoder vectors")
        return self.decoder(quantized)

    def forward(self, vectors: torch.Tensor) -> RQVAEOutput:
        latent = self.encode(vectors)
        quantization = self.quantize(latent)
        reconstruction = self.decode(quantization.quantized)
        reconstruction_loss = (vectors - reconstruction).square().sum(1).mean()
        loss = (
            reconstruction_loss
            + quantization.codebook_loss
            + self.config.commitment_weight * quantization.commitment_loss
        )
        return RQVAEOutput(reconstruction, latent, quantization, reconstruction_loss, loss)

    @torch.no_grad()
    def initialize_codebooks(
        self,
        training_vectors: torch.Tensor,
        seed: int = 42,
        max_samples: int = 4096,
        batch_size: int = 256,
    ) -> None:
        """Initialize each level with k-means on the preceding residuals.

        Only the supplied fitting vectors are used. Tiny or identical catalogs
        repeat available centroids rather than inventing variation or jitter.
        """
        _positive_integer(max_samples, "max_samples")
        _positive_integer(batch_size, "batch_size")
        _validate_matrix(training_vectors, self.config.input_dim, "Initialization vectors")
        chosen = np.random.default_rng(seed).permutation(len(training_vectors))[:max_samples]
        device = next(self.parameters()).device
        residuals = torch.cat(
            [
                self.encode(training_vectors[chosen[start : start + batch_size]].to(device)).cpu()
                for start in range(0, len(chosen), batch_size)
            ]
        ).numpy()
        for level, codebook in enumerate(self.codebooks):
            distinct = np.unique(residuals, axis=0)
            count = min(self.config.codebook_size, len(distinct))
            if count == len(distinct):
                centers = distinct
            else:
                centers = (
                    KMeans(
                        n_clusters=count,
                        n_init=10,
                        random_state=(seed + level) % 2**32,
                        algorithm="lloyd",
                    )
                    .fit(residuals)
                    .cluster_centers_
                )
            centers = centers[np.arange(self.config.codebook_size) % count]
            codebook.weight.copy_(torch.as_tensor(centers, device=device))
            distances = (residuals[:, None, :] - centers[None, :, :]) ** 2
            selected = distances.sum(-1).argmin(1)
            residuals = residuals - centers[selected]


def indices_to_codebook(
    indices: np.ndarray,
    catalog_sha256: str,
    codebook_size: int = 16,
    padded_vocab_size: int | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> SemanticCodebook:
    """Add level offsets and catalog-order suffixes without changing raw codes."""
    _positive_integer(codebook_size, "codebook_size")
    indices = np.asarray(indices)
    if indices.ndim != 2 or min(indices.shape) < 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Raw codes must be a nonempty item-by-level integer matrix.")
    if np.any((indices < 0) | (indices >= codebook_size)):
        raise ValueError("Raw code indices must belong to their level's codebook.")
    n_items, depth = indices.shape
    counts = Counter(tuple(row) for row in indices.tolist())
    max_leaf_size = max(counts.values())
    suffix_offset = 3 + depth * codebook_size
    natural_vocab_size = suffix_offset + max_leaf_size
    vocab_size = natural_vocab_size if padded_vocab_size is None else padded_vocab_size
    _positive_integer(vocab_size, "padded_vocab_size")
    if vocab_size < natural_vocab_size:
        raise ValueError("padded_vocab_size cannot be smaller than the required suffix range.")

    next_suffix: Counter = Counter()
    item_codes = []
    for row in indices.tolist():
        prefix = tuple(row)
        tokens = [3 + level * codebook_size + index for level, index in enumerate(row)]
        item_codes.append([*tokens, suffix_offset + next_suffix[prefix]])
        next_suffix[prefix] += 1
    return SemanticCodebook(
        item_codes=item_codes,
        vocab_size=int(vocab_size),
        code_length=depth + 1,
        catalog_sha256=catalog_sha256,
        branching_factor=codebook_size,
        depth=depth,
        diagnostics={
            **(diagnostics or {}),
            "n_items": n_items,
            "leaf_count": len(counts),
            "max_leaf_size": max_leaf_size,
            "suffix_offset": suffix_offset,
            "suffix_vocab_size": max_leaf_size,
            "suffix_capacity": int(vocab_size) - suffix_offset,
            "natural_vocab_size": natural_vocab_size,
            "semantic_vocab_sizes": [codebook_size] * depth,
            "semantic_collision_rate": 1 - len(counts) / n_items,
            "colliding_item_fraction": sum(size for size in counts.values() if size > 1) / n_items,
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    """Hash large artifacts in chunks without retaining another embedding copy."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(paths: dict[str, str]) -> dict[str, str]:
    """The aliases in the public result refer to a single set of saved files."""
    files = {Path(path) for name, path in paths.items() if name != "completion_manifest"}
    return {path.name: _file_sha256(path) for path in sorted(files)}


def _verify_completed_artifacts(paths: dict[str, str], fingerprint: str) -> None:
    marker = json.loads(Path(paths["completion_manifest"]).read_text(encoding="utf-8"))
    if marker.get("fingerprint") != fingerprint:
        raise ValueError("Completed tokenizer configuration fingerprint does not match.")
    expected_files = {
        Path(path).name for name, path in paths.items() if name != "completion_manifest"
    }
    recorded_hashes = marker.get("artifact_sha256", {})
    if set(recorded_hashes) != expected_files:
        raise ValueError("Completed tokenizer manifest does not list every saved artifact.")
    for filename, expected_hash in recorded_hashes.items():
        path = Path(paths["completion_manifest"]).parent / filename
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(
                f"Completed tokenizer artifact failed integrity verification: {filename}"
            )


def _write_metrics(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


@torch.inference_mode()
def _continuous_error(
    model: ResidualQuantizationVAE, vectors: torch.Tensor, batch_size: int
) -> float:
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    for batch in vectors.split(batch_size):
        batch = batch.to(device)
        reconstruction = model.decode(model.encode(batch))
        total += float((batch - reconstruction).square().sum())
    return total / len(vectors)


@torch.inference_mode()
def tokenizer_diagnostics(
    model: ResidualQuantizationVAE, vectors: torch.Tensor, batch_size: int = 256
) -> tuple[dict[str, Any], np.ndarray]:
    """Separate continuous reconstruction, quantized reconstruction, and code use."""
    _positive_integer(batch_size, "batch_size")
    _validate_matrix(vectors, model.config.input_dim, "Diagnostic vectors")
    model.eval()
    device = next(model.parameters()).device
    sums = dict.fromkeys(
        (
            "total_loss",
            "quantized_reconstruction",
            "continuous_reconstruction",
            "codebook_loss",
            "commitment_loss",
            "quantized_cosine",
            "continuous_cosine",
        ),
        0.0,
    )
    norms = np.zeros(model.config.num_levels + 1)
    all_indices = []
    for batch in vectors.split(batch_size):
        batch = batch.to(device)
        output = model(batch)
        continuous = model.decode(output.latent)
        size = len(batch)
        sums["total_loss"] += float(output.loss) * size
        sums["quantized_reconstruction"] += float(output.reconstruction_loss) * size
        sums["continuous_reconstruction"] += float((batch - continuous).square().sum())
        sums["codebook_loss"] += float(output.quantization.codebook_loss) * size
        sums["commitment_loss"] += float(output.quantization.commitment_loss) * size
        sums["quantized_cosine"] += float(F.cosine_similarity(batch, output.reconstruction).sum())
        sums["continuous_cosine"] += float(F.cosine_similarity(batch, continuous).sum())
        norms += output.quantization.residual_norms.sum(0).cpu().numpy()
        all_indices.append(output.indices.cpu().numpy())
    metrics: dict[str, Any] = {key: value / len(vectors) for key, value in sums.items()}
    metrics["quantized_reconstruction_mse"] = (
        metrics["quantized_reconstruction"] / model.config.input_dim
    )
    metrics["continuous_reconstruction_mse"] = (
        metrics["continuous_reconstruction"] / model.config.input_dim
    )
    indices = np.concatenate(all_indices)
    used, entropies, perplexities, code_counts = [], [], [], []
    for level in range(model.config.num_levels):
        counts = np.bincount(indices[:, level], minlength=model.config.codebook_size)
        probabilities = counts[counts > 0] / len(indices)
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        code_counts.append(counts.tolist())
        used.append(int(np.count_nonzero(counts)))
        entropies.append(entropy)
        perplexities.append(float(np.exp(entropy)))
    metrics.update(
        {
            "used_codes_by_level": used,
            "code_counts_by_level": code_counts,
            "entropy_nats_by_level": entropies,
            "perplexity_by_level": perplexities,
            "mean_residual_norm_by_level": (norms / len(vectors)).tolist(),
        }
    )
    return metrics, indices


def _fit_stage(
    vectors: torch.Tensor,
    model_config: RQVAEConfig,
    training: RQVAETrainingConfig,
    fit_indices: np.ndarray,
    monitor_indices: np.ndarray,
    epochs: int,
    metrics_path: Path,
    select_epoch: bool,
) -> tuple[ResidualQuantizationVAE, dict]:
    """Warm up an ordinary autoencoder, initialize residual books, then train VQ."""
    stage_started = time.perf_counter()
    seed_everything(training.seed)
    device = resolve_device(training.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = ResidualQuantizationVAE(model_config).to(device)
    fit_vectors, monitor_vectors = vectors[fit_indices], vectors[monitor_indices]
    loader = DataLoader(
        TensorDataset(fit_vectors),
        batch_size=training.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(training.seed),
        num_workers=0,
    )
    records = []
    continuous_parameters = [*model.encoder.parameters(), *model.decoder.parameters()]
    optimizer = torch.optim.AdamW(
        continuous_parameters, lr=training.learning_rate, weight_decay=training.weight_decay
    )
    warmup_started = time.perf_counter()
    for epoch in range(1, training.warmup_epochs + 1):
        started = time.perf_counter()
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model.decode(model.encode(batch))
            loss = (batch - reconstruction).square().sum(1).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Autoencoder warmup loss became nonfinite.")
            loss.backward()
            nn.utils.clip_grad_norm_(
                continuous_parameters, training.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        monitor = _continuous_error(model, monitor_vectors, training.batch_size)
        records.append(
            {
                "phase": "warmup",
                "epoch": epoch,
                "train_loss": total / len(fit_vectors),
                "train_reconstruction": total / len(fit_vectors),
                "train_codebook_loss": None,
                "train_commitment_loss": None,
                "monitor_quantized_reconstruction": None,
                "monitor_continuous_reconstruction": monitor,
                "monitor_total_loss": None,
                "epoch_seconds": time.perf_counter() - started,
            }
        )
        _write_metrics(metrics_path, records)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started
    initialization_started = time.perf_counter()
    with threadpool_limits(limits=training.torch_threads):
        model.initialize_codebooks(
            fit_vectors,
            seed=training.seed,
            max_samples=training.init_samples,
            batch_size=training.batch_size,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialization_seconds = time.perf_counter() - initialization_started
    quantized_started = time.perf_counter()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    best_reconstruction, best_epoch, stale = math.inf, 0, 0
    best_monitor_diagnostics = None
    best_selection_statistics = None
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        totals = np.zeros(4)
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            if not bool(torch.isfinite(output.loss)):
                raise FloatingPointError("Residual VQ training loss became nonfinite.")
            output.loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), training.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            totals += np.array(
                [
                    float(output.loss.detach()),
                    float(output.reconstruction_loss.detach()),
                    float(output.quantization.codebook_loss.detach()),
                    float(output.quantization.commitment_loss.detach()),
                ]
            ) * len(batch)
        monitor, _ = tokenizer_diagnostics(model, monitor_vectors, training.batch_size)
        totals /= len(fit_vectors)
        records.append(
            {
                "phase": "quantized",
                "epoch": epoch,
                "train_loss": float(totals[0]),
                "train_reconstruction": float(totals[1]),
                "train_codebook_loss": float(totals[2]),
                "train_commitment_loss": float(totals[3]),
                "monitor_quantized_reconstruction": monitor["quantized_reconstruction"],
                "monitor_continuous_reconstruction": monitor["continuous_reconstruction"],
                "monitor_total_loss": monitor["total_loss"],
                "epoch_seconds": time.perf_counter() - started,
            }
        )
        reconstruction = monitor["quantized_reconstruction"]
        if reconstruction < best_reconstruction:
            best_reconstruction, best_epoch, stale = reconstruction, epoch, 0
            best_monitor_diagnostics = monitor
            if select_epoch:
                # These are metadata-selection statistics, not final refit weights.
                fit_diagnostics, _ = tokenizer_diagnostics(model, fit_vectors, training.batch_size)
                best_selection_statistics = {
                    "epoch": epoch,
                    "fit_metadata": fit_diagnostics,
                    "heldout_metadata": monitor,
                }
        else:
            stale += 1
        records[-1]["epoch_seconds"] = time.perf_counter() - started
        _write_metrics(metrics_path, records)
        if training.verbose and (epoch == 1 or epoch % 25 == 0):
            print(
                f"RQ-VAE epoch {epoch:03d}: monitor reconstruction={reconstruction:.6f}", flush=True
            )
        if select_epoch and stale >= training.patience:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, {
        "best_epoch": best_epoch,
        "best_monitor_quantized_reconstruction": best_reconstruction,
        "quantized_epochs_completed": epoch,
        "warmup_epochs": training.warmup_epochs,
        "fit_items": len(fit_indices),
        "monitor_items": len(monitor_indices),
        "best_monitor_diagnostics": best_monitor_diagnostics,
        "final_monitor_diagnostics": monitor,
        "best_selection_epoch_statistics": best_selection_statistics,
        "warmup_seconds": warmup_seconds,
        "kmeans_initialization_seconds": initialization_seconds,
        "quantized_training_seconds": time.perf_counter() - quantized_started,
        "stage_seconds": time.perf_counter() - stage_started,
        # CUDA's peak includes all live allocations in this process on the device.
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def load_rqvae_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[ResidualQuantizationVAE, dict]:
    """Load final all-catalog tokenizer weights with the recorded architecture."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("method") != "rqvae":
        raise ValueError("This checkpoint is not a residual-VQ tokenizer.")
    model = ResidualQuantizationVAE(RQVAEConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"])
    return model.eval(), payload


def train_rqvae_tokenizer(
    processed_dir: Path,
    artifact_dir: Path,
    output_dir: Path,
    model_config: RQVAEConfig | dict | None = None,
    training_config: RQVAETrainingConfig | dict | None = None,
    padded_vocab_size: int | None = None,
) -> dict[str, Any]:
    """Select epochs on metadata holdout, then refit on every warm-catalog item.

    Neither stage reads user interactions or recommendation validation/test
    targets. The final codebook freezes the full-catalog refit, and its copied
    embedding manifest remains compatible with the existing recommender runner.
    """
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    if artifact_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "Use a separate output directory to preserve existing tokenizer artifacts."
        )
    catalog, catalog_sha256 = read_catalog(processed_dir)
    manifest_path = artifact_dir / "embedding_config.json"
    embedding_config = json.loads(manifest_path.read_text(encoding="utf-8"))
    if embedding_config.get("catalog_sha256") != catalog_sha256:
        raise ValueError("Frozen embeddings belong to a different catalog.")
    if not embedding_config.get("frozen", False):
        raise ValueError("RQ-VAE requires frozen product metadata embeddings.")
    vectors_array = np.load(artifact_dir / "embeddings.npy", allow_pickle=False)
    if (
        vectors_array.ndim != 2
        or vectors_array.shape[0] != len(catalog)
        or vectors_array.shape[1] < 1
        or not np.issubdtype(vectors_array.dtype, np.floating)
        or not np.isfinite(vectors_array).all()
    ):
        raise ValueError("Embedding vectors must be finite and catalog-aligned.")
    if len(catalog) < 2:
        raise ValueError("Metadata holdout selection requires at least two catalog items.")
    if model_config is None:
        model_config = RQVAEConfig(input_dim=vectors_array.shape[1])
    elif isinstance(model_config, dict):
        model_config = RQVAEConfig(**model_config)
    if training_config is None:
        training = RQVAETrainingConfig()
    elif isinstance(training_config, dict):
        training = RQVAETrainingConfig(**training_config)
    else:
        training = training_config
    if model_config.input_dim != vectors_array.shape[1]:
        raise ValueError("Model input_dim differs from the frozen embedding dimension.")
    torch.set_num_threads(training.torch_threads)
    vectors = torch.from_numpy(vectors_array.astype(np.float32))
    fingerprints = {
        "catalog_sha256": catalog_sha256,
        "embeddings_sha256": _file_sha256(artifact_dir / "embeddings.npy"),
        "embedding_config_sha256": _file_sha256(manifest_path),
    }
    config = {
        **fingerprints,
        "model": asdict(model_config),
        "training": asdict(training),
        "padded_vocab_size": padded_vocab_size,
        "protocol": "metadata_holdout_epoch_selection_then_full_catalog_refit",
        "optimizer": "AdamW",
        "loss_reduction": "batch mean of squared vector norms",
        "selection_objective": "metadata holdout quantized reconstruction only",
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    paths = {
        "checkpoint": str((output_dir / "rqvae.pt").resolve()),
        "model_weights": str((output_dir / "rqvae.pt").resolve()),
        "semantic_ids": str((output_dir / "semantic_ids.json").resolve()),
        "training_config": str((output_dir / "rqvae_config.json").resolve()),
        "selection_config": str((output_dir / "rqvae_selection_config.json").resolve()),
        "selection_metrics": str((output_dir / "rqvae_selection_metrics.csv").resolve()),
        "refit_config": str((output_dir / "rqvae_refit_config.json").resolve()),
        "refit_metrics": str((output_dir / "rqvae_refit_metrics.csv").resolve()),
        "metrics": str((output_dir / "rqvae_refit_metrics.csv").resolve()),
        "summary": str((output_dir / "rqvae_summary.json").resolve()),
        "embedding_config": str((output_dir / "embedding_config.json").resolve()),
        "embeddings": str((output_dir / "embeddings.npy").resolve()),
        "completion_manifest": str((output_dir / "rqvae_completed.json").resolve()),
    }
    config_path = Path(paths["training_config"])
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            raise ValueError(
                "Existing tokenizer configuration differs; use a new output directory."
            )
        if Path(paths["completion_manifest"]).exists():
            _verify_completed_artifacts(paths, fingerprint)
            summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
            return {
                **paths,
                "reused": True,
                "selected_quantized_epochs": summary["selected_quantized_epochs"],
            }
    elif (output_dir / "semantic_ids.json").exists():
        raise ValueError("The output directory already contains unrelated semantic IDs.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, {**config, "fingerprint": fingerprint})

    permutation = np.random.default_rng(training.split_seed).permutation(len(vectors))
    heldout_count = min(
        len(vectors) - 1, max(1, round(len(vectors) * training.validation_fraction))
    )
    heldout_indices, fit_indices = (
        np.sort(permutation[:heldout_count]),
        np.sort(permutation[heldout_count:]),
    )
    _write_json(
        Path(paths["selection_config"]),
        {
            **config,
            "fit_item_ids": fit_indices.tolist(),
            "monitor_item_ids": heldout_indices.tolist(),
            "monitor_split": "held-out warm-catalog metadata",
        },
    )
    selection_model, selection = _fit_stage(
        vectors,
        model_config,
        training,
        fit_indices,
        heldout_indices,
        training.epochs,
        Path(paths["selection_metrics"]),
        select_epoch=True,
    )
    chosen_epochs = selection["best_epoch"]
    del selection_model
    all_indices = np.arange(len(vectors))
    _write_json(
        Path(paths["refit_config"]),
        {
            **config,
            "selected_quantized_epochs": chosen_epochs,
            "fit_item_ids": all_indices.tolist(),
            "monitor_split": "fitting metadata; diagnostic only, no checkpoint selection",
        },
    )
    model, refit = _fit_stage(
        vectors,
        model_config,
        training,
        all_indices,
        all_indices,
        chosen_epochs,
        Path(paths["refit_metrics"]),
        select_epoch=False,
    )
    diagnostics, raw_indices = tokenizer_diagnostics(model, vectors, training.batch_size)
    codebook = indices_to_codebook(
        raw_indices,
        catalog_sha256,
        model_config.codebook_size,
        padded_vocab_size,
        diagnostics={
            **diagnostics,
            "method": "rqvae",
            "implementation": "deterministic-residual-vq-autoencoder",
            "seed": training.seed,
            "split_seed": training.split_seed,
            "selected_quantized_epochs": chosen_epochs,
            "final_fit_items": len(vectors),
            "embedding_backend": embedding_config.get("backend"),
            "diagnostic_only": embedding_config.get("diagnostic_only", False),
        },
    )
    _write_json(Path(paths["semantic_ids"]), codebook.to_dict())
    payload = {
        **fingerprints,
        "method": "rqvae",
        "model_config": asdict(model_config),
        "training_config": asdict(training),
        "selected_quantized_epochs": chosen_epochs,
        "final_fit_items": len(vectors),
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    temporary_checkpoint = output_dir / "rqvae.pt.tmp"
    torch.save(payload, temporary_checkpoint)
    temporary_checkpoint.replace(Path(paths["checkpoint"]))
    shutil.copy2(manifest_path, output_dir / "embedding_config.json")
    shutil.copy2(artifact_dir / "embeddings.npy", output_dir / "embeddings.npy")
    summary = {
        **fingerprints,
        "selected_quantized_epochs": chosen_epochs,
        "selection": selection,
        "refit": refit,
        "final_diagnostics": diagnostics,
        "semantic_code_diagnostics": codebook.diagnostics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selection_uses_recommendation_labels": False,
        "final_checkpoint": "last epoch of the fixed all-catalog refit",
        "selection_seconds": selection["stage_seconds"],
        "refit_seconds": refit["stage_seconds"],
        "kmeans_initialization_seconds": (
            selection["kmeans_initialization_seconds"] + refit["kmeans_initialization_seconds"]
        ),
        "peak_cuda_allocated_bytes": (
            max(selection["peak_cuda_allocated_bytes"], refit["peak_cuda_allocated_bytes"])
            if next(model.parameters()).device.type == "cuda"
            else None
        ),
    }
    _write_json(Path(paths["summary"]), summary)
    _write_json(
        config_path,
        {**config, "fingerprint": fingerprint, "selected_quantized_epochs": chosen_epochs},
    )
    # A directory is reusable only after every output has been saved and hashed.
    _write_json(
        Path(paths["completion_manifest"]),
        {"fingerprint": fingerprint, "artifact_sha256": _artifact_hashes(paths)},
    )
    return {**paths, "reused": False, "selected_quantized_epochs": chosen_epochs}
