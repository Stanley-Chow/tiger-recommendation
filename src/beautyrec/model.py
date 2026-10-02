"""A compact encoder-decoder Transformer for next-item semantic codes.

The encoder reads a user's item history. The decoder predicts the next item's
code one token at a time; it cannot look at future target tokens during training.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Model dimensions, including the maximum encoded sequence lengths."""

    vocab_size: int
    max_source_length: int
    max_target_length: int
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    pad_token: int = 0

    def __post_init__(self) -> None:
        positive = (
            self.vocab_size,
            self.max_source_length,
            self.max_target_length,
            self.d_model,
            self.nhead,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.dim_feedforward,
        )
        if any(value < 1 for value in positive):
            raise ValueError("Model dimensions must be positive.")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions, so ordering does not require extra parameters."""

    def __init__(self, d_model: int, max_length: int, dropout: float) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: d_model // 2])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.size(1) > self.encoding.size(1):
            raise ValueError("Sequence exceeds the configured positional encoding length.")
        return self.dropout(embeddings + self.encoding[:, : embeddings.size(1)])


class GenerativeRecommender(nn.Module):
    """Predict semantic tokens using teacher forcing or autoregressive decoding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, config.pad_token)
        self.source_positions = PositionalEncoding(
            config.d_model,
            config.max_source_length,
            config.dropout,
        )
        self.target_positions = PositionalEncoding(
            config.d_model,
            config.max_target_length,
            config.dropout,
        )
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output = nn.Linear(config.d_model, config.vocab_size)

    def encode(self, source_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_padding = source_tokens.eq(self.config.pad_token)
        source = self.source_positions(
            self.embedding(source_tokens) * math.sqrt(self.config.d_model)
        )
        memory = self.transformer.encoder(source, src_key_padding_mask=source_padding)
        return memory, source_padding

    def decode(
        self,
        decoder_tokens: torch.Tensor,
        memory: torch.Tensor,
        source_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self.target_positions(
            self.embedding(decoder_tokens) * math.sqrt(self.config.d_model)
        )
        # True means "blocked" for both masks. A boolean mask avoids mixed-mask warnings.
        causal_mask = torch.ones(
            target.size(1),
            target.size(1),
            dtype=torch.bool,
            device=target.device,
        ).triu(diagonal=1)
        hidden = self.transformer.decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_tokens.eq(self.config.pad_token),
            memory_key_padding_mask=source_padding_mask,
        )
        return self.output(hidden)

    def forward(self, source_tokens: torch.Tensor, decoder_tokens: torch.Tensor) -> torch.Tensor:
        memory, padding = self.encode(source_tokens)
        return self.decode(decoder_tokens, memory, padding)


def build_source_tokens(
    history: list[int],
    item_codes: list[list[int]],
    max_history: int,
    bos_token: int = 1,
    eos_token: int = 2,
) -> list[int]:
    """Keep recent complete items, then flatten their semantic codes into tokens."""
    if max_history < 1 or not history:
        raise ValueError("A nonempty history and positive max_history are required.")
    if any(item < 0 or item >= len(item_codes) for item in history):
        raise ValueError("History contains an item outside the catalog.")
    tokens = [token for item in history[-max_history:] for token in item_codes[item]]
    return [bos_token, *tokens, eos_token]
