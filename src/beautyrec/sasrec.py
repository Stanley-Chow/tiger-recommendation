"""A readable SASRec-style sequential recommendation baseline.

The model combines causal self-attention, learned positions, and an item
embedding table shared by the history encoder and the output scorer. Each
expanding-prefix example trains the final real history state with full-catalog
cross entropy. This is a full-softmax adaptation, rather than a reproduction of
the original SASRec implementation's sampled positive/negative BCE objective.

References: https://arxiv.org/abs/1808.09781 and https://github.com/kang205/SASRec.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


@dataclass
class SASRecConfig:
    """Item catalog and dimensions for a causal history encoder."""

    num_items: int
    max_history: int = 20
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "num_items",
            "max_history",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
        ):
            _require_positive_integer(getattr(self, name), name)
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")


def build_history_tokens(
    history: Sequence[int], num_items: int, max_history: int = 20
) -> list[int]:
    """Keep recent items and shift catalog IDs by one to reserve PAD=0.

    Dataset item IDs and output class IDs remain zero based. Only input tokens
    use the shift: item 0 is token 1, and item N-1 is token N.
    """
    _require_positive_integer(num_items, "num_items")
    _require_positive_integer(max_history, "max_history")
    if len(history) == 0:
        raise ValueError("A nonempty history is required.")
    if any(
        isinstance(item, bool) or not isinstance(item, Integral) or not 0 <= item < num_items
        for item in history
    ):
        raise ValueError("History contains an item outside the zero-based catalog.")
    return [int(item) + 1 for item in history[-max_history:]]


def prepare_history_batch(
    histories: Sequence[Sequence[int]],
    num_items: int,
    max_history: int = 20,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a right-padded token batch from chronological item histories."""
    if len(histories) == 0:
        raise ValueError("An inference batch must contain at least one history.")
    tokens = [
        torch.tensor(build_history_tokens(history, num_items, max_history), dtype=torch.long)
        for history in histories
    ]
    return pad_sequence(tokens, batch_first=True, padding_value=0).to(device=device)


class SASRecBlock(nn.Module):
    """Pre-normalized causal attention followed by a pointwise feed-forward layer."""

    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model, config.nhead, dropout=config.dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.feedforward_norm = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, hidden: torch.Tensor, causal_mask: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = hidden + self.attention_dropout(attended)
        hidden = hidden + self.feedforward(self.feedforward_norm(hidden))
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SASRecRecommender(nn.Module):
    """Encode a batch of histories and score every zero-based catalog item."""

    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(config.num_items + 1, config.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_history, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([SASRecBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.max_history, config.max_history, dtype=torch.bool).triu(diagonal=1),
            persistent=False,
        )
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def _validate_tokens(self, history_tokens: torch.Tensor) -> torch.Tensor:
        if history_tokens.ndim != 2 or min(history_tokens.shape) < 1:
            raise ValueError("History tokens must be a nonempty batch-by-length matrix.")
        if history_tokens.dtype != torch.long:
            raise ValueError("History tokens must use torch.long integer values.")
        if history_tokens.size(1) > self.config.max_history:
            raise ValueError("Token length exceeds max_history; truncate complete items first.")
        if torch.any((history_tokens < 0) | (history_tokens > self.config.num_items)):
            raise ValueError("History tokens must be PAD=0 or item tokens in [1, num_items].")
        padding = history_tokens.eq(0)
        if torch.any(padding.all(dim=1)):
            raise ValueError("Every history must contain at least one real item.")
        if torch.any(padding[:, :-1] & ~padding[:, 1:]):
            raise ValueError("Histories must use right padding after their final real item.")
        return padding

    def encode(self, history_tokens: torch.Tensor) -> torch.Tensor:
        """Return causal hidden states, with padded positions reset to zero.

        Right padding ensures that each attention query has a real earlier key;
        left padding could leave the first queries fully masked and produce NaNs.
        """
        padding = self._validate_tokens(history_tokens)
        length = history_tokens.size(1)
        positions = torch.arange(length, device=history_tokens.device).unsqueeze(0)
        hidden = self.item_embedding(history_tokens) * math.sqrt(self.config.d_model)
        hidden = self.input_dropout(hidden + self.position_embedding(positions))
        hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        for block in self.blocks:
            hidden = block(hidden, self.causal_mask[:length, :length], padding)
        return self.final_norm(hidden).masked_fill(padding.unsqueeze(-1), 0.0)

    def forward(self, history_tokens: torch.Tensor) -> torch.Tensor:
        """Return B-by-N full-catalog logits for ``CrossEntropyLoss`` or ranking.

        The final real history state predicts the next item. Input/output item
        embeddings are tied, and the PAD row is omitted from output classes.
        Seen-item exclusions belong in evaluation, not this training objective.
        """
        hidden = self.encode(history_tokens)
        last_positions = history_tokens.ne(0).sum(dim=1) - 1
        rows = torch.arange(history_tokens.size(0), device=history_tokens.device)
        final_states = hidden[rows, last_positions]
        return final_states @ self.item_embedding.weight[1:].transpose(0, 1)

    @torch.inference_mode()
    def score_histories(self, histories: Sequence[Sequence[int]]) -> torch.Tensor:
        """Score one inference batch of raw item histories on the model's device."""
        self.eval()
        tokens = prepare_history_batch(
            histories,
            self.config.num_items,
            self.config.max_history,
            device=self.item_embedding.weight.device,
        )
        return self(tokens)


class SASRecDataset(Dataset):
    """Reuse the project's expanding-prefix histories and zero-based targets."""

    def __init__(self, examples: list[dict], num_items: int, max_history: int = 20) -> None:
        _require_positive_integer(num_items, "num_items")
        _require_positive_integer(max_history, "max_history")
        self.examples = examples
        self.num_items = num_items
        self.max_history = max_history

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        target = example["target"]
        if (
            isinstance(target, bool)
            or not isinstance(target, Integral)
            or not 0 <= target < self.num_items
        ):
            raise ValueError("Targets must be zero-based catalog item IDs.")
        tokens = build_history_tokens(example["history"], self.num_items, self.max_history)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(int(target), dtype=torch.long)


def collate_sasrec_examples(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad histories while keeping one unshifted target class per row."""
    if not batch:
        raise ValueError("A training batch cannot be empty.")
    histories, targets = zip(*batch)
    return (
        pad_sequence(histories, batch_first=True, padding_value=0),
        torch.stack(targets),
    )
