"""
NextItNet Model Architecture
Dilated causal convolutional network for sequential recommendations
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with dilated causal convolution."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.ln = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out = out[:, : x.size(1), :]
        out = self.ln(out)
        out = self.relu(out)
        return out + x if out.size(-1) == x.size(-1) else out


class NextItNet(nn.Module):
    """NextItNet model for sequential recommendations."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        num_blocks: int = 6,
        kernel_size: int = 3,
        dilation_rates: list | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim

        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 1, 2, 4]

        self.embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(embedding_dim, embedding_dim, kernel_size, d)
                for d in dilation_rates
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embedding_dim, num_items)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
            x = self.dropout(x)
        return self.output_layer(x)

    def predict(
        self,
        input_seq: torch.Tensor,
        top_k: int = 10,
        exclude_items: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k items for given sequence.

        Args:
            input_seq: Input sequence tensor [batch_size, seq_len]
            top_k: Number of recommendations
            exclude_items: List of item indices to exclude

        Returns:
            Tuple of (top_items, top_scores)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_seq)
            last_logits = logits[:, -1, :]

            if exclude_items is not None:
                for item in exclude_items:
                    if 0 <= item < self.num_items:
                        last_logits[:, item] = float("-inf")

            last_logits[:, 0] = float("-inf")

            scores = torch.softmax(last_logits, dim=-1)
            top_scores, top_items = torch.topk(scores, top_k, dim=-1)

        return top_items, top_scores
