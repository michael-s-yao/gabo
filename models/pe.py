"""
Defines and implements a positional encoder module for 1D sequences.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        self.model_dim = model_dim
        self.max_len = max_len

        position = torch.unsqueeze(torch.arange(self.max_len), dim=-1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2) * (-math.log(10_000.0) / model_dim)
        )
        pe = torch.zeros(self.max_len, 1, self.model_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x_embed: torch.Tensor) -> torch.Tensor:
        """
        Injects information about the position of the tokens in the sequence.
        Input:
            x_embed: an embedded sequence of shape BND, where B is the batch
                size, N the sequence length, and D the embedding dimension.
        Returns:
            PE(x_embed).
        """
        x_encode = x_embed.permute(1, 0, 2) + self.pe[:x_embed.size(dim=1)]
        return x_encode.permute(1, 0, 2)
