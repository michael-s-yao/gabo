"""
Implements a simple fully connected neural net (FCNN) model.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn as nn
from typing import Optional, Sequence

from models.block import Block


class FCNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.1,
        final_activation: Optional[str] = "Sigmoid"
    ):
        """
        Args:
            in_dim: dimensions of input data.
            out_dim: dimensions of mode output.
            hidden_dims: dimensions of the hidden intermediate layers.
            dropout: dropout. Default 0.1.
            final_activation: final activation function. One of [`Sigmoid`,
                `LeakyReLU`, None].
        """
        super().__init__()
        layers, dims = [], [in_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            func = "LeakyReLU" if i < len(dims) - 2 else final_activation
            layers.append(
                Block(dims[i], dims[i + 1], normalize=False, activation=func)
            )
            if i < len(dims) - 2 and dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FCNN model.
        Input:
            X: input tensor of shape Bx(in_dim), where B is the batch size.
        Returns:
            Output tensor of shape Bx(out_dim), where B is the batch size.
        """
        return self.model(X)
