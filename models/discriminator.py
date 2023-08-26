"""
Source discriminator network implementation.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

from models.block import Block


class Discriminator(nn.Module):
    """Discriminator network implementation."""

    def __init__(
        self,
        x_dim: Sequence[int] = (1, 28, 28),
        intermediate_layers: Sequence[int] = (512, 256)
    ):
        """
        Args:
            x_dim: dimensions CHW of the input image to the discriminator D.
                Default MNIST dimensions (1, 28, 28).
            intermediate_layers: intermediate layer output dimensions. Default
                (512, 256).
        """
        super().__init__()
        self.x_dim = x_dim
        self.intermediate_layers = intermediate_layers
        self.model = [
            (
                "layer0",
                Block(
                    in_dim=int(np.prod(self.x_dim)),
                    out_dim=self.intermediate_layers[0],
                    normalize=False,
                    activation="LeakyReLU"
                ),
            )
        ]
        for i in range(1, len(self.intermediate_layers)):
            self.model.append(
                (
                    "layer" + str(i),
                    Block(
                        in_dim=self.intermediate_layers[i - 1],
                        out_dim=self.intermediate_layers[i],
                        normalize=False,
                        activation="LeakyReLU"
                    ),
                )
            )
        self.model.append(
            (
                "layer" + str(len(self.intermediate_layers)),
                Block(
                    in_dim=self.intermediate_layers[-1],
                    out_dim=1,
                    normalize=False,
                    activation="Sigmoid"
                )
            )
        )

        self.model = nn.Sequential(OrderedDict(self.model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the discriminator D.
        Input:
            x: input image as input to the discriminator.
        Returns:
            Probability of x being from source distribution.
        """
        return self.model(x.view(x.size(0), -1))
