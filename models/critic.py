"""
Source critic network implementation.

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


class Critic(nn.Module):
    """Source critic network implementation."""

    def __init__(
        self,
        x_dim: Sequence[int] = (1, 28, 28),
        intermediate_layers: Sequence[int] = (512, 256),
        use_sigmoid: bool = True
    ):
        """
        Args:
            x_dim: dimensions CHW of the input image to the source critic.
                Default MNIST dimensions (1, 28, 28).
            intermediate_layers: intermediate layer output dimensions. Default
                (512, 256).
            use_sigmoid: whether to apply sigmoid activation function as the
                final layer. Default True.
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
        if use_sigmoid:
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
        Forward propagation through the source critic network.
        Input:
            x: input image as input to the source critic.
        Returns:
            Probability of x being from source distribution.
        """
        x = x.to(torch.float)
        return torch.squeeze(self.model(x.view(x.size(0), -1)), dim=-1)


class WeightClipper:
    """Object to clip the weights of a neural network to a finite range."""

    def __init__(self, c: float = 0.01):
        """
        Args:
            c: weight clipping parameter to clip all weights between [-c, c].
        """
        self.c = c

    def __call__(self, module: nn.Module) -> None:
        """
        Clips the weights of an input neural network to between [-c, c].
        Input:
            module: neural network to clip the weights of.
        Returns:
            None.
        """
        _ = [p.data.clamp_(-self.c, self.c) for p in module.parameters()]
