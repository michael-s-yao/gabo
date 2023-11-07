"""
Defines a CNN-based generator network.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import torch
import torch.nn as nn
from typing import Sequence

sys.path.append(".")
from models.block import ConvBlock


class ConvGenerator(nn.Module):
    """Implements a CNN-based generator network."""

    def __init__(
        self,
        latent_dim: int = 128,
        x_dim: int = 28,
        up_channels: Sequence[int] = [32, 128, 256, 512, 1024],
        down_channels: Sequence[int] = [1024, 256, 64, 16, 1]
    ):
        """
        Args:
            latent_dim: number of input latent space dimensions. Default 128.
            x_dim: dimensions of generated images. Default (1, 28, 28).
            up_channels: hidden ConvTranspose2d channels.
            down_channels: hidden Conv2d channels.
        """
        super().__init__()
        self.latent_dim, self.x_dim = latent_dim, x_dim
        self.up_channels, self.down_channels = up_channels, down_channels

        self.target_dim = 1 << (self.x_dim - 1).bit_length()
        self.diff = (self.target_dim - self.x_dim) // 2
        self.input_dim = self.target_dim // (2 ** (len(self.up_channels) - 2))
        self.input_channels = self.up_channels[0]
        self.dense = nn.Sequential(
            nn.Linear(
                self.latent_dim, (
                    int(self.input_dim ** 2) * self.input_channels
                )
            ),
            nn.ReLU(),
            nn.BatchNorm1d(int(self.input_dim ** 2) * self.input_channels)
        )

        model = [
            ConvBlock(
                self.up_channels[i],
                self.up_channels[i + 1],
                stride=(1 + int(i > 0)),
                activation="ReLU",
                use_batch_norm=True,
                transpose=bool(i > 0)
            )
            for i in range(len(self.up_channels) - 1)
        ]
        model += [
            ConvBlock(
                self.down_channels[i],
                self.down_channels[i + 1],
                stride=1,
                activation=(
                    "ReLU" if i < len(self.up_channels) - 2 else "Sigmoid"
                ),
                use_batch_norm=bool(i < len(self.up_channels) - 2),
                transpose=False
            )
            for i in range(len(self.up_channels) - 1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Forward pass through the convolutional block.
        Input:
            batch_size: batch size to generate.
        Returns:
            model(X) with dimensions B1HW.
        """
        z = torch.randn((batch_size, self.latent_dim)).to(
            self.dense[0].weight.device
        )
        z = self.dense(z).reshape(
            -1, self.input_channels, self.input_dim, self.input_dim
        )
        return self.model(z)[:, :, self.diff:-self.diff, self.diff:-self.diff]
