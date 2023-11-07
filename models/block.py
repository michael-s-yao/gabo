"""
Defines the building blocks for the FCNN and CNN model architectures.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn as nn
from typing import Optional


def Block(
    in_dim: int,
    out_dim: int,
    normalize: bool,
    activation: Optional[str] = None
) -> nn.Module:
    """
    Generates a layer of a network consisting of a linear transformation,
    optional batch normalization, and activation.
    Input:
        in_dim: number of input dimensions.
        out_dim: number of output dimensions.
        normalize: whether to apply batch normalization.
        activation: activation function. One of [`LeakyReLU`, `Tanh`,
            `Sigmoid`, `ReLU`, None].
    Output:
        Layer consisting of a linear transformation, optional batch
            normalization, and activation.
    """
    layer = [nn.Linear(in_dim, out_dim)]

    if normalize:
        layer.append(nn.BatchNorm1d(out_dim))

    if activation is None:
        pass
    elif activation.lower() == "relu":
        layer.append(nn.ReLU(inplace=False))
    elif activation.lower() == "leakyrelu":
        layer.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
    elif activation.lower() == "gelu":
        layer.append(nn.GELU())
    elif activation.lower() == "tanh":
        layer.append(nn.Tanh())
    elif activation.lower() == "sigmoid":
        layer.append(nn.Sigmoid())
    else:
        raise NotImplementedError(
            "`activation` must be one of [`LeakyReLU`, `Tanh`, `Sigmoid`]."
        )

    return nn.Sequential(*layer)


class ConvBlock(nn.Module):
    """Implements a convolutional block for a CNN model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "LeakyReLU",
        use_batch_norm: bool = True,
        transpose: bool = False
    ):
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: kernel size. Default 3.
            stride: stride. Default 1.
            padding: padding. Default 1.
            activation: activation function. Default `LeakyReLU`.
            use_batch_norm: whether to apply batch normalization.
            transpose: whether to use the transposed convolution operator.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.transpose = transpose
        if not self.transpose:
            self.conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.padding
            )

        if self.activation is None:
            self.activation = nn.Identity()
        elif self.activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif self.activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif self.activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        elif self.activation.lower() == "gelu":
            self.activation = nn.GELU()

        if self.use_batch_norm:
            self.bn = nn.BatchNorm2d(self.out_channels)
        else:
            self.bn = nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.
        Input:
            X: input to the block with dimensions BCHW.
        Returns:
            model(X) with dimensions BC'HW.
        """
        return self.bn(self.activation(self.conv(X)))
