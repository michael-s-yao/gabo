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


class ResCNNBlock(nn.Module):
    """Defines a 1D convolutional residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "ReLU"
    ):
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: kernel size. Default 3.
            stride: stride. Default 1.
            padding: padding. Default 1.
            activation: activation. Default ReLU.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        self.conv_1 = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        self.layer_norm_1 = nn.LayerNorm(self.out_channels)
        self.activation_1 = getattr(nn, self.activation)()
        self.conv_2 = nn.Conv1d(
            self.out_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        self.layer_norm_2 = nn.LayerNorm(self.out_channels)
        self.activation_2 = getattr(nn, self.activation)()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual CNN block.
        Input:
            X: a 1D signal of shape BCN or CN.
        Returns:
            model(X) of shape BCN or CN.
        """
        h = self.activation_1(
            self.layer_norm_1(self.conv_1(X.permute(0, 2, 1)).permute(0, 2, 1))
        )
        h = self.activation_2(
            self.layer_norm_2(self.conv_1(X.permute(0, 2, 1)).permute(0, 2, 1))
        )
        return X + h
