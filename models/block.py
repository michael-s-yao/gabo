"""
Defines a utility function `block()` to generate a layer of a network.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
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
