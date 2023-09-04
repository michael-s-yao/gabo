"""
Defines a utility function `block()` to generate a layer of a network.

Author(s):
    Michael Yao
"""
import torch.nn as nn


def Block(
    in_dim: int, out_dim: int, normalize: bool, activation: str
) -> nn.Module:
    """
    Generates a layer of a network consisting of a linear transformation,
    optional batch normalization, and activation.
    Input:
        in_dim: number of input dimensions.
        out_dim: number of output dimensions.
        normalize: whether to apply batch normalization.
        activation: activation function. One of [`LeakyReLU`, `Tanh`,
            `Sigmoid`].
    Output:
        Layer consisting of a linear transformation, optional batch
            normalization, and activation.
    """
    layer = [nn.Linear(in_dim, out_dim)]

    if normalize:
        layer.append(nn.BatchNorm1d(out_dim))

    if activation.lower() == "leakyrelu":
        layer.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
    elif activation.lower() == "tanh":
        layer.append(nn.Tanh())
    elif activation.lower() == "sigmoid":
        layer.append(nn.Sigmoid())
    else:
        raise NotImplementedError(
            "`activation` must be one of [`LeakyReLU`, `Tanh`, `Sigmoid`]."
        )

    return nn.Sequential(*layer)
