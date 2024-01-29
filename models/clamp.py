"""
Implements a method for weight clamping of neural nets.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Arjovsky M, Chintala S, Bottou L. Wasserstein generative adversarial
        networks. Proc ICML. PMLR 70:214-23. (2017). https://proceedings.mlr.
        press/v70/arjovsky17a.html

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import torch.nn as nn


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
