"""
Converts FCNN models from PyTorch to NumPy.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import inspect
import numpy as np
import torch.nn as nn
from typing import Tuple, Union


class NumPyFCNN:
    def __init__(self, func: nn.Module):
        """
        Args:
            func: a simple FCNN to represent as a NumPy function. Can only have
                linear layers and non-linear activation functions.
        """
        self.func = func
        self.weights = []
        self.biases = []
        self.activations = []
        for name, module in self.func.named_modules():
            if isinstance(module, nn.Linear):
                self.weights.append(module.weight.detach().cpu().numpy())
                self.biases.append(module.bias.detach().cpu().numpy())
            elif any([
                f == type(module)
                for _, f in inspect.getmembers(nn.modules.activation)
            ]):
                self.activations.append(self.activation_function(module))
        self.activations += ["identity"] * (
            len(self.weights) - len(self.activations)
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neural network.
        Input:
            x: input to the neural network.
        Returns:
            Output from the neural network.
        """
        for W, b, f in zip(self.weights, self.biases, self.activations):
            x = x @ W.T + b[np.newaxis]
            if f[0] == "leaky_relu":
                x = np.where(x > 0.0, x, x * f[1])
            elif f == "relu":
                x = np.where(x > 0.0, x, 0.0)
            elif f != "identity":
                raise NotImplementedError
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neural network.
        Input:
            x: input to the neural network.
        Returns:
            Output from the neural network.
        """
        return self.forward(x)

    def activation_function(
        self, module: nn.Module
    ) -> Union[str, Tuple[str, float]]:
        """
        Returns the activation function in a module.
        Input:
            module: an activation function module.
        Returns:
            A string or a tuple representing the activation function.
        """
        if isinstance(module, nn.LeakyReLU):
            return ("leaky_relu", module.negative_slope)
        elif isinstance(module, nn.ReLU):
            return "relu"
        elif isinstance(module, nn.Identity):
            return "identity"
        else:
            raise NotImplementedError
