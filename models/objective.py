"""
Objective function to optimize.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import torch
import torch.nn as nn
import torchmetrics.functional.image as F
from typing import Tuple


class SELFIESObjective(nn.Module):
    """Objective function to optimize for molecule generation."""

    def __init__(self):
        """
        Args:
            None.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation to calculate the objective function value.
        Input:
            x: input tensor.
        Returns:
            Value of the objective function f(x).
        """
        return torch.mean(x)  # TODO


class Objective(nn.Module):
    """Objective function implementations to optimize."""

    def __init__(self, objective: str, x_dim: Tuple[int] = (1, 28, 28)):
        """
        Args:
            objective: objective function. One of [`gradient`, `energy`].
            x_dim: dimensions CHW of the output image from the generator G.
                Default MNIST dimensions (1, 28, 28).
        """
        super().__init__()
        self.objective = objective.lower()
        if self.objective not in ["gradient", "grad", "energy"]:
            raise NotImplementedError(
                f"Unrecognized objective function {self.objective}."
            )

        _, H, W = x_dim
        self._max_energy = H * W
        self._max_grad = np.sqrt(2.0 * H * (W - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation to calculate the objective function value.
        Input:
            x: input tensor.
        Returns:
            Value of the objective function f(x).
        """
        if self.objective in ["gradient", "grad"]:
            return self._gradient(x) / self._max_grad
        elif self.objective == "energy":
            return self._energy(x) / self._max_energy

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the L2 norm of the spatial gradient of an image.
        Input:
            x: input image.
        Returns:
            L2 norm of the spatial gradient of x.
        """
        dy, dx = F.image_gradients(x)
        return torch.mean(
            torch.sqrt(
                torch.sum(torch.square(dx) + torch.square(dy), dim=(1, 2, 3))
            )
        )

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the L2 energy of an input image.
        Input:
            x: input image.
        Returns:
            L2 energy of x.
        """
        return torch.mean(torch.sum(torch.square(x), dim=(1, 2, 3)))

    def _argmax_grad(self, n: int = 28) -> torch.Tensor:
        """
        Returns a tensor that represents the nxn image that maximizes the L2
        norm of the spatial gradient.
        Input:
            n: dimensions of the tensor. Default is the dimensions of the MNIST
                dataset.
        Returns:
            A 1x1xnxn tensor that maximizes the gradient objective function.
        """
        x = torch.zeros((28, 28))
        x[::2, 1::2] = 1.0
        x[1::2, ::2] = 1.0
        return torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
