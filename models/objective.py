"""
Objective function to optimize.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn as nn
import torchmetrics.functional.image as F


class Objective(nn.Module):
    """Objective function implementations to optimize."""

    def __init__(self, objective: str):
        """
        Args:
            objective: objective function. One of [`gradient`, `energy`].
        """
        super().__init__()
        self.objective = objective.lower()
        if self.objective not in ["gradient", "grad", "energy"]:
            raise NotImplementedError(
                f"Unrecognized objective function {self.objective}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation to calculate the objective function value.
        Input:
            x: input tensor.
        Returns:
            Value of the objective function f(x).
        """
        if self.objective in ["gradient", "grad"]:
            return self._gradient(x)
        elif self.objective == "energy":
            return self._energy(x)

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
