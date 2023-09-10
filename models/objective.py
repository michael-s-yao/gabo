"""
Objective function to optimize.

Author(s):
    Michael Yao @michael-s-yao
    Yimeng Zeng @yimeng-zeng

Adapted from Haydn Jones @haydn-jones molformers repo.

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchmetrics.functional.image as F
from typing import Dict, Tuple, Union

from MolOOD.molformers.models.BaseRegressor import BaseRegressor


class SELFIESObjective(nn.Module):
    """Black-box objective function to optimize for molecule generation."""

    def __init__(
        self,
        vocab: Dict[str, int],
        surrogate_ckpt: Union[Path, str],
        encoder_dim: int = 256,
        encoder_nhead: int = 8,
        encoder_dim_ff: int = 1024,
        encoder_num_layers: int = 6
    ):
        """
        Args:
            vocab: vocabulary dictionary.
            surrogate_ckpt: path to the ckpt for the trained surrogate
                objective function.
            encoder_dim: output dimensions from encoder. Default 256.
            encoder_nhead: number of heads of the encoder. Default 8.
            encoder_dim_ff: feed-forward encoder dimension. Default 1024.
            encoder_num_layers: number of encoder layers. Default 6.
        """
        super().__init__()
        self.model = BaseRegressor(
            vocab,
            d_enc=encoder_dim,
            encoder_nhead=encoder_nhead,
            encoder_dim_ff=encoder_dim_ff,
            encoder_dropout=0.0,
            encoder_num_layers=encoder_num_layers
        )
        self.surrogate_ckpt = surrogate_ckpt
        model_state_dict = torch.load(self.surrogate_ckpt, map_location="cpu")
        try:
            self.model.load_state_dict(model_state_dict["state_dict"])
        except RuntimeError:
            prefix = "model."
            alt_state_dict = {}
            for key, item in model_state_dict["state_dict"].items():
                key = key[len(prefix):] if key.startswith(prefix) else key
                alt_state_dict[key] = item
            self.model.load_state_dict(alt_state_dict)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation to calculate the objective function value.
        Input:
            tokens: input batch of token representations of molecules.
        Returns:
            Value of the objective function for the input molecules.
        """
        encoding, pad_mask = self.model.encode(tokens)

        # Average encoding over all tokens, excluding padding.
        encoding = encoding.masked_fill(torch.unsqueeze(pad_mask, dim=-1), 0.0)
        encoding = torch.sum(encoding, dim=1) / torch.sum(
            ~pad_mask, dim=1, keepdim=True
        )

        return torch.squeeze(self.model.regressor(encoding), dim=1)


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
