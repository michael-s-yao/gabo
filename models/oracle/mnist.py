"""
Defines the oracle objective for the MNIST energy optimization task.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Union


class MNISTOracle(nn.Module):
    def __init__(self):
        """
        Args:
            None.
        """
        super().__init__()

    def forward(
        self, X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Computes the energy of a specified MNIST image or batch of images.
        Input:
            X: an image or batch of images.
        Returns:
            The squared L2 norm of X.
        """
        X = X[np.newaxis] if X.ndim % 2 else X
        X = X.reshape(X.shape[0], -1)
        if isinstance(X, np.ndarray):
            return np.mean(np.square(X.reshape), axis=-1).astype(X.dtype)
        return torch.mean(torch.square(X.reshape), dim=-1).to(X.dtype)
