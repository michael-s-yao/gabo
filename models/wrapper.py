"""
PyTorch wrapper for sklearn MLPRegressor models.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union


class FrozenMLPRegressor(nn.Module):
    """Frozen PyTorch MLPRegressor from a trained sklearn MLPRegressor."""

    def __init__(self, surrogate_cost: Union[Path, str]):
        """
        Args:
            surrogate_cost: path to surrogate cost function.
        """
        super().__init__()
        self.surrogate_cost = surrogate_cost
        with open(self.surrogate_cost, "rb") as f:
            self.np_model = pickle.load(f)
        model = []
        for i, (W, bias) in enumerate(
            zip(self.np_model._best_coefs, self.np_model._best_intercepts)
        ):
            model.append(nn.Linear(*W.shape))
            with torch.no_grad():
                model[-1].weight.copy_(torch.tensor(W.T).to(model[-1].weight))
                model[-1].bias.copy_(torch.tensor(bias).to(model[-1].bias))
            if i < len(self.np_model._best_coefs) - 1:
                model.append(nn.ReLU())
        self.torch_model = nn.Sequential(*model)

    def forward(
        self, X: Union[torch.Tensor, np.ndarray, pd.DataFrame]
    ) -> Union[torch.Tensor, np.ndarray, pd.DataFrame]:
        """
        Forward propagation through the network.
        Input:
            X: input tensor of shape BN, where B is the batch size and N is the
                number of input features.
        Returns:
            model(X): output tensor of shape B, where B is the batch size.
        """
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            return self.np_model.predict(X)
        return self.torch_model(X)
