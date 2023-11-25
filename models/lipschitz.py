"""
Utility functions for computing the Lipschitz constants of neural networks.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Zhang H, Zhang P, Hsieh C. RecurJac: An efficient recursive algorithm
        for bounding jacobian matrix of neural networks and its applications.
        Proc 33rd AAAI Conf AI 706:5757-64. (2019). https://doi.org/10.1609/
        aaai.v33i01.33015757
    [2] Weng T, Zhang H, Chen H, Song Z, Hsieh C, Boning D, Dhillon IS, Daniel
        L. Towards fast computation of certified robustness for ReLU networks.
        ICML 80:5273-82. (2018). https://proceedings.mlr.press/v80/weng18a.html
    [3] Xu K, Shi Z, Zhang H, Wang Y, Chang K, Huang M, Kailkhura B, Lin X,
        Hsieh C. Automatic perturbation analysis for scalable certified
        robustness and beyond. Proc NeurIPS. (2020). https://doi.org/10.48550/
        arXiv.2002.12920
    [4] auto_LiRPA GitHub repo from @Verified-Intelligence.
        https://github.com/Verified-Intelligence/auto_LiRPA

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
