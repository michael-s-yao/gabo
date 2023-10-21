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
from typing import Optional, Union

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


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


class Lipschitz(nn.Module):
    """Computes an upper bound on the Lipschitz constant."""

    def __init__(
        self,
        model: nn.Module,
        mode: Optional[str] = "global",
        p: int = torch.inf,
        eps: float = 0.1
    ):
        """
        Args:
            model: neural network to compute an upper bound on the Lipschitz
                constant of.
            mode: one of [`global`, `local`, None].
        """
        super().__init__()
        self.model = model
        self.mode = mode
        if self.mode not in ["global", "local"]:
            raise NotImplementedError(f"Unreocgnized mode {mode}.")
        elif self.mode == "global":
            self._lipshitz = self._global_lipschitz
        elif self.mode == "local":
            self._lipshitz = self._local_lipschitz
        self.p = p
        self.eps = eps

    def forward(self, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes an upper bound on the Lipschitz constant.
        Input:
            X: batch of model inputs.
        Returns:
            The estimated upper bound of the Lipschitz constant(s).
        """
        if self.mode is None:
            return torch.zeros(X.size(dim=0)).to(X)
        return self._lipshitz(X)

    def _local_lipschitz(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes an upper bound on the local Lipschitz constant at the input.
        Input:
            X: batch of model inputs.
        Returns:
            The estimated upper bound of the local Lipschitz constants.
        """
        model = BoundedModule(self.model, X)
        model.augment_gradient_graph(X)
        X = BoundedTensor(X, PerturbationLpNorm(norm=self.p, eps=self.eps))
        Lb, Ub = model.compute_jacobian_bounds(X)
        Lb, Ub = Lb.reshape(*X.size()), Ub.reshape(*X.size())
        Mb = torch.maximum(torch.abs(Lb), torch.abs(Ub))
        return torch.norm(Mb, p=self.p, dim=-1).detach()

    def _global_lipschitz(self, *args, **kwargs) -> torch.Tensor:
        """
        Computes an upper bound on the global Lipschitz constant of the model.
        Input:
            None.
        Returns:
            The estimated upper bound of the global Lipschitz constant.
        """
        return np.prod([
            self._spectral_norm(param)
            for name, param in self.model.named_parameters()
            if "weight" in name
        ])

    def _spectral_norm(self, W: torch.Tensor, num_iter: int = 100) -> float:
        """
        Approximates the spectral norm of a matrix using the power method.
        Input:
            W: input matrix to approximate the spectral norm of.
            num_iter: maximum number of iterations for the power method.
        Returns:
            The estimated spectral norm of the input matrix.
        """
        b_k = torch.randn(W.size(dim=-1)).to(W)
        for _ in range(num_iter):
            b_k1 = torch.squeeze(W.T @ W @ b_k)
            b_k = b_k1 / torch.norm(b_k1, p=self.p)
        eig = torch.norm(W.T @ W @ b_k, p=self.p) / torch.norm(b_k, p=self.p)
        return torch.sqrt(eig).item()
