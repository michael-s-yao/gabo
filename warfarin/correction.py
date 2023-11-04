"""
Conservative out-of-distribution objective correction algorithm implementation
for warfarin counterfactual generation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Union


class ObjectiveCorrection:
    def __init__(
        self,
        P: pd.DataFrame,
        num_in_distribution: int,
        random_state: Union[int, np.random.RandomState] = 42,
        device: torch.device = torch.device("cpu"),
        maximize_objective: bool = False
    ):
        """
        Args:
            P: a reference dataset of in-distribution examples.
            num_in_distribution: number of in-distribution examples to use.
            random_state: random seed or a RandomState object. Default 42.
            device: device. Default CPU.
            maximize_objective: whether we are maximizing the objective.
                Default False, corresponding to the task of minimizing a cost.
        """
        self.P = P
        self.n = min(num_in_distribution, len(P))
        self.random_state = random_state
        self.device = device
        self.sign = -1.0 if maximize_objective else 1.0

    def __call__(
        self,
        Q: pd.DataFrame,
        critic: nn.Module,
        K: float,
        L: Union[np.ndarray, float]
    ) -> Union[np.ndarray, float]:
        """
        Calculates the out-of-distribution correction factor for a set of input
        datums Q.
        Input:
            Q: a set of input generated datums.
            critic: source critic callable model.
            K: global Lipschitz constant of the critic function.
            L: Lipschitz constant(s) of the cost/objective function. Can be
                either local or global constants.
        Returns:
            df = (L / K) * Wd, where Wd is the Wasserstein distance.
        """
        P = self.P.sample(
            n=self.n, random_state=self.random_state, replace=False
        )
        P = torch.from_numpy(P.to_numpy().astype(np.float32)).to(self.device)
        Q = torch.from_numpy(Q.to_numpy().astype(np.float32)).to(self.device)
        Wd = np.squeeze(
            (torch.mean(critic(P)) - critic(Q)).detach().cpu().numpy()
        )
        return (self.sign * L / K) * np.maximum(Wd, 0.0)
