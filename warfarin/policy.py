"""
Implements the sampler policy to optimize patient warfarin dose.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from math import isclose
from tqdm import tqdm
from typing import Optional

sys.path.append(".")
from models.fcnn import FCNN
from models.critic import WeightClipper


class DosingPolicy:
    """Implements a random sampler policy for warfarin dose optimization."""

    def __init__(
        self,
        ref_dataset: pd.DataFrame,
        surrogate: nn.Module,
        min_z_dose: float,
        max_z_dose: float,
        alpha: Optional[float] = None,
        seed: int = 42,
        save_best_on_reset: bool = True,
        num_restarts: int = 10,
        patience: int = 100,
        verbose: bool = True,
        **critic_hparams
    ):
        """
        Args:
            ref_dataset: a reference dataset of true patient observations.
            surrogate: a trained surrogate estimator for the objective.
            min_z_dose: minimum warfarin dose (after normalization is applied).
            max_z_dose: maximum warfarin dose (after normalization is applied).
            alpha: optional constant value for alpha.
            seed: random seed. Default 42.
            save_best_on_reset: whether to save the best losses on reset.
            num_restarts: number of starting points for multistart acquisition
                function optimization.
            patience: patience for source critic training. Default 100.
            verbose: whether to print verbose outputs to `stdout`.
        """
        self.dataset = ref_dataset
        self.surrogate = surrogate
        self.min_z_dose, self.max_z_dose = min_z_dose, max_z_dose
        self.constant = alpha
        self.seed = seed
        self.save_best_on_reset = save_best_on_reset
        self.num_restarts = num_restarts
        self.patience = patience
        self.verbose = verbose
        self.rng = np.random.RandomState(seed=self.seed)
        self.dose_key = "Therapeutic Dose of Warfarin"
        self.arg_best, self.best_loss, self.cache = None, None, []

        self.critic = FCNN(**critic_hparams)
        self.critic = self.critic
        self.critic.eval()
        self.clipper = WeightClipper(c=0.1)
        self.optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.1)

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perturbs the warfarin dose of the table of patients and returns the
        the resulting perturbed sample.
        Input:
            X: a DataFrame of patent data.
        Returns:
            A DataFrame of perturbed patient data.
        """
        X[self.dose_key] = self._generate_doses(len(X))
        self.cache.append(X[self.dose_key])
        return X

    def __len__(self) -> int:
        """
        Returns the number of times that the sampler has been called.
        Input:
            None.
        Returns:
            The number of times that the sampler has been called.
        """
        return len(self.cache)

    def reset(self) -> None:
        """
        Resets the sampler.
        Input:
            None.
        Returns:
            None.
        """
        if self.save_best_on_reset:
            self.best_loss, self.cache = self.optimum(), [self.optimum()]
            self.arg_best = np.zeros(len(self.arg_best), dtype=int)
            return
        self.arg_best, self.best_loss, self.cache = None, None, []

    def feedback(self, loss: np.ndarray) -> None:
        """
        Keeps track of the best generated warfarin doses for each patient based
        on an input updated metric.
        Input:
            loss: predicted loss of the most recently generated warfarin dose.
        Returns:
            None.
        """
        if self.arg_best is None or self.best_loss is None:
            self.best_loss = loss
            self.arg_best = np.zeros_like(loss, dtype=int)
            return
        self.arg_best = np.where(
            loss < self.best_loss, len(self) - 1, self.arg_best
        )

    def optimum(self) -> np.ndarray:
        """
        Returns the optimal warfarin dosing found for each patient based on
        the available sampling history.
        Input:
            None.
        Returns:
            A vector containing the optimal warfarin dosing for each patient.
        """
        opt, cache = np.zeros(len(self.arg_best)), np.array(self.cache)
        for i in range(len(self.arg_best)):
            opt[i] = cache[self.arg_best[i], i]
        return opt

    def _generate_doses(self, n: int) -> np.ndarray:
        """
        Generates a vector of n valid random doses.
        Input:
            n: number of random doses to generate.
        Returns:
            A vector of n valid random doses.
        """
        m, b = self.max_z_dose - self.min_z_dose, self.min_z_dose
        return (m * self.rng.rand(n)) + b

    def fit_critic(self, X: pd.DataFrame) -> None:
        """
        Trains the critic until the Wasserstein distance stops improving
        according to a preset relative tolerance.
        Input:
            X: a generated dataset of shape BN, where B is the batch size and
                N is the number of patient features.
        Returns:
            None.
        """
        self.critic.train()
        num_steps = 0
        cache = [-1e12] * self.patience
        P = torch.from_numpy(self.dataset.to_numpy().astype(np.float64))
        Q = torch.from_numpy(X.to_numpy().astype(np.float64))
        param = next(self.critic.parameters())
        P, Q = P.to(param), Q.to(param)
        with tqdm(
            generator(),
            desc="Training Source Critic",
            disable=(not self.verbose)
        ) as pbar:
            for _ in pbar:
                self.critic.zero_grad()
                negWd = -1.0 * torch.mean(self.wasserstein(P, Q))
                negWd.backward()
                self.optimizer.step()
                self.clipper(self.critic)
                num_steps += 1
                Wd = -negWd.item()
                if isclose(Wd, min(cache), rel_tol=1e-3) or Wd <= min(cache):
                    break
                cache = cache[1:] + [Wd]
                pbar.set_postfix(Wd=Wd)
        self.critic.eval()

    def wasserstein(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Wasserstein distance contribution from each datum in the
        batch of generated samples Q.
        Input:
            P: a reference dataset of shape BN, where B is the batch size and
                N is the input dimension into the critic function.
            Q: a dataset of generated samples of shape BN, where B is the batch
                size and N is the input dimension into the critic function.
        Returns:
            The Wasserstein distance contribution from each datum in Q.
        """
        return torch.mean(self.critic(P)) - self.critic(Q).squeeze(dim=-1)

    def alpha(self) -> float:
        """
        Returns the optimal value of alpha according to the dual optimization
        problem.
        Input:
            None.
        Returns:
            The optimal value of alpha as a float.
        """
        if self.constant is not None:
            return self.constant
        alphas = np.linspace(0.0, 1.0, num=201)
        return alphas[
            np.argmax([self._score(a) for a in alphas])
        ]

    def _score(self, alpha: float) -> float:
        """
        Scores a particular value of alpha according to the Lagrange dual
        function g(alpha).
        Input:
            alpha: the particular value of alpha to score.
        Returns:
            g(alpha).
        """
        P = torch.from_numpy(self.dataset.to_numpy().astype(np.float64))
        zstar = self._search(alpha).detach()
        score = ((1.0 - alpha) * self.surrogate(zstar)) + (
            alpha * self.wasserstein(P, zstar.unsqueeze(dim=0))
        )
        return score.item()

    def _search(self, alpha: float, budget: int = 4096) -> np.ndarray:
        """
        Approximates z* for the Lagrange dual function by searching over
        the standard normal distribution.
        Input:
            alpha: value of alpha to find z* for.
            budget: number of samples to sample from the normal distribution.
        Returns:
            The optimal z* from the sampled latent space points.
        """
        z = self.rng.randn(budget, self.dataset.shape[-1])
        z = self.rng.rand(budget, self.dataset.shape[-1])
        dose = list(self.dataset.columns).index(self.dose_key)
        z[:, dose] = self.min_z_dose + (
            self.rng.randn(budget) * (self.max_z_dose - self.min_z_dose)
        )
        z = torch.from_numpy(z).requires_grad_(True)
        self.surrogate(z).backward(torch.ones((budget, 1)))
        Df = z.grad[dose]
        z.requires_grad_(True)
        self.critic(z).backward(torch.ones((budget, 1)))
        Dc = z.grad[dose]
        L = ((1.0 - alpha) * Df) - (alpha * Dc)
        return z[torch.argmin(torch.linalg.norm(L, dim=-1))]


def generator() -> None:
    """
    Defines a dummy generator for an infinite loop.
    Input:
        None.
    Returns:
        None.
    """
    while True:
        yield
