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
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize, unnormalize
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood
)

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
        device: torch.device = torch.device("cpu"),
        batch_size: int = 4,
        num_restarts: int = 10,
        raw_samples: int = 256,
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
            device: device. Default CPU.
            batch_size: batch size. Default 4.
            num_restarts: number of starting points for multistart acquisition
                function optimization.
            raw_samples: number of samples for initialization.
            patience: patience for source critic training. Default 100.
            verbose: whether to print verbose outputs to `stdout`.
        """
        self.dataset = ref_dataset
        self.min_z_dose, self.max_z_dose = min_z_dose, max_z_dose
        self.constant = alpha
        self.seed = seed
        self.device = device
        self.surrogate = surrogate.to(self.device)
        self.batch_size = batch_size
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.patience = patience
        self.verbose = verbose
        self.rng = np.random.RandomState(seed=self.seed)
        self.dose_key = "Therapeutic Dose of Warfarin"
        self.bounds = torch.tensor(
            [[self.min_z_dose], [self.max_z_dose]], device=self.device
        )

        self.critic = FCNN(**critic_hparams)
        self.critic = self.critic
        self.critic.eval()
        self.clipper = WeightClipper(c=0.05)
        self.optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.01)

        self.state_dicts = None

    def __call__(
        self, X: pd.DataFrame, y: torch.Tensor, step: Optional[int] = None
    ) -> torch.Tensor:
        """
        Acquires a new set of warfarin doses to acquire.
        Input:
            X: a DataFrame of patient data.
            y: an BN tensor of the patient cost values, where B is the number
                of patients and N is the number of observed cost values so far.
            step: optional optimization step.
        Returns:
            new_z: a set of new candidate values of shape Bx(batch_size).
        """
        new_z = torch.empty(
            (len(self.bo_models), self.batch_size, 1), device=self.device
        )
        # Optimize the acquisition function.
        for idx, (model, yy) in enumerate(
            tqdm(
                zip(self.bo_models, -y),
                desc=f"Optimizing Warfarin Doses (Step {step})",
                total=len(self.bo_models),
                leave=False
            )
        ):
            best_y = torch.max(y.to(self.device))
            candidates, _ = optimize_acqf(
                acq_function=qExpectedImprovement(model=model, best_f=best_y),
                bounds=torch.stack([
                    torch.zeros(1).to(self.device),
                    torch.ones(1).to(self.device)
                ]),
                q=self.batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
            new_z[idx] = unnormalize(candidates.detach(), bounds=self.bounds)
        # Return new dose values to sample.
        return torch.squeeze(new_z, dim=-1)

    def fit(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit a SingleTaskGP model to a set of input observations.
        Input:
            z: previous normalized dose observations.
            y: corresponding observations of the surrogate cost.
        Returns:
            None.
        """
        self.bo_models = [
            SingleTaskGP(zz.unsqueeze(dim=-1), yy.unsqueeze(dim=-1))
            for zz, yy in zip(normalize(z, bounds=self.bounds), y)
        ]
        if self.state_dicts is not None and len(self.state_dicts) > 0:
            [
                model.load_state_dict(state_dict)
                for model, state_dict in zip(self.bo_models, self.state_dicts)
            ]
        [
            fit_gpytorch_mll(
                ExactMarginalLogLikelihood(
                    likelihood=model.likelihood, model=model
                ).to(self.device)
            )
            for model in self.bo_models
        ]
        self.state_dicts = [model.state_dict() for model in self.bo_models]

    def fit_critic(self, X: pd.DataFrame, z: torch.Tensor) -> None:
        """
        Trains the critic until the Wasserstein distance stops improving
        according to a preset relative tolerance.
        Input:
            X: a generated dataset of shape BN, where B is the batch size and
                N is the number of patient features.
            z: a generated dataset of shape BD, where B is the batch size and
                D is the number of dose observations.
        Returns:
            None.
        """
        self.critic.train()
        num_steps = 0
        cache = [-1e12] * self.patience
        with tqdm(
            generator(),
            desc="Training Source Critic",
            disable=(not self.verbose)
        ) as pbar:
            for _ in pbar:
                self.critic.zero_grad()
                negWd = -1.0 * torch.mean(self.wasserstein(X, z))
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

    def wasserstein(self, X: pd.DataFrame, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Wasserstein distance contribution from each datum in the
        batch of generated samples Q.
        Input:
            X: a generated dataset of shape BN, where B is the batch size and
                N is the number of patient features.
            z: a generated dataset of shape BD, where B is the batch size and
                D is the number of dose observations.
        Returns:
            The Wasserstein distance contribution from each datum in Q.
        """
        X = pd.concat([X] * z.size(dim=-1))
        X[self.dose_key] = z.flatten().detach().cpu().numpy()
        P = torch.from_numpy(self.dataset.to_numpy().astype(np.float64))
        Q = torch.from_numpy(X.to_numpy().astype(np.float64))
        param = next(self.critic.parameters())
        P, Q = P.to(param), Q.to(param)
        Wd = torch.mean(self.critic(P)) - torch.squeeze(self.critic(Q), dim=-1)
        return Wd.reshape(*z.size())

    def surrogate_cost(self, X: pd.DataFrame, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the unpenalized cost estimate according to the surrogate
        function given a set of patient doses and covariates.
        Input:
            X: a DataFrame containing the set of patient covariates.
            z: a tensor containing the normalized dose estimates for each
                patient.
        Returns:
            The unpenalized cost estimates for each dose and patient.
        """
        X = pd.concat([X] * z.size(dim=-1))
        X[self.dose_key] = z.flatten().detach().cpu().numpy()
        cost = self.surrogate(
            torch.from_numpy(X.to_numpy().astype(np.float64)).to(self.device)
        )
        return cost.reshape(*z.size())

    def alpha(self, dataset: pd.DataFrame) -> torch.Tensor:
        """
        Returns the optimal value of alpha according to the dual optimization
        problem.
        Input:
            dataset: the dataset of patients for which to calculate alpha for.
        Returns:
            The optimal values of alpha for each patient in the dataset.
        """
        if self.constant is not None:
            return self.constant
        alphas = torch.from_numpy(np.linspace(0.0, 1.0, num=201))
        alphas = alphas.to(self.device)
        scores = self._score(alphas, dataset)
        return alphas[torch.argmax(scores, dim=-1)]

    def _score(self, alpha: torch.Tensor, X: pd.DataFrame) -> torch.Tensor:
        """
        Scores a particular value of alpha according to the Lagrange dual
        function g(alpha).
        Input:
            alpha: the particular values of alpha to score.
            X: the dataset of patients for which to calculate the score for.
        Returns:
            g(alpha | X) for each datum in X. The returned tensor has shape
            NA, where N is the number of patients in X and A is the number of
            values of alpha tested.
        """
        P = torch.from_numpy(self.dataset.to_numpy().astype(np.float64)).to(
            self.device
        )
        zstar = self._search(alpha, X).detach()
        Wd = (torch.mean(self.critic(P)) - self.critic(zstar)).squeeze(dim=-1)
        return ((1.0 - alpha) * self.surrogate(zstar).squeeze(dim=-1)) + (
            alpha * Wd
        )

    def _search(
        self, alpha: torch.Tensor, X: pd.DataFrame, budget: int = 4096
    ) -> torch.Tensor:
        """
        Approximates z* for the Lagrange dual function by searching over
        the standard normal distribution.
        Input:
            alpha: values of alpha to find z* for.
            X: the dataset of patients for which to find z* for.
            budget: number of samples to sample from the normal distribution.
        Returns:
            The optimal z* from the sampled latent space points for each datum.
            The returned tensor has shape NAD, where N is the number of
            patients in X, A is the number of values of alpha tested, and D is
            the dimensions of the patient features.
        """
        dose_idx = list(self.dataset.columns).index(self.dose_key)
        zstar = torch.empty(
            (X.shape[0], alpha.size(dim=0), X.shape[-1]), device=self.device
        )
        z_min = self.min_z_dose - (5.0 * (self.max_z_dose - self.min_z_dose))
        z_max = self.max_z_dose + (5.0 * (self.max_z_dose - self.min_z_dose))
        alpha = torch.from_numpy(np.linspace(0, 1, 201)).to(self.device)
        alpha = alpha.repeat(budget, 1).T
        for i in range(len(X)):
            z = pd.concat([X.iloc[i]] * budget, ignore_index=True).to_numpy()
            z = z.reshape(-1, X.shape[-1]).astype(np.float64)
            z[:, dose_idx] = z_min + (self.rng.rand(budget) * (z_max - z_min))
            z = torch.from_numpy(z).to(self.device).requires_grad_(True)
            Df = torch.autograd.grad(self.surrogate(z).sum(), z)[0][
                :, dose_idx
            ]
            z.requires_grad_(True)
            Dc = torch.autograd.grad(self.critic(z).sum(), z)[0][:, dose_idx]
            DL = ((1.0 - alpha) * Df) - (alpha * Dc)
            zstar[i] = z[torch.argmin(torch.abs(DL), dim=-1)]
        return zstar


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
