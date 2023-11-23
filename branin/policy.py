"""
Generative adversarial Bayesian optimization for the toy Branin
optimization task.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood
)
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from math import isclose
from tqdm import tqdm
from typing import Optional, Tuple

sys.path.append(".")
from models.fcnn import FCNN
from models.critic import WeightClipper


class GABOPolicy:
    def __init__(
        self,
        surrogate: nn.Module,
        alpha: Optional[float] = None,
        x1_range: Tuple[float] = (-5.0, 10.0),
        x2_range: Tuple[float] = (0.0, 15.0),
        num_restarts: int = 10,
        raw_samples: int = 128,
        patience: int = 100,
        verbose: bool = True,
        device: torch.device = torch.device("cpu"),
        seed: int = 42
    ):
        """
        Args:
            surrogate: trained surrogate objective model.
            alpha: optional constant value for alpha.
            x1_range: range of valid inputs for the argument x1.
            x2_range: range of valid inputs for the argument x2.
            num_restarts: number of starting points for multistart acquisition
                function optimization.
            raw_samples: number of samples for initialization.
            patience: patience used for source critic training. Default 100.
            verbose: whether to print verbose outputs to `stdout`.
            device: device. Default CPU.
            seed: random seed. Default 42.
        """
        self.surrogate = surrogate
        self.constant = alpha
        self.y_min, self.y_max = self.surrogate.y_min, self.surrogate.y_max
        self.device = device
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.patience = patience
        self.verbose = verbose
        self.bounds = torch.tensor(
            [[min(x1_range), min(x2_range)], [max(x1_range), max(x2_range)]],
            device=self.device
        )
        self.state_dict = None
        self.critic = FCNN(
            in_dim=next(self.surrogate.parameters()).size(dim=-1),
            out_dim=1,
            hidden_dims=([64] * 4),
            dropout=0.0,
            final_activation=None,
            hidden_activation="ReLU",
            use_batch_norm=False
        )
        self.critic = self.critic.to(self.device)
        self.critic.eval()
        self.clipper = WeightClipper(c=0.05)
        self.optimizer = optim.SGD(self.critic.parameters(), lr=0.002)

    def __call__(
        self, model: SingleTaskGP, batch_size: int, y: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Optimizes the acquisition function and returns the next sampled batch.
        Input:
            model: model used to define the acquisition function.
            batch_size: number of training datums to generate.
            y: a tensor of the past objective observations.
        Returns:
            X: an Nx2 array of N (x1, x2) observations, where N is the batch
                size.
            y: an N vector of N corresponding function values.
        """
        candidates, _ = optimize_acqf(
            acq_function=qExpectedImprovement(
                model=model, best_f=(torch.max(y) - self.y_min)
            ),
            bounds=torch.stack([
                torch.zeros(2, device=self.device),
                torch.ones(2, device=self.device),
            ]),
            q=batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples
        )
        X = unnormalize(candidates.detach(), bounds=self.bounds)
        return X.detach(), self.surrogate(X).detach().to(self.device)

    def generate_initial_data(self, n: int) -> Tuple[torch.Tensor]:
        """
        Generate the initial training data.
        Input:
            n: number of training datums to generate.
        Returns:
            X: an Nx2 array of N (x1, x2) observations.
            y: an N vector of N corresponding function values.
        """
        X = unnormalize(
            torch.from_numpy(self.rng.rand(n, 2)).to(self.device),
            bounds=self.bounds
        )
        return X.detach(), self.surrogate(X).detach().to(self.device)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
        """
        Fit a SingleTaskGP model to a set of input observations.
        Input:
            X: a set of observations.
            y: corresponding observations of the surrogate objective.
        Returns:
            None.
        """
        X, y = X.to(self.device), y.to(self.device)
        model = SingleTaskGP(
            train_X=normalize(X, self.bounds),
            train_Y=y,
            outcome_transform=Standardize(m=1)
        )
        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll = mll.to(self.device)
        fit_gpytorch_mll(mll)
        self.state_dict = model.state_dict()
        return model

    def fit_critic(self, X: torch.Tensor, X_ref: torch.Tensor) -> None:
        """
        Trains the critic until the Wasserstein distance stops improving
        according to a preset relative tolerance.
        Input:
            X: a generated dataset of shape N2 where N is the minibatch size.
            X_ref: a reference dataset of shape N2.
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
                negWd = -1.0 * torch.mean(self.wasserstein(X_ref, X))
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

    def wasserstein(
        self, X_ref: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the Wasserstein distance contribution from each datum in the
        batch of generated samples Q.
        Input:
            X_ref: a reference dataset of shape N2 where N is the batch size.
            X: a dataset of generated samples of shape N2.
        Returns:
            The Wasserstein distance contribution from each datum in X.
        """
        return torch.mean(self.critic(X_ref)) - torch.squeeze(
            self.critic(X), dim=-1
        )

    def alpha(self, X_ref: torch.Tensor) -> float:
        """
        Returns the optimal value of alpha according to the dual optimization
        problem.
        Input:
            X_ref: a tensor of reference samples from nature.
        Returns:
            The optimal value of alpha as a float.
        """
        if self.constant is not None:
            return self.constant
        alphas = np.linspace(0.0, 1.0, num=201)
        return alphas[np.argmax([self._score(a, X_ref) for a in alphas])]

    def _score(self, alpha: float, X_ref: torch.Tensor) -> float:
        """
        Scores a particular value of alpha according to the Lagrange dual
        function g(alpha).
        Input:
            alpha: the particular value of alpha to score.
            X_ref: a tensor of reference samples from nature.
        Returns:
            g(alpha).
        """
        Xstar = self._search(alpha).detach()
        score = ((alpha - 1.0) * self.surrogate(Xstar)) + (
            alpha * self.wasserstein(X_ref, torch.unsqueeze(Xstar, dim=0))
        )
        return score.item()

    def _search(self, alpha: float, budget: int = 4096) -> torch.Tensor:
        """
        Approximates z* for the Lagrange dual function by searching over
        the standard normal distribution.
        Input:
            alpha: value of alpha to find z* for.
            budget: number of samples to sample from the normal distribution.
        Returns:
            The optimal z* from the sampled latent space points.
        """
        X = torch.randn((budget, 2))
        X = unnormalize(X.detach(), bounds=self.bounds).requires_grad_(True)
        self.surrogate(X).backward(torch.ones((budget, 1)))
        Df = X.grad
        X.requires_grad_(True)
        self.critic(X).backward(torch.ones((budget, 1)))
        Dc = X.grad
        L = ((alpha - 1.0) * Df) - (alpha * Dc)
        return X[torch.argmin(torch.linalg.norm(L, dim=-1))]


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
