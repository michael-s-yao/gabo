"""
Estimates a Lagrange multiplier-associated variable alpha through solving the
dual problem of the primal optimization task.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import isclose
from tqdm import tqdm
from typing import Optional

sys.path.append(".")
from models.fcnn import FCNN
from models.critic import WeightClipper


class Alpha:
    def __init__(
        self,
        surrogate: nn.Module,
        critic_lr: float = 0.001,
        critic_patience: int = 100,
        constant: Optional[float] = None,
        maximize_surrogate: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            surrogate: learned objective function modeled as a neural net.
            critic_lr: learning rate for the source critic.
            critic_patience: patience in training the source critic to
                determine training convergence. Default 100.
            constant: if specified, the constant value provided is returned
                every time.
            maximize_surrogate: whether we are trying to maximize the
                surrogate objective (and hence hopefully the oracle objective).
            verbose: whether to print verbose outputs for the source critic
                training.
        """
        self.surrogate = surrogate.eval()
        self.in_dim = next(self.surrogate.model.parameters()).size(dim=-1)
        self.critic = FCNN(
            in_dim=self.in_dim,
            out_dim=1,
            hidden_dims=[4 * self.in_dim, self.in_dim],
            dropout=0.0,
            final_activation=None,
            hidden_activation="ReLU"
        )
        self.critic.eval()
        self.lr = critic_lr
        self.patience = critic_patience
        self.constant = constant
        self.maximize_surrogate = maximize_surrogate
        self.verbose = verbose

        self.clipper = WeightClipper(c=0.01)
        self.optimizer = optim.SGD(
            self.critic.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-3,
            nesterov=True
        )

    def __call__(self, P: torch.Tensor) -> float:
        """
        Returns the optimal value of alpha according to the dual optimization
        problem.
        Input:
            P: a tensor of reference true samples.
        Returns:
            The optimal value of alpha as a float.
        """
        if self.constant is not None:
            return self.constant
        alphas = self.sigspace(n=200, T=5)
        return alphas[np.argmax([self._score(a, P) for a in alphas])]

    def _score(self, alpha: float, P: torch.Tensor) -> float:
        """
        Scores a particular value of alpha according to the Lagrange dual
        function g(alpha).
        Input:
            alpha: the particular value of alpha to score.
            P: a tensor of reference true samples.
        Returns:
            g(alpha).
        """
        xstar = self._search(alpha).detach()
        sign = -1.0 if self.maximize_surrogate else 1.0
        score = (sign * (1.0 - alpha) * self.surrogate(xstar)) - (
            alpha * self._wasserstein(P, torch.unsqueeze(xstar, dim=0))
        )
        return score.item()

    def _search(self, alpha: float, budget: int = 4096) -> torch.Tensor:
        """
        Approximates x* for the Lagrange dual function by searching over
        the standard normal distribution.
        Input:
            alpha: value of alpha to find x* for.
            budget: number of samples to sample from the normal distribution.
        Returns:
            The optimal x* from the sampled latent space points.
        """
        sign = -1.0 if self.maximize_surrogate else 1.0

        x = torch.randn((budget, self.in_dim)).requires_grad_(True)
        self.surrogate(x).backward(torch.ones((budget, 1)))
        Df = sign * x.grad
        x.requires_grad_(True)
        self.critic(x).backward(torch.ones((budget, 1)))
        Dc = x.grad
        L = ((1.0 - alpha) * Df) - (alpha * Dc)
        obj = 0.5 * torch.linalg.norm(L, dim=-1) ** 2

        return x[torch.argmin(obj)]

    def fit_critic(self, P: torch.Tensor, Q: torch.Tensor) -> None:
        """
        Trains the critic using gradient descent with Nesterov's acceleration
        until the Wasserstein distance stops improving according to a preset
        relative tolerance.
        Input:
            P: a reference dataset of shape BN, where B is the batch size and
                N is the input dimension into the critic function.
            Q: a dataset of generated samples of shape BN, where B is the batch
                size and N is the input dimension into the critic function.
        Returns:
            None.
        """
        self.critic.train()
        num_steps = 0
        cache = [-1e12] * self.patience
        with tqdm(
            self._generator(),
            desc="Training Source Critic",
            disable=(not self.verbose)
        ) as pbar:
            for _ in pbar:
                self.critic.zero_grad()
                negWd = -1.0 * torch.mean(self._wasserstein(P, Q))
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

    def penalize(
        self, P: torch.Tensor, Q: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the penalty term for the surrogate objective function.
        Input:
            P: a reference dataset of shape BN, where B is the batch size and
                N is the input dimension into the critic function.
            Q: a dataset of generated samples of shape BN, where B is the batch
                size and N is the input dimension into the critic function.
        Returns:
            The penalty term to be added to the surrogate objective function.
        """
        sign = -1.0 if self.maximize_surrogate else 1.0
        return sign * self._wasserstein(P, Q)

    def _wasserstein(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
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
        return torch.mean(self.critic(P)) - torch.squeeze(
            self.critic(Q), dim=-1
        )

    def _generator(self):
        """
        Defines a generator for an infinite loop.
        Input:
            None.
        Returns:
            None.
        """
        while True:
            yield

    def sigspace(self, n: int, T: float = 1) -> torch.Tensor:
        """
        Returns a nonlinear sampling of points near 0 and 1.
        Input:
            n: number of points to sample.
            T: shape parameter. Default 1.
        Returns:
            A sorted tensor of values between 0 and 1 with increased density at
            near 0 and 1.
        """
        samples = torch.tanh(torch.from_numpy(np.linspace(-T, T, n)))
        return (samples - samples[0]) / (samples[-1] - samples[0])
