"""
Sampler policy for MNIST generative adversarial Bayesian optimization (GABO).

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import sys
import torch
import torch.nn as nn
from math import isclose
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood
)
from tqdm import tqdm
from typing import Optional

sys.path.append(".")
from models.fcnn import FCNN
from models.clamp import WeightClipper


class MNISTPolicy:
    def __init__(
        self,
        bounds: torch.Tensor,
        ref_dataset: torch.utils.data.Dataset,
        surrogate: nn.Module,
        alpha: Optional[float] = None,
        batch_size: int = 16,
        z_dim: int = 16,
        device: torch.device = torch.device("cpu"),
        num_restarts: int = 10,
        raw_samples: int = 256,
        patience: int = 100,
        verbose: bool = True
    ):
        """
        Args:
            bounds: a 2xN tensor of the lower and upper sampling bounds.
            ref_dataset: a reference dataset of true datums from nature.
            surrogate: a trained surrogate estimator for the objective.
            alpha: optional constant value for alpha.
            batch_size: batch size. Default 16.
            z_dim: dimensions of the VAE latent space. Default 64.
            device: device. Default CPU.
            num_restarts: number of starting points for multistart acquisition
                function optimization.
            raw_samples: number of samples for initialization.
            patience: patience used for source critic training. Default 100.
            verbose: whether to print verbose outputs to `stdout`.
        """
        self.bounds = bounds
        self.ref_dataset = ref_dataset
        self.surrogate = surrogate
        self.constant = alpha
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.device = device
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.patience = patience
        self.verbose = verbose
        self.model, self.state_dict = None, None

        self.critic = FCNN(
            in_dim=self.z_dim,
            out_dim=1,
            hidden_dims=[4 * self.z_dim, self.z_dim],
            dropout=0.0,
            final_activation=None,
            hidden_activation="ReLU",
            use_batch_norm=False
        )
        self.critic = self.critic.to(self.device)
        self.critic.eval()
        self.clipper = WeightClipper(c=0.05)
        self.optimizer = torch.optim.SGD(self.critic.parameters(), lr=1e-3)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """
        Optimize the acquisition function and return a set of new candidates
        in the VAE latent space to sample.
        Input:
            y: previous observations of the surrogate objective.
        Returns:
            A batch of new latent space points to sample.
        """
        # Optimize the acquisition function.
        candidates, _ = optimize_acqf(
            acq_function=qExpectedImprovement(
                model=self.model, best_f=torch.max(y)
            ),
            bounds=torch.stack(
                [
                    torch.zeros(self.z_dim, device=self.device),
                    torch.ones(self.z_dim, device=self.device),
                ]
            ),
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        # Return new values to sample from the VAE latent space.
        return unnormalize(candidates.detach(), bounds=self.bounds)

    def fit(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit a SingleTaskGP model to a set of input observations.
        Input:
            z: previous observations from the VAE latent space.
            y: corresponding observations of the surrogate objective.
        Returns:
            None.
        """
        model = SingleTaskGP(
            train_X=normalize(z, self.bounds),
            train_Y=y,
            outcome_transform=Standardize(m=1)
        )
        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(z)
        fit_gpytorch_mll(mll)
        self.model = model

    def save_current_state_dict(self) -> None:
        """
        Save the current model state dict.
        Input:
            None.
        Returns:
            None.
        """
        self.state_dict = self.model.state_dict()

    def fit_critic(self, z: torch.Tensor, z_ref: torch.Tensor) -> None:
        """
        Trains the critic until the Wasserstein distance stops improving
        according to a preset relative tolerance.
        Input:
            z: a generated dataset of shape BN, where B is the batch size and
                N is the VAE latent space dimension.
            z_ref: a reference dataset of shape BN, where B is the batch size
                and N is the VAE latent space dimension.
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
                negWd = -1.0 * torch.mean(self.wasserstein(z_ref, z))
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
        return torch.mean(self.critic(P)) - torch.squeeze(
            self.critic(Q), dim=-1
        )

    def reference_sample(self, n: int) -> torch.Tensor:
        """
        Samples a batch of reference images from the reference dataset.
        Input:
            n: number of reference images to sample.
        Returns:
            A batch of reference images with shape nCHW.
        """
        idxs = torch.randint(len(self.ref_dataset), (n,))
        X = torch.cat([self.ref_dataset[i][0] for i in idxs], dim=0)
        return X.to(self.device)

    def alpha(self, z_ref: torch.Tensor) -> float:
        """
        Returns the optimal value of alpha according to the dual optimization
        problem.
        Input:
            z_ref: a tensor of reference true samples.
        Returns:
            The optimal value of alpha as a float.
        """
        if self.constant is not None:
            return self.constant
        alpha = torch.from_numpy(np.linspace(0.0, 1.0, num=201))
        alpha = alpha.to(self.device)
        return alpha[torch.argmax(self._score(alpha, z_ref))]

    def _score(self, alpha: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
        """
        Scores alpha values according to the Lagrange dual function g(alpha).
        Input:
            alpha: the particular values of alpha to score.
            z_ref: a tensor of reference true samples.
        Returns:
            g(alpha) as a tensor with the same shape as the alpha input.
        """
        zstar = self._search(alpha).detach()
        return torch.where(
            torch.linalg.norm(zstar, dim=-1) > 0.0,
            ((alpha - 1.0) * self.surrogate(zstar).squeeze(dim=-1)) + (
                alpha * self.wasserstein(z_ref, zstar)
            ),
            -1e12
        )

    def _search(
        self, alpha: torch.Tensor, budget: int = 4096, thresh: float = 1e-4
    ) -> torch.Tensor:
        """
        Approximates z* for the Lagrange dual function by searching over
        the standard normal distribution.
        Input:
            alpha: values of alpha to find z* for.
            budget: number of samples to sample from the normal distribution.
            thresh: threshold value the norm of the Lagrangian gradient.
        Returns:
            The optimal z* from the sampled latent space points. The returned
            tensor has shape AD, where A is the number of values of alpha
            tested and D is the dimensions of the latent space points.
        """
        alpha = alpha.repeat(budget, self.z_dim, 1).permute(2, 0, 1)
        z = torch.randn((budget, self.z_dim)).to(self.device)
        z = unnormalize(z.detach(), bounds=self.bounds).requires_grad_(True)
        Df = torch.autograd.grad(self.surrogate(z).sum(), z)[0]
        z.requires_grad_(True)
        Dc = torch.autograd.grad(self.critic(z).sum(), z)[0]
        DL = ((alpha - 1.0) * Df) - (alpha * Dc)
        norm_DL, best_idxs = torch.min(torch.linalg.norm(DL, dim=-1), dim=-1)
        z = z[best_idxs]
        best_idxs = torch.where(norm_DL < thresh, best_idxs, -1)
        with torch.no_grad():
            for bad_idx in torch.where(best_idxs < 0)[0]:
                z[bad_idx] = torch.zeros_like(z[bad_idx])
        return z


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
