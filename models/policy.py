"""
Sampler policy for conservative offline model-based optimization over latent
spaces via source critic regularization (COMBO-SCR). Our method estimates the
Lagrange multiplier through solving the dual problem of the primal optimization
task.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from collections import namedtuple
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood
)
from tqdm import tqdm
from typing import Optional

sys.path.append(".")
from models.fcnn import FCNN
from models.clamp import WeightClipper


class COMBOSCRPolicy:
    def __init__(
        self,
        task_name: str,
        z_dim: int,
        bounds: torch.Tensor,
        ref_dataset: torch.utils.data.Dataset,
        surrogate: nn.Module,
        alpha: Optional[float] = None,
        batch_size: int = 16,
        c: float = 0.01,
        critic_lr: float = 0.001,
        critic_patience: int = 100,
        device: torch.device = torch.device("cpu"),
        search_budget: int = 4096,
        norm_thresh: float = 1e-3,
        num_restarts: int = 10,
        raw_samples: int = 256,
        patience: int = 100,
        verbose: bool = True,
        **kwargs
    ):
        """
        Args:
            task_name: name of the MBO task.
            z_dim: input dimensions.
            bounds: a 2x(z_dim) tensor of the lower and upper sampling bounds.
            ref_dataset: a reference dataset of true datums from nature.
            surrogate: a trained surrogate estimator for the objective.
            alpha: optional constant value for alpha.
            batch_size: batch size. Default 16.
            c: weight clipping parameter. Default 0.01.
            critic_lr: source critic learning rate. Default 0.001.
            critic_patience: patience to determine source critic convergence.
                Default 100.
            device: device. Default CPU.
            search_budget: number of samples to sample from the normal
                distribution in solving for alpha.
            norm_thresh: threshold value the norm of the Lagrangian gradient.
            num_restarts: number of starting points for multistart acquisition
                function optimization.
            raw_samples: number of samples for initialization.
            patience: patience used for source critic training. Default 100.
            verbose: whether to print verbose outputs to `stdout`.
        """
        self.hparams = locals()
        self.save_hyperparameters()
        self.model, self.state_dict = None, None
        self.NINF = -1e12

        self.critic = FCNN(
            in_dim=self.hparams.z_dim,
            out_dim=1,
            hidden_dims=[4 * self.hparams.z_dim, self.hparams.z_dim],
            dropout=0.0,
            final_activation=None,
            hidden_activation="ReLU",
            use_batch_norm=False
        )
        self.critic = self.critic.to(self.hparams.device)
        self.critic.eval()
        self.clipper = WeightClipper(c=self.hparams.c)
        self.optimizer = torch.optim.SGD(
            self.critic.parameters(), lr=self.hparams.critic_lr
        )

    def save_hyperparameters(self) -> None:
        """
        Save input hyperparameters of the sampling policy.
        Input:
            None.
        Returns:
            None.
        """
        self.hparams = namedtuple("hparams", list(self.hparams.keys()))(
            **self.hparams
        )
        for hparam in self.hparams:
            if isinstance(hparam, (nn.Module, torch.Tensor)):
                hparam = hparam.to(self.hparams.device)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """
        Optimize the acquisition function and return a set of new candidates
        to sample.
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
                    torch.zeros(
                        self.hparams.z_dim, device=self.hparams.device
                    ),
                    torch.ones(self.hparams.z_dim, device=self.hparams.device),
                ]
            ),
            q=self.hparams.batch_size,
            num_restarts=self.hparams.num_restarts,
            raw_samples=self.hparams.raw_samples,
        )
        # Return new values to sample.
        return unnormalize(candidates.detach(), bounds=self.hparams.bounds)

    def fit(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit a SingleTaskGP model to a set of input observations.
        Input:
            z: previous observations.
            y: corresponding observations of the surrogate objective.
        Returns:
            None.
        """
        model = SingleTaskGP(
            train_X=normalize(z, self.hparams.bounds),
            train_Y=y,
            outcome_transform=Standardize(m=1)
        )
        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(z)
        fit_gpytorch_model(mll)
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
        cache, Wd = [-1e12] * self.hparams.patience, 0.0

        def generator():
            while not np.isclose(Wd, min(cache), rtol=1e-3) or Wd < min(cache):
                yield

        with tqdm(
            generator(),
            desc="Training Source Critic",
            disable=(not self.hparams.verbose)
        ) as pbar:
            for _ in pbar:
                self.critic.zero_grad()
                negWd = -1.0 * torch.mean(self.wasserstein(z_ref, z))
                negWd.backward()
                self.optimizer.step()
                self.clipper(self.critic)
                num_steps += 1
                Wd = -negWd.item()
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
        P, Q = P.reshape(P.size(dim=0), -1), Q.reshape(Q.size(dim=0), -1)
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
        idxs = torch.randint(len(self.hparams.ref_dataset), (n,))
        X = self.hparams.ref_dataset[idxs]
        return X.to(self.hparams.device, dtype=torch.float64)

    def alpha(self, z_ref: torch.Tensor, **kwargs) -> float:
        """
        Returns the optimal value of alpha according to the dual optimization
        problem.
        Input:
            z_ref: a tensor of reference true samples.
        Returns:
            The optimal value of alpha as a float.
        """
        if self.hparams.alpha is not None:
            return self.hparams.alpha
        alpha = torch.from_numpy(np.linspace(0.0, 1.0, num=201))
        alpha = alpha.to(self.hparams.device)
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
            ((alpha - 1.0) * self.hparams.surrogate(zstar).squeeze(dim=-1)) + (
                alpha * self.wasserstein(z_ref, zstar)
            ),
            self.NINF
        )

    def _search(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Approximates z* for the Lagrange dual function by searching over
        the standard normal distribution.
        Input:
            alpha: values of alpha to find z* for.
        Returns:
            The optimal z* from the sampled latent space points. The returned
            tensor has shape AD, where A is the number of values of alpha
            tested and D is the dimensions of the latent space points.
        """
        alpha = alpha.repeat(self.hparams.search_budget, self.hparams.z_dim, 1)
        alpha = alpha.permute(2, 0, 1).to(self.hparams.device)
        z = self.prior().to(self.hparams.device).detach()
        z = unnormalize(z, bounds=self.hparams.bounds).requires_grad_(True)
        Df = torch.autograd.grad(self.hparams.surrogate(z).sum(), z)[0]
        z = z.requires_grad_(True)
        Dc = torch.autograd.grad(self.critic(z).sum(), z)[0]
        DL = ((alpha - 1.0) * Df) - (alpha * Dc)
        norm_DL, best_idxs = torch.min(torch.linalg.norm(DL, dim=-1), dim=-1)
        z = z[best_idxs]
        best_idxs = torch.where(
            norm_DL < self.hparams.norm_thresh, best_idxs, -1
        )
        with torch.no_grad():
            for bad_idx in torch.where(best_idxs < 0)[0]:
                z[bad_idx] = torch.zeros_like(z[bad_idx])
        return z

    def prior(self) -> torch.Tensor:
        """
        Returns samples over the prior of the latent space distribution.
        Input:
            None.
        Returns:
            Samples from the prior of the latent space distribution.
        """
        if self.hparams.task_name == os.environ["BRANIN_TASK"]:
            return torch.rand(self.hparams.search_budget, self.hparams.z_dim)
        return torch.randn((self.hparams.search_budget, self.hparams.z_dim))
