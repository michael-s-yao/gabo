"""
Implements the sampler policy for conservative offline model-based
optimization over latent spaces via source critic regularization (COMBO-SCR)
over a conditioned design space whereby some input design dimensions are
optimized over and others are treated as frozen condition attributes.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import sys
import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood
)
from typing import Optional

sys.path.append(".")
from models.policy import COMBOSCRPolicy


class ConditionalCOMBOSCRPolicy(COMBOSCRPolicy):
    def __init__(
        self,
        grad_mask: np.ndarray,
        task_name: str,
        z_dim: int,
        bounds: torch.Tensor,
        ref_dataset: torch.utils.data.Dataset,
        surrogate: nn.Module,
        continuous_condition_idxs: Optional[np.ndarray] = None,
        discrete_relaxation: float = 1.0,
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
            grad_mask: a mask of input design dimensions that can be optimized
                over. The mask should be True for dimensions that will be
                optimized over and False for frozen condition dimensions.
            task_name: name of the MBO task.
            z_dim: input dimensions.
            bounds: a 2x(num_opt_dims) tensor of the (unstandardized) lower
                and upper sampling bounds.
            ref_dataset: a reference dataset of true datums from nature.
            surrogate: a trained surrogate estimator for the objective.
            continuous_condition_idxs: an optional array of indices that
                specifies the input dimensions that should be treated as
                continuous conditions. All other conditions are treated as
                discrete conditions.
            discrete_relaxation: relaxation parameter for discrete boolean
                variables.
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
        super(ConditionalCOMBOSCRPolicy, self).__init__(
            task_name=task_name,
            z_dim=z_dim,
            bounds=bounds,
            ref_dataset=ref_dataset,
            surrogate=surrogate,
            alpha=alpha,
            batch_size=batch_size,
            c=c,
            critic_lr=critic_lr,
            critic_patience=critic_patience,
            device=device,
            search_budget=search_budget,
            norm_thresh=norm_thresh,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            patience=patience,
            verbose=verbose,
            **kwargs
        )
        self.grad_mask = torch.from_numpy(grad_mask).to(self.hparams.device)
        self.num_opt_dims = int(torch.sum(self.grad_mask).item())
        self.discrete_relaxation = discrete_relaxation
        self.continuous_condition_idxs = continuous_condition_idxs

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """
        Optimize the acquisition function and return a set of new candidates
        to sample.
        Input:
            y: previous observations of the surrogate objective.
        Returns:
            A batch of new candidates points to sample.
        """
        new_z = torch.empty(
            (len(self.model), self.hparams.batch_size, self.num_opt_dims),
            device=self.hparams.device
        )
        for idx, (model, yy) in enumerate(zip(self.model, y)):
            # Optimize the acquisition function.
            candidates, _ = optimize_acqf(
                acq_function=qExpectedImprovement(
                    model=model, best_f=torch.max(yy)
                ),
                bounds=torch.stack(
                    [
                        torch.zeros(
                            self.num_opt_dims, device=self.hparams.device
                        ),
                        torch.ones(
                            self.num_opt_dims, device=self.hparams.device
                        ),
                    ]
                ),
                q=self.hparams.batch_size,
                num_restarts=self.hparams.num_restarts,
                raw_samples=self.hparams.raw_samples,
            )
            new_z[idx] = unnormalize(
                candidates.detach(),
                bounds=self.hparams.bounds[:, :self.num_opt_dims]
            )
        # Return new values to sample.
        return new_z

    def fit(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit SingleTaskGP models to a set of input observations.
        Input:
            z: previous observations.
            y: corresponding observations of the surrogate objective.
        Returns:
            None.
        """
        self.model = [
            SingleTaskGP(
                train_X=zz[..., :self.num_opt_dims],
                train_Y=yy,
                outcome_transform=Standardize(m=1)
            )
            for zz, yy in zip(normalize(z, self.hparams.bounds), y)
        ]
        if self.state_dict is not None:
            [
                model.load_state_dict(state_dict)
                for model, state_dict in zip(self.model, self.state_dict)
            ]
        for model in self.model:
            mll = ExactMarginalLogLikelihood(model.likelihood, model).to(z)
            fit_gpytorch_model(mll)

    def save_current_state_dict(self) -> None:
        """
        Save the current model state dicts.
        Input:
            None.
        Returns:
            None.
        """
        self.state_dict = [model.state_dict() for model in self.model]

    def wasserstein(
        self, z_ref: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the Wasserstein distance contribution from each datum in the
        batch of generated samples z.
        Input:
            z_ref: a generated dataset of shape BD, where B is the batch size
                and D is the number of input dimensions.
            z: a generated dataset of shape NBD, where N is the number of
                datums, B the batch size, and D the number of input dimensions.
        Returns:
            The Wasserstein distance contribution from each datum in z as a
            tensor with shape NB.
        """
        return torch.squeeze(
            torch.mean(self.critic(z_ref)) - self.critic(z), dim=-1
        )

    def alpha(self, z_ref: torch.Tensor, X: np.ndarray) -> torch.Tensor:
        """
        Returns the optimal value of alpha according to the dual optimization
        problem.
        Input:
            z_ref: a tensor of reference true samples.
            X: the dataset for which to find z* for.
        Returns:
            The optimal values of alpha for each datum in the dataset as a
            tensor with shape Nx1x1, where N is the number of datums in X.
        """
        if self.hparams.alpha is not None:
            return self.hparams.alpha
        alpha = torch.from_numpy(np.linspace(0.0, 1.0, num=201))
        alpha = alpha.to(self.hparams.device)
        scores = self._score(
            alpha,
            torch.from_numpy(X).to(self.hparams.device),
            z_ref
        )
        return alpha[torch.argmax(scores, dim=-1)].unsqueeze(dim=-1).unsqueeze(
            dim=-1
        )

    def _score(
        self, alpha: torch.Tensor, X: torch.Tensor, z_ref: torch.Tensor
    ) -> torch.Tensor:
        """
        Scores alpha values according to the Lagrange dual function g(alpha).
        Input:
            alpha: the particular values of alpha to score.
            X: the dataset for which to find z* for.
            z_ref: a tensor of reference true samples.
        Returns:
            g(alpha | X) for each datum in X. The returned tensor has shape
            NA, where N is the number of datums in X and A is the number of
            values of alpha tested.
        """
        zstar = self._search(alpha, X).detach()
        return torch.where(
            torch.linalg.norm(zstar, dim=-1) > 0.0,
            ((alpha - 1.0) * self.hparams.surrogate(zstar).squeeze(dim=-1)) + (
                alpha * self.wasserstein(z_ref, zstar)
            ),
            self.NINF
        )

    def _search(self, alpha: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Approximates z* for the Lagrange dual function by searching over
        the standard normal distribution.
        Input:
            alpha: values of alpha to find z* for.
            X: the dataset for which to find z* for.
        Returns:
            The optimal z* from the sampled latent space points. The returned
            tensor has shape NAD, where N is the number of datums from X, A
            is the number of values of alpha tested, and D is the dimensions
            of the latent space points.
        """
        zstar = torch.empty(
            (X.size(dim=0), alpha.size(dim=0), X.size(dim=-1)),
            device=self.hparams.device
        )
        alpha = alpha.repeat(
            self.hparams.search_budget, X.size(dim=-1), self.num_opt_dims
        )
        alpha = alpha.permute(2, 0, 1).to(self.hparams.device)
        for i in range(X.size(dim=0)):
            # Relax the search space over discrete variables from 0 to 1 by a
            # factor of discrete_relaxation.
            z = -self.discrete_relaxation + (
                (1.0 + 2.0 * self.discrete_relaxation) * (
                    torch.rand(self.hparams.search_budget, self.hparams.z_dim)
                )
            )
            z = z.to(self.hparams.device)

            z[:, :self.num_opt_dims] = unnormalize(
                self.prior().to(z),
                bounds=self.hparams.bounds[:, :self.num_opt_dims]
            )

            # Assumes a standard normal distribution of the continuous
            # conditional dimensions that are frozen.
            for continuous_idx in self.continuous_condition_idxs:
                z[:, continuous_idx] = (
                    self.hparams.bounds[:, continuous_idx].max() * (
                        torch.rand((self.hparams.search_budget,)).to(z)
                    )
                )

            z = z.requires_grad_(True)
            Df = torch.autograd.grad(self.hparams.surrogate(z).sum(), z)[0]
            z = z.requires_grad_(True)
            Dc = torch.autograd.grad(self.critic(z).sum(), z)[0]
            Df, Dc = self.grad_mask * Df, self.grad_mask * Dc
            DL = ((alpha - 1.0) * Df) - (alpha * Dc)
            norm_DL, best_idxs = torch.min(
                torch.linalg.norm(DL, dim=-1), dim=-1
            )
            zstar[i] = z[best_idxs]
            best_idxs = torch.where(
                norm_DL < self.hparams.norm_thresh, best_idxs, -1
            )
            with torch.no_grad():
                for bad_idx in torch.where(best_idxs < 0)[0]:
                    zstar[i, bad_idx] = torch.zeros_like(zstar[i, bad_idx])
        return zstar

    def prior(self) -> torch.Tensor:
        """
        Returns samples over the prior of the latent space distribution.
        Input:
            None.
        Returns:
            Samples from the prior of the latent space distribution.
        """
        return torch.rand((self.hparams.search_budget, self.num_opt_dims))
