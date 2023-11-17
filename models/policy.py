"""
Generative adversarial Bayesian optimization (GABO) implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import botorch
from pathlib import Path
from typing import Optional, Tuple, Union

sys.path.append(".")
from models.turbostate import TurboState
from models.dual import Alpha


class BOPolicy:
    """Implements a Bayesian optimization policy for latent space sampling."""

    def __init__(
        self,
        maximize: bool,
        num_restarts: int = 10,
        raw_samples: int = 512,
        device: torch.device = torch.device("cpu"),
        **kwargs
    ):
        """
        Args:
            maximize: whether we are maximizing an objective versus minimizing
                a cost.
            num_restarts: number of starting points for multistart acquisition
                function optimization.
            raw_samples: number of samples for initialization.
            device: device. Default CPU.
        """
        self.maximize = maximize
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.device = device
        self.state = TurboState(**kwargs)
        self.init_region_size = 6
        self.eps = torch.finfo(torch.float64).eps

    def __call__(
        self,
        model: botorch.models.model.Model,
        z: torch.Tensor,
        y: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Generate a set of candidates with the trust region via multi-start
        optimization.
        Input:
            model: a single-task variational GP model.
            z: prior observations of digits in the autoencoder latent space.
            y: objective values of the prior observations of digits.
            batch_size: number of candidates to return.
        """
        z_center = z[torch.argmax(y), :].clone()
        tr_lb = z_center - (self.init_region_size * self.state.length / 2)
        tr_lb = torch.maximum(tr_lb, torch.full_like(tr_lb, self.eps))
        tr_ub = z_center + (self.init_region_size * self.state.length / 2)
        tr_ub = torch.minimum(tr_ub, torch.full_like(tr_ub, 1.0 - self.eps))
        z_next, _ = botorch.optim.optimize_acqf(
            botorch.acquisition.qExpectedImprovement(
                model, torch.max(y), maximize=self.maximize
            ),
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return z_next

    def update_state(self, y: torch.Tensor) -> None:
        """
        Updates the state internal variables given objective values y.
        Input:
            y: input objective values.
        Returns:
           None.
        """
        return self.state.update(y)

    @property
    def restart_triggered(self) -> bool:
        """
        Returns whether a restart has been triggered during the optimization.
        Input:
            None.
        Returns:
            Whether a restart has been triggered during the optimization.
        """
        return self.state.restart_triggered


class BOAdversarialPolicy(BOPolicy):
    def __init__(
        self,
        maximize: bool,
        ref_dataset: torch.utils.data.Dataset,
        surrogate: Union[nn.Module, pl.LightningModule],
        autoencoder: nn.Module,
        alpha: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
        num_restarts: int = 10,
        raw_samples: int = 512,
        critic_config: Optional[Union[Path, str]] = None,
        verbose: bool = True,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            maximize: whether we are maximizing an objective versus minimizing
                a cost.
            ref_dataset: a reference dataset of real digit images.
            autoencoder: trained encoder model with both encode and decode
                methods implemented.
            alpha: if specified, a constant value of alpha is used.
            surrogate: surrogate function for objective estimation. Only
                required if the alpha argument is `Lipschitz`.
            device: device. Default CPU.
            num_restarts: number of starting points for multistart acquisition
                function optimization.
            raw_samples: number of samples for initialization.
            critic_config: JSON file with source critic hyperparameters.
            verbose: whether to print verbose outputs to `stdout`.
            seed: random seed. Default 42.
        """
        super().__init__(maximize, num_restarts, raw_samples, device, **kwargs)
        self.ref_dataset = ref_dataset
        self.autoencoder = autoencoder.to(self.device)
        self.surrogate = surrogate
        self.alpha = Alpha(
            self.surrogate,
            constant=alpha,
            verbose=verbose,
            maximize_surrogate=maximize
        )
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)

    def penalize(
        self, y: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, float]]:
        """
        Penalizes the input objective values y.
        Input:
            y: input objective values.
            z: input generated digts in the latent space of the autoencoder
                corresponding to the input objective values.
        Returns:
            A tuple of the penalized objective values and the associated
            correction factors.
        """
        if self.alpha_ == 0.0:
            return y, self.alpha_
        zref = self.reference_sample(z.size(dim=0)).to(z.device)
        Wd = torch.mean(self.critic(zref)) - self.critic(z)
        Wd = torch.maximum(Wd, torch.zeros_like(y))
        if self.maximize:
            Wd = -Wd
        if self.alpha_ == 1.0:
            return Wd, self.alpha_
        alpha = self.corr_factor(z)
        penalized_objective = torch.squeeze(y, dim=-1) + (
            alpha * torch.squeeze(Wd, dim=-1)
        )
        return torch.unsqueeze(penalized_objective, dim=-1), alpha

    def reference_sample(self, num: int) -> torch.Tensor:
        """
        Samples a batch of random datums from a reference dataset.
        Input:
            num: number of datums to sample from the reference dataset.
        Returns:
            A batch of real examples from the reference dataset encoded in
            the autoencoder latent space.
        """
        raise NotImplementedError
