"""
Generative adversarial Bayesian optimization (GABO) implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import botorch
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union

sys.path.append(".")
from models.turbostate import TurboState
from models.fcnn import FCNN
from models.critic import WeightClipper
from models.lipschitz import Lipschitz


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
        tr_ub = z_center + (self.init_region_size * self.state.length / 2)
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
        autoencoder: nn.Module,
        alpha: Union[float, str],
        surrogate: Optional[Union[nn.Module, pl.LightningModule]] = None,
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
            alpha: a float between 0 and 1, or `Lipschitz` for our method.
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
        self.alpha_ = alpha
        if self.alpha_.replace(".", "", 1).isnumeric():
            self.alpha_ = float(self.alpha_)
        self.surrogate = surrogate
        self.critic_config = {}
        if critic_config is not None:
            with open(critic_config, "rb") as f:
                self.critic_config = json.load(f)
        self.verbose = verbose
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)

        if isinstance(self.alpha_, str) or self.alpha_ > 0.0:
            self.critic = FCNN(
                in_dim=self.critic_config["in_dim"],
                out_dim=1,
                hidden_dims=self.critic_config["hidden_dims"],
                dropout=0.0,
                final_activation=None,
                hidden_activation="ReLU"
            )
            self.clipper = WeightClipper(c=self.critic_config["c"])
            self.critic_optimizer = self.configure_critic_optimizers()
        if isinstance(self.alpha_, str):
            self.L = Lipschitz(self.surrogate, mode="local", p=2)
            self.K = Lipschitz(self.critic, mode="global")

    def update_state(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Updates the state internal variables given objective values y.
        Input:
            y: input objective values.
            z: input generated digts in the latent space of the autoencoder
                corresponding to the input objective values.
        Returns:
            A tensor of the penalized objective values.
        """
        if self.alpha_ == 0.0:
            return y
        zref = self.reference_sample(z.size(dim=0))
        Wd = torch.mean(self.critic(zref)) - self.critic(z)
        Wd = torch.maximum(Wd, torch.zeros_like(y))
        if self.alpha_ == 1.0:
            return Wd
        penalized_objective = torch.squeeze(y, dim=-1) - (
            self.corr_factor(z) * torch.squeeze(Wd, dim=-1)
        )
        penalized_objective = torch.unsqueeze(penalized_objective, dim=-1)
        self.state.update(penalized_objective)
        return penalized_objective

    def corr_factor(self, Zq: torch.Tensor) -> Union[float, torch.Tensor]:
        """
        Calculates the correction factor for regularization weighting.
        Input:
            Xq: generated digits in the latent space of the autoencoder.
        Returns:
            The value(s) of alpha for regularization weighting.
        """
        if isinstance(self.alpha_, float):
            return self.alpha_ / (1.0 - self.alpha_)
        return torch.from_numpy(self.L(Zq) / self.K()).to(Zq)

    def update_critic(
        self,
        model: botorch.models.model.Model,
        z: torch.Tensor,
        y: torch.Tensor
    ) -> None:
        """
        Trains the source critic according to the allocated training budget.
        Input:
            model: a single-task variational GP model.
            z: prior observations in the autoencoder latent space.
            y: objective values of the prior latent space observations.
        Returns:
            None.
        """
        if isinstance(self.alpha_, float) and self.alpha_ == 0.0:
            return
        for _ in tqdm(
            range(self.critic_config["max_steps"]),
            desc="Training Source Critic",
            leave=False,
            disable=(not self.verbose)
        ):
            Zp = self.reference_sample(self.critic_config["batch_size"])
            Zq = self(model, z, y, batch_size=self.critic_config["batch_size"])
            self.critic.zero_grad()
            negWd = torch.mean(self.critic(Zq)) - torch.mean(self.critic(Zp))
            negWd.backward()
            self.critic_optimizer.step()
            self.clipper(self.critic)
        return

    def configure_critic_optimizers(self) -> optim.Optimizer:
        """
        Returns the optimizer for the source critic.
        Input:
            None.
        Returns:
            The optimizer for the source critic.
        """
        return optim.SGD(self.critic.parameters(), lr=self.critic_config["lr"])

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
