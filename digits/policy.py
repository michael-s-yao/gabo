"""
Sampler policy for MNIST generative adversarial Bayesian optimization (GABO).

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import torch
import botorch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Union

sys.path.append(".")
from models.policy import BOAdversarialPolicy
from models.convae import ConvAutoEncLightningModule
from digits.surrogate import SurrogateObjective


class MNISTAdversarialPolicy(BOAdversarialPolicy):
    """Generative adversarial Bayesian optimization policy for digit images."""

    def __init__(
        self,
        ref_dataset: torch.utils.data.Dataset,
        autoencoder: ConvAutoEncLightningModule,
        alpha: Union[float, str],
        surrogate: SurrogateObjective,
        device: torch.device = torch.device("cpu"),
        num_restarts: int = 10,
        raw_samples: int = 512,
        critic_config: Union[Path, str] = "./digits/critic_config.json",
        verbose: bool = True,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            ref_dataset: a reference dataset of real digit images.
            autoencoder: trained convolutional autoencoder.
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
        super().__init__(
            maximize=True,
            ref_dataset=ref_dataset,
            autoencoder=autoencoder,
            alpha=alpha,
            surrogate=surrogate,
            device=device,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            critic_config=critic_config,
            verbose=verbose,
            seed=seed
        )
        Xdummy, y = self.ref_dataset[0]
        self.x_dim = Xdummy.size()
        _, self.z_dim = self.encode(
            torch.unsqueeze(Xdummy.to(self.device), dim=0)
        )

    def reference_sample(self, num: int) -> torch.Tensor:
        """
        Samples a batch of random real digit images from a reference dataset.
        Input:
            num: number of images to sample from the reference dataset.
        Returns:
            A batch of real digit images from the reference dataset.
        """
        idxs = self.rng.choice(len(self.ref_dataset), num, replace=False)
        Xp = [
            self.ref_dataset[int(i)][0].unsqueeze(dim=0).to(self.device)
            for i in idxs
        ]
        return torch.cat(Xp, dim=0).flatten(start_dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a tensor of autoencoder latent space represntations to image
        space representations.
        Input:
            z: a tensor of the encoded digits in the autoencoder latent space.
        Returns:
            A tensor of the digits in image space with shape BCHW.
        """
        return self.autoencoder.decode(z.reshape(-1, *self.z_dim))

    def encode(self, X: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int]]:
        """
        Encodes a batch of images into the autoencoder latent space.
        Input:
            X: a tensor of the digits in image space to encode with shape BCHW.
        Returns:
            z: a tensor of the encoded digits in the autoencoder latent space.
            z_dim: the shape of the latent space representation as a tuple of
                CHW to feed into the decoder.
        """
        z = self.autoencoder.encode(X)
        z_dim = tuple(z.size())[1:]
        return z.reshape(z.size(dim=0), -1), z_dim

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
        with tqdm(
            range(self.critic_config["max_steps"]),
            desc="Training Source Critic",
            leave=False,
            disable=(not self.verbose)
        ) as pbar:
            for _ in pbar:
                Xp = self.reference_sample(self.critic_config["batch_size"])
                Xp = Xp.to(self.device)
                Zq = self(model, z, y, 8 * self.critic_config["batch_size"])
                Xq = self.decode(Zq.to(self.device)).flatten(start_dim=1)
                self.critic.zero_grad()
                negWd = torch.mean(self.critic(Xq)) - torch.mean(
                    self.critic(Xp)
                )
                negWd.backward()
                self.critic_optimizer.step()
                self.clipper(self.critic)
                pbar.set_postfix(Wd=-negWd.item())
        return
