"""
Sampler policy for MNIST generative adversarial Bayesian optimization (GABO).

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import numpy as np
import torch

sys.path.append(".")
from models.policy import BOPolicy
from digits.surrogate import SurrogateObjective


class MNISTPolicy(BOPolicy):
    """Latent space sampling policy for digit images."""

    def __init__(
        self,
        ref_dataset: torch.utils.data.Dataset,
        surrogate: SurrogateObjective,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            ref_dataset: a reference dataset of real digit images.
            autoencoder: trained convolutional autoencoder.
            surrogate: surrogate function for objective estimation. Only
                required if the alpha argument is `Lipschitz`.
            device: device. Default CPU.
            seed: random seed. Default 42.
        """
        super().__init__(maximize=True, device=device, **kwargs)
        self.ref_dataset = ref_dataset
        self.surrogate = surrogate
        self.autoencoder = surrogate.convae
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)

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
        return self.encode(torch.cat(Xp, dim=0))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a tensor of autoencoder latent space represntations to image
        space representations.
        Input:
            z: a tensor of the encoded digits in the autoencoder latent space.
        Returns:
            A tensor of the digits in image space with shape BCHW.
        """
        return self.autoencoder.model.decode(z)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of images into the autoencoder latent space.
        Input:
            X: a tensor of the digits in image space to encode with shape BCHW.
        Returns:
            z: a tensor of the encoded digits in the autoencoder latent space.
        """
        z, _, _ = self.autoencoder.model.encode(X)
        return z

    def oracle(self, z: torch.tensor) -> torch.Tensor:
        """
        Computes the ground truth image energy of the autoencoder latent space.
        Input:
            z: a tensor of the encoded digits in the autoencoder latent space.
        Returns:
            A tensor of the ground truth image energy values.
        """
        return torch.mean(
            torch.square(self.decode(z).flatten(start_dim=(z.ndim - 1))),
            dim=-1,
            keepdim=True
        )
