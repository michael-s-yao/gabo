"""
Defines and implements a fully-connected VAE model.

Author(s):
    Michael Yao @michael-s-yao

Adapted from the @pytorch examples GitHub repository at
https://github.com/pytorch/examples/blob/main/vae/main.py

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple, Union


class VAE(nn.Module):
    def __init__(
        self,
        in_dim: int = 784,
        hidden_dims: Sequence[int] = [256, 64, 16],
        **kwargs
    ):
        """
        Args:
            in_dim: number of flattened input dimensions into the VAE.
            hidden_dims: hidden dimensions of the encoder and decoder.
        """
        super().__init__()
        self.in_dim, self.hidden_dims = in_dim, hidden_dims
        self.hidden_dims = [self.in_dim] + self.hidden_dims
        self.latent_size = self.hidden_dims[-1]

        self.encoder, self.decoder = [], []
        for i in range(len(self.hidden_dims) - 1):
            if i < len(self.hidden_dims) - 2:
                self.encoder += [
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                    nn.LeakyReLU(negative_slope=0.2)
                ]
            else:
                self.mu = nn.Linear(
                    self.hidden_dims[i], self.hidden_dims[i + 1]
                )
                self.logvar = nn.Linear(
                    self.hidden_dims[i], self.hidden_dims[i + 1]
                )
            self.decoder += [
                nn.Linear(self.hidden_dims[-i - 1], self.hidden_dims[-i - 2]),
                (
                    nn.LeakyReLU(negative_slope=0.2)
                    if i < len(self.hidden_dims) - 2
                    else nn.Identity()
                )
            ]
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

    def encode(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Encodes an input design into the VAE latent space.
        Input:
            X: an input design or batch of designs.
        Returns:
            z: a vector of point(s) from the VAE latent space (N or BN), where
                N is the dimensions of the VAE latent space.
            mu: tensor of means in the latent space (N or BN).
            logvar: tensor of log variances in the latent space (N or BN).
        """
        h = self.encoder(X)
        mu, logvar = self.mu(h), self.logvar(h)
        return self.reparameterize(mu, logvar), mu, logvar

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Reconstructs a tensor of point(s) from the VAE latent space into the
        flattened design space.
        Input:
            z: a vector of point(s) from the VAE latent space (N or BN), where
                N is the dimensions of the VAE latent space.
        Returns:
            Rconstructed flattened design or batch of designs.
        """
        return self.decoder(z)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparametrization trick to sample from the VAE latent space.
        Input:
            mu: tensor of means in the latent space (N or BN), where N is the
                dimensions of the VAE latent space.
            logvar: tensor of log variances in the latent space (N or BN),
                where N is the dimensions of the VAE latent space.
        Returns:
            A vector of point(s) from the VAE latent space (N or BN).
        """
        std = torch.exp(0.5 * logvar)
        return mu + (torch.randn_like(std) * std)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass through the variational autoencoder.
        Input:
            X: an input design or batch of designs.
        Returns:
            logits: logits of the reconstructed design or batch of designs.
            mu: tensor of means in the latent space (N or BN), where N is the
                dimensions of the VAE latent space.
            logvar: tensor of log variances in the latent space (N or BN),
                where N is the dimensions of the VAE latent space.
        """
        z, mu, logvar = self.encode(X.view(-1, self.in_dim))
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(
        self,
        n: Optional[int] = 1,
        z: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Samples and decodes n datums from the VAE latent space distribution.
        Input:
            n: number of points to sample.
            z: optional specified latent space points to decode.
            return_logits: whether to return the decoder logits in addition
                to the sampled molecules. Default False.
        Returns:
            sample: sampled decoded designs.
            logits: logits that are returned if `return_logits` is True.
        """
        model_state = self.training
        self.eval()

        if z is None:
            z = torch.randn((n, self.latent_size))
        else:
            n = z.shape[0] if z.ndim > 1 else 1
        logits = self.decode(z)
        sample = torch.sigmoid(logits)

        self.train(model_state)
        if return_logits:
            return sample, logits
        return sample


class IdentityVAE(nn.Module):
    """
    A dummy VAE class where the encoding and decoding layers are the identity
    function.
    """

    def __init__(self, in_dim: int, **kwargs):
        """
        Args:
            in_dim: number of flattened input dimensions into the VAE.
        """
        super().__init__()
        self.in_dim, self.latent_size = in_dim, in_dim
        self.encoder, self.decoder = nn.Identity(), nn.Identity()

    def encode(self, X: torch.Tensor) -> Tuple[Optional[torch.Tensor]]:
        """
        Encodes an input design into the VAE latent space.
        Input:
            X: an input design or batch of designs.
        Returns:
            X: the same unaltered batch of input designs.
            mu: None.
            logvar: None.
        """
        return self.encoder(X), None, None

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Reconstructs a tensor of point(s) from the VAE latent space into the
        flattened design space.
        Input:
            z: an input design or batch of designs.
        Returns:
            The same unaltered batch of input designs.
        """
        return self.decoder(z)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass through the model.
        Input:
            X: an input design or batch of designs.
        Returns:
            X: the same batch of input designs with flattened shape.
            mu: None.
            logvar: None.
        """
        z, mu, logvar = self.encode(X.view(-1, X.size(dim=-1)))
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Samples and decodes n datums from the VAE latent space distribution.
        Input:
            z: optional specified latent space points to decode.
        Returns:
            sample: sampled decoded designs.
        """
        return self.decode(z)
