"""
Defines and implements a sequential VAE model for 1D sequences.

Author(s):
    Michael Yao @michael-s-yao

Adapted from the design-baselines GitHub repository by @brandontrabucco at
https://github.com/brandontrabucco/design-baselines

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

sys.path.append(".")
from models.block import ResCNNBlock
from models.pe import PositionalEncoding


class SequentialVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        vocab_size: int,
        hidden_size: int = 64,
        latent_size: int = 16,
        activation: str = "ReLU",
        kernel_size: int = 3,
        num_layers: int = 4,
        **kwargs
    ):
        """
        Args:
            in_dim: length of the input sequences.
            vocab_size: size of the vocabulary for each token.
            hidden_size: number of embedding dimensions.
            latent_size: number of latent space dimensions for the VAE.
            activation: nonlinear activation function.
            kernel_size: kernel size for the residual layers.
            num_layers: number of residual layer blocks.
        """
        super().__init__()
        self.in_dim = in_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.activation = activation
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.encoder = SequentialVAEEncoder(
            in_dim=self.in_dim,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            activation=self.activation,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers
        )

        self.decoder = SequentialVAEDecoder(
            in_dim=self.in_dim,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            activation=self.activation,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers
        )

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass through the VAE.
        Input:
            seq: a sequence or batch of sequences each of length in_dim.
        Returns:
            logits: the reconstructed sequence logits of shape (B)NC, where B
                is the batch size, N is the sequence length, and C is the
                number of classes.
            mu: the multidimensional mean of the latent space point(s).
            logvar: the log of the variance of the latent space point(s).
        """
        z, mu, logvar = self.encode(seq)
        return self.decode(z), mu, logvar

    def encode(self, seq: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Encodes a sequence(s) into the latent space of the VAE.
        Input:
            seq: a sequence or batch of sequences each of length in_dim.
        Returns:
            z: the latent space representation(s) of the input(s).
            mu: the multidimensional mean of the latent space point(s).
            logvar: the log of the variance of the latent space point(s).
        """
        return self.encoder(seq.to(torch.int32))

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decodes latent space point(s) into sequence logits.
        Input:
            z: input latent space point(s).
        Returns:
            logits: the reconstructed sequence logits of shape (B)NC, where B
                is the batch size, N is the sequence length, and C is the
                number of classes.
        """
        return self.decoder(z)

    @torch.no_grad()
    def sample(
        self,
        n: Optional[int] = 1,
        z: Optional[torch.Tensor] = None,
        differentiable: bool = False,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Samples and decodes n datums from the VAE latent space distribution.
        Input:
            n: number of points to sample.
            z: optional specified latent space points to decode.
            differentiable: whether function outputs should be differentiable.
            return_logits: whether to return the decoder logits in addition
                to the sampled designs. Default False.
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
        sample = torch.exp(F.log_softmax(logits, dim=-1))

        self.train(model_state)
        if not differentiable:
            sample = torch.argmax(sample, dim=-1)
        if return_logits:
            return sample, logits
        return sample


class SequentialVAEEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        vocab_size: int = 4,
        hidden_size: int = 64,
        latent_size: int = 16,
        activation: str = "ReLU",
        kernel_size: int = 3,
        num_layers: int = 4
    ):
        """
        Args:
            in_dim: length of the input sequences.
            vocab_size: size of the vocabulary for each token.
            hidden_size: number of embedding dimensions.
            latent_size: number of latent space dimensions for the VAE.
            activation: nonlinear activation function.
            kernel_size: kernel size for the residual layers.
            num_layers: number of residual layer blocks.
        """
        super().__init__()
        self.in_dim = in_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.activation = activation
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=self.hidden_size
        )
        self.pe = PositionalEncoding(model_dim=self.hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size)

        self.encoder = []
        for block_idx in range(self.num_layers):
            if block_idx > 0:
                self.encoder.append(nn.AvgPool1d(kernel_size=2))
            self.encoder.append(
                ResCNNBlock(
                    self.hidden_size // 2 ** block_idx,
                    self.hidden_size // 2 ** block_idx,
                    kernel_size=self.kernel_size,
                    activation=self.activation
                )
            )
        self.encoder = nn.Sequential(*self.encoder)

        self.mu = nn.Linear(
            self.in_dim * self.hidden_size // 2 ** (self.num_layers - 1),
            self.latent_size
        )
        self.logvar = nn.Linear(
            self.in_dim * self.hidden_size // 2 ** (self.num_layers - 1),
            self.latent_size
        )

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass through the sequential VAE encoder model.
        Input:
            seq: a sequence or batch of sequences each of length in_dim.
        Returns:
            z: the latent space representation(s) of the input(s).
            mu: the multidimensional mean of the latent space point(s).
            logvar: the log of the variance of the latent space point(s).
        """
        hidden = self.encoder(self.layer_norm(self.pe(self.embed(seq))))
        hidden = hidden.flatten(start_dim=(seq.ndim - 1))
        mu, logvar = self.mu(hidden), self.logvar(hidden)
        z = mu + (torch.randn_like(mu) * torch.exp(-0.5 * logvar))
        return z, mu, logvar


class SequentialVAEDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        vocab_size: int = 4,
        hidden_size: int = 64,
        latent_size: int = 16,
        activation: str = "ReLU",
        kernel_size: int = 3,
        num_layers: int = 4
    ):
        """
        Args:
            in_dim: length of the input sequences.
            vocab_size: size of the vocabulary for each token.
            hidden_size: number of embedding dimensions.
            latent_size: number of latent space dimensions for the VAE.
            activation: nonlinear activation function.
            kernel_size: kernel size for the residual layers.
            num_layers: number of residual layer blocks.
        """
        super().__init__()
        self.in_dim = in_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.activation = activation
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.fc_input = nn.Linear(
            self.latent_size,
            self.in_dim * self.hidden_size // 2 ** (self.num_layers - 1)
        )
        self.decoder = []
        for block_idx in reversed(range(self.num_layers)):
            if block_idx > 0:
                self.decoder.append(AvgUnpool1d(kernel_size=2))
            num_features = int(self.hidden_size // 2 ** max(block_idx - 1, 0))
            self.decoder += [
                PositionalEncoding(model_dim=num_features),
                nn.LayerNorm(normalized_shape=num_features),
                ResCNNBlock(
                    num_features,
                    num_features,
                    kernel_size=self.kernel_size,
                    activation=self.activation
                )
            ]
        self.decoder = nn.Sequential(*self.decoder)
        self.fc_output = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass through the sequential VAE decoder model.
        Input:
            z: input latent space point(s).
        Returns:
            logits: the reconstructed sequence logits of shape (B)NC, where B
                is the batch size, N is the sequence length, and C is the
                number of classes.
        """
        hidden = self.fc_input(z).reshape(
            -1, self.in_dim, self.hidden_size // 2 ** (self.num_layers - 1)
        )
        return self.fc_output(self.decoder(hidden))


class AvgUnpool1d(nn.Module):
    def __init__(self, kernel_size: int):
        """
        Args:
            kernel_size: the size of the window.
        """
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the unpooling module.
        Input:
            X: an input 1D signal or batch of 1D signals of shape (B)CN.
        Returns:
            The unpooled signal of shape (B)C(self.kernel_size * N).
        """
        X = X.unsqueeze(dim=-1).expand(*([-1] * X.ndim), self.kernel_size)
        return X.flatten(start_dim=-2)
