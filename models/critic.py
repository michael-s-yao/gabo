"""
Source critic network implementation.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Sequence

from models.block import Block


class Critic(nn.Module):
    """Source critic network implementation."""

    def __init__(
        self,
        x_dim: Sequence[int] = (1, 28, 28),
        intermediate_layers: Sequence[int] = (512, 256),
        use_sigmoid: bool = True
    ):
        """
        Args:
            x_dim: dimensions CHW of the input image to the source critic.
                Default MNIST dimensions (1, 28, 28).
            intermediate_layers: intermediate layer output dimensions. Default
                (512, 256).
            use_sigmoid: whether to apply sigmoid activation function as the
                final layer. Default True.
        """
        super().__init__()
        self.x_dim = x_dim
        self.intermediate_layers = intermediate_layers
        self.model = [
            (
                "layer0",
                Block(
                    in_dim=int(np.prod(self.x_dim)),
                    out_dim=self.intermediate_layers[0],
                    normalize=False,
                    activation="LeakyReLU"
                ),
            )
        ]
        for i in range(1, len(self.intermediate_layers)):
            self.model.append(
                (
                    "layer" + str(i),
                    Block(
                        in_dim=self.intermediate_layers[i - 1],
                        out_dim=self.intermediate_layers[i],
                        normalize=False,
                        activation="LeakyReLU"
                    ),
                )
            )
        if use_sigmoid:
            self.model.append(
                (
                    "layer" + str(len(self.intermediate_layers)),
                    Block(
                        in_dim=self.intermediate_layers[-1],
                        out_dim=1,
                        normalize=False,
                        activation="Sigmoid"
                    )
                )
            )

        self.model = nn.Sequential(OrderedDict(self.model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the source critic network.
        Input:
            x: input image as input to the source critic.
        Returns:
            Probability of x being from source distribution.
        """
        return torch.squeeze(self.model(x.view(x.size(0), -1)), dim=-1)


class SeqGANCritic(nn.Module):
    """
    SeqGan critic network as a bidirectional GRU.

    Citation(s):
        [1] Yu L, Zhang W, Wang J, Yu Y. SeqGAN: Sequence generative
            adversarial nets with policy gradient. AAAI. (2017).
            https://doi.org/10.48550/arXiv.1609.05473
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        embedding_dim: int,
        hidden_dim: int,
        max_molecule_length: int = 109,
        padding_token: Optional[str] = "[pad]",
        dropout: float = 0.1,
        use_sigmoid: bool = True
    ):
        """
        Args:
            vocab: dictionary of vocabulary.
            embedding_dim: the size of each embedding vector.
            hidden_dim: dimensions of the hidden vector in the RNN.
            max_molecule_length: maximum molecule length. Default 109.
            padding_token: optional padding token. Default `[pad]`.
            dropout: dropout parameter. Default 0.1.
            use_sigmoid: whether to apply sigmoid activation function
                to final output.
        """
        super().__init__()
        self.vocab = vocab
        self.num_embeddings = len(self.vocab.keys())
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_molecule_length = max_molecule_length
        self.padding_token = padding_token
        self.dropout = dropout
        self.use_sigmoid = use_sigmoid

        self.embed = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            padding_idx=self.vocab[self.padding_token]
        )
        self.rnn = nn.GRU(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=self.dropout
        )
        self.w1 = nn.Linear(2 * 2 * self.hidden_dim, hidden_dim)
        self.drop = nn.Dropout(p=self.dropout)
        self.w2 = nn.Linear(hidden_dim, 1)

    def init_hidden(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Initializes the hidden state of the discriminator.
        Input:
            batch_size: batch size.
            device: optional device to place the hidden state on.
        Returns:
            hidden: initial hidden state.
        """
        hidden = torch.zeros((2 * 2 * 1, batch_size, self.hidden_dim))
        if device:
            hidden = hidden.to(device)
        return hidden

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the source critic network.
        Input:
            X: input batch of molecules with dimensions BN, where B is the
                batch size and N is the maximum molecule length.
        Returns:
            Output probabilities that the molecules are from the source
            distribution as a tensor of length B.
        """
        self.embed.requires_grad_(True)
        hidden = self.init_hidden(X.size(0), device=X.device)
        _, hidden = self.rnn(self.embed(X).permute(1, 0, 2), hidden)
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden = hidden.view(-1, 2 * 2 * self.hidden_dim)
        logits = self.w2(self.drop(torch.tanh(self.w1(hidden)))).view(-1)
        return torch.sigmoid(logits) if self.use_sigmoid else logits


class WeightClipper:
    """Object to clip the weights of a neural network to a finite range."""

    def __init__(self, c: float = 0.01):
        """
        Args:
            c: weight clipping parameter to clip all weights between [-c, c].
        """
        self.c = c

    def __call__(self, module: nn.Module) -> None:
        """
        Clips the weights of an input neural network to between [-c, c].
        Input:
            module: neural network to clip the weights of.
        Returns:
            None.
        """
        _ = [p.data.clamp_(-self.c, self.c) for p in module.parameters()]
