"""

"""
import torch
import torch.nn as nn
from typing import Callable, Dict, Sequence


class HighwayNN(nn.Module):
    """
    Highway network implementation.
    Citation:
        [1] Srivastava RK, Greff K, Schmidhuber J. Highway networks. ICML Deep
            Learning Workshop. (2015). doi: 10.48550/arXiv.1505.00387
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
        initial_bias_towards_carry: bool = True
    ):
        """
        Args:
            in_dim: flattened input dimensions of the network.
            num_layers: number of layers. Default 1.
            activation: activation function for the nonlinear transform.
            initial_bias_towards_carry: bias the initial bheavior of the
                network towards carry behavior versus nonlinear transform.
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.activation = activation
        self.initial_bias_towards_carry = initial_bias_towards_carry

        self.H = nn.ModuleList([
            nn.Linear(self.in_dim, self.in_dim) for _ in range(self.num_layers)
        ])
        self.T_gate = nn.ModuleList([
            nn.Linear(self.in_dim, self.in_dim) for _ in range(self.num_layers)
        ])
        if self.initial_bias_towards_carry:
            for layer in self.T_gate:
                layer.bias.data.fill_(-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the highway network.
        Input:
            X: input tensor with shape BN, where B is the batch size and N is
                the flattened input size.
        Returns:
            Output tensor of shape BN.
        """
        for H, T_gate in zip(self.H, self.T_gate):
            gate = torch.sigmoid(T_gate(X))
            transform = self.activation(H(X))
            X = (gate * transform) + ((1.0 - gate) * X)
        return X


class SourceCritic(nn.Module):
    """A CNN-based discriminator for molecule sequence classification."""

    def __init__(
        self,
        kernel_num: int,
        kernel_sizes: Sequence[int],
        vocab: Dict[str, int],
        embedding_layer_size: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
        max_molecule_length: int = 111,
        padding_token: str = " ",
        dropout: float = 0.1,
        device: torch.device = torch.device("cpu")
    ):
        """
        Args:
            kernel_num: number of kernels for the convolutional layers.
            kernel_sizes: kernel sizes for the convolutional layers.
            vocab: vocabulary dict.
            embedding_layer_size: size of the embedding layer.
            activation: activation function for the nonlinear transform and
                between convolutional layers.
            max_molecule_length: maximum molecule length. Default 111.
            padding_token: padding token in vocab. Default ` `.
            dropout: dropout probability. Default 0.1.
            device: device to place the embedding weights on.
        """
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.vocab = vocab
        self.embedding_layer_size = embedding_layer_size
        self.max_molecule_length = max_molecule_length
        self.padding_token = padding_token

        self.embedding = nn.Embedding(
            len(self.vocab.keys()),
            self.embedding_layer_size,
            padding_idx=self.vocab[self.padding_token]
        )
        self.embedding.weight = nn.Parameter(self.embedding.weight.to(device))
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.kernel_num, (K, self.embedding_layer_size))
            for K in self.kernel_sizes
        ])
        self.activation = activation
        self.highway = HighwayNN(
            len(self.kernel_sizes) * self.kernel_num,
            activation=self.activation
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(len(self.kernel_sizes) * self.kernel_num, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the source critic.
        Input:
            X: a tensor of tokens with dimensions BN, where B is the batch size
                and N is the maximum molecule length.
        Returns:
            A vector of probabilities with dimensions B.
        """
        X = torch.unsqueeze(self.embedding(X), dim=1)
        X = [
            torch.squeeze(self.activation(conv(X)), dim=-1)
            for conv in self.convs
        ]
        X = [
            torch.squeeze(torch.max_pool1d(X, X.size(dim=1)), dim=-1)
            for x in X
        ]
        X = self.dropout(self.highway(torch.cat(X, dim=-1)))
        return torch.sigmoid(self.linear(X))


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from data.molecule import QM9DataModule
    from data.molecule_utils import one_hot_encodings_to_tokens

    dm = QM9DataModule(batch_size=4)
    dm.prepare_data()
    dm.setup()

    model = SourceCritic(
        kernel_num=128,
        kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
        vocab=dm.vocab,
        embedding_layer_size=64
    )
    for batch in dm.train_dataloader():
        batch = one_hot_encodings_to_tokens(batch)
        print(model(batch))
        break
    print(dm.vocab)
