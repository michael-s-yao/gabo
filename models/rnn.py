"""
RNN as LSTM or GRU including and embedding layer and output linear layer.

Author(s):
    Michael Yao

Implementation adapted from @undeadpixel at https://github.com/undeadpixel/rein
vent-randomized.

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple


class RNN(nn.Module):
    """
    Multi-layer RNN including an embedding layer and output linear layer back
    to the original vocabulary.

    Implementation adapted from @undeadpixel at https://github.com/undeadpixel/
    reinvent-randomized.
    """

    def __init__(
        self,
        cell_type: str,
        out_dim: int,
        vocab: Dict[str, int],
        num_dimensions: int,
        num_layers: int,
        embedding_layer_size: int,
        dropout: float,
        padding_token: str = "[pad]",
        use_bidirectional: bool = False,
        use_sigmoid: bool = False,
        return_hidden: bool = True
    ):
        """
        Args:
            cell_type: cell type to use. One of [`GRU`, `LSTM`].
            out_dim: output dimensions from the final linear layer.
            vocab: vocabulary dict.
            num_dimensions: number of dimensions of the RNN hidden states.
            num_layers: number of RNN layers.
            embedding_layer_size: size of the embedding layer.
            dropout: dropout parameter.
            padding_token: padding token in vocab. Default `[pad]`.
            use_bidirectional: whether to use a bidirectional RNN.
            use_sigmoid: whether to apply sigmoid activation to model output.
            return_hidden: whether to also return the final hidden state.
        """
        super().__init__()

        self.cell_type = cell_type
        self.out_dim = out_dim
        self.vocab = vocab
        self.pad = padding_token
        self.num_dimensions = num_dimensions
        self.embedding_layer_size = embedding_layer_size
        self.num_layers = num_layers
        self.use_bidirectional = use_bidirectional
        self.use_sigmoid = use_sigmoid
        self.return_hidden = return_hidden

        self.embedding = nn.Embedding(
            len(self.vocab.keys()), self.embedding_layer_size
        )
        self.dropout = nn.Dropout(dropout)
        if self.cell_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.embedding_layer_size,
                hidden_size=self.num_dimensions,
                num_layers=self.num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=self.use_bidirectional
            )
        elif self.cell_type.upper() == "GRU":
            self.rnn = nn.GRU(
                input_size=self.embedding_layer_size,
                hidden_size=self.num_dimensions,
                num_layers=self.num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=self.use_bidirectional
            )
        else:
            raise ValueError(f"Unrecognized RNN type {self.cell_type}.")
        self.linear = nn.Linear(
            self.num_dimensions * (1 + int(self.use_bidirectional)),
            self.out_dim
        )

    def forward(
        self,
        X: torch.Tensor,
        h: Optional[Tuple[torch.Tensor]] = None
    ) -> Any:
        """
        Forward pass through the RNN model.
        Input:
            X: padded input tensor with dimensions BN, where B is the batch
                size and N is the padded sequence length.
            h: optional hidden and/or cell state tensors each with dimensions
                DBH, where D is self.num_layers if self.use_bidirectional is
                False and 2 * self.num_layers if self.use_bidirectional is
                True, B is the batch size, and H (self.num_dimensions) is the
                hidden state size.
        Returns:
            logits: RNN(X).
            h: a tuple of final hidden and/or cell state tensors with the same
                dimensions as described in the input.
        """
        B, N = X.size()
        if h is None:
            D = self.num_layers * (1 + int(self.use_bidirectional))
            if self.cell_type.upper() == "LSTM":
                h = (
                    torch.zeros((D, B, self.num_dimensions)).to(X.device),
                    torch.zeros((D, B, self.num_dimensions)).to(X.device)
                )
            elif self.cell_type.upper() == "GRU":
                h = torch.zeros((D, B, self.num_dimensions)).to(X.device)
        seq_lengths = torch.ones(B)

        self.embedding = self.embedding.to(X.device)
        self.rnn = self.rnn.to(X.device)
        self.linear = self.linear.to(X.device)

        encoded_X = nn.utils.rnn.pack_padded_sequence(
            self.dropout(self.embedding(X)),
            seq_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        X, h = self.rnn(encoded_X, h)
        X, _ = nn.utils.rnn.pad_packed_sequence(
            X, batch_first=True, padding_value=self.vocab[self.pad]
        )
        logits = self.linear(X) * torch.unsqueeze(X[..., 0] != 0, dim=-1)
        if self.return_hidden:
            return torch.sigmoid(logits) if self.use_sigmoid else logits, h
        return torch.sigmoid(logits) if self.use_sigmoid else logits
