import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, Sequence, Union

from data.molecule import one_hot_encodings_to_tokens, tokens_to_selfies
from models.objective import SELFIESObjective
from models.rnn import RNN


class MolGeneratorModule(pl.LightningModule):
    """Molecule generation to optimize against an objective."""

    def __init__(
        self,
        vocab: Dict[str, int],
        max_molecule_length: int = 109,
        alpha: float = 0.0,
        num_dimensions: int = 1_024,
        num_layers: int = 3,
        embedding_layer_size: int = 128,
        dropout: float = 0.2,
        use_bidirectional: bool = False,
        start_token: str = "[start]",
        padding_token: str = "[pad]",
        lr: float = 0.001,
        optimizer: str = "Adam",
        beta: Union[float, Sequence[float]] = (0.9, 0.999),
        weight_decay: float = 0.001,
    ):
        """
        Args:
            vocab: vocabulary dict.
            max_molecule_length: maximum molecule length. Default 109.
            alpha: in-distribution weighting term. Default 1.0.
            num_dimensions: number of dimensions of the RNN hidden states.
            num_layers: number of RNN layers.
            embedding_layer_size: size of the embedding layer.
            dropout: dropout parameter.
            use_bidirectional: whether to use a bidirectional RNN.
            start_token: start token in vocab. Default `[start]`.
            padding_token: padding token in vocab. Default `[pad]`.
            lr: learning rate. Default 0.001.
            optimizer: optimizer algorithm. One of [`SGD`, `Adam`, `RMSProp`].
            beta: betas/momentum optimizer hyperparameter.
            weight_decay: weight decay. Default 0.001.
        """
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.optimizer.title() != "Adam":
            self.hparams.beta, _ = self.hparams.beta
        self.hparams.alpha = min(max(alpha, 0.0), 1.0)

        self.model = RNN(
            cell_type="LSTM",
            out_dim=len(self.hparams.vocab.keys()),
            vocab=self.hparams.vocab,
            num_dimensions=self.hparams.num_dimensions,
            num_layers=self.hparams.num_layers,
            embedding_layer_size=self.hparams.embedding_layer_size,
            dropout=self.hparams.dropout,
            use_bidirectional=False,
            padding_token=self.hparams.padding_token,
            device=self.device
        )

        self.objective = None
        if self.hparams.alpha < 1.0:
            self.objective = SELFIESObjective(
                self.hparams.vocab,
                surrogate_ckpt="./MolOOD/checkpoints/regressor.ckpt",
            )
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        X: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Sequence[torch.Tensor]:
        """
        Forward pass through the RNN molecule generator.
        Input:
            X: a batch of input sequences with dimensions BNV, where B is the
                batch size, N is any number less than the maximum molecule
                length, and V is the vocab size.
            hidden: optional hidden state of the RNN with dimensions BN, where
                H is the dimension of the hidden state.
        Returns:
            logits: logits of the next token in the sequence with dimensions
                BV, where V is the maximum size of the vocab.
            hidden: hidden state of the RNN with dimensions BH, where H is the
                dimension of the hidden state.
        """
        return self.model(
            one_hot_encodings_to_tokens(X).to(torch.int), hidden
        )

    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for generative model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            loss: training loss.
        """
        B, N, _ = batch.size()
        tokens = one_hot_encodings_to_tokens(batch)
        generated = torch.empty_like(tokens)
        generated[:, 0] = torch.full(
            (B,), self.hparams.vocab[self.hparams.start_token]
        )
        loss, hidden = 0.0, None
        for input_length in range(1, N):
            logits, hidden = self(batch[:, :input_length, :], hidden)
            logits = torch.squeeze(logits, dim=1)
            y = tokens[:, input_length]
            loss += self.loss(logits, y)
            ypred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            generated[:, input_length] = ypred
        id_loss = loss / (N - 1)
        objective = torch.mean(self.objective(generated))
        train_loss = ((self.hparams.alpha - 1.0) * objective) + (
            self.hparams.alpha * id_loss
        )

        self.log("train_loss", train_loss, prog_bar=True, sync_dist=True)

        return train_loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, float]:
        """
        Validation step for generative model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            val_loss: validation loss.
            val_acc: validation reconstruction accuracy.
        """
        B, N, _ = batch.size()
        tokens = one_hot_encodings_to_tokens(batch)
        generated = torch.empty_like(tokens)
        generated[:, 0] = torch.full(
            (B,), self.hparams.vocab[self.hparams.start_token]
        )
        loss, acc, hidden = 0.0, 0.0, None
        for input_length in range(1, N):
            logits, hidden = self(batch[:, :input_length, :], hidden)
            logits = torch.squeeze(logits, dim=1)
            y = tokens[:, input_length]
            loss += self.loss(logits, y)
            ypred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            generated[:, input_length] = ypred
            acc += torch.mean((ypred == y).to(torch.float))
        id_loss = loss / (N - 1)
        objective = torch.mean(self.objective(generated))
        val_loss = ((self.hparams.alpha - 1.0) * objective) + (
            self.hparams.alpha * id_loss
        )
        val_acc = acc.item() / (N - 1)

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", val_acc, prog_bar=True, sync_dist=True)

        return {
            "val_loss": val_loss,
            "val_id_loss": id_loss,
            "val_objective": objective,
            "val_acc": val_acc
        }

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure manual optimization.
        Input:
            None.
        Returns:
            Sequence of optimizer(s).
        """
        if self.hparams.optimizer.upper() == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.beta,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.title() == "Adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
                betas=self.hparams.beta,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.beta,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unrecognized optimizer {self.hparams.optimizer} specified."
            )
