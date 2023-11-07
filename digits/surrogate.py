"""
Defines a surrogate objective to estimate the energy in a grayscale image.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from typing import Sequence, Tuple

sys.path.append(".")
from models.fcnn import FCNN


class SurrogateObjective(pl.LightningModule):
    def __init__(
        self,
        in_dim: int = 784,
        hidden_dims: Sequence[int] = [784, 196],
        dropout: float = 0.0,
        optimizer: str = "Adam",
        lr: float = 0.001,
        weight_decay: float = 1e-6,
    ):
        """
        Args:
            in_dim: number of input dimensions into the model. Default 784.
            hidden_dims: hidden dimensions into the model. Default [784, 196].
            dropout: dropout probability. Default 0.0.
            optimizer: optimizer algorithm. One of [`Adam`, `SGD`].
            lr: learning rate. Default 0.001.
            weight_decay: weight decay. Default 1e-6.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = FCNN(
            in_dim=self.hparams.in_dim,
            out_dim=1,
            hidden_dims=self.hparams.hidden_dims,
            dropout=self.hparams.dropout,
            final_activation=None,
            hidden_activation="ReLU"
        )
        self.loss = nn.MSELoss()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the surrogate objective model.
        Input:
            X: input batch of flattened image data.
        Returns:
            Estimated objective values for each input datum.
        """
        return self.model(X)

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Defines the training step.
        Input:
            batch: input batch of data for model training.
            batch_idx: batch index.
        Returns:
            The training loss over the batch.
        """
        X, _ = batch
        X = X.reshape(-1, self.hparams.in_dim)
        y = torch.unsqueeze(torch.mean(torch.square(X), dim=-1), dim=-1)
        loss = self.loss(self(X), y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Defines the validation step.
        Input:
            batch: input batch of data for model validation.
            batch_idx: batch index.
        Returns:
            The validation loss over the batch.
        """
        X, _ = batch
        X = X.reshape(-1, self.hparams.in_dim)
        y = torch.unsqueeze(torch.mean(torch.square(X), dim=-1), dim=-1)
        loss = self.loss(self(X), y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Defines the test step.
        Input:
            batch: input batch of data for model testing.
            batch_idx: batch index.
        Returns:
            The test loss over the batch.
        """
        X, _ = batch
        X = X.reshape(-1, self.hparams.in_dim)
        y = torch.unsqueeze(torch.mean(torch.square(X), dim=-1), dim=-1)
        ypred = self(X)
        loss = self.loss(ypred, y)
        self.log("test_mse", loss, prog_bar=False)
        self.log("test_error", torch.mean(ypred - y), prog_bar=False)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        if self.hparams.optimizer.title() == "Adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.upper() == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.hparams.optimizer} not implemented."
            )
