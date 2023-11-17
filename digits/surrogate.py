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
from pathlib import Path
from typing import Sequence, Tuple, Union

sys.path.append(".")
from models.fcnn import FCNN
from models.convae import ConvAutoEncLightningModule


class SurrogateObjective(pl.LightningModule):
    def __init__(
        self,
        autoencoder: Union[Path, str],
        hidden_dims: Sequence[int] = [128, 32],
        lr: float = 0.001,
        weight_decay: float = 1e-6,
    ):
        """
        Args:
            autoencoder: path to autoencoder model checkpoint.
            hidden_dims: hidden dimensions into the model. Default [128, 32].
            lr: learning rate. Default 0.001.
            weight_decay: weight decay. Default 1e-6.
        """
        super().__init__()
        self.save_hyperparameters()

        self.convae = ConvAutoEncLightningModule.load_from_checkpoint(
            self.hparams.autoencoder
        )
        self.convae = self.convae.to(self.device)
        self.convae.eval()

        self.model = FCNN(
            in_dim=self.convae.model.z_dim,
            out_dim=1,
            hidden_dims=self.hparams.hidden_dims,
            dropout=0.0,
            final_activation=None,
            hidden_activation="ReLU"
        )
        self.loss = nn.MSELoss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the surrogate objective model.
        Input:
            z: input batch of image data in the autoencoder latent space.
        Returns:
            Estimated objective values for each input datum.
        """
        return self.model(z)

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
        z, _, _ = self.convae.model.encode(X)
        y = torch.mean(
            torch.square(
                self.convae.model.decode(z).flatten(start_dim=(X.ndim - 3))
            ),
            dim=-1
        )
        loss = self.loss(torch.squeeze(self(z), dim=-1), y)
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
        z, _, _ = self.convae.model.encode(X)
        y = torch.mean(
            torch.square(
                self.convae.model.decode(z).flatten(start_dim=(X.ndim - 3))
            ),
            dim=-1
        )
        loss = self.loss(torch.squeeze(self(z), dim=-1), y)
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
        z, _, _ = self.convae.model.encode(X)
        y = torch.mean(
            torch.square(
                self.convae.model.decode(z).flatten(start_dim=(X.ndim - 3))
            ),
            dim=-1
        )
        ypred = torch.squeeze(self(z), dim=-1)
        loss = self.loss(ypred, y)
        self.log("test_mse", loss, prog_bar=False)
        self.log("test_error", torch.mean(ypred - y), prog_bar=False)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
