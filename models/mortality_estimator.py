"""
Patient mortality estimator as a function of clinical features and INR.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from typing import Any, Callable, Dict, Sequence, Union

from models.fcnn import FCNN


class WarfarinMortalityLightningModule(pl.LightningModule):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        invert_continuous_transform: Callable[[torch.Tensor], torch.Tensor],
        dropout: float = 0.1,
        lr: float = 0.001,
        lr_milestones: Sequence[int] = [20, 40],
        lr_gamma: float = 0.1,
        optimizer: str = "Adam",
        beta: Union[float, Sequence[float]] = (0.9, 0.999),
        weight_decay: float = 0.001,
    ):
        """
        Args:
            in_dim: dimensions of input data.
            hidden_dims: dimensions of the hidden intermediate layers.
            invert_continuous_transform: a function to invert the original
                BayesianGMM transforms.
            dropout: dropout. Default 0.1.
            lr: learning rate. Default 0.001.
            lr_milestones: milestones for LR scheduler. Default [20, 40].
            lr_gamma: LR decay rate. Default 0.1.
            optimizer: optimizer algorithm. One of [`SGD`, `Adam`, `RMSProp`].
            beta: betas/momentum optimizer hyperparameter.
            weight_decay: weight decay. Default 0.001.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = FCNN(
            in_dim=self.hparams.in_dim,
            out_dim=1,
            hidden_dims=self.hparams.hidden_dims,
            dropout=self.hparams.dropout,
            final_activation=None
        )

        self.loss = nn.MSELoss()

        # Initialize weights.
        [param.data.fill_(0.01) for param in self.model.parameters()]

    def forward(self, patient: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Input:
            patient: a batch of patient features.
        Returns:
            Estimated patient costs as predicted by the model.
        """
        return torch.square(torch.squeeze(self.model(patient), dim=-1))

    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for cost model training.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            train_loss: training loss.
        """
        raw = self.hparams.invert_continuous_transform(
            batch.X, batch.X_attributes
        )
        train_loss = self.loss(self(raw), batch.cost)
        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.X.size(dim=0)
        )
        return train_loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step for cost model training.
        Input:
            batch: batch of input data for model validation.
            batch_idx: batch index.
        Returns:
            val_loss: validation loss.
        """
        raw = self.hparams.invert_continuous_transform(
            batch.X, batch.X_attributes
        )
        val_loss = self.loss(self(raw), batch.cost)
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.X.size(dim=0)
        )
        return val_loss

    def test_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """
        Test step for cost model evaluation.
        Input:
            batch: batch of input data for model testing.
            batch_idx: batch index.
        Returns:
            test_loss: test loss.
        """
        return self.loss(self(batch.X), batch.cost)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and lr_scheduler.
        Input:
            None.
        Returns:
            A dict of the optimizer and learning rate scheduler.
        """
        if self.hparams.optimizer.upper() == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.beta[0],
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.title() == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
                betas=self.hparams.beta,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.beta[0],
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unrecognized optimizer {self.hparams.optimizer} specified."
            )
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, self.hparams.lr_milestones, self.hparams.lr_gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
