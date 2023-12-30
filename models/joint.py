"""
Defines and implements a joint VAE-property prediction model.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Gomez-Bombarelli R, Wei JN, Duvenaud D, Hernandez-Lobato J, Sanchez-
        Lengeling B, Sheberla D, Aguilera-Iparraguirre J, Hirzel TD, Adams
        RP, Aspuru-Guzik A. Automatic chemical design using a data-driven
        continuous representation of molecules. ACS Cent Sci 4(2): 268-76.
        (2018). https://doi.org/10.1021/acscentsci.7b00572

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from typing import Dict, Tuple

sys.path.append(".")
from models.fcnn import FCNN
from models.vae import VAE
from models.seqvae import SequentialVAE
from design_bench.task import Task


class JointVAESurrogate(pl.LightningModule):
    def __init__(
        self,
        task: Task,
        lr: float = 1e-3,
        alpha: float = 1e-5,
        beta: float = 1.0,
        **kwargs
    ):
        """
        Args:
            task: offline MBO task.
            lr: learning rate. Default 1e-3.
            alpha: relative weighting for the KLD term in the ELBO loss.
            beta: relative weighting for the regressor loss term.
        """
        super().__init__()
        self.save_hyperparameters(ignore="task")
        self.task = task

        if self.task.is_discrete:
            self.vae = SequentialVAE(in_dim=self.task.input_shape[0], **kwargs)
        else:
            self.vae = VAE(in_dim=self.task.input_shape[0], **kwargs)

        self.surrogate = FCNN(
            in_dim=self.vae.latent_size,
            out_dim=1,
            hidden_dims=[2048, 2048],
            dropout=0.0,
            final_activation=None,
            hidden_activation="LeakyReLU",
            use_batch_norm=False
        )

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass through the VAE model.
        Input:
            X: a batch of input designs.
        Returns:
            z: the latent space representation(s) of the input(s).
            y: the predicted objective scores from the surrogate model.
            recon: the log softmax of the logits of the reconstructed sequence.
            mu: the mean tensor of shape Bx(self.vae.latent_size).
            logvar: the log variance tensor of shape Bx(self.vae.latent_size).
        """
        z, mu, logvar = self.vae.encode(X)
        return z, self.surrogate(z), self.vae.decode(z), mu, logvar

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Defines a training step for model training.
        Input:
            batch: batch of input datums for model training.
            batch_idx: index of the batch.
        Returns:
            The loss over the input training batch.
        """
        X, y = batch
        X, y = X.to(self.dtype), y.to(self.dtype)
        z, ypred, logits, mu, logvar = self(X)
        ce, kld = self.recon_loss(logits, X), self.kld(mu, logvar)
        mse = self.obj_loss(y, ypred)
        train_loss = ce + (self.hparams.alpha * kld) + (
            self.hparams.beta * mse
        )
        self.log("train_loss", train_loss, sync_dist=True, prog_bar=True)
        return train_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Defines a validation step for model validation.
        Input:
            batch: batch of input datums for model validation.
            batch_idx: index of the batch.
        Returns:
            A dictionary containing the validation step results.
        """
        X, y = batch
        X, y = X.to(self.dtype), y.to(self.dtype)
        z, ypred, logits, mu, logvar = self(X)
        ce, kld = self.recon_loss(logits, X), self.kld(mu, logvar)
        mse = self.obj_loss(y, ypred)
        val_loss = ce + (self.hparams.alpha * kld) + (self.hparams.beta * mse)
        self.log("val/ce", ce, sync_dist=True, prog_bar=True)
        self.log("val/objective_mse", mse, sync_dist=True, prog_bar=True)
        self.log("val/kld", kld, sync_dist=True, prog_bar=True)
        self.log("val_loss", val_loss, sync_dist=True, prog_bar=False)
        return {
            "ce": ce, "objective_mse": mse, "kld": kld, "val_loss": val_loss
        }

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Defines and configures the optimizer for model training.
        Input:
            None.
        Returns:
            The optimizer for model training.
        """
        return optim.Adam(
            list(self.vae.parameters()) + list(self.surrogate.parameters()),
            lr=self.hparams.lr
        )

    def recon_loss(
        self, recon: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the reconstruction loss term of the ELBO loss.
        Input:
            logits: reconstruction for continuous tasks or logits of the
                reconstructed sequence for discrete tasks.
            X: the original input batch of designs.
        Returns:
            The mean reconstruction loss term of the ELBO loss.
        """
        if self.task.is_discrete:
            return F.cross_entropy(
                recon.reshape(-1, self.vae.vocab_size),
                X.reshape(-1).to(torch.int64)
            )
        else:
            return F.binary_cross_entropy(recon, X.view(recon.size()))

    def kld(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL divergence loss term of the ELBO loss.
        Input:
            mu: the mean tensor of shape Bx(self.latent_size).
            logvar: the log variance tensor of shape Bx(self.latent_size).
        Returns:
            The mean KL divergence loss term of the ELBO loss.
        """
        return -0.5 * torch.sum(
            1.0 + logvar - torch.pow(mu, 2) - torch.exp(logvar)
        )

    def obj_loss(self, y: torch.Tensor, ypred: torch.Tensor) -> torch.Tensor:
        """
        Computes the MSE loss for the objective.
        Input:
            y: the ground truth objective values.
            ypred: the predicted objective values.
        Returns:
            The MSE loss term.
        """
        return F.mse_loss(
            y.squeeze(dim=-1).to(ypred.dtype), ypred.squeeze(dim=-1)
        )
