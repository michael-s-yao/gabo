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

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from typing import Any, Dict, Tuple, Union

sys.path.append(".")
from data.molecules.selfies import SELFIESDataset
from data import SELFIESChEMBLDataset
from models.fcnn import FCNN
from models.vae import VAE, IdentityVAE
from models.seqvae import SequentialVAE
from models.transformer_vae import InfoTransformerVAE
from design_bench.task import Task


class JointVAESurrogate(pl.LightningModule):
    def __init__(
        self,
        task: Task,
        task_name: str,
        lr: float = 1e-3,
        alpha: float = 1e-5,
        beta: float = 1.0,
        **kwargs
    ):
        """
        Args:
            task: offline MBO task.
            task_name: name of the offline MBO task.
            lr: learning rate. Default 1e-3.
            alpha: relative weighting for the KLD term in the ELBO loss.
            beta: relative weighting for the regressor loss term.
        """
        super().__init__()
        self.save_hyperparameters(ignore="task")
        self.task = task

        if self.hparams.task_name in (
            os.environ["BRANIN_TASK"], os.environ["WARFARIN_TASK"]
        ):
            self.vae = IdentityVAE(in_dim=self.task.input_shape[0], **kwargs)
            self.hparams.beta = 1.0
        elif self.hparams.task_name in (
            os.environ["MOLECULE_TASK"], os.environ["CHEMBL_TASK"]
        ):
            if self.hparams.task_name == os.environ["MOLECULE_TASK"]:
                _dataset = SELFIESDataset()
            elif self.hparams.task_name == os.environ["CHEMBL_TASK"]:
                _dataset = SELFIESChEMBLDataset()
            self.vae = InfoTransformerVAE(
                vocab2idx=_dataset.vocab2idx,
                start=_dataset.start,
                stop=_dataset.stop,
                **kwargs
            )
        elif self.task.is_discrete:
            self.vae = SequentialVAE(
                in_dim=self.task.input_shape[0],
                vocab_size=self.task.dataset.num_classes,
                **kwargs
            )
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
            logits: the logits of the reconstructed sequence.
            mu: the mean tensor of shape Bx(self.vae.latent_size).
            logvar: the log variance tensor of shape Bx(self.vae.latent_size).
        """
        z, mu, logvar = self.vae.encode(X)
        logits = self.vae.decode(z, tokens=X)
        if z.ndim > X.ndim:
            z = z.reshape(z.size(dim=0), -1) if X.ndim > 1 else z.flatten()
        return z, self.surrogate(z), logits, mu, logvar

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
        if self.hparams.task_name == os.environ["MNIST_TASK"]:
            recon = torch.sigmoid(logits)
        else:
            recon = logits
        ce, kld = self.recon_loss(recon, X), self.kld(mu, logvar)
        mse = self.obj_loss(y, ypred)
        train_loss = ce + (self.hparams.alpha * kld) + (
            self.hparams.beta * mse
        )
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
        recon = logits if self.task.is_discrete else torch.sigmoid(logits)
        ce, kld = self.recon_loss(recon, X), self.kld(mu, logvar)
        mse = self.obj_loss(y, ypred)
        val_loss = ce + (self.hparams.alpha * kld) + (self.hparams.beta * mse)
        self.log("val/ce", ce, sync_dist=True, prog_bar=True)
        self.log("val/objective_mse", mse, sync_dist=True, prog_bar=True)
        self.log("val/kld", kld, sync_dist=True, prog_bar=True)
        self.log("val_loss", val_loss, sync_dist=True, prog_bar=False)
        return {
            "ce": ce, "objective_mse": mse, "kld": kld, "val_loss": val_loss
        }

    def configure_optimizers(self) -> Union[optim.Optimizer, Dict[str, Any]]:
        """
        Defines and configures the optimizer for model training.
        Input:
            None.
        Returns:
            The optimizer(s) for model training.
        """
        if self.hparams.task_name == os.environ["MOLECULE_TASK"]:
            encoder_params, decoder_params, surrogate_params = [], [], []
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "encoder" in name:
                        encoder_params.append(param)
                    elif "decoder" in name:
                        decoder_params.append(param)
                    else:
                        surrogate_params.append(param)

            optimizer = optim.Adam([
                {"params": encoder_params, "lr": self.hparams.encoder_lr},
                {"params": decoder_params, "lr": self.hparams.decoder_lr},
                {"params": surrogate_params, "lr": self.hparams.lr}
            ])
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                [
                    self._encoder_lr_sched,
                    self._decoder_lr_sched,
                    lambda step: 1.0
                ]
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }

        params = list(self.surrogate.parameters())
        if not isinstance(self.vae, IdentityVAE):
            params += list(self.vae.parameters())
        return optim.Adam(params, lr=self.hparams.lr)

    def recon_loss(
        self, recon: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the reconstruction loss term of the ELBO loss.
        Input:
            recon: probabilities for continuous tasks or logits of the
                reconstructed sequence for discrete tasks.
            X: the original input batch of designs.
        Returns:
            The mean reconstruction loss term of the ELBO loss.
        """
        if self.hparams.task_name in (
            os.environ["BRANIN_TASK"], os.environ["WARFARIN_TASK"]
        ):
            return 0.0
        if self.task.is_discrete:
            return F.cross_entropy(
                recon.reshape(-1, self.vae.vocab_size),
                X.flatten().to(torch.int64)
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
        if mu is None or logvar is None:
            return 0.0
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

    def _encoder_lr_sched(self, step: int) -> float:
        """
        Simple linear warmup LR scheduler for the encoder.
        Input:
            step: optimization step.
        Returns:
            LR weighting factor for the encoder.
        """
        return min(step / self.hparams.encoder_warmup_steps, 1.0)

    def _decoder_lr_sched(self, step: int) -> float:
        """
        Decoder LR scheduler.
        Input:
            step: optimization step.
        Returns:
            LR weighting factor for the decoder.
        """
        if step < self.hparams.encoder_warmup_steps or (
            (step - self.hparams.encoder_warmup_steps + 1) %
            self.hparams.aggressive_steps != 0
        ):
            return 0.0
        return min(
            (step - self.hparams.encoder_warmup_steps) / (
                self.hparams.decoder_warmup_steps *
                self.hparams.aggressive_steps
            ),
            1.0
        )
