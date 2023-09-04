"""
Defines a generator that optimizes against an objective with optional source
discriminator regularization.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pytorch_lightning as pl
from typing import Optional, Sequence, Union

from models.block import Block
from models.objective import Objective
from models.metric import FID
from models.regularization import Regularization


class Generator(nn.Module):
    """Vanilla generator network implementation."""

    def __init__(
        self,
        z_dim: int = 128,
        x_dim: Sequence[int] = (1, 28, 28),
        intermediate_layers: Sequence[int] = (128, 256, 512, 1024)
    ):
        """
        Args:
            z_dim: number of latent dimensions as input to the generator G.
                Default 128.
            x_dim: dimensions CHW of the output image from the generator G.
                Default MNIST dimensions (1, 28, 28).
            intermediate_layers: intermediate layer output dimensions. Default
                (128, 256, 512, 1024).
        """
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.intermediate_layers = intermediate_layers

        self.model = [
            (
                "layer0",
                Block(
                    in_dim=self.z_dim,
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
                        normalize=True,
                        activation="LeakyReLU"
                    ),
                )
            )
        self.model.append(
            (
                "layer" + str(len(self.intermediate_layers)),
                Block(
                    in_dim=self.intermediate_layers[-1],
                    out_dim=int(np.prod(self.x_dim)),
                    normalize=False,
                    activation="Tanh"
                )
            )
        )

        self.model = nn.Sequential(OrderedDict(self.model))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the generator G.
        Input:
            z: latent space vector as input to the generator.
        Returns:
            Generated sample x = G(z), where pixel intensities are scaled to be
            between 0 and 1.
        """
        x = self.model(z)
        return (0.5 * x.view(x.size(0), *self.x_dim)) + 0.5


class GeneratorModule(pl.LightningModule):
    """
    Generator that optimizes against an objective with optional source
    discriminator regularization.
    """

    def __init__(
        self,
        objective: str,
        alpha: float = 0.0,
        regularization: str = "gan_loss",
        z_dim: int = 128,
        x_dim: Sequence[int] = (1, 28, 28),
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        num_images_logged: int = 8,
        **kwargs
    ):
        """
        Args:
            objective: objective function to optimize the generator against.
            alpha: source discriminator regularization weighting term.
                Default no source discriminator regularization.
            regularization: method of regularization. One of [`None`, `fid`,
                `gan_loss`, `importance_weighting`, `wasserstein_distance`].
            z_dim: number of latent dimensions as input to the generator G.
                Default 128.
            x_dim: dimensions CHW of the output image from the generator G.
                Default MNIST dimensions (1, 28, 28).
            lr: learning rate. Default 0.0002.
            beta1: beta_1 parameter in Adam optimizer algorithm. Default 0.5.
            beta2: beta_2 parameter in Adam optimizer algorithm. Default 0.999.
            num_images_logged: number of images to log per training and
                validation step. Can also use this parameter to set the *total*
                number of images to log during model testing.
        """
        super().__init__()
        self.save_hyperparameters()
        self.hparams.alpha = min(max(alpha, 0.0), 1.0)
        self.automatic_optimization = False

        self.generator = Generator(
            z_dim=self.hparams.z_dim, x_dim=self.hparams.x_dim
        )

        self.regularization = None
        if self.hparams.alpha > 0.0:
            self.regularization = Regularization(
                method=regularization,
                x_dim=self.hparams.x_dim
            )
            self.loss_D = self.regularization.loss_D

        self.objective = None
        if self.hparams.alpha < 1.0:
            self.objective = Objective(objective, x_dim=x_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the network.
        Input:
            z: input latent space vector.
        Returns:
            Generated sample x = G(z).
        """
        return self.generator(z)

    def training_step(
        self, batch: Sequence[torch.Tensor], batch_idx: int
    ) -> None:
        """
        Training step for generative model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            None.
        """
        xp, _ = batch
        B, C, H, W = xp.size()

        optimizer_G, optimizer_D = self.optimizers()

        z = torch.randn((B, self.hparams.z_dim)).to(xp)

        self.toggle_optimizer(optimizer_G)
        xq = self(z)

        if self.hparams.num_images_logged and self.logger:
            self.logger.log_image(
                "generated_images",
                images=[xq[i] for i in range(self.hparams.num_images_logged)]
            )

        loss_G = 0.0
        if self.objective:
            loss_G += (1.0 - self.hparams.alpha) * self.objective(xq)
        if self.regularization:
            loss_G += (self.hparams.alpha) * self.regularization(xp, xq)
        self.log("loss_G", loss_G, prog_bar=True, sync_dist=True)
        self.manual_backward(loss_G, retain_graph=bool(optimizer_D))
        optimizer_G.step()
        optimizer_G.zero_grad()
        self.untoggle_optimizer(optimizer_G)

        if optimizer_D:
            self.toggle_optimizer(optimizer_D)
            loss_D = self.loss_D(xp, xq.detach())
            self.log("loss_D", loss_D, prog_bar=True, sync_dist=True)
            self.manual_backward(loss_D)
            optimizer_D.step()
            optimizer_D.zero_grad()
            self.untoggle_optimizer(optimizer_D)

    def validation_step(
        self, batch: Sequence[torch.Tensor], batch_idx: int
    ) -> None:
        """
        Validation step for generative model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            None.
        """
        xp, _ = batch
        B, C, H, W = xp.size()

        z = torch.randn((B, self.hparams.z_dim)).to(xp)
        xq = self(z)

        if self.hparams.num_images_logged and self.logger:
            self.logger.log_image(
                "val_generated_images",
                images=[xq[i] for i in range(self.hparams.num_images_logged)]
            )

        if self.objective:
            self.log(
                "val_obj", self.objective(xq), prog_bar=True, sync_dist=True
            )
        else:
            self.log("val_obj", 0.0, prog_bar=True, sync_dist=True)

        self.log("val_fid", FID(xp, xq), prog_bar=True, sync_dist=True)

    def test_step(
        self, batch: Sequence[torch.Tensor], batch_idx: int
    ) -> None:
        """
        Testing step for generative model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            None.
        """
        if self.hparams.num_images_logged <= 0:
            return

        xp, _ = batch
        B, C, H, W = xp.size()

        num_images_to_log = self.hparams.num_images_logged - (
            batch_idx * min(B, self.hparams.num_images_logged)
        )
        if num_images_to_log > 0:
            xq = self(torch.randn((B, self.hparams.z_dim)).to(xp))
            self.log_image([xq[i] for i in range(min(B, num_images_to_log))])

        return

    def configure_optimizers(self) -> Sequence[Optional[optim.Optimizer]]:
        """
        Configure manual optimization.
        Input:
            None.
        Returns:
            Sequence of optimizer(s).
        """
        optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )

        if self.regularization:
            optimizer_D = optim.Adam(
                self.regularization.D.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2)
            )
        else:
            optimizer_D = None

        return [optimizer_G, optimizer_D]

    def log_image(
        self,
        images: Sequence[torch.Tensor],
        savedir: Union[Path, str] = "./output",
        with_colorbar: bool = False
    ) -> None:
        """
        Saves a series of images to a specified directory.
        Input:
            images: sequence of images to save.
            savedir: directory to save to. Default `./output`.
            with_colorbar: whether to save image with colorbar. Default False.
        Returns:
            None.
        """
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        for i, img in enumerate(images):
            plt.figure()
            plt.imshow(
                img[0, ...].detach().cpu().numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0
            )
            plt.axis("off")
            if with_colorbar:
                plt.colorbar()
            plt.savefig(
                os.path.join(savedir, f"{i}.png"),
                transparent=True,
                bbox_inches="tight",
                format="png",
                dpi=600
            )
            plt.close()
        return
