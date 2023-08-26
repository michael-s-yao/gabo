"""
Defines a generator that optimizes against an objective with optional source
discriminator regularization.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
from typing import Sequence

from models.block import Block
from models.discriminator import Discriminator
from models.objective import Objective


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
            Generated sample x = G(z).
        """
        img = self.model(z)
        return img.view(img.size(0), *self.x_dim)


class GeneratorModule(pl.LightningModule):
    """
    Generator that optimizes against an objective with optional source
    discriminator regularization.
    """

    def __init__(
        self,
        objective: str,
        alpha: float = 0.0,
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
            z_dim: number of latent dimensions as input to the generator G.
                Default 128.
            x_dim: dimensions CHW of the output image from the generator G.
                Default MNIST dimensions (1, 28, 28).
            lr: learning rate. Default 0.0002.
            beta1: beta_1 parameter in Adam optimizer algorithm. Default 0.5.
            beta2: beta_2 parameter in Adam optimizer algorithm. Default 0.999.
            num_images_logged: number of images to log per training and
                validation step.
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(
            z_dim=self.hparams.z_dim, x_dim=self.hparams.x_dim
        )
        self.discriminator = None
        if 0.0 < self.hparams.alpha < 1.0:
            self.discriminator = Discriminator(x_dim=self.hparams.x_dim)
        self.objective = Objective(objective)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the network.
        Input:
            z: input latent space vector.
        Returns:
            Generated sample x = G(z).
        """
        return self.generator(z)

    def src_discriminator_loss(
        self, src_pred: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor:
        """
        Source discriminator D loss.
        Input:
            src_pred: discriminator predicted probability that sample is from
                source distribution.
            src: ground truth whether sample is from source distribution.
        Returns:
            Binary cross entropy loss of source discriminator.
        """
        return F.binary_cross_entropy(src_pred, src)

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

        if self.discriminator:
            optimizer_G, optimizer_D = self.optimizers()
        else:
            optimizer_G, optimizer_D = self.optimizers(), None

        z = torch.randn((B, self.hparams.z_dim)).to(xp)

        self.toggle_optimizer(optimizer_G)
        xq = self(z)

        if self.hparams.num_images_logged and self.logger:
            self.logger.experiment.add_image(
                "generated_images",
                torchvision.utils.make_grid(
                    xq[:self.hparams.num_images_logged]
                ),
                0
            )

        src = torch.ones(B).to(xp)

        loss_G = -1.0 * self.objective(xq)
        if self.hparams.alpha:
            loss_G += self.hparams.alpha * self.src_discriminator_loss(
                self.discriminator(xq), src
            )
        self.log("loss_G", loss_G, prog_bar=True)
        self.manual_backward(loss_G)
        optimizer_G.step()
        optimizer_G.zero_grad()
        self.untoggle_optimizer(optimizer_G)

        if optimizer_D:
            self.toggle_optimizer(optimizer_D)

            src_id = torch.ones(B).to(xp)
            src_loss = self.src_discriminator_loss(
                self.discriminator(xq), src_id
            )

            gen_id = torch.zeros(B).to(xp)
            gen_loss = self.src_discriminator_loss(
                self.discriminator(xq.detach()), gen_id
            )

            loss_D = (src_loss + gen_loss) / 2
            self.log("loss_D", loss_D, prog_bar=True)
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

        xq = self(torch.randn((B, self.hparams.z_dim)).to(xp))

        if self.hparams.num_images_logged and self.logger:
            self.logger.experiment.add_image(
                "generated_images",
                torchvision.utils.make_grid(
                    xq[:self.hparams.num_images_logged]
                ),
                0
            )

        src = torch.ones(B).to(xp)

        objective_G, loss_D = self.objective(xq), 0.0
        if self.discriminator:
            loss_D = self.src_discriminator_loss(self.discriminator(xq), src)
        self.log("val_objective_G", objective_G, prog_bar=True)
        self.log("val_loss_D", loss_D, prog_bar=True)

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
        return

    def configure_optimizers(self) -> Sequence[optim.Optimizer]:
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

        if self.discriminator is None:
            return [optimizer_G]

        optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )

        return [optimizer_G, optimizer_D]
