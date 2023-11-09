"""
Defines a MNIST image generator that optimizes against an objective function.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Gulrajani I, Ahmed F, Arjovsky M, Dumoulin V, Courville A. Improved
        training of Wasserstein GANs. Proc NeurIPS. (2017).
        https://doi.org/10.48550/arXiv.1704.00028

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pytorch_lightning as pl
from typing import Optional, Sequence, Union

sys.path.append(".")
from digits.surrogate import SurrogateObjective
from models.cnn import ConvGenerator
from models.critic import WeightClipper
from models.fcnn import FCNN
from models.metric import FID
from models.lipschitz import Lipschitz


class WGANModule(pl.LightningModule):
    """Vanilla WGAN network implementation for image generation."""

    def __init__(
        self,
        alpha: str = "lipschitz",
        latent_dim: int = 128,
        x_dim: int = 28,
        lr: float = 0.0001,
        weight_decay: float = 1e-6,
        clip: Optional[float] = None,
        n_critic_per_generator: int = 5,
        num_images_logged: int = 8,
    ):
        """
        Args:
            alpha: relative regularization weighting. Use `Lipschitz` for our
                method, otherwise specify a float between 0 and 1.
            latent_dim: number of input latent space dimensions. Default 128.
            x_dim: dimensions of generated images. Default (1, 28, 28).
            lr: learning rate. Default 0.0001.
            weight_decay: weight decay. Default 1e-6.
            clip: gradient clipping. Default no clipping.
            n_critic_per_generator: number of times to optimize the critic
                versus the generator. Default 5.
            num_images_logged: number of images to log per training and
                validation step. Can also use this parameter to set the *total*
                number of images to log during model testing.
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = ConvGenerator(
            self.hparams.latent_dim, self.hparams.x_dim
        )
        self.surrogate = SurrogateObjective.load_from_checkpoint(
            os.path.join(os.path.dirname(__file__), "ckpt", "surrogate.ckpt")
        )
        self.surrogate.eval()
        self.surrogate = self.surrogate.to(self.device)
        if not self.hparams.alpha.replace(".", "", 1).isnumeric():
            self.L = Lipschitz(self.surrogate, mode="local", p=2)
        self.critic = FCNN(
            self.hparams.x_dim * self.hparams.x_dim,
            1,
            hidden_dims=[256, 128, 64, 32, 16, 1],
            final_activation=None,
            hidden_activation="GELU",
            use_batch_norm=True
        )
        self.critic_clipper = WeightClipper()
        if not self.hparams.alpha.replace(".", "", 1).isnumeric():
            self.K = Lipschitz(self.critic, mode="global")
        self.objective = nn.MSELoss()

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Forward propagation through the generator G.
        Input:
            z: latent space vector as input to the generator.
        Returns:
            Generated sample x = G(z), where pixel intensities are scaled to be
            between 0 and 1.
        """
        return self.generator(batch_size)

    def alpha(self, Xq: torch.Tensor) -> Union[float, torch.Tensor]:
        """
        Calculates the value of alpha for regularization weighting.
        Input:
            Xq: generated images.
        Returns:
            The value of alpha for regularization weighting.
        """
        if self.hparams.alpha.replace(".", "", 1).isnumeric():
            return float(self.hparams.alpha)
        return 1.0 / (1.0 + (self.K() / self.L(Xq)))

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
        Xp, _ = batch
        B, C, H, W = Xp.size()
        Xq = self(B)
        optimizer_G, optimizer_C = self.optimizers()

        self.toggle_optimizer(optimizer_C)
        loss_C = torch.mean(self.critic(Xq.reshape(B, -1))) - torch.mean(
            self.critic(Xp.reshape(B, -1))
        )
        self.log("loss_C", loss_C, prog_bar=True, sync_dist=True)
        self.manual_backward(loss_C, retain_graph=True)
        if self.hparams.clip:
            self.clip_gradients(
                optimizer_C,
                gradient_clip_val=self.hparams.clip,
                gradient_clip_algorithm="norm"
            )
        optimizer_C.step()
        self.critic_clipper(self.critic)
        optimizer_C.zero_grad()
        self.untoggle_optimizer(optimizer_C)

        if batch_idx % self.hparams.n_critic_per_generator:
            return

        self.toggle_optimizer(optimizer_G)
        if self.hparams.num_images_logged and self.logger and batch_idx == 0:
            self.logger.log_image(
                "generated_images",
                images=[
                    Xq[i] for i in range(self.hparams.num_images_logged)
                ]
            )
        alpha_ = self.alpha(Xq)
        Xq = Xq.detach()
        loss_G = (alpha_ - 1.0) * self.surrogate(Xq.reshape(B, -1))
        loss_G += alpha_ * (
            torch.mean(self.critic(Xp.reshape(B, -1))) - self.critic(
                Xq.reshape(B, -1)
            )
        )
        loss_G = torch.mean(loss_G)
        self.log("loss_G", loss_G, prog_bar=True, sync_dist=True)
        self.manual_backward(loss_G)
        if self.hparams.clip:
            self.clip_gradients(
                optimizer_G,
                gradient_clip_val=self.hparams.clip,
                gradient_clip_algorithm="norm"
            )
        optimizer_G.step()
        optimizer_G.zero_grad()
        self.surrogate.zero_grad()
        self.untoggle_optimizer(optimizer_G)

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
        Xp, _ = batch
        B, C, H, W = Xp.size()
        Xq = self(B)

        Wd = torch.mean(self.critic(Xp.reshape(B, -1))) - self.critic(
            Xq.reshape(B, -1)
        )
        self.log(
            "val_wasserstein", torch.mean(Wd), prog_bar=True, sync_dist=True
        )

        alpha_ = self.alpha(Xq)
        obj = self.surrogate(Xq.reshape(B, -1))
        loss_G = ((alpha_ - 1.0) * obj) + (alpha_ * Wd)
        self.log("val_obj", torch.mean(obj), prog_bar=True, sync_dist=True)
        self.log(
            "val_loss_G", torch.mean(loss_G), prog_bar=True, sync_dist=True
        )
        self.log("val_fid", FID(Xp, Xq), prog_bar=False, sync_dist=True)

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
            weight_decay=self.hparams.weight_decay
        )
        optimizer_C = optim.Adam(
            self.critic.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return [optimizer_G, optimizer_C]

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
