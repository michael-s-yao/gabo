"""
Implements a vanilla autoencoder with a convolutional backbone.

Author(s):
    Michael Yao

Adapted from the pytorch-beginner repository from @L1aoXingyu at
https://github.com/L1aoXingyu/pytorch-beginner/tree/master

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pathlib import Path
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
)
from typing import Dict, Optional, Sequence, Tuple, Union


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        encoding_channels: Sequence[int] = [16, 8],
        encoding_kernels: Sequence[int] = [3, 3],
        encoding_maxpool_windows: Sequence[int] = [2, 2],
        encoding_strides: Sequence[int] = [3, 2],
        encoding_maxpool_strides: Sequence[int] = [2, 1],
        decoding_channels: Sequence[int] = [8, 16, 8],
        decoding_kernels: Sequence[int] = [3, 5, 2],
        decoding_strides: Sequence[int] = [2, 3, 2],
        decoding_padding: Sequence[int] = [0, 1, 1]
    ):
        """
        Args:
            in_channels: number of input (and output) channels into (and out
                of) the network.
            encoding_channels: sequence of channels to build the encoder.
            encoding_kernels: sequence of kernel sizes to build the encoder.
            encoding_maxpool_windows: sequence of window sizes to take a max
                over.
            encoding_strides: sequence of strides for the convolutional
                layers for the encoder.
            encoding_maxpool_strides: strides for the max pooling layers for
                the encoder.
            decoding_channels: sequence of channels to build the decoder.
            decoding_kernels: sequence of kernel sizes to build the decoder.
            decoding_strides: sequence of strides for the transpose
                convolutional layers for the decoder.
            decoding_padding: sequence of paddings for the convolutional
                layers for the decoder.
        """
        super().__init__()
        self.in_channels = in_channels

        encoding_channels = [self.in_channels] + encoding_channels
        decoding_channels = decoding_channels + [self.in_channels]

        encoder = []
        for i in range(len(encoding_channels) - 1):
            encoder += ConvolutionalAutoEncoder.encoder_layer(
                in_channels=encoding_channels[i],
                out_channels=encoding_channels[i + 1],
                kernel_size=encoding_kernels[i],
                maxpool_window=encoding_maxpool_windows[i],
                stride=encoding_strides[i],
                maxpool_stride=encoding_maxpool_strides[i]
            )
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        for i in range(len(decoding_channels) - 1):
            decoder += ConvolutionalAutoEncoder.decoder_layer(
                in_channels=decoding_channels[i],
                out_channels=decoding_channels[i + 1],
                kernel_size=decoding_kernels[i],
                stride=decoding_strides[i],
                padding=decoding_padding[i],
                activation=(
                    "ReLU" if i < len(decoding_channels) - 2 else "Sigmoid"
                )
            )
        self.decoder = nn.Sequential(*decoder)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional autoencoder.
        Input:
            X: input image with shape BCHW.
        Returns:
            Reconstructed image with shape BCHW.
        """
        return self.decoder(self.encoder(X))

    @staticmethod
    def encoder_layer(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        maxpool_window: int = 2,
        stride: int = 1,
        maxpool_stride: int = 1,
        padding: int = 1
    ) -> Sequence[nn.Module]:
        """
        Builds a single convolutional layer for the encoder.
        Input:
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: size of the convolving kernel. Default 3.
            maxpool_window: size of the window to take a max over. Default 2.
            stride: stride of the convolution. Default 1.
            maxpool_stride: stride of the maxpooling window. Default 1.
            padding: padding on the boundaries of the input. Default 1.
        Returns:
            A sequence of a 2D convolution, ReLU activation, and maxpooling.
        """
        return [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_window, stride=maxpool_stride)
        ]

    @staticmethod
    def decoder_layer(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        activation: Optional[str] = "ReLU"
    ) -> Sequence[nn.Module]:
        """
        Builds a single convolutional layer for the decoder.
        Input:
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: size of the convolving kernel. Default 3.
            stride: stride of the convolution. Default 1.
            padding: padding on the boundaries of the input. Default 1.
        Returns:
            A sequence of a 2D transpose convolution and ReLU activation.
        """
        if activation is None:
            activation = nn.Identity()
        elif activation.lower() == "relu":
            activation = nn.ReLU()
        elif activation.lower() == "sigmoid":
            activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
                f"Unrecognized activation function {activation}."
            )
        return [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            activation
        ]


class ConvAutoEncLightningModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        """
        Args:
            lr: learning rate. Default 0.001.
            weight_decay: weight decay. Default 1e-6.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = ConvolutionalAutoEncoder(**kwargs)
        self.loss = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(gaussian_kernel=True)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional autoencoder.
        Input:
            X: input image with shape BCHW.
        Returns:
            Reconstructed image with shape BCHW.
        """
        return self.model(X)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encodes an input from image space into the latent space.
        Input:
            X: input image with shape BCHW.
        Returns:
            Encoded image in the latent space.
        """
        return self.model.encoder(X)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes an input latent space vector back into image space.
        Input:
            z: input batch of latent space vectors.
        Returns:
            Reconstructed image with shape BCHW.
        """
        return self.model.decoder(z)

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Defines the training step for model training.
        Input:
            batch: an input batch of images and associated labels.
            batch_idx: index of the batch.
        Returns:
            The reconstruction loss of the model over the batch.
        """
        X, _ = batch
        loss = self.loss(self(X), X)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Defines the validation step for model training.
        Input:
            batch: an input batch of images and associated labels.
            batch_idx: index of the batch.
        Returns:
            Validation step statistics over the batch.
        """
        X, _ = batch
        recon = self(X)
        loss = self.loss(recon, X)
        ssim = self.ssim(recon, X)
        psnr = self.psnr(recon, X)
        self.log("val_mse", loss, prog_bar=True, sync_dist=True)
        self.log("val_ssim", ssim, prog_bar=True, sync_dist=True)
        self.log("val_psnr", psnr, prog_bar=True, sync_dist=True)
        return {"mse": loss.item(), "ssim": ssim.item(), "psnr": psnr.item()}

    def test_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Defines the validation step for model training.
        Input:
            batch: an input batch of images and associated labels.
            batch_idx: index of the batch.
        Returns:
            Validation step statistics over the batch.
        """
        X, _ = batch
        recon = self(X)
        return {
            "input": X.detach(),
            "recon": recon.detach(),
            "mse": self.loss(recon, X).item(),
            "ssim": self.ssim(recon, X).item(),
            "psnr": self.psnr(recon, X).item()
        }

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure model optimizer.
        Input:
            None.
        Returns:
            The model optimizer.
        """
        return optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

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
