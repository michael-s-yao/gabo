"""
Defines and trains a MNIST VAE model.

Author(s):
    Michael Yao

Adapted from the @pytorch examples GitHub repository at
https://github.com/pytorch/examples/blob/main/vae/main.py

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import matplotlib.pyplot as plt
import math
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as thv
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Sequence, Tuple, Union

sys.path.append(".")
from experiment.utility import seed_everything, get_device, plot_config


class VAE(nn.Module):
    def __init__(
        self,
        in_dim: int = 784,
        hidden_dims: Sequence[int] = [256, 64, 16]
    ):
        """
        Args:
            in_dim: number of flattened input dimensions into the VAE.
            hidden_dims: hidden dimensions of the encoder and decoder.
        """
        super().__init__()
        self.in_dim, self.hidden_dims = in_dim, hidden_dims
        self.hidden_dims = [self.in_dim] + self.hidden_dims

        self.encoder, self.decoder = [], []
        for i in range(len(self.hidden_dims) - 1):
            if i < len(self.hidden_dims) - 2:
                self.encoder += [
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                    nn.ReLU()
                ]
            else:
                self.mu = nn.Linear(
                    self.hidden_dims[i], self.hidden_dims[i + 1]
                )
                self.logvar = nn.Linear(
                    self.hidden_dims[i], self.hidden_dims[i + 1]
                )
            self.decoder += [
                nn.Linear(self.hidden_dims[-i - 1], self.hidden_dims[-i - 2]),
                nn.ReLU() if i < len(self.hidden_dims) - 2 else nn.Sigmoid()
            ]
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

    def encode(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Encodes an image or batch of images into the VAE latent space.
        Input:
            X: input image CHW or batch of images BCHW.
        Returns:
            z: a vector of point(s) from the VAE latent space (N or BN), where
                N is the dimensions of the VAE latent space.
            mu: tensor of means in the latent space (N or BN).
            logvar: tensor of log variances in the latent space (N or BN).
        """
        h = self.encoder(X)
        mu, logvar = self.mu(h), self.logvar(h)
        return self.reparameterize(mu, logvar), mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs a tensor of point(s) from the VAE latent space into the
        flattened image space.
        Input:
            z: a vector of point(s) from the VAE latent space (N or BN), where
                N is the dimensions of the VAE latent space.
        Returns:
            Reconstructed flattened image C*H*W or batch of images (B(C*H*W)).
        """
        return self.decoder(z)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparametrization trick to sample from the VAE latent space.
        Input:
            mu: tensor of means in the latent space (N or BN), where N is the
                dimensions of the VAE latent space.
            logvar: tensor of log variances in the latent space (N or BN),
                where N is the dimensions of the VAE latent space.
        Returns:
            A vector of point(s) from the VAE latent space (N or BN).
        """
        std = torch.exp(0.5 * logvar)
        return mu + (torch.randn_like(std) * std)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass through the variational autoencoder.
        Input:
            X: input image CHW or batch of images BCHW.
        Returns:
            recon: reconstructed flattened image C*H*W or batch of images
                (B(C*H*W)).
            mu: tensor of means in the latent space (N or BN), where N is the
                dimensions of the VAE latent space.
            logvar: tensor of log variances in the latent space (N or BN),
                where N is the dimensions of the VAE latent space.
        """
        z, mu, logvar = self.encode(X.view(-1, self.in_dim))
        return self.decode(z), mu, logvar


def VAELoss(
    recon: torch.Tensor,
    X: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor
) -> torch.Tensor:
    """
    Defines the VAE loss function for VAE training.
    Input:
        recon: reconstructed flattened image C*H*W or batch of images
            (B(C*H*W)).
        X: input image CHW or batch of images BCHW.
        mu: tensor of means in the latent space (N or BN), where N is the
            dimensions of the VAE latent space.
        logvar: tensor of log variances in the latent space (N or BN),
            where N is the dimensions of the VAE latent space.
    """
    BCE = F.binary_cross_entropy(recon, X.view(recon.size()), reduction="sum")
    KLD = -0.5 * torch.sum(1.0 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
    return BCE + KLD


def fit(
    root: Union[Path, str] = "./mnist/data",
    batch_size: int = 128,
    lr: float = 1e-3,
    num_epochs: int = 50,
    device: torch.device = torch.device("cpu"),
    savepath: Union[Path, str] = "./mnist/checkpoints/mnist_vae.pt"
) -> None:
    """
    Main VAE training function.
    Input:
        root: root directory containing the MNIST dataset.
        batch_size: batch size. Default 128.
        lr: learning rate. Default 1e-3.
        num_epochs: number of epochs to train. Default 50.
        device: device for model training. Default CPU.
        savepath: path to save the model weights to.
    Returns:
        None.
    """
    train = DataLoader(
        thv.datasets.MNIST(
            root,
            train=True,
            download=True,
            transform=thv.transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator(device=device)
    )

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        with tqdm(train, desc=f"Epoch {epoch}", leave=False) as pbar:
            for batch_idx, (X, _) in enumerate(pbar):
                X = X.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(X)
                loss = VAELoss(recon, X, mu, logvar)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(train_loss=loss.item())
    torch.save(model.state_dict(), savepath)


def test(
    checkpoint: Union[Path, str],
    mode: str,
    root: Union[Path, str] = "./mnist/data",
    batch_size: int = 128,
    savepath: Optional[Union[Path, str]] = None,
    device: torch.device = torch.device("cpu"),
    num_plot: int = 16,
    num_per_row: int = 4
) -> None:
    """
    Main VAE evaluation function.
    Input:
        checkpoint: path to the trained model checkpoint.
        mode: one of [`recon`, `sample`]. If `recon`, then the image
            reconstruction efficacy is evaluated on the MNIST test set.
            If `sample`, then the image generation efficacy from latent space
            sampling is evaluated.
        root: root directory containing the MNIST dataset.
        savepath: path to save the image plots to.
        device: device for model training. Default CPU.
        num_plot: total number of images to plot.
        num_per_row: total number of images to plot per row of the figure.
    Returns:
        None.
    """
    model = VAE().to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    test = DataLoader(
        thv.datasets.MNIST(
            root,
            train=False,
            download=True,
            transform=thv.transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    plot_config()
    if mode.lower() == "recon":
        fig, axs = plt.subplots(
            math.ceil(num_plot / num_per_row),
            2 * num_per_row,
            figsize=(20, 10)
        )
        fig.tight_layout()
        axs = axs.flatten()
        X, _ = next(iter(test))
        X = X.to(device)
        recon = model(X)[0].reshape(X.size())
        for i, img in enumerate(X[:num_plot]):
            axs[(2 * i)].imshow(img[0].detach().cpu().numpy(), cmap="gray")
            axs[(2 * i)].axis("off")
            axs[(2 * i) + 1].imshow(
                recon[i, 0].detach().cpu().numpy(), cmap="gray"
            )
            axs[(2 * i) + 1].axis("off")
        if savepath is None:
            plt.show()
        else:
            plt.savefig(
                savepath, dpi=600, transparent=True, bbox_inches="tight"
            )
        plt.close()
    elif mode.lower() == "sample":
        fig, axs = plt.subplots(
            math.ceil(num_plot / (2 * num_per_row)),
            2 * num_per_row,
            figsize=(20, 5)
        )
        fig.tight_layout()
        axs = axs.flatten()
        X = model.decode(torch.randn((16, model.hidden_dims[-1])).to(device))
        X = X.reshape(-1, *tuple(next(iter(test))[0].size())[1:])
        for i, img in enumerate(X.detach().cpu().numpy()):
            axs[i].imshow(img[0], cmap="gray")
            axs[i].axis("off")
        if savepath is None:
            plt.show()
        else:
            plt.savefig(
                savepath, dpi=600, transparent=True, bbox_inches="tight"
            )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST VAE Training")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device. Default CPU."
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_default_device(args.device)
    seed_everything(
        args.seed, use_deterministic=("cuda" not in args.device.lower())
    )
    device = get_device(args.device)
    model_path = f"./mnist/checkpoints/mnist_vae_{args.seed}.pt"
    fit(savepath=model_path, device=device)
    test(
        model_path,
        mode="recon",
        savepath=f"./mnist/docs/vae_recon_{args.seed}.png"
    )
    test(
        model_path,
        mode="sample",
        savepath=f"./mnist/docs/vae_sample_{args.seed}.png"
    )
