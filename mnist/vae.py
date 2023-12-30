"""
Trains and evaluates a fully-connected VAE model for the MNIST dataset.

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
import torch.optim as optim
import torchvision as thv
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Union

sys.path.append(".")
from experiment.utility import seed_everything, get_device, plot_config
from models.vae import VAE, VAELoss


def build_args() -> argparse.Namespace:
    """
    Defines the experimental arguments for MNIST VAE model training.
    Input:
        None.
    Returns:
        A namespace containing the experimental argument values.
    """
    parser = argparse.ArgumentParser(description="MNIST VAE Training")

    parser.add_argument(
        "--root",
        type=str,
        default="data/mnist",
        help="Root directory containing the MNIST dataset. Default data/mnist"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size. Default 128."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate. Default 1e-3."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs. Default 50."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default="./checkpoints/mnist_vae.pt",
        help="Savepath for VAE checkpoint. Default ./checkpoints/mnist_vae.pt"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device. Default CPU."
    )

    return parser.parse_args()


def fit():
    torch.set_default_dtype(torch.float64)
    args = build_args()
    seed_everything(
        args.seed, use_deterministic=("cuda" not in args.device.lower())
    )
    device = get_device(args.device)

    train = DataLoader(
        thv.datasets.MNIST(
            args.root,
            train=True,
            download=True,
            transform=thv.transforms.ToTensor()
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator(device=device)
    )

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.num_epochs):
        with tqdm(train, desc=f"Epoch {epoch}", leave=False) as pbar:
            for batch_idx, (X, _) in enumerate(pbar):
                X = X.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(X)
                loss = VAELoss(recon, X, mu, logvar)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(train_loss=loss.item())
    torch.save(model.state_dict(), args.savepath)


def test(
    checkpoint: Union[Path, str],
    mode: str,
    root: Union[Path, str] = "./data/mnist",
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
    fit()
