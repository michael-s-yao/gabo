"""
Defines and trains a surrogate image energy estimator.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as thv
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Sequence, Union

sys.path.append(".")
from mnist.vae import VAE
from experiment.utility import seed_everything, get_device, plot_config


class Surrogate(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        hidden_dims: Sequence[int] = [256, 64]
    ):
        """
        Args:
            in_dim: number of flattened input dimensions into surrogate.
            hidden_dims: hidden dimensions of the MLP surrogate model.
        """
        super().__init__()
        self.in_dim, self.hidden_dims = in_dim, hidden_dims
        self.hidden_dims = [self.in_dim] + self.hidden_dims + [1]
        self.model = []
        for i in range(len(self.hidden_dims) - 1):
            self.model += [
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                nn.ReLU() if i < len(self.hidden_dims) - 2 else nn.Sigmoid()
            ]
        self.model = nn.Sequential(*self.model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Estimates the energy of the decoded image from the the latent space.
        Input:
            z: latent space point(s) with shape N or BN where N is in_dim.
        Returns:
            Estimates of the energy of the decoded image with shape 1 or B1.
        """
        return self.model(z)


def fit(
    root: Union[Path, str] = "./mnist/data",
    vae: Union[Path, str] = "./mnist/checkpoints/mnist_vae.pt",
    batch_size: int = 128,
    lr: float = 1e-3,
    num_epochs: int = 50,
    device: torch.device = torch.device("cpu"),
    savepath: Union[Path, str] = "./mnist/checkpoints/mnist_surrogate.pt"
) -> None:
    """
    Main surrogate energy estimator training function.
    Input:
        root: root directory containing the MNIST dataset.
        vae: path containing the trained MNIST VAE model.
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
        num_workers=0
    )

    autoencoder = VAE().to(device)
    autoencoder.load_state_dict(torch.load(vae))
    autoencoder.eval()

    model = Surrogate(autoencoder.hidden_dims[-1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        with tqdm(train, desc=f"Epoch {epoch}", leave=False) as pbar:
            for batch_idx, (X, _) in enumerate(pbar):
                X = X.to(device).flatten(start_dim=(X.ndim - 3))
                optimizer.zero_grad()
                z, _, _ = autoencoder.encode(X)
                ypred = torch.squeeze(model(z), dim=-1)
                y = torch.mean(torch.square(autoencoder.decode(z)), dim=-1)
                loss = F.mse_loss(ypred, y)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(train_mse=loss.item())
    torch.save(model.state_dict(), savepath)


def test(
    checkpoint: Union[Path, str],
    vae: Union[Path, str] = "./mnist/checkpoints/mnist_vae.pt",
    root: Union[Path, str] = "./mnist/data",
    batch_size: int = 128,
    savepath: Optional[Union[Path, str]] = None,
    device: torch.device = torch.device("cpu"),
    num_plot: int = 16,
    num_per_row: int = 4
) -> None:
    model = Surrogate().to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    autoencoder = VAE().to(device)
    autoencoder.load_state_dict(torch.load(vae))
    autoencoder.eval()

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

    error = []
    for X, _ in test:
        X = X.to(device).flatten(start_dim=(X.ndim - 3))
        z, _, _ = autoencoder.encode(X)
        ypred = torch.squeeze(model(z), dim=-1)
        y = torch.mean(torch.square(autoencoder.decode(z)), dim=-1)
        error += (ypred - y).detach().cpu().numpy().tolist()

    plot_config()
    plt.figure(figsize=(20, 6))
    plt.hist(error, bins=np.linspace(-0.015, 0.015, 100), color="k", alpha=0.5)
    plt.xlabel("Surrogate Energy - Oracle Energy")
    plt.ylabel("Count")
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
    seed = parser.parse_args().seed

    torch.set_default_dtype(torch.float64)
    seed_everything(seed)
    device = get_device("cpu")
    model_path = f"./mnist/checkpoints/mnist_surrogate_{seed}.pt"
    fit(savepath=model_path, device=device)
    # test(
    #     checkpoint=model_path,
    #     device=device,
    #     savepath="./mnist/docs/surrogate.png"
    # )
