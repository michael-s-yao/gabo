"""
Script to explore the distribution of objective values in the training and
test distributions.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as thv
from pathlib import Path
from typing import Optional, Union

sys.path.append(".")
from mnist.vae import VAE


def main(
    checkpoint: Union[Path, str],
    root: Union[Path, str] = "./mnist/data",
    savepath: Optional[Union[Path, str]] = None,
    device: torch.device = torch.device("cpu")
):
    test = thv.datasets.MNIST(
        root,
        train=False,
        download=True,
        transform=thv.transforms.ToTensor()
    )

    model = VAE().to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    y = []
    for X, _ in test:
        y.append(torch.mean(torch.square(model(X)[0])).item())

    plt.plot(figsize=(10, 5))
    bins = np.linspace(0, 1, 100)
    plt.hist(y, bins=bins, alpha=0.5, label="Test Dataset", color="k")
    plt.xlabel(r"Energy $||x||_2^2$")
    plt.ylabel("Count")
    percentile_95 = sorted(y)[round(0.95 * len(y))]
    percentile_100 = sorted(y)[-1]
    plt.axvline(
        percentile_95,
        color="k",
        alpha=0.5,
        ls="--",
        label=f"95th Percentile: {percentile_95:.3f}"
    )
    plt.axvline(
        percentile_100,
        color="k",
        alpha=1.0,
        ls="--",
        label=f"100th Percentile: {percentile_100:.3f}"
    )
    plt.legend()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=600, transparent=True, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main(
        checkpoint="./mnist/checkpoints/mnist_vae.pt",
        savepath="./mnist/docs/distribution.png"
    )
